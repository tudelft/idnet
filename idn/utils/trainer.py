import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from types import GeneratorType
from collections import namedtuple
from torchinfo import summary
from torch.optim.lr_scheduler import OneCycleLR

from .torch_environ import config_torch
from .helper_functions import move_batch_to_cuda
from .model_utils import get_model_by_name
from .loss_utils import compute_seq_loss, get_loss_fn_by_name
from .validation import Validator
from .callbacks import CallbackBridge
from .exp_tracker import ExpTracker
from .retrieval_fn import get_retreival_fn
from ..loader.loader_dsec import (
    Sequence,
    RepresentationType,
    DatasetProvider,
    assemble_dsec_sequences,
    assemble_dsec_test_set,
    train_collate,
    rec_train_collate
)
from ..loader.loader_mvsec import (
    MVSEC,
    MVSECRecurrent
)


class Trainer(CallbackBridge):
    def __init__(self, config, model=None):
        super().__init__()
        self.config = config
        config_torch(config.torch)
        self.model = model if model is not None else \
            get_model_by_name(config.model.name, config.model)
        if config.model.get("pretrain_ckpt", None):
            self.resume_model_from_ckpt(config.model.pretrain_ckpt)

        if not self.config.get("eval_only", False):
            self.train_dataloader = self.configure_train_dataloader()
            self.configure_loss()
            self.optimizer = self.configure_optimizer()
            if config.optim.get("scheduler", None):
                self.scheduler = OneCycleLR(
                    self.optimizer, max_lr=self.config.optim.lr,
                    steps_per_epoch=len(self.train_dataloader), 
                    epochs=self.config.num_epoch,
                    pct_start=0.05, cycle_momentum=False, anneal_strategy='linear',
                )
            else:
                self.scheduler = None

        self.epoch = 0
        self.step = 0
        self.batches_seen = 0
        self.samples_seen = 0
        if config.get("resume_ckpt", None):
            resume_only_model = config.get("finetune", False)
            self.resume_from_ckpt(config.resume_ckpt, resume_only_model)
        self.logger = self.configure_tracker()
        
        self.configure_callbacks(config.callbacks)

        self.execute_callbacks("on_init_end")

    def configure_train_dataloader(self):
        if self.config.dataset.dataset_name == "dsec":
            train_set = assemble_dsec_sequences(
                self.config.dataset.common.data_root,
                exclude_seq=set(
                    [val_seq for x in self.config.get("validation", dict()).values() for val_seq in x.dataset.val.seq]),
                require_gt=True,
                config=self.config.dataset.train,
                representation_type=self.config.dataset.get("representation_type", None),
                num_bins=self.config.dataset.get("num_voxel_bins", None)
            )

        elif self.config.dataset.dataset_name == "mvsec":
            train_set = MVSEC("outdoor_day2", num_bins=self.config.dataset.get("num_voxel_bins", None), dt=None) #20Hz
        elif self.config.dataset.dataset_name == "mvsec_recurrent":
            train_set = MVSECRecurrent("outdoor_day2", augment=False, 
                                       sequence_length=self.config.dataset.train.sequence_length)
        else:
            raise NotImplementedError
        collate_fn = rec_train_collate \
            if hasattr(self.config.dataset.train, "recurrent") \
            and self.config.dataset.train.recurrent else train_collate
        return DataLoader(
            train_set, collate_fn=collate_fn, **self.config.data_loader.train.args)

    def configure_optimizer(self):
        if self.config.optim.optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.config.optim.lr)
        elif self.config.optim.optimizer == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.config.optim.lr)
        else:
            raise NotImplementedError

    def resume_model_from_ckpt(self, ckpt):
        ckpt = torch.load(ckpt)
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        elif "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            try:
                self.model.load_state_dict(ckpt)
            except:
                raise ValueError("Invalid checkpoint")
    
    def resume_from_ckpt(self, ckpt, resume_only_model=False):
        ckpt = torch.load(ckpt, map_location='cpu')
        self.model.load_state_dict(ckpt['model_state_dict'])
        if not resume_only_model:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.epoch = ckpt['epoch']
            if self.scheduler and 'scheduler_state_dict' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if "tracker" in ckpt:
            self.logged_tracker = ckpt['tracker']

    def configure_tracker(self):
        return ExpTracker()

    def configure_loss(self):
        lc = namedtuple("loss_config",
                        ["retrieval_fn", "loss_fn", "weight", "seq_weight", "seq_norm"])
        self.loss_config = dict()
        for quantity, config in self.config.loss.items():
            self.loss_config[quantity] = lc(
                get_retreival_fn(quantity),
                get_loss_fn_by_name(config.loss_type),
                config.get("weight", 1.0),
                config.get("seq_weight", None),
                config.get("seq_norm", False))

    def train_epoch(self):
        self.execute_callbacks("on_epoch_begin")
        self.model.train()
        self.model.cuda(self.config.data_loader.train.gpu)
        # This is necessary to sync optimizer parameter device with model device
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        for batch in tqdm(self.train_dataloader):
            self.execute_callbacks("on_batch_begin")
            self.optimizer.zero_grad()
            batch = move_batch_to_cuda(
                batch, self.config.data_loader.train.gpu)
            out = self.model(batch)
            if isinstance(out, GeneratorType):
                loss_item = []
                for i, ret in enumerate(out):
                    seq_len = len(ret["flow_trajectory"])
                    loss, loss_breakdown = self.compute_loss(
                        ret, batch[i*seq_len:(i+1)*seq_len])
                    loss.backward()
                    loss_item.append(loss.detach().item())
                self.loss = sum(loss_item)/len(loss_item)
                self.loss_1 = \
                loss_item[0]
                for loss_type, l in loss_breakdown.items():
                    setattr(self, 'loss_'+loss_type, l.item())

            else:
                self.loss, _ = self.compute_loss(out, batch)
                self.loss.backward()
            self.execute_callbacks("on_step_begin")
            self.optimizer.step()
            self.execute_callbacks("on_step_end")
            self.execute_callbacks("on_batch_end")
            self.step += 1
            if self.scheduler:
                self.scheduler.step()
                self.lr = self.scheduler.get_last_lr()[0]
        self.execute_callbacks("on_epoch_end")
        self.epoch += 1

    def compute_loss(self, ret, batch):
        loss = dict()
        total_loss = 0
        for quantity, config in self.loss_config.items():
            estimate, ground_truth = config.retrieval_fn(ret, batch)
            if isinstance(estimate, list):
                loss_fn = lambda estimate, ground_truth: \
                    compute_seq_loss(config.seq_weight, config.loss_fn,
                                     estimate, ground_truth)
            else:
                loss_fn = config.loss_fn
            
            loss[quantity] = loss_fn(estimate, ground_truth)
            total_loss += config.weight * loss[quantity]
        return total_loss, loss

    def fit(self, epochs=None):
        num_epochs = epochs if epochs is not None else self.config.num_epoch
        self.execute_callbacks("on_train_begin")
        try:
            while self.epoch < num_epochs:
                self.train_epoch()
        except:
            raise Exception("Training failed")
        finally:
            self.execute_callbacks("on_train_end")

