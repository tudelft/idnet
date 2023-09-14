from collections import namedtuple
from ..utils.logger import Logger
from ..loader.loader_dsec import assemble_dsec_sequences, rec_train_collate, \
    train_collate
from ..utils.helper_functions import move_batch_to_cuda
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from contextlib import contextmanager
from .eval import Evaluator


class Test:
    fm = namedtuple("frame_metric", ["n_frame", "value"])

    def __init__(self, test_spec):
        self.spec = test_spec
        self.device = self.spec.data_loader.gpu
        self.data_loader = self.configure_dataloader()
        self.evaluator = Evaluator(test_spec.metrics)
        self.logger = Logger(self.spec.logger, self.spec.name)


    def configure_dataloader(self):
        valid_set = assemble_dsec_sequences(
            self.spec.dataset.common.data_root,
            include_seq=self.spec.dataset.val.seq,
            require_gt=True,
            config=self.spec.dataset.val,
            representation_type=self.spec.dataset.get(
                "representation_type", None),
        )
        collate_fn = rec_train_collate if self.spec.dataset.val.get("recurrent", False) \
            else train_collate
        assert self.spec.data_loader.args.shuffle is False, \
            "shuffle must be false for val run."
        if isinstance(valid_set, list):
            val_dataloader = [DataLoader(
                seq, collate_fn=collate_fn, **self.spec.data_loader.args
            ) for seq in valid_set]
        else:
            val_dataloader = DataLoader(
                valid_set, collate_fn=collate_fn, **self.spec.data_loader.args
            )
        return val_dataloader

    def assemble_postprocess_fn(self, postprocess):
        pass
        return None

    @staticmethod
    def configure_model_forward_fn(model):
        return model.forward

    @staticmethod
    def configure_forward_pass_fn(forward_fn):
        return forward_fn

    def configure_model(self, model):
        pass

    def cleanup_model(self, model):
        pass

    @contextmanager
    def evaluate_model(self, model):
        istrain = model.training
        original_device = next(model.parameters()).device
        try:
            model.cuda(self.device)
            self.configure_model(model)
            model.eval()
            yield model
        finally:
            self.cleanup_model(model)
            model.to(original_device)
            model.train(mode=istrain)

    @torch.no_grad()
    def execute_test(self, model_eval):
        try:
            with self.evaluate_model(model_eval) as model:
                model_forward_fn = self.configure_model_forward_fn(model)
                forward_pass_fn = self.configure_forward_pass_fn(
                    model_forward_fn)
                with self.logger.log_test(model) as log_path:
                    if not isinstance(self.data_loader, list):
                        self.data_loader = [self.data_loader]
                    for seq_loader in self.data_loader:
                        with self.logger.record_sequence(seq_loader.dataset.seq_name,
                                                         log_path) as rec:
                            for idx, batch in enumerate(tqdm(seq_loader, position=1)):
                                batch = move_batch_to_cuda(batch, self.device)
                                out = forward_pass_fn(batch)
                                self.evaluator.evaluate(batch, out, idx)
                                rec.log_tensors(batch, out, idx)
                            rec.log_metrics(self.evaluator.results)

            return 0, self.logger.summary()
        except Exception as e:
            return e, None
