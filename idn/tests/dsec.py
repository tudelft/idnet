
from torch.utils.data import DataLoader
from ..loader.loader_dsec import assemble_dsec_sequences, assemble_dsec_test_set, rec_train_collate, \
    train_collate
from ..loader.loader_mvsec import MVSEC
from .test import Test
import torch
from tqdm import tqdm
from ..utils.helper_functions import move_batch_to_cuda
import time


class TestCO(Test):
    def __init__(self, test_spec):
        super().__init__(test_spec)

    def configure_dataloader(self):
        self.spec.data_loader.args.batch_size = 1
        return super().configure_dataloader()

    @staticmethod
    def configure_model_forward_fn(model):
        return super(type(model), model).forward

    @staticmethod
    def configure_forward_pass_fn(model_forward_fn):
        return lambda list_batch: model_forward_fn(list_batch[0])

    def configure_model(self, model):
        self.original_co_mode = model.co_mode
        model.co_mode = True
        model.reset_continuous_flow()

    def cleanup_model(self, model):
        model.co_mode = self.original_co_mode
        model.reset_continuous_flow()


class TestRESET(Test):
    def __init__(self, test_spec):
        super().__init__(test_spec)

    @staticmethod
    def configure_model_forward_fn(model):
        return model.forward_inference

    def configure_dataloader(self):
        return super().configure_dataloader()

    def configure_model(self, model):
        self.original_co_mode = model.co_mode
        model.co_mode = False

    def cleanup_model(self, model):
        model.co_mode = self.original_co_mode


class TestNONREC(Test):
    def __init__(self, test_spec):
        super().__init__(test_spec)

    def configure_dataloader(self):
        return super().configure_dataloader()


def assemble_dsec_test_cls(test_type_name=None):
    test_cls = {
        "co": TestCO,
        "reset": TestRESET,
    }
    test_type = test_cls.get(test_type_name, Test)

    class TestDSEC(test_type):

        # DSEC Test set
        def configure_dataloader(self):
            test_set = assemble_dsec_test_set(self.spec.dataset.common.test_root,
                                              seq_len=self.spec.dataset.val.get(
                                                  "sequence_length", None),
                                              representation_type=self.spec.dataset.get("representation_type", None))
            if isinstance(test_set, list):
                val_dataloader = [DataLoader(
                    seq, batch_size=1, shuffle=False, num_workers=0,
                ) for seq in test_set]
            else:
                val_dataloader = DataLoader(
                    test_set, batch_size=1, shuffle=False, num_workers=0,
                )
            return val_dataloader

        @torch.no_grad()
        def execute_test(self, model_eval, save_all=False):
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
                                    if isinstance(batch, list):
                                        assert 'save_submission' in batch[-1]
                                    else:
                                        assert 'save_submission' in batch
                                        # for non-recurrent loading, we skip samples
                                        # not for eval
                                        if not batch['save_submission'].cpu().item() and not save_all:
                                            continue

                                    batch = move_batch_to_cuda(
                                        batch, self.device)
                                    start_time = time.perf_counter()

                                    out = forward_pass_fn(batch)
                                    end_time = time.perf_counter()
                                    #print(f"Forward pass time: {(end_time - start_time):.4f}")
                                    self.evaluator.evaluate(batch, out, idx)
                                    rec.log_tensors(batch, out, idx, save_all)
                                rec.log_metrics(self.evaluator.results)

                    self.pack_submission_to_zip(log_path)

                return 0, self.logger.summary()
            except Exception as e:
                return e, None

        @staticmethod
        def pack_submission_to_zip(log_path):
            import zipfile
            import os
            import tempfile
            from pathlib import Path
            from ..check_submission import check_submission
            with zipfile.ZipFile(os.path.join(log_path, 'submission.zip'), 'w') as zip:
                for seq in os.listdir(log_path):
                    if os.path.isdir(os.path.join(log_path, seq, "submission")):
                        for file in os.listdir(os.path.join(log_path, seq, "submission")):
                            zip.write(os.path.join(
                                log_path, seq, "submission", file), os.path.join(seq, file))
            with tempfile.TemporaryDirectory() as tempdir:
                zipfile.ZipFile(os.path.join(
                    log_path, 'submission.zip')).extractall(tempdir)
                assert check_submission(
                    Path(tempdir), Path("data/test_forward_optical_flow_timestamps")), \
                    "submission did not pass check"
    return TestDSEC

class TestMVSEC(Test):
    def __init__(self, test_spec):
        super().__init__(test_spec)

    def configure_dataloader(self):
        valid_set = MVSEC("outdoor_day1", filter=(4356, 4706), num_bins=15, augment=False)

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
