import os
import torch
from ..callbacks import Callback

class CBLogger(Callback):
    def __init__(self, logger_config, logger_type="exp_tracker"):
        super().__init__()
        self.logger_config = logger_config
        self.logger = None

    def on_init_end(self, caller):
        self.logger = caller.logger
        try:
            run_id = caller.logged_tracker["run_id"]
        except:
            run_id = None
        self.logger.on_init_end(caller.config, run_id=run_id)

    def on_train_begin(self, caller):
        self.logger.on_exp_begin(caller.model)

    def on_batch_end(self, caller):
        log_dict = dict()
        for key in self.logger_config.log_keys["batch_end"]:
            log_dict[key] = getattr(caller, key, None)
        self.logger.log_dict_at_step(log_dict)

    def on_epoch_end(self, caller):
        if self.logger.log_dir is not None:
            torch.save({
                'epoch': caller.epoch,
                'model_state_dict': caller.model.state_dict(),
                'optimizer_state_dict': caller.optimizer.state_dict(),
                'scheduler_state_dict': caller.scheduler.state_dict() \
                    if caller.scheduler is not None else None,
                'loss': caller.loss,
                'tracker': self.logger.summary(),
            }, os.path.join(self.logger.log_dir, "model.ckpt"))
