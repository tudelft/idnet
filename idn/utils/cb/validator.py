from pickle import NONE
from ..callbacks import Callback
from ..validation import Validator
from warnings import warn

class CBValidator(Callback):
    def __init__(self, config):
        super().__init__()
        self.validator = None
        self.config = config
        self.logger = None

    def run_validation(self, caller, sanity_check_run=False):
        validator = Validator(caller.config.validation)
        results = validator(caller.model)
        if self.logger is not None and not sanity_check_run:
            self.logger.log_dict_at_step(results)
        else:
            print(results)

    def on_train_begin(self, caller):
        self.logger = caller.logger

    def on_batch_end(self, caller):
        if caller.step == self.config.get("sanity_run_step", None):
            self.run_validation(caller, sanity_check_run=True)
        if self.time_to_validate(step=caller.step):
            self.run_validation(caller)

    def on_epoch_end(self, caller):
        if self.time_to_validate(epoch=caller.epoch):
            self.run_validation(caller)

    def time_to_validate(self, step=None, epoch=None):
        if self.config.frequency_type == "epoch":
            return epoch != 0 and epoch % self.config.frequency == 0 \
                if epoch is not None else False
        elif self.config.frequency_type == "step":
            return step != 0 and step % self.config.frequency == 0 \
                if step is not None else False
        else:
            warn(f"Frequency type {self.config.frequency_type} not recognized.")
            return False
