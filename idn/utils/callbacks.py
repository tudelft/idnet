class CallbackBridge:
    def __init__(self):
        self.callbacks = list()

    def configure_callbacks(self, config):
        def get_cb_from_name(callback):
            if callback == "logger":
                from .cb.logger import CBLogger
                return CBLogger
            elif callback == "validator":
                from .cb.validator import CBValidator
                return CBValidator
        if config is not None:
            for callback, callback_config in config.items():
                if "enable" in callback_config.keys():
                    self.callbacks.append(
                        get_cb_from_name(callback)(callback_config))

    def execute_callbacks(self, callback_type):
        for callback in sorted(self.callbacks,
                               key=lambda x: x.call_order[callback_type]):
            getattr(callback, callback_type)(self)


class Callback:
    callback_types = [
        "on_init_end",
        "on_train_begin",
        "on_train_end",
        "on_epoch_begin",
        "on_epoch_end",
        "on_batch_begin",
        "on_batch_end",
        "on_step_begin",
        "on_step_end",
    ]

    def __init__(self):
        self.call_order = dict.fromkeys(self.callback_types, 0)

    def on_init_end(self, caller):
        pass

    def on_train_begin(self, caller):
        pass

    def on_train_end(self, caller):
        pass

    def on_epoch_begin(self, caller):
        pass

    def on_epoch_end(self, caller):
        pass

    def on_batch_begin(self, caller):
        pass

    def on_batch_end(self, caller):
        pass

    def on_step_begin(self, caller):
        pass

    def on_step_end(self, caller):
        pass
