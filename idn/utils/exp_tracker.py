class ExpTracker:
    def __init__(self) -> None:
        self.log_dir = None

    def on_init_end(self, *args, **kwargs):
        pass

    def on_exp_begin(self, *args, **kwargs):
        pass

    def log_dict_at_step(self, dict, step=None):
        pass

    def summary(self):
        return {
            "id": "exp_tracker",
        }
