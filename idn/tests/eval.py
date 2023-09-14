from collections import namedtuple
from ..utils.retrieval_fn import get_retreival_fn
from ..model.loss import sparse_lnorm, compute_npe

fm = namedtuple("frame_metric", ["n_frame", "value"])


class Evaluator:
    def __init__(self, spec):
        spec = dict() if spec is None else spec
        self.spec = spec
        self.assemble_eval_fn()

    def evaluate(self, batch, out, idx):
        for quantity, metrics in self.spec.items():
            for metric in metrics:
                try:
                    q = self.quantity_retrieval_fn[quantity](out, batch)
                except KeyError:
                    continue
                eval_result = self.eval_fn[metric](*q)
                eval_metric = self.extract_metric(eval_result)
                self.results[quantity][metric].append(fm(idx, eval_metric))

    def assemble_eval_fn(self):
        self.results = self.initialize_metrics_dict(self.spec)
        self.eval_fn = dict()
        self.quantity_retrieval_fn = dict()
        for quantity, metrics in self.spec.items():
            if quantity not in self.quantity_retrieval_fn:
                self.quantity_retrieval_fn[quantity] = get_retreival_fn(
                    quantity)
            for metric in metrics:
                if metric not in self.eval_fn:
                    self.eval_fn[metric] = self.get_eval_fn(metric)


    @staticmethod
    def initialize_metrics_dict(spec):
        results = dict()
        for quantity, metrics in spec.items():
            results[quantity] = {metric: [] for metric in metrics}
        return results

    @staticmethod
    def get_eval_fn(metric):
        if metric == "L1":
            return lambda estimate, ground_truth: \
                sparse_lnorm(1, estimate, ground_truth.frame, ground_truth.mask,
                             per_frame=True)
        if metric == "L2":
            return lambda estimate, ground_truth: \
                sparse_lnorm(2, estimate, ground_truth.frame, ground_truth.mask,
                             per_frame=True)
        if metric == "1PE":
            return lambda estimate, ground_truth: \
                compute_npe(1, estimate, ground_truth.frame, ground_truth.mask)
        if metric == "3PE":
            return lambda estimate, ground_truth: \
                compute_npe(3, estimate, ground_truth.frame, ground_truth.mask)

    @staticmethod
    def extract_metric(eval_result):
        assert "metric" in eval_result, "metric not found in eval result"
        if isinstance(eval_result["metric"], list):
            assert len(eval_result["metric"]) == 1, "multiple metrics found"
            return eval_result["metric"][0]
        else:
            return eval_result["metric"]
