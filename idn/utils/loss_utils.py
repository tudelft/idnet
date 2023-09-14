from ..model.loss import sparse_l1

def get_loss_fn_by_name(loss_name):
    if loss_name == 'sparse_l1':
        return lambda estimate, ground_truth: sparse_l1(estimate, \
            ground_truth.frame, valid_mask=ground_truth.mask)
    else:
        assert False, f"loss {loss_name} not implemented"


def get_valid_loss_fn_by_name(config):
    valid_loss_fn = dict()
    #
    if config.name == 'sparse_l1':
        return sparse_l1

def compute_seq_loss(weight, loss_fn, estimate, ground_truth):
    assert isinstance(estimate, list)
    if weight == "last":
        return loss_fn(estimate[-1], ground_truth[-1])
    else:
        seq_loss = list(map(loss_fn, estimate, ground_truth))
        if weight == "sum":
            return sum(seq_loss)

        elif weight == "avg":
            return sum(seq_loss)/len(seq_loss)

        elif hasattr(weight, '__getitem__') and isinstance(weight[0], float):
            assert len(weight) == len(estimate)
            return sum(map(lambda x, y: x*y, seq_loss, weight))
        else:
            assert False, f"weight {weight} for seq loss not supported"