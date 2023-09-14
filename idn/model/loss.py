import torch


def sparse_l1_seq(estimated, ground_truth, valid_mask=None):
    assert isinstance(estimated, list)
    assert isinstance(ground_truth, list)
    if valid_mask is not None:
        assert isinstance(valid_mask, list)
    assert len(estimated) == len(ground_truth) == len(valid_mask)
    loss = 0.
    for i in range(len(estimated)):
        loss += sparse_l1(estimated[i], ground_truth[i], valid_mask[i])
    return loss/len(estimated)


def sparse_l1(estimated, ground_truth, valid_mask=None):
    """Return L1 loss.
    This loss ignores difference in pixels where mask is False.
    If all pixels are marked as False, the loss is equal to zero.
    Args:
        estimated: is tensor with predicted values of size
                   batch_size x height x width.
        ground_truth: is tensor with ground truth values of
                      size batch_size x height x width. 
        mask: mask of size batch_size x height x width. Only
              pixels with True values are used. If "valid_mask"
              is None, than we use all pixels.              
    """
    if valid_mask is not None:
        valid_mask = valid_mask.bool()
    pixelwise_diff = (estimated - ground_truth).abs()
    if valid_mask is not None:
        if valid_mask.size() == pixelwise_diff.size():
            pixelwise_diff = pixelwise_diff[valid_mask]
        else:
            try:
                pixelwise_diff = pixelwise_diff[valid_mask.expand(
                    pixelwise_diff.size())]
            except:
                raise Exception("Mask auto expand failed.")
    if pixelwise_diff.numel() == 0:
        return torch.Tensor([0]).type(estimated.type())
    return pixelwise_diff.mean()


def sparse_lnorm(order, estimated, ground_truth, valid_mask=None, per_frame=False):
    """Return L1 loss.
    This loss ignores difference in pixels where mask is False.
    If all pixels are marked as False, the loss is equal to zero.
    Args:
        estimated: is tensor with predicted values of size
                   batch_size x height x width.
        ground_truth: is tensor with ground truth values of
                      size batch_size x height x width. 
        mask: mask of size batch_size x height x width. Only
              pixels with True values are used. If "valid_mask"
              is None, than we use all pixels.              
    """
    if valid_mask is not None:
        valid_mask = valid_mask.bool()
    pixelwise_diff = torch.norm(
        estimated - ground_truth, p=order, keepdim=True, dim=(1))
    # make sure valid_mask is the same shape as diff
    if valid_mask is not None:
        if valid_mask.size() != pixelwise_diff.size():
            try:
                valid_mask = valid_mask.expand(pixelwise_diff.size())
            except:
                raise Exception("Mask auto expand failed.")
    if per_frame:
        error = []
        if valid_mask is not None:
            for diff, mask in zip(pixelwise_diff, valid_mask):
                error.append(diff[mask].mean().item())
        else:
            error = [e.mean().item() for e in pixelwise_diff]
        emap = estimated - ground_truth
        emask = valid_mask.expand(emap.size())
        emap[~emask] = 0
        return {
            "metric": error,
            "t_emap": emap
        }
    else:
        if valid_mask is not None:
            pixelwise_diff = pixelwise_diff[valid_mask]
        if pixelwise_diff.numel() == 0:
            return torch.Tensor([0]).type(estimated.type())
        return pixelwise_diff.mean()


def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    """
    Robust Charbonnier loss, as defined in equation (4) of the paper.
    """
    loss = torch.mean(torch.pow((delta ** 2 + epsilon ** 2), alpha))
    return loss


def compute_smoothness_loss(flow):
    """
    Local smoothness loss, as defined in equation (5) of the paper.
    The neighborhood here is defined as the 8-connected region around each pixel.
    """
    flow_ucrop = flow[..., 1:]
    flow_dcrop = flow[..., :-1]
    flow_lcrop = flow[..., 1:, :]
    flow_rcrop = flow[..., :-1, :]

    flow_ulcrop = flow[..., 1:, 1:]
    flow_drcrop = flow[..., :-1, :-1]
    flow_dlcrop = flow[..., :-1, 1:]
    flow_urcrop = flow[..., 1:, :-1]

    smoothness_loss = charbonnier_loss(flow_lcrop - flow_rcrop) + \
        charbonnier_loss(flow_ucrop - flow_dcrop) + \
        charbonnier_loss(flow_ulcrop - flow_drcrop) + \
        charbonnier_loss(flow_dlcrop - flow_urcrop)
    smoothness_loss /= 4.

    return smoothness_loss


def compute_npe(n, estimated, ground_truth, valid_mask=None):
    if valid_mask is not None:
        valid_mask = valid_mask.bool()
    pixelwise_diff = torch.norm(
        estimated - ground_truth, p=2, keepdim=True, dim=(1))

    # make sure valid_mask is the same shape as diff
    if valid_mask is not None:
        if valid_mask.size() != pixelwise_diff.size():
            try:
                valid_mask = valid_mask.expand(pixelwise_diff.size())
            except:
                raise Exception("Mask auto expand failed.")

    if valid_mask is not None:
        pixelwise_diff = pixelwise_diff[valid_mask]
    if pixelwise_diff.numel() == 0:
        return torch.Tensor([0]).type(estimated.type())
    return {
        "metric": torch.numel(pixelwise_diff[pixelwise_diff >= n]) /
        torch.numel(pixelwise_diff)
    }
