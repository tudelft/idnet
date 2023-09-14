from collections import namedtuple
import torch.nn.functional as F


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def get_retreival_fn(quantity):
    fmask = namedtuple("masked_frame", ["frame", "mask"])
    if quantity == "final_prediction":
        return retreival_pred_seq_1
    elif quantity == "pred_flow_seq":
        return retreival_pred_seq
    elif quantity == "final_prediction_nonseq":
        return retreival_pred_nonseq
    elif quantity == "pred_flow_next_seq":
        return retreival_pred_nextflow_seq
    elif quantity == "next_flow":
        return retreival_next_flow
    elif quantity == "pred_lowres_flow_next_seq":
        return retreival_pred_lowres_nextflow_seq

    assert False, f"quantity {quantity} not implemented"


def retreival_pred_nonseq(out, batch):
    fmask = namedtuple("masked_frame", ["frame", "mask"])
    return (out["final_prediction"], fmask(batch["flow_gt_event_volume_new"],
            batch["flow_gt_event_volume_new_valid_mask"]))


def retreival_pred_seq_1(out, batch):
    fmask = namedtuple("masked_frame", ["frame", "mask"])
    return (out["final_prediction"], fmask(batch[-1]["flow_gt_event_volume_new"],
            batch[-1]["flow_gt_event_volume_new_valid_mask"]))


def retreival_pred_seq(out, batch):
    fmask = namedtuple("masked_frame", ["frame", "mask"])
    return (out["flow_trajectory"], [fmask(x["flow_gt_event_volume_new"],
            x["flow_gt_event_volume_new_valid_mask"]) for x in batch])


def retreival_pred_nextflow_seq(out, batch):
    fmask = namedtuple("masked_frame", ["frame", "mask"])
    return (out["flow_next_trajectory"], [fmask(x["flow_gt_next"],
            x["flow_gt_next_valid_mask"]) for x in batch])


def retreival_next_flow(out, batch):
    fmask = namedtuple("masked_frame", ["frame", "mask"])
    return (out["next_flow"], fmask(batch[-1]["flow_gt_next"],
            batch[-1]["flow_gt_next_valid_mask"]))


def retreival_pred_lowres_nextflow_seq(out, batch):
    fmask = namedtuple("masked_frame", ["frame", "mask"])
    return ([upflow8(x) for x in out["flow_next_trajectory"][:-1]], [fmask(x["flow_gt_event_volume_new"],
            x["flow_gt_event_volume_new_valid_mask"]) for x in batch[1:]])
