import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
import importlib

from .helper_functions import move_batch_to_cuda, move_list_to_cuda, move_tensor_to_cuda
from ..model.loss import sparse_lnorm


def log_tensors(idx, dict_of_tensors, tmp_folder):
    for key, value in dict_of_tensors.items():
        if isinstance(value, torch.Tensor):
            tensor = value.cpu().numpy()
            np.save(os.path.join(tmp_folder, f"{key}_{idx}.npy"), tensor)
            continue
        if isinstance(value, np.ndarray):
            np.save(os.path.join(tmp_folder, f"{key}_{idx}.npy"), value)
            continue



class Validator:
    def __init__(self, config):
        self.test = dict()
        # self.logger = Logger.from_config(config.logger, config.name)
        for test_name, test_spec in config.items():
            # create a test object
            self.test[test_name] = self.get_test_type(test_name)(test_spec)

    @staticmethod
    def get_test_type(test_name, test_type=None):
        from ..tests import dsec as dsec_tests
        if test_name == "dsec":
            return dsec_tests.assemble_dsec_test_cls(test_type)
        if test_name == "mvsec_day1":
            return dsec_tests.TestMVSEC
        if test_name == "mvsec_day1_rec":
            return dsec_tests.TestMVSECCO
        try:
            return getattr(dsec_tests, f"Test{test_name.upper()}")
        except:
            raise ValueError(f"Test {test_name} not found.")

    def __call__(self, model, run_all=True):
        # run the corresponding validator
        results = dict()
        for name, test in self.test.items():
            state, results[name] = test.execute_test(model)
        return results


def validate_model_warm(data_loader, model, config, return_dict, gpu_id, test_mode=False,
                        log_field_dict=None, log_dir=None):
    model.cuda(gpu_id)
    model.eval()

    metrics = dict()

    with torch.no_grad():
        if not isinstance(data_loader, list):
            data_loader = [data_loader]
        data_loader = data_loader[0:1]
        for seq_loader in data_loader:
            for idx, batch in enumerate(tqdm(seq_loader, position=1)):
                # if idx == 730:
                #     break
                if isinstance(batch, list):
                    batch = move_list_to_cuda(batch, gpu_id)
                    loss_item = batch[-1]
                else:
                    move_tensor_to_cuda(batch, gpu_id)
                    loss_item = batch
                if idx == 0:
                    out = model(batch)
                else:
                    pre_flow = model.forward_flow(pred)
                    if isinstance(batch, list):
                        batch[0]["pre_flow"] = pre_flow
                    else:
                        batch["pre_flow"] = pre_flow
                    out = model(batch, pre_flow, 1)
                if isinstance(batch, list):
                    batch[-1] = {**batch[-1], **out}
                    pred = batch[-1]["final_prediction"]
                else:
                    batch = {**batch, **out}
                    pred = batch["final_prediction"]

                if not test_mode and idx != 0:
                    loss_l1, emap = sparse_lnorm(1, pred, loss_item['flow_gt_event_volume_new'],
                                                 loss_item['flow_gt_event_volume_new_valid_mask'],
                                                 per_frame=True)
                    loss_l2, _ = sparse_lnorm(2, pred, loss_item['flow_gt_event_volume_new'],
                                              loss_item['flow_gt_event_volume_new_valid_mask'],
                                              per_frame=True)
                    loss_l1_pre, pre_emap = sparse_lnorm(1, pre_flow, loss_item['flow_gt_event_volume_new'],
                                                         loss_item['flow_gt_event_volume_new_valid_mask'],
                                                         per_frame=True)
                    loss_l2_pre, _ = sparse_lnorm(2, pre_flow, loss_item['flow_gt_event_volume_new'],
                                                  loss_item['flow_gt_event_volume_new_valid_mask'],
                                                  per_frame=True)
                    if isinstance(batch, list):
                        batch = batch[-1]
                    batch['emap'] = emap
                    batch['pre_emap'] = pre_emap
                    for i, sample_seq in enumerate(batch['seq_name']):
                        if sample_seq not in metrics.keys():
                            metrics[sample_seq] = dict()
                            metrics[sample_seq]["l2"] = []
                            metrics[sample_seq]["l1"] = []
                            metrics[sample_seq]["l2_pre"] = []
                            metrics[sample_seq]["l1_pre"] = []
                        metrics[sample_seq]["l2"].append(loss_l2[i])
                        metrics[sample_seq]["l1"].append(loss_l1[i])
                        metrics[sample_seq]["l2_pre"].append(loss_l2_pre[i])
                        metrics[sample_seq]["l1_pre"].append(loss_l1_pre[i])

                if log_dir:
                    log_tensors_dict = dict()
                    for key, value in log_field_dict.items():
                        if isinstance(batch, list):
                            batch = batch[-1]
                        if value in batch.keys():
                            log_tensors_dict[key] = batch[value]
                    log_tensors(idx, log_tensors_dict, log_dir)

    for seqs in metrics.keys():
        metrics[seqs]["l2_avg"] = np.array(metrics[seqs]["l2"]).mean()
        metrics[seqs]["l1_avg"] = np.array(metrics[seqs]["l1"]).mean()
        return_dict[seqs] = metrics[seqs]

    if log_dir:
        with open(os.path.join(log_dir, "metrics.pkl"), "wb") as f:
            pickle.dump(metrics, f)


def validate_model(data_loader, model, config, return_dict, gpu_id, test_mode=False,
                   log_field_dict=None, log_dir=None):
    model.cuda(gpu_id)
    model.eval()

    metrics = dict()

    with torch.no_grad():
        if not isinstance(data_loader, list):
            data_loader = [data_loader]
        data_loader = data_loader[0:1]  # TODO: remove hard coded first seq
        for seq_loader in data_loader:
            for idx, batch in enumerate(tqdm(seq_loader, position=1)):
                if isinstance(batch, list):
                    batch = move_list_to_cuda(batch, gpu_id)
                    loss_item = batch[-1]
                else:
                    move_tensor_to_cuda(batch, gpu_id)
                    loss_item = batch
                out = model(batch)
                if not isinstance(batch, list):
                    batch = {**batch, **out}
                    pred = batch["final_prediction"]
                else:
                    pred = out["final_prediction"]
                    batch = batch[-1]
                if not test_mode:
                    loss_l1, _ = sparse_lnorm(1, pred, loss_item['flow_gt_event_volume_new'],
                                              loss_item['flow_gt_event_volume_new_valid_mask'],
                                              per_frame=True)
                    loss_l2, _ = sparse_lnorm(2, pred, loss_item['flow_gt_event_volume_new'],
                                              loss_item['flow_gt_event_volume_new_valid_mask'],
                                              per_frame=True)
                    for i, sample_seq in enumerate(batch['seq_name']):
                        if sample_seq not in metrics.keys():
                            metrics[sample_seq] = dict()
                            metrics[sample_seq]["l2"] = []
                            metrics[sample_seq]["l1"] = []
                        metrics[sample_seq]["l2"].append(loss_l2[i])
                        metrics[sample_seq]["l1"].append(loss_l1[i])

                if log_dir:
                    log_tensors_dict = dict()
                    for key, value in log_field_dict.items():
                        if value in batch.keys():
                            log_tensors_dict[key] = batch[value]
                    log_tensors(idx, log_tensors_dict, log_dir)

    for seqs in metrics.keys():
        metrics[seqs]["l2_avg"] = np.array(metrics[seqs]["l2"]).mean()
        metrics[seqs]["l1_avg"] = np.array(metrics[seqs]["l1"]).mean()
        return_dict[seqs] = metrics[seqs]

    if log_dir:
        with open(os.path.join(log_dir, "metrics.pkl"), "wb") as f:
            pickle.dump(metrics, f)
