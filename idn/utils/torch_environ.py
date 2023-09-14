import torch
import numpy as np
import random


def config_torch(config):
    torch.backends.cudnn.benchmark = True
    if config.get("debug_grad", False):
        torch_set_gradient_debug()
    if config.get("deterministic", True):
        torch_set_deterministic()


def torch_set_deterministic(seed=141021):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # because of .put_ in forward_interpolate_pytorch
    torch.use_deterministic_algorithms(False)


def torch_set_gradient_debug():
    torch.autograd.set_detect_anomaly(True)
