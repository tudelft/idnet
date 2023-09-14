import numpy as np
import torch
import torch.nn as nn
import re
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF


def dictionary_of_numpy_arrays_to_tensors(sample):
    """Transforms dictionary of numpy arrays to dictionary of tensors."""
    if isinstance(sample, dict):
        return {
            key: dictionary_of_numpy_arrays_to_tensors(value)
            for key, value in sample.items()
        }
    if isinstance(sample, np.ndarray):
        if len(sample.shape) == 2:
            return torch.from_numpy(sample).float().unsqueeze(0)
        else:
            return torch.from_numpy(sample).float()
    return sample


class EventSequenceToVoxelGrid_Pytorch(object):
    # Source: https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py#L480
    def __init__(self, num_bins, gpu=False, gpu_nr=1, normalize=True, forkserver=True):
        if forkserver:
            try:
                torch.multiprocessing.set_start_method('forkserver')
            except RuntimeError:
                pass
        self.num_bins = num_bins
        self.normalize = normalize
        if gpu:
            if not torch.cuda.is_available():
                print('Warning: There\'s no CUDA support on this machine!')
            else:
                self.device = torch.device('cuda:' + str(gpu_nr))
        else:
            self.device = torch.device('cpu')

    def __call__(self, event_sequence):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.
        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        :param device: device to use to perform computations
        :return voxel_grid: PyTorch event tensor (on the device specified)
        """

        events = event_sequence.features.astype('float')

        width = event_sequence.image_width
        height = event_sequence.image_height

        assert (events.shape[1] == 4)
        assert (self.num_bins > 0)
        assert (width > 0)
        assert (height > 0)

        with torch.no_grad():

            events_torch = torch.from_numpy(events)
            # with DeviceTimer('Events -> Device (voxel grid)'):
            events_torch = events_torch.to(self.device)

            # with DeviceTimer('Voxel grid voting'):
            voxel_grid = torch.zeros(
                self.num_bins, height, width, dtype=torch.float32, device=self.device).flatten()

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]

            assert last_stamp.dtype == torch.float64, 'Timestamps must be float64!'
            # assert last_stamp.item()%1 == 0, 'Timestamps should not have decimals'

            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (self.num_bins - 1) * \
                (events_torch[:, 0] - first_stamp) / deltaT
            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1

            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < self.num_bins
            valid_indices &= tis >= 0

            if events_torch.is_cuda:
                datatype = torch.cuda.LongTensor
            else:
                datatype = torch.LongTensor

            voxel_grid.index_add_(dim=0,
                                  index=(xs[valid_indices] + ys[valid_indices]
                                         * width + tis_long[valid_indices] * width * height).type(
                                      datatype),
                                  source=vals_left[valid_indices])

            valid_indices = (tis + 1) < self.num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                  index=(xs[valid_indices] + ys[valid_indices] * width
                                         + (tis_long[valid_indices] + 1) * width * height).type(datatype),
                                  source=vals_right[valid_indices])

            voxel_grid = voxel_grid.view(self.num_bins, height, width)

        if self.normalize:
            mask = torch.nonzero(voxel_grid, as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = voxel_grid[mask].mean()
                std = voxel_grid[mask].std()
                if std > 0:
                    voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                else:
                    voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid


def apply_transform_to_field(sample, func, field_name):
    """
    Applies a function to a field of a sample.
    :param sample: a sample
    :param func: a function that takes a numpy array and returns a numpy array
    :param field_name: the name of the field to transform
    :return: the transformed sample
    """
    if isinstance(sample, dict):
        for key, value in sample.items():
            if bool(re.search(field_name, key)):
                sample[key] = func(value)

    if isinstance(sample, np.ndarray) or isinstance(sample, torch.Tensor):
        return func(sample)
    return sample


def apply_randomcrop_to_sample(sample, crop_size):
    """
    Applies a random crop to a sample.
    :param sample: a sample
    :param crop_size: the size of the crop
    :return: the cropped sample
    """
    i, j, h, w = RandomCrop.get_params(
        sample["event_volume_old"], output_size=crop_size)
    keys_to_crop = ["event_volume_old", "event_volume_new",
                    "flow_gt_event_volume_old", "flow_gt_event_volume_new", "reverse_flow_gt_event_volume_old", "reverse_flow_gt_event_volume_new"]

    for key, value in sample.items():
        if key in keys_to_crop:
            if isinstance(value, torch.Tensor):
                sample[key] = TF.crop(value, i, j, h, w)
            elif isinstance(value, list) or isinstance(value, tuple):
                sample[key] = [TF.crop(v, i, j, h, w) for v in value]
    return sample


def downsample_spatial(x, factor):
    """
    Downsample a given tensor spatially by a factor.
    :param x: PyTorch tensor of shape [batch, num_bins, height, width]
    :param factor: downsampling factor
    :return: PyTorch tensor of shape [batch, num_bins, height/factor, width/factor]
    """
    assert (factor > 0), 'Factor must be positive!'

    assert (x.shape[-1] %
            factor == 0), 'Width of x must be divisible by factor!'
    assert (x.shape[-2] %
            factor == 0), 'Height of x must be divisible by factor!'

    return nn.AvgPool2d(kernel_size=factor, stride=factor)(x)


def downsample_spatial_mask(x, factor):
    """
    Downsample a given mask (boolean) spatially by a factor.
    :param x: PyTorch tensor of shape [batch, num_bins, height, width]
    :param factor: downsampling factor
    :return: PyTorch tensor of shape [batch, num_bins, height/factor, width/factor]
    """
    assert (factor > 0), 'Factor must be positive!'

    assert (x.shape[-1] %
            factor == 0), 'Width of x must be divisible by factor!'
    assert (x.shape[-2] %
            factor == 0), 'Height of x must be divisible by factor!'

    return nn.AvgPool2d(kernel_size=factor, stride=factor)(x.float()) >= 0.5
