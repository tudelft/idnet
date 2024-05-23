# import h5pickle as h5py
import h5py
import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from idn.utils.mvsec_utils import EventSequence
from idn.utils.dsec_utils import RepresentationType, VoxelGrid
from idn.utils.transformers import EventSequenceToVoxelGrid_Pytorch, apply_randomcrop_to_sample

class MVSEC(Dataset):
    def __init__(self, seq_name, seq_path="data/", representation_type=None, \
        rate=20, num_bins=5, transforms=[], filter=None, augment=True, dt=None):
        self.seq_name = seq_name
        self.dt = dt
        if self.dt is None:
            self.event_h5 = h5py.File(os.path.join(seq_path, f"{seq_name}_data.hdf5"), "r")
            self.event = self.event_h5['davis']['left']['events']
            self.gt_h5 = h5py.File(os.path.join(seq_path, f"{seq_name}_gt.hdf5"), "r")
            self.gt_flow = self.gt_h5['davis']['left']['flow_dist']
            self.timestamps = self.gt_h5['davis']['left']['flow_dist_ts']
        else:
            assert self.dt == 1 or self.dt == 4
            self.h5 = h5py.File(os.path.join(seq_path, f"{seq_name}.h5"), "r")
            self.event = self.h5['events']
            self.timestamps = self.h5['flow']['dt={}'.format(self.dt)]['timestamps'][:, 0]
            self.gt_flow = list(self.h5['flow']['dt={}'.format(self.dt)].keys())
            self.gt_flow.remove('timestamps')
            assert sorted(self.gt_flow) == self.gt_flow
        
        if representation_type is None:
            self.representation_type = VoxelGrid
        else:
            self.representation_type = representation_type
        
        if filter is not None:
            assert isinstance(filter, tuple) and isinstance(filter[0], int)\
                and isinstance(filter[1], int)
            self.timestamps = self.timestamps[slice(*filter)]
            self.gt_flow = self.gt_flow[slice(*filter)]

        self.raw_gt_len = self.timestamps.shape[0]
        self.event_ts_to_idx = self.build_event_idx()
        self.voxel = EventSequenceToVoxelGrid_Pytorch(
            num_bins=num_bins,
            normalize=True,
            gpu=False,
        )
        self.image_width, self.image_height = 346, 260
        self.cropper = T.CenterCrop((256, 256))
        self.augment = augment
        pass

    def __len__(self):
        return self.raw_gt_len - 2

    def __getitem__(self, idx):
        idx += 1
        sample = {}
        if self.dt is None:
            # get events
            events = self.event[self.event_ts_to_idx[idx-1]:self.event_ts_to_idx[idx]]
            events = events[:, [2, 0, 1, 3]]  # make it (t, x, y, p)
            sample["event_volume_old"] = \
                self.voxel(EventSequence(events,
                                     params={'width': self.image_width,
                                             'height': self.image_height},
                                     timestamp_multiplier=1e6,
                                     convert_to_relative=True,
                                     features=events))
            
            # get events
            events = self.event[self.event_ts_to_idx[idx]:self.event_ts_to_idx[idx+1]]
            events = events[:, [2, 0, 1, 3]] # make it (t, x, y, p)

            sample["event_volume_new"] = \
                self.voxel(EventSequence(events, 
                                    params={'width': self.image_width, 
                                            'height': self.image_height},
                                    timestamp_multiplier=1e6,
                                    convert_to_relative=True,
                                    features = events))
            # get flow
            flow = self.gt_flow[idx] # -1 yields the same gt flow as E-RAFT, but likely incorrect
            flow_next = self.gt_flow[idx+1]
        else:
            old_p = self.event['ps'][self.event_ts_to_idx[idx-1]:self.event_ts_to_idx[idx]]
            old_t = self.event['ts'][self.event_ts_to_idx[idx-1]:self.event_ts_to_idx[idx]]
            old_x = self.event['xs'][self.event_ts_to_idx[idx-1]:self.event_ts_to_idx[idx]]
            old_y = self.event['ys'][self.event_ts_to_idx[idx-1]:self.event_ts_to_idx[idx]]

            old_events = np.column_stack((old_t, old_x, old_y, old_p))
            sample["event_volume_old"] = \
                self.voxel(EventSequence(old_events,
                                     params={'width': self.image_width,
                                             'height': self.image_height},
                                     timestamp_multiplier=1e6,
                                     convert_to_relative=True,
                                     features=old_events))
            
            new_p = self.event['ps'][self.event_ts_to_idx[idx]:self.event_ts_to_idx[idx+1]]
            new_t = self.event['ts'][self.event_ts_to_idx[idx]:self.event_ts_to_idx[idx+1]]
            new_x = self.event['xs'][self.event_ts_to_idx[idx]:self.event_ts_to_idx[idx+1]]
            new_y = self.event['ys'][self.event_ts_to_idx[idx]:self.event_ts_to_idx[idx+1]]

            new_events = np.column_stack((new_t, new_x, new_y, new_p))
            sample["event_volume_new"] = \
                self.voxel(EventSequence(new_events,
                                     params={'width': self.image_width,
                                             'height': self.image_height},
                                     timestamp_multiplier=1e6,
                                     convert_to_relative=True,
                                     features=new_events))

            # get flow
            flow = np.transpose(self.h5['flow']['dt={}'.format(self.dt)][self.gt_flow[idx]][:], (2, 0, 1))
            flow_next = np.transpose(self.h5['flow']['dt={}'.format(self.dt)][self.gt_flow[idx+1]][:], (2, 0, 1))
        

        sample["flow_gt_event_volume_new"] = self.process_flow_gt(flow)
        sample["flow_gt_next"] = self.process_flow_gt(flow_next)

        sample["event_volume_old"] = self.cropper(sample["event_volume_old"])
        sample["event_volume_new"] = self.cropper(sample["event_volume_new"])

        if self.augment:
            # augmentation
            if random.random() > 0.5:
                for key in sample:
                    if isinstance(sample[key], torch.Tensor):
                        sample[key] = T.functional.hflip(sample[key])
                    if key.startswith("flow_gt"):
                        sample[key] = [T.functional.hflip(
                            mask) for mask in sample[key]]
                        sample[key][0][0, :] = -sample[key][0][0, :]
            

        return sample

    def process_flow_gt(self, flow):
        flow_valid = (flow[0] != 0) | (flow[1] != 0)
        flow_valid[193:, :] = False
        flow = torch.from_numpy(flow)
        valid_mask = torch.from_numpy(
            np.stack([flow_valid]*1, axis=0))

        return (self.cropper(flow), self.cropper(valid_mask))
        
    def build_event_idx(self):
        if self.dt is None:
            events_ts = self.event_h5['davis']['left']['events'][:, 2]
        else:
            events_ts = self.h5['events']['ts']
        return np.searchsorted(events_ts, self.timestamps, side='left')


class MVSECRecurrent(MVSEC):
    def __init__(self, seq_name, seq_path="/scratch", representation_type=None,
                 rate=20, num_bins=15, transforms=[], filter=None, augment=True, sequence_length=1):
        super(MVSECRecurrent, self).__init__(seq_name, seq_path, representation_type,
                                             rate, num_bins, transforms, filter, augment)
        self.sequence_length = sequence_length
        self.valid_indices = self.get_continuous_sequences()

    def get_continuous_sequences(self):
        # MVSEC is continuous without breaks
        continuous_seq_idcs = list(
            (range(self.raw_gt_len - 2 - self.sequence_length)))
        return continuous_seq_idcs
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        assert idx >= 0
        assert idx < len(self)

        valid_idx = self.valid_indices[idx]
        sequence = []
        j = valid_idx

        for i in range(self.sequence_length):
            sample = super(MVSECRecurrent, self).__getitem__(j)
            sequence.append(sample)
            j += 1

        
        # Check if the current sample is the first sample of a continuous sequence
        if idx == 0 or self.valid_indices[idx]-self.valid_indices[idx-1] != 1:
            sequence[0]['new_sequence'] = 1
        else:
            sequence[0]['new_sequence'] = 0

        return sequence

