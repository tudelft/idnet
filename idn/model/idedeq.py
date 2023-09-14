import torch
import torch.nn as nn
from torch.nn.functional import unfold, grid_sample, interpolate

from .extractor import LiteEncoder
from .update import LiteUpdateBlock

from math import sqrt


class IDEDEQIDO(nn.Module):
    def __init__(self, config):
        super(IDEDEQIDO, self).__init__()
        self.hidden_dim = getattr(config, 'hidden_dim', 96)
        self.input_dim = 64
        self.downsample = getattr(config, 'downsample', 8)
        self.input_flowmap = getattr(config, 'input_flowmap', False)
        self.pred_next_flow = getattr(config, 'pred_next_flow', False)
        self.fnet = LiteEncoder(
            output_dim=self.input_dim//2, dropout=0, n_first_channels=2, stride=2 if self.downsample == 8 else 1)
        self.update_net = LiteUpdateBlock(
            hidden_dim=self.hidden_dim, input_dim=self.input_dim,
            num_outputs=2 if self.pred_next_flow else 1,
            downsample=self.downsample)
        self.deblur_iters = config.update_iters
        self.zero_init = config.zero_init
        self._deq = getattr(config, "deq_mode", False)
        self.hook = None
        self.co_mode = getattr(config, "co_mode", False)
        self.conr_mode = getattr(config, "conr_mode", False)
        self.deblur = getattr(config, "deblur", True)
        self.add_delta = getattr(config, "add_delta", False)
        self.deblur_mode = getattr(config, "deblur_mode", "voxel")
        self.reset_continuous_flow()
        if self.input_flowmap:
            self.cnet = LiteEncoder(
                output_dim=self.hidden_dim // 2, dropout=0, n_first_channels=2, stride=2 if self.downsample == 8 else 1)
        else:
            self.cnet = None

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        _, D, H, W = mask.shape
        upsample_ratio = int(sqrt(D/9))
        mask = mask.view(N, 1, 9, upsample_ratio, upsample_ratio, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, upsample_ratio*H, upsample_ratio*W)

    @staticmethod
    def upflow8(flow, mode='bilinear'):
        new_size = (8 * flow.shape[-2], 8 * flow.shape[-1])
        return 8 * interpolate(flow, size=new_size, mode=mode, align_corners=True)

    @staticmethod
    def create_identity_grid(H, W, device):
        i, j = map(lambda x: x.float(), torch.meshgrid(
            [torch.arange(0, H), torch.arange(0, W)], indexing='ij'))
        return torch.stack([j, i], dim=-1).to(device)

    def deblur_tensor(self, raw_input, flow, mask=None):
        # raw: [N, T, C, H, W]
        raw = raw_input.unsqueeze(2) if raw_input.ndim == 4 else raw_input
        N, T, C, H, W = raw.shape
        deblurred_tensor = torch.zeros_like(raw)
        identity_grid = self.create_identity_grid(H, W, raw.device)
        for t in range(T):
            if self.deblur_mode == "voxel":
                delta_p = flow*t/(T-1)
            else:
                delta_p = flow*((t+0.5)/T)
            sampling_grid = identity_grid + torch.movedim(delta_p, 1, -1)
            sampling_grid[..., 0] = sampling_grid[..., 0] / (W-1) * 2 - 1
            sampling_grid[..., 1] = sampling_grid[..., 1] / (H-1) * 2 - 1
            deblurred_tensor[:, t,
                             ] = grid_sample(raw[:, t, ], sampling_grid, align_corners=False)
        if raw_input.ndim == 4:
            deblurred_tensor = deblurred_tensor.squeeze(2)
        return deblurred_tensor

    def reset_continuous_flow(self, reset=True):
        if reset:
            self.flow_init = None
            self.last_net_co = None

    def forward(self, event_bins, flow_init=None, deblur_iters=None, net_co=None):
        if self.co_mode:
            # continuous mode
            # take flow_init from state and forward propagate it
            assert flow_init is None, "flow_init should be None in continuous mode"
            if self.flow_init is None:
                print("No last flow, using zero flow")
            elif 'new_sequence' in event_bins and event_bins['new_sequence'][0] == 1:
                print("Got new sequence, resetting flow")
                flow_init = None
            else:
                flow_init = self.flow_init
        if self.conr_mode:
            if self.last_net_co is None:
                print("No last net_co, using zero flow")
            elif event_bins['new_sequence'][0] == 1:
                print("Got new sequence, resetting flow")
                net_co = None
            else:
                net_co = self.last_net_co

        deblur_iters = self.deblur_iters if deblur_iters is None else deblur_iters
        # x_old, x_new = event_bins["event_volume_old"], event_bins["event_volume_new"]
        x_raw = event_bins["event_volume_new"]

        B, V, H, W = x_raw.shape
        flow_total = torch.zeros(B, 2, H, W).to(
            x_raw.device) if flow_init is None else flow_init.clone()

        delta_flow = flow_total
        flow_history = torch.zeros(B, 0, 2, H, W).to(x_raw.device)
        x_deblur_history = x_raw.clone().unsqueeze(1)
        delta_flow_history = delta_flow.clone().unsqueeze(1)


        x_deblur = x_raw.clone()
        for iter in range(deblur_iters):
            if self.deblur:
                x_deblur = self.deblur_tensor(x_deblur, delta_flow)
                x = torch.stack([x_deblur, x_deblur], dim=1)
                x_deblur_history = torch.cat(
                    [x_deblur_history, x_deblur.unsqueeze(1)], dim=1)
            else:
                x = torch.stack([x_raw, x_raw], dim=1)

            if net_co is not None:
                net = net_co
            else:
                if self.input_flowmap:
                    assert self.cnet is not None, "cnet non initialized in flowmap mode"
                    if flow_init is not None or iter >= 1:
                        net = self.cnet(flow_total)
                    else:
                        net = torch.zeros(
                            (B, self.hidden_dim,
                                H//self.downsample, W//self.downsample)).to(x.device)
                else:
                    if self.cnet is not None:
                        net = self.cnet(x)
                    else:
                        net = torch.zeros(
                            (B, self.hidden_dim,
                                H//self.downsample, W//self.downsample)).to(x.device)
            for i, slice in enumerate(x.permute(2, 0, 1, 3, 4)):
                f = self.fnet(slice)
                net = self.update_net(net, f)

            dflow = self.update_net.compute_deltaflow(net)
            up_mask = self.update_net.compute_up_mask(net)
            delta_flow = self.upsample_flow(dflow, up_mask)
            delta_flow_history = torch.cat(
                [delta_flow_history, delta_flow.unsqueeze(1)], dim=1)
            if self.pred_next_flow:
                nflow = self.update_net.compute_nextflow(net)
                up_mask_next_flow = self.update_net.compute_up_mask2(net)
                next_flow = self.upsample_flow(nflow, up_mask_next_flow)
            else:
                next_flow = None

            if self.deblur or self.add_delta:
                flow_total = flow_total + delta_flow
            else:
                flow_total = delta_flow
            flow_history = torch.cat(
                [flow_history, flow_total.unsqueeze(1)], dim=1)


        if self.co_mode:
            if self.pred_next_flow:
                assert 'next_flow' in locals()
                self.flow_init = next_flow
            else:
                self.flow_init = self.forward_flow(flow_total)
        if self.conr_mode:
            self.last_net_co = net
        return {'final_prediction': flow_total,
                'next_flow': next_flow,
                'delta_flow': delta_flow_history,
                'deblurred_event_volume_new': x_deblur_history,
                'flow_history': flow_history,
                'net': net}

    def forward_flowmap(self, event_bins, flow_init=None, deblur_iters=None):
        deblur_iters = self.deblur_iters if deblur_iters is None else deblur_iters
        
        x = event_bins["event_volume_new"]

        B, V, H, W = x.shape
        flow_total = torch.zeros(B, 2, H, W).to(
            x.device) if flow_init is None else flow_init.clone()

        delta_flow = flow_total
        flow_history = torch.zeros(B, 0, 2, H, W).to(x.device)

        for _ in range(deblur_iters):
            x_deblur = self.deblur_tensor(x, delta_flow)
            x = torch.stack([x_deblur, x_deblur], dim=1)
            
            if flow_init is not None and self.cnet is not None:
                net = self.cnet(flow_total)
            else:
                net = torch.zeros(
                    (B, self.hidden_dim, H//8, W//8)).to(x.device)
            for i, slice in enumerate(x.permute(2, 0, 1, 3, 4)):
                f = self.fnet(slice)
                net = self.update_net(net, f)

            dflow = self.update_net.compute_deltaflow(net)
            up_mask = self.update_net.compute_up_mask(net)
            delta_flow = self.upsample_flow(dflow, up_mask)
            flow_total = flow_total + delta_flow
            
            flow_history = torch.cat(
                [flow_history, flow_total.unsqueeze(1)], dim=1)


        return {'final_prediction': flow_total,
                'flow_history': flow_history}


class RecIDE(IDEDEQIDO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch, flow_init=None, deblur_iters=None):
        deblur_iters = self.deblur_iters if deblur_iters is None else deblur_iters

        flow_trajectory = []
        flow_next_trajectory = []

        for t, x in enumerate(batch):
            out = super().forward(x, flow_init=flow_init)
            flow_pred = out['final_prediction']
            if 'next_flow' in out:
                flow_next = out['next_flow']
                flow_init = flow_next
                flow_next_trajectory.append(flow_next)
            else:
                flow_init = self.forward_flow(flow_pred)
            
            flow_trajectory.append(flow_pred)

            if (t+1) % 4 == 0:
                flow_init = flow_init.detach()
                yield {'final_prediction': flow_pred,
                       'flow_trajectory': flow_trajectory,
                       'flow_next_trajectory': flow_next_trajectory, }
                flow_trajectory = []
                flow_next_trajectory = []

    def forward_inference(self, batch, flow_init=None, deblur_iters=None):
        deblur_iters = self.deblur_iters if deblur_iters is None else deblur_iters
        

        flow_trajectory = []

        for t, x in enumerate(batch):
            out = super().forward(x, flow_init=flow_init)
            flow_pred = out['final_prediction']
            flow_init = self.forward_flow(flow_pred)
            flow_trajectory.append(flow_pred)

        return {'final_prediction': flow_pred,
                'flow_trajectory': flow_trajectory}

    def backward_neg_flow(self, x):
        x["event_volume_new"] = -torch.flip(x["event_volume_new"], [1])
        back_flow = -super().forward(x)['final_prediction']
        return back_flow
