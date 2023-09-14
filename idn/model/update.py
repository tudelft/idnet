import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead2(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead2, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 10*self.tanh(self.conv2(self.relu(self.conv1(x))))


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class LiteUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=32, input_dim=16, num_outputs=1, downsample=8):
        super(LiteUpdateBlock, self).__init__()
        self.upsample_mask_dim = downsample * downsample
        self.num_outputs = num_outputs
        assert self.num_outputs in [1, 2]
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=hidden_dim)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.upsample_mask_dim*9, 1, padding=0))
        if self.num_outputs == 2:
            self.flow_head2 = FlowHead(hidden_dim, hidden_dim=hidden_dim)
            self.mask2 = nn.Sequential(
                nn.Conv2d(hidden_dim, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.upsample_mask_dim*9, 1, padding=0))

    def forward(self, net, inp):
        return self.gru(net, inp)

    def compute_deltaflow(self, net):
        return self.flow_head(net)

    def compute_nextflow(self, net):
        if self.num_outputs == 2:
            return self.flow_head2(net)
        else:
            raise NotImplementedError

    def compute_up_mask(self, net):
        return self.mask(net)

    def compute_up_mask2(self, net):
        if self.num_outputs == 2:
            return self.mask2(net)
        else:
            raise NotImplementedError
