import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class DLC(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(
            output_dim,
            output_dim,
            kernel_size=[lorder + lorder - 1, 1],
            stride=[1, 1],
            groups=output_dim,
            bias=False
        )

    def forward(self, input):
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)

        x = torch.unsqueeze(p1, 1)
        x_permuted = x.permute(0, 3, 2, 1)
        y = F.pad(x_permuted, [0, 0, self.lorder - 1, self.lorder - 1])

        out = x_permuted + self.conv1(y)
        output = out.permute(0, 3, 2, 1).squeeze()

        return input + output


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, lorder=20, in_channels=64):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.twidth = 2 * lorder - 1
        self.kernel_size = (self.twidth, 1)

        self.pads = []
        self.convs = []
        self.norms = []
        self.prelus = []

        for i in range(depth):
            dil = 2 ** i
            pad_length = lorder + (dil - 1) * (lorder - 1) - 1

            self.pads.append(nn.ConstantPad2d((0, 0, pad_length, pad_length), value=0.))
            self.convs.append(
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                    groups=self.in_channels,
                    bias=False
                )
            )
            self.norms.append(nn.InstanceNorm2d(in_channels, affine=True))
            self.prelus.append(nn.PReLU(self.in_channels))

        self.pads = nn.ModuleList(self.pads)
        self.convs = nn.ModuleList(self.convs)
        self.norms = nn.ModuleList(self.norms)
        self.prelus = nn.ModuleList(self.prelus)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = self.pads[i](skip)
            out = self.convs[i](out)
            out = self.norms[i](out)
            out = self.prelus[i](out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DLC_dilated(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_size),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * hidden_size, input_dim),
            nn.Dropout(0.1)
        )

        self.conv = DilatedDenseNet(depth=2, lorder=lorder, in_channels=output_dim)

    def forward(self, input):
        x1 = self.net(input)
        x_unsqueezed = torch.unsqueeze(x1, 1).permute(0, 3, 2, 1)

        out = self.conv(x_unsqueezed)
        output = out.permute(0, 3, 2, 1).squeeze()

        return input + output