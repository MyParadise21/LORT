import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        # [B, C, H, W] to [B, 1, H, W]
        return torch.mean(x,1).unsqueeze(1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.kernel_size = 5
        self.compress = ChannelPool()
        self.spatial_conv = BasicConv(1, 1, self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)

        x_out = self.spatial_conv(x_compress)

        scale = torch.sigmoid(x_out)

        return scale

class Attention1(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=int(3 / 2), bias=False)

    def forward(self, x):
        y = self.avg_pool(x)

        y = y.squeeze(-1).transpose(-1, -2)

        y = self.conv1d(y)

        y = y.transpose(-1, -2).unsqueeze(-1)

        return torch.sigmoid(y)

class Attention2(nn.Module):

    def __init__(self):
        super().__init__()

        self.SpatialGate = SpatialGate()

    def forward(self, x):
        """
        Input X: [B,C,N,T]
        Ouput X: [B,1,N,T]
        """

        x = self.SpatialGate(x)

        return x