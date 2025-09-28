import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.init as init
import torch.nn.functional as F

class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)

class DepthwiseConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv2d, self).__init__()
        assert out_channels % in_channels == 0
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 11,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        assert expansion_factor == 2

        self.sequential = nn.Sequential(
            DepthwiseConv2d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.sequential(inputs)