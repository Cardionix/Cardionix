"""
ResNet is class-builder for residual network with shortcut connetions by params.
This implementation residual network for signal processing
and use 1D convolution instead of 2D convolution as original implementation.
"""

__all__ = ["ResNet"]

from typing import Optional, Literal
import torch
from torch import nn


class ResidualUnite(nn.Module):
    """
    This class implements residual block with shortcut connections.

    Note.
    This is pre-activation residual unite architecture.
    It means that architecture seems like below:

    >>> unite = nn.Sequential(
    >>>     BatchNorm1d()
    >>>     ReLU()
    >>>     Conv1d()
    >>>     BatchNorm1d()
    >>>     ReLU()
    >>>     Conv1d()
    >>> )

    :param in_channels (int): number of input channels
    :param out_channels (int): number of output channels
    :param downsample (boolean): if true, then first convolution downsample
    :param activation (str): activation function which was used to apply before convolution (default: relu)
    """

    activations = nn.ModuleDict(
        {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "relu6": nn.ReLU6()
        }
    )

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 downsample: bool,
                 activation: Literal["relu", "leaky_relu", "relu6"] = "relu"
                 ):
        super().__init__()

        self.shortcut = nn.Sequential()
        self.activation = self.activations[activation]

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2 if downsample else 1),
            padding=1
        )

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        if downsample:
            self.shortcut = nn.Sequential(
                nn.BatchNorm1d(in_channels),
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=2
                )
            )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        shortcut = self.shortcut(x)
        x = self.conv1(self.activation(self.bn1(x)))
        x = self.conv2(self.activation(self.bn2(x)))
        return x + shortcut


class GlobalAvgPool1d(nn.Module):
    """
    Average tensor by 1 dimension.
    This pooling does not have kernel size and
    just average all features vector for each Conv1d kernel output.
    Note. This layer support batch processing.

    Input:
        Tensor with shape (N, K, W), where N is batch size,
        K is number of channels and W is feature vector for each convolution.

    Example:
         >>> y = torch.empty((10, 512, 40))
         >>> conv = nn.Conv1d(512, 1024, kernel_size=5)
         >>> avg = GlobalAvgPool1d()
         >>> c_y = conv(y) # Out shape is torch.Size([10, 1024, 36])
         >>> avg(c_y) # Out shape is torch.Size([10, 1024])
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.FloatTensor | torch.Tensor) -> torch.FloatTensor:
        if x.dim() == 3:
            return torch.mean(x, dim=2)
        elif x.dim() == 2:
            return torch.mean(x, dim=1)
        else:
            raise ValueError(
                f"Expected tensor with 2 or 3 dimensions, "
                f"but got {x.dim()}"
            )


class ResNet(nn.Module):
    """
    ResNet is class-builder for residual network with shortcut connetions by params.
    This implementation residual network for signal processing
    and use 1D convolution instead of 2D convolution as original implementation.

    :param input_shape (tuple): shape of input data
    :param stem_channels: number of channels in first convolution block
    :param backbone: config for network architecture building,
        where key is number of channels in one residual group
        and values it's number of residual units in this group
    """

    __backbones: dict = {
        "resnet18": {
            64: 2,
            128: 2,
            256: 2,
            512: 2
        },
        "resnet34": {
            64: 3,
            128: 4,
            256: 6,
            512: 3
        }
    }

    def __init__(self,
                 input_shape: tuple[int, int],
                 stem_channels: Optional[int] = None,
                 backbone: Optional[dict[int, int]] = None
                 ):
        super().__init__()
        if not isinstance(backbone, dict):
            backbone = self.__backbones["resnet18"]
        self.backbone = backbone

        if not isinstance(stem_channels, int):
            stem_channels = list(self.backbone.keys())[0]
        self.stem_channels = stem_channels

        self.input_shape = input_shape
        self.stem = self.__define_stem()
        self.residual_groups = self.__build_groups()
        self.avgpool = GlobalAvgPool1d()

    @property
    def out_features(self) -> int:
        return list(self.backbone.keys())[-1]

    @property
    def groups(self) -> zip:
        kernels = list([self.stem_channels])
        kernels.extend(list(self.backbone.keys()))
        channels_per_group = list(zip(kernels, kernels[1:]))
        unites_per_group = list(self.backbone.values())
        return zip(channels_per_group, unites_per_group)

    def __define_stem(self, kernel_size: Optional[int] = 7) -> nn.Sequential:
        conv = nn.Conv1d(
            in_channels=self.input_shape[0],
            out_channels=self.stem_channels,
            kernel_size=kernel_size,
            stride=2
        )
        maxpol = nn.MaxPool1d(kernel_size=3, stride=2)
        return nn.Sequential(conv, maxpol)

    @staticmethod
    def __build_group(unites: int, downsample: bool, channels: tuple[int, int]) -> nn.Sequential:
        group = nn.ModuleList()
        in_channels, out_channels = channels
        for index in range(unites):
            if index == 0:
                unite = ResidualUnite(in_channels, out_channels, downsample)
            else:
                unite = ResidualUnite(out_channels, out_channels, False)
            group.append(unite)
        return nn.Sequential(*group)

    def __build_groups(self) -> nn.Sequential:
        groups = nn.ModuleList()
        for index, (channels, units) in enumerate(self.groups):
            if index == 0:
                group = self.__build_group(
                    unites=units,
                    channels=channels,
                    downsample=False
                )
            else:
                group = self.__build_group(
                    unites=units,
                    channels=channels,
                    downsample=True
                )
            groups.append(group)
        return nn.Sequential(*groups)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.stem(x)
        x = self.residual_groups(x)
        return self.avgpool(x)
