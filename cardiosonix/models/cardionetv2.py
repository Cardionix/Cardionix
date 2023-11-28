"""
This module contain implementation of multimodal neural network
for audio and tabular features inference.
"""

__all__ = ["CardioNetV2"]

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

    @staticmethod
    def batch(x: torch.FloatTensor) -> list:
        samples: list = []
        for sample in x:
            sample = sample.mean(1)
            samples.append(sample)
        return samples

    def forward(self, x: torch.FloatTensor | torch.Tensor) -> torch.FloatTensor:
        if x.dim() == 3:
            x = self.batch(x)
            return torch.vstack(x)
        elif x.dim() == 2:
            return x.mean(1)
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


class Concat1d(nn.Module):
    """
    Concatenate each tensors by dimision.
    Note.
    Support filtering of iterable object and batch processing.

    Example:
        >>> concat = Concat1d()
        >>> y = torch.empty((10, 512))
        >>> y1 = torch.empty((10, 512))
        >>> out = concat((y, y1))
        >>> out.shape # torch.Size([10, 1024])
    """
    def __init__(self, dim: Optional[int] = 0):
        super().__init__()
        self.dim = dim

    @staticmethod
    def __check_sanity(x: list | tuple) -> list:
        return [
            data.squeeze()
            for data in x
            if isinstance(data, (torch.FloatTensor, torch.Tensor))
        ]

    def __batch_concat(self,
                       x: list[torch.FloatTensor, torch.FloatTensor] | tuple[torch.FloatTensor, torch.FloatTensor]
                       ) -> torch.FloatTensor:
        samples: list = []
        batch_size = x[0].shape[0]
        for i in range(batch_size):
            to_concat = [x[0][i], x[1][i]]
            sample = torch.concatenate(to_concat, dim=self.dim)
            sample = sample.unsqueeze(self.dim)
            samples.append(sample)
        return torch.concatenate(samples, dim=self.dim)

    @staticmethod
    def __is_batch(x: list[torch.FloatTensor, ...] | tuple[torch.FloatTensor, ...]) -> bool:
        if x[0].dim() == 1:
            return False
        return True

    def __concat(self,
                 x: list[torch.FloatTensor, torch.FloatTensor] | list[torch.FloatTensor]
                 ) -> torch.FloatTensor:
        if len(x) == 1:
            return x[0]
        if len(x) == 2 and not self.__is_batch(x):
            return torch.concatenate(x, dim=self.dim)
        return self.__batch_concat(x)

    def forward(self,
                x: torch.FloatTensor | list[torch.FloatTensor | None] | list[torch.FloatTensor | torch.FloatTensor]
                ) -> torch.FloatTensor:
        if isinstance(x, torch.FloatTensor):
            return x
        x = self.__check_sanity(x)
        return self.__concat(x)


class DenseMixer(nn.Module):
    """
    Mixer for multimodal data inputs.
    This mixer build unique dense networks for audio and tabular features,
        do forward and concatenate them outputs for mixed inference.

    :param input_features: dictionary with number of audio and tabular features
    :param num_classes: number of classes
    :param depth: dictionary with lists of integers where each integer is number of features for linear layer
    :param from_logites: return logites if True and probabilities if False
    """
    def __init__(self,
                 input_features: dict[Literal["audio", "tabular"], int],
                 num_classes: int,
                 depth: dict[Literal["audio", "tabular", "mixer"], list],
                 from_logites: Optional[bool] = True
                 ):
        super().__init__()
        if not isinstance(depth, dict):
            depth = {"mixer": [1024]}

        self.__from_logites = from_logites
        self.__audio_features = input_features.get("audio", 0)
        self.__tabular_features = input_features.get("tabular", 0)
        self.__check_features()
        self.__audio_depth = [self.__audio_features, *depth.get("audio", [])]
        self.__tabular_depth = [self.__tabular_features, *depth.get("tabular", [])]
        self.__mixer_depth = [self.mix_features, *depth["mixer"], num_classes]
        self.audio_fc = self.__build_fc(self.audio_depth)
        self.tabular_fc = self.__build_fc(self.tabular_depth)
        self.mixer = self.__build_mixer()

    @property
    def mix_features(self) -> int:
        return self.__audio_depth[-1] + self.__tabular_depth[-1]

    @property
    def mixer_depth(self) -> list:
        return self.__define_depth(self.__mixer_depth)

    @property
    def audio_depth(self) -> list:
        return self.__define_depth(self.__audio_depth)

    @property
    def tabular_depth(self) -> list | None:
        if self.__tabular_features != 0:
            return self.__define_depth(self.__tabular_depth)
        return None

    @staticmethod
    def __define_depth(depth: list | tuple) -> list:
        return list(zip(depth, depth[1:]))

    def __check_features(self):
        if self.__audio_features + self.__tabular_features == 0:
            raise ValueError(
                f"Expected at lest one number of features "
                f"in 'input_features' dict, but got "
                f"{self.__audio_features} audio "
                f"and {self.__tabular_features} tabular features"
            )

    def __build_mixer(self) -> nn.Sequential:
        head = nn.ModuleList([Concat1d()])
        for index, (in_features, out_features) in enumerate(self.mixer_depth):
            if index != (len(self.mixer_depth) - 1):
                head.append(self.get_fcc(in_features, out_features))
            else:
                head.append(nn.Linear(in_features, out_features))
        if not self.__from_logites:
            head.append(nn.Softmax(dim=0))
        return nn.Sequential(*head)

    def __build_fc(self, depth: list) -> nn.Sequential:
        if isinstance(depth, list):
            return nn.Sequential(*[
                self.get_fcc(in_features, out_features)
                for in_features, out_features in depth
            ])
        return nn.Sequential()

    @staticmethod
    def get_fcc(in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )

    def forward(self,
                audio_features: torch.FloatTensor,
                tabular_features: Optional[torch.FloatTensor] = None
                ) -> torch.FloatTensor:
        audio_features = self.audio_fc(audio_features)
        tabular_features = self.tabular_fc(tabular_features)
        return self.mixer((audio_features, tabular_features))


class CardioNetV2(nn.Module):
    """
    Multi-modal neural network for audio and tabular inference.
    This network consists from:
        * Reccurent layrs (LSTM)
        * Residual groups (ResNet for audio signal inference)
        * Fully connected layers for mixing tabular and audio features (DenseMixer)

    :param num_classes: number of classes
    :param audio_features_shape: shape of audio features
    :param tabular_features: number of tabular features
    :param rnn_layers: number of LSTM layers
    :param rnn_hidden: size of hidden LSTM layers
    :param dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
        LSTM layer except the last layer, with dropout probability equal to
        :attr:`dropout`. Default: 0
    :param bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
    :param mixer_depth: dictionary with lists of integers where each integer is number of features for linear layer
    :param stem_channels: number of channels in first ResNet convolution block
    :param backbone: config for network architecture building,
        where key is number of channels in one residual group
        and values it's number of residual units in this group.
    :param from_logites: return logites if True and probabilities if False
    """
    def __init__(self,
                 num_classes: int,
                 audio_features_shape: tuple[int, int],
                 tabular_features: Optional[int] = None,
                 rnn_layers: Optional[int] = 2,
                 rnn_hidden: Optional[int] = None,
                 bidirectional: Optional[bool] = True,
                 dropout: Optional[float] = 0.0,
                 resnet_backbone: Optional[dict] = None,
                 mixer_depth: Optional[dict[Literal["audio", "tabular", "mixer"], int]] = None,
                 stem_channels: Optional[int] = None,
                 from_logites: Optional[bool] = True
                 ):
        super().__init__()
        self.audio_features_shape = audio_features_shape
        self.tabular_features = tabular_features
        self.bidirectional = bidirectional
        self.rnn_hidden = rnn_hidden if rnn_hidden else self.__get_hidden_size()

        self.rnn = nn.LSTM(
            input_size=audio_features_shape[1],
            hidden_size=self.rnn_hidden,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=rnn_layers,
            dropout=dropout
        )

        self.resnet = ResNet(
            self.__get_resnet_input_shape(),
            stem_channels,
            resnet_backbone
        )

        self.mixer = DenseMixer(
            self.__get_mixer_input_features(),
            num_classes=num_classes,
            depth=mixer_depth,
            from_logites=from_logites
        )

    @property
    def example_input_array(self) -> torch.FloatTensor | list[torch.FloatTensor, torch.FloatTensor]:
        example_input = [torch.zeros(size=(1, *self.audio_features_shape))]
        if self.tabular_features:
            example_input.append(torch.zeros(size=(1, 1, self.tabular_features)))
        return example_input

    def __get_resnet_input_shape(self) -> tuple[int, int]:
        features = self.rnn_hidden * 2 if self.bidirectional else self.rnn_hidden
        return self.audio_features_shape[0], features

    def __get_mixer_input_features(self) -> dict[int, int]:
        return {
            "audio": self.resnet.out_features,
            "tabular": self.tabular_features if self.tabular_features else 0
        }

    def __get_hidden_size(self) -> int:
        features = self.audio_features_shape[1]
        return int(features * (2 / 3)) + features

    def audio_forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        output, (cell, hidden) = self.rnn(x)
        return self.resnet(output)

    def sanity_check(self, tabular_features: torch.FloatTensor | None) -> torch.FloatTensor | None:
        if not isinstance(tabular_features, (torch.Tensor, torch.FloatTensor)) and self.tabular_features:
            raise ValueError(
                f"Expected 'tabular_features' must be a torch.FloatTensor or torch.Tensor "
                f"for multimodal forward, but got {type(tabular_features)}"
            )
        if isinstance(tabular_features, (torch.Tensor, torch.FloatTensor)) and not self.tabular_features:
            raise ValueError(
                f"You must define number of tabular features "
                f"before calling forward for multimodal forward support, "
                f"but number of tabular features is {self.tabular_features}"
            )
        return tabular_features

    def forward(self,
                audio_features: torch.FloatTensor,
                tabular_features: Optional[torch.FloatTensor] = None
                ) -> torch.FloatTensor:
        tabular_features = self.sanity_check(tabular_features)
        audio_features = self.audio_forward(audio_features)
        return self.mixer(audio_features, tabular_features)
