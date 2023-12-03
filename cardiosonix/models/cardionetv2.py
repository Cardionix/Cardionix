"""
This module contain implementation of multimodal neural network
for audio and tabular features inference.
"""

__all__ = ["CardioNetV2"]

from typing import Optional, Literal
import torch
from torch import nn
from .utils import DenseMixer, ResNet


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
