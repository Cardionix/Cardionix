"""
Mixer for multimodal data inputs.
This mixer build unique dense networks for audio and tabular features,
do forward and concatenate them outputs for mixed inference.
"""

__all__ = ["DenseMixer"]

from typing import Optional, Literal
import torch
from torch import nn


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
            depth = {}

        self.__from_logites = from_logites
        self.__audio_features = input_features.get("audio", 0)
        self.__tabular_features = input_features.get("tabular", 0)
        self.__check_features()
        self.__audio_depth = [self.__audio_features, *depth.get("audio", [])]
        self.__tabular_depth = [self.__tabular_features, *depth.get("tabular", [])]
        self.__mixer_depth = [self.mix_features, *depth.get("mixer", []), num_classes]
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
            nn.ReLU(),
        )

    def forward(self,
                audio_features: torch.FloatTensor,
                tabular_features: Optional[torch.FloatTensor] = None
                ) -> torch.FloatTensor:
        audio_features = self.audio_fc(audio_features)
        tabular_features = self.tabular_fc(tabular_features)
        return self.mixer((audio_features, tabular_features))
