"""
Docstring
"""

__all__ = ["ETLPipeline"]

from typing import Literal
import numpy as np
import torch
from torch.nn import Module
import librosa


class ETLPipeline(Module):
    """
    Docstring
    """
    def __init__(self,
                 transform_params: TransformParams,
                 stage: Literal["train", "val", "test"]
                 ):
        super().__init__()

        self.__n_mfcc = transform_params.n_mfcc
        self.__duration = transform_params.duration
        self.__sample_rate = transform_params.sample_rate
        self.__mono = transform_params.mono
        self.__length = self.__duration * self.__sample_rate
        self._stage = stage

    def check_duration(self, waveform: np.ndarray) -> torch.Tensor:
        duration = librosa.get_duration(y=waveform, sr=self.__sample_rate)
        if duration < self.__duration:
            waveform = librosa.util.fix_length(waveform, size=self.__length)
        return waveform

    def extract_mfcc(self, waveform: np.ndarray) -> np.ndarray:
        mfcc = np.mean(
            librosa.feature.mfcc(
                y=waveform,
                sr=self.__sample_rate,
                n_mfcc=self.__n_mfcc).T, axis=0
        )
        return mfcc.reshape([-1, 1])

    def forward(self, filepath: str) -> torch.Tensor:
        waveform, _ = librosa.load(
            path=filepath,
            sr=self.__sample_rate,
            mono=self.__mono,
            duration=self.__duration
        )

        waveform = self.check_duration(waveform)
        features = self.extract_mfcc(waveform)
        features = torch.tensor(features, dtype=torch.float32)
        return features
