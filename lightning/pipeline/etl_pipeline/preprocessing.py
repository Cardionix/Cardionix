"""
Contains a ``ETLPipeline`` class which is a pipeline
for data preprocessing, feature extraction and augmentation.
"""

__all__ = ["Upload"]

from typing import Optional
import numpy as np
import torch
from torch.nn import Module
import librosa


class Upload(Module):
    def __init__(self,
                 sample_rate: Optional[int] = 22050,
                 duration: Optional[int] = 10,
                 mono: Optional[bool] = True,
                 ):

        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono
        self.length = self.duration * self.sample_rate

    def check_duration(self, waveform: np.ndarray) -> np.ndarray:
        """
        Сhecks the duration of the audio recording.
        If the duration is not equal to the specified duration argument,
        then the audio sample is scaled to the specified length and returned.
        Otherwise, the audio sample will return unchanged.

        Args:
            waveform: (np.ndarray) audio sample represented as an array with a set of amplitudes.
        """

        duration = librosa.get_duration(y=waveform, sr=self.sample_rate)
        if duration != self.duration:
            waveform = librosa.util.fix_length(waveform, size=self.length)
        return waveform

    def forward(self, filepath: str) -> torch.Tensor:
        """
        Performs loading of an audio sample, data preprocessing, augmentation and feature extraction step by step.
        Finally returns a set of features cast to a tensor type

        Args:
            filepath: (str) path to the audio sample file
        """

        waveform, _ = librosa.load(
            path=filepath,
            sr=self.sample_rate,
            mono=self.mono,
        )

        waveform = self.check_duration(waveform)
        return torch.tensor(np.expand_dims(waveform, axis=0), dtype=torch.float32)
