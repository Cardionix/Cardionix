"""
Contains a ``ETLPipeline`` class which is a pipeline
for data preprocessing, feature extraction and augmentation.
"""

__all__ = ["ETLPipeline"]

from typing import Literal
import numpy as np
import torch
from torch.nn import Module
import librosa

from ..config import ETLPipelineParams


class ETLPipeline(Module):
    """
    ETLPipeline is a high-level API that implements methods for
    data preprocessing, feature extraction and augmentation.
    Thus, it allows you to transform the data before issuing it.
    This class must be integrated into any ``Dataset`` subclass
    to process an audio sample before it is emitted, along with a class label.

    Args:
        etl_pipeline_params: (ETLPipelineParams) subclass of ``BaseModel``
            containing parameters (configuration) for ``ETLPipeline`` initialization.

        stage: (Literal["train", "val", "test"]) depending on the stage,
            the transformations that will take place with the data will be determined.
            The stage argument passed will determine the transformations that will be applied to the data.
            For example, with stage ``train``, we want to apply data augmentation,
            but during ``validation``, we want to check the accuracy of the model
            on a clean data without additional augmentation.
            Stages: ``training``, ``validation``, ``testing``.
    """
    def __init__(self,
                 etl_pipeline_params: ETLPipelineParams,
                 stage: Literal["train", "val", "test"]
                 ):
        super().__init__()

        self.__n_mfcc = etl_pipeline_params.n_mfcc
        self.__duration = etl_pipeline_params.duration
        self.__sample_rate = etl_pipeline_params.sample_rate
        self.__mono = etl_pipeline_params.mono
        self.__length = self.__duration * self.__sample_rate
        self._stage = stage

    def check_duration(self, waveform: np.ndarray) -> np.ndarray:
        """
        Сhecks the duration of the audio recording.
        If the duration is not equal to the specified duration argument,
        then the audio sample is scaled to the specified length and returned.
        Otherwise, the audio sample will return unchanged.

        Args:
            waveform: (np.ndarray) audio sample represented as an array with a set of amplitudes.
        """
        duration = librosa.get_duration(y=waveform, sr=self.__sample_rate)
        if duration != self.__duration:
            waveform = librosa.util.fix_length(waveform, size=self.__length)
        return waveform

    def extract_mfcc(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract MFCC coefficients and average them.

        Args:
            waveform: (np.ndarray) audio sample represented as an array with a set of amplitudes.
        """
        return np.mean(
            librosa.feature.mfcc(
                y=waveform,
                sr=self.__sample_rate,
                n_mfcc=self.__n_mfcc).T, axis=0
        )

    def forward(self, filepath: str) -> torch.Tensor:
        """
        Performs loading of an audio sample, data preprocessing, augmentation and feature extraction step by step.
        Finally returns a set of features cast to a tensor type

        Args:
            filepath: (str) path to the audio sample file
        """
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
