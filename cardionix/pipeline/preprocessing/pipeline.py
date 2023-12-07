"""
Contains a ``ETLPipeline`` class which is a pipeline
for data preprocessing, feature extraction and augmentation.
"""

__all__ = ["ETLPipeline"]

from pathlib import Path
from typing import Literal, Optional
import torch
from torch.nn import Module
import numpy as np
import pandas as pd
from cardionix.configs import ETLPipelineParams
from .extractors import MFCCExtractor
from .preprocessors import AudioPreprocessor, TabularPreprocessor


class ETLPipeline(AudioPreprocessor, TabularPreprocessor):
    """
    ETLPipeline is a high-level API that implements methods for
    data preprocessing, feature extracting and augmentation audio and tabular data.
    Thus, it allows you to transform the data before issuing it.
    This class can be integrated into any ``Dataset`` subclass
    to process an audio sample before it is emitted, along with a class label.

    :param extractor: type of extractor which will be used to extract features from audi samples
    :param extractor_kwargs: extractor keyword arguments
    :param sample_rate: sample rate for all audio samples
    :param duration: duration for all audio samples
    :param mono: convert signal to mono
    :param pad_mode: pad mode, which will be used to pad audio samples
    :param augment_kwargs: dict, which key it name of type augmentation and values it probability of augmentation action
    :param merge_rules: dict, where keys is a class label and value is a dictionary with mergeable classes and probabilities
    :param merge_map: DataFrame with columns duration, class labels and fullpath for audio samples for current loop stage
    :param tabular: dataset, which will be used for define preprocessors and preprocessing data on train loop
    :param scaler: name of scaling function for scaling numerical features on train loop
    :param encoder: name of encoder function for encoding categorical features on train loop
    """

    def __init__(self,
                 extractor: Literal["MFCC"],
                 extractor_kwargs: Optional[dict] = None,
                 augment_kwargs: Optional[dict] = None,
                 merge_rules: Optional[dict] = None,
                 merge_map: Optional[pd.DataFrame] = None,
                 sample_rate: Optional[int] = 22050,
                 duration: Optional[int] = 10,
                 mono: Optional[bool] = True,
                 pad_mode: Literal["constant", "edge", "linear_ramp", "reflect", "symmetric", "merge_neighbors"] = "constant",
                 tabular: Optional[pd.DataFrame] = None,
                 scaler: Optional[Literal["StandardScaler", "Normalizer", "MinMaxScaler"]] = None,
                 encoder: Optional[Literal["OneHotEncoder"]] = None
                 ):

        super().__init__(
            extractor, extractor_kwargs,
            sample_rate, duration,
            mono, pad_mode
        )

        self.__augment_kwargs = augment_kwargs if augment_kwargs else {}
        self.__merge_map = merge_map
        self.__merge_rules = merge_rules
        self._define_preprocessors(tabular, scaler, encoder)

    @property
    def merge_map(self) -> pd.DataFrame:
        return self.__merge_map

    @property
    def merge_rules(self) -> dict:
        return self.__merge_rules

    def _define_preprocessors(self, tabular: pd.DataFrame, scaler: str, encoder: str) -> None:
        if isinstance(tabular, pd.DataFrame) and all([scaler, encoder]):
            super()._define_preprocessors(tabular, scaler, encoder)
        if isinstance(tabular, pd.DataFrame) and not all([scaler, encoder]):
            raise ValueError(
                f"Preprocessing error! "
                f"Expected 'scaler' and 'encoder' if data was passed, "
                f"but got {scaler=} and {encoder=}"
            )
        if not isinstance(tabular, pd.DataFrame) and any([scaler, encoder]):
            raise TypeError(
                f"Preprocessing error! "
                f"Expected 'tabular' type pd.Dataframe if 'scaler' and 'encoder' was passed, "
                f"but got {type(tabular)}"
            )

    def __get_compatible_label(self, label: str) -> str:
        merge_classes = self.__merge_rules[label]
        if isinstance(merge_classes, str):
            return merge_classes
        classes = list(merge_classes.keys())
        probs = list(merge_classes.values())
        return np.random.choice(classes, p=probs)

    def __get_compatible_neighbors(self, label: str, duration: float | int) -> pd.DataFrame:
        return self.__merge_map[
            (self.__merge_map["duration"] >= duration) &
            (self.__merge_map["label"] == label)
            ]

    def __get_neighbor(self, waveform: np.ndarray, label: str) -> np.ndarray:
        if self._is_proportional(waveform):
            label = self.__get_compatible_label(label)
        duration = self._get_pad_duration(waveform)
        neighbors = self.__get_compatible_neighbors(label, duration)
        filepath = neighbors["filepath"].sample().values[0]
        return self._load(filepath)

    def __find_label(self, filepath: str | Path) -> str:
        row = self.__merge_map[self.__merge_map["filepath"] == filepath]
        if len(row) == 0:
            raise FileNotFoundError(
                f"Audio sample with filepath '{filepath}' "
                f"was not found in merge map!"
            )
        return row["label"].values[0]

    def merge_available(self) -> bool:
        return all([
            isinstance(self.__merge_map, pd.DataFrame),
            isinstance(self.__merge_rules, dict)
        ])

    def merge_neighbors(self, filepath: str | Path) -> np.ndarray:
        if not self.merge_available():
            raise AttributeError(
                f"Before merging audio samples "
                f"is needed to specify merge_rules and merge_map, "
                f"but merge_rules is {self.__merge_rules} and merge_map is {self.__merge_map}"
            )
        waveform = self._load(filepath)
        label = self.__find_label(filepath)
        neighbor = self.__get_neighbor(waveform, label)
        return self._merge_samples(waveform, neighbor)

    def __noise_by(self, waveform: np.ndarray, p: float) -> np.ndarray:
        if not self.merge_available():
            return waveform
        duration = self.get_duration(waveform)
        neighbors = self.__get_compatible_neighbors(label="artifact", duration=duration)
        filepath = neighbors["filepath"].sample().values[0]
        noise = self._load(filepath)
        return self.noise_by(waveform, by=noise, p=p)

    def augment(self,
                waveform: np.ndarray,
                time_stretch: float = 0.5,
                value_augment: float = 0.5,
                pitch_shift: float = 0.5,
                noise_by: float = 0.5,
                gaussian_noise: float = 0.5,
                hpss: float = 0.5
                ) -> np.ndarray:

        waveform = self.time_stretch(waveform, p=time_stretch)
        waveform = self.value_augment(waveform, p=value_augment)
        waveform = self.pitch_shift(waveform, p=pitch_shift)
        waveform = self.__noise_by(waveform, p=noise_by)
        waveform = self.gaussian_noise(waveform, p=gaussian_noise)
        return self.hpss(waveform, p=hpss)

    def preprocess_audio(self, filepath: str | Path) -> torch.FloatTensor:
        """
        Performs step by step:
            - loading of an audio sample
            - ata preprocessing
            - augmentation
            - feature extraction
        Finally returns a set of features cast to a tensor type.
        """
        waveform = self.load_safely(filepath)
        waveform = self.augment(waveform, **self.__augment_kwargs)
        waveform = self.to_tensor_waveform(waveform)
        return self.extract_features(waveform)
