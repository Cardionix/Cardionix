"""
This module contains classes for preprocessing and standardization tabular and audio data.
"""

__all__ = [
    "TabularPreprocessor",
    "AudioPreprocessor"
]

from pathlib import Path
from typing import Optional, Literal

import numpy as np
import torch

from librosa.util import fix_length
from librosa import get_duration, load
from librosa.effects import hpss, time_stretch, pitch_shift


class AudioAugmentations:
    @staticmethod
    def __augment_this(p: float = 0.5) -> bool:
        return np.random.choice([True, False], p=[p, 1.0-p]) 

    def hpss(self, waveform: np.ndarray, p: float = 0.5) -> np.ndarray:
        if not self.__augment_this(p):
            return waveform
        coeff_percussive = np.random.uniform(low=0, high=1)
        y_harmonic, y_percussive = hpss(waveform)
        return (1 - coeff_percussive) * y_harmonic + y_percussive * coeff_percussive

    def value_augment(self, waveform: np.ndarray, p: float = 0.5) -> np.ndarray:
        if not self.__augment_this(p):
            return waveform
        value = np.random.uniform(low=0.5, high=3)
        return waveform * value
    
    def time_stretch(self, waveform: np.ndarray, keep_size: bool = True, p: float = 0.5) -> np.ndarray:
        if not self.__augment_this(p):
            return waveform
        rate = np.random.uniform(low=0.5, high=1.8)
        time_stretched = time_stretch(y=waveform, rate=rate)
        if keep_size:
            return fix_length(time_stretched, size=len(waveform), mode="constant") 
        return time_stretched
    
    def gaussian_noise(self, waveform: np.ndarray, p: float = 0.5) -> np.ndarray:
        if not self.__augment_this(p):
            return waveform
        noise_coeff = 0.005 * np.random.uniform() * np.amax(waveform)
        noise = np.random.normal(size=len(waveform))
        return waveform + (noise * noise_coeff)

    def noise_by(self, waveform: np.ndarray, by: np.ndarray, p: float = 0.5) -> np.ndarray:
        if not self.__augment_this(p):
            return waveform
        noise_coeff = np.random.uniform(low=0.05, high=0.3)
        by = fix_length(by, size=len(waveform), mode="constant")
        return (1 - noise_coeff) * waveform + by * noise_coeff 

    def _pitch_shift(self, waveform: np.ndarray, sr: int, p: float = 0.5) -> np.ndarray:
        if not self.__augment_this(p):
            return waveform
        n_steps =  np.random.uniform(low=-5, high=5)
        return pitch_shift(
            waveform, sr=sr,
            n_steps=n_steps, 
        )
    

class AudioPreprocessor(AudioAugmentations):
    __extractors: dict = {
        "MFCC": MFCCExtractor
    }
    
    __padding: list = [
        "constant",
        "edge", 
        "linear_ramp", 
        "reflect", 
        "symmetric", 
        "nearest_neighbors"
    ]
    
    def __init__(self, 
                 extractor: Literal["MFCC"],
                 extractor_kwargs: Optional[dict] = None,
                 sample_rate: Optional[int] = 22050,
                 duration: Optional[int] = 10,
                 mono: Optional[bool] = True,
                 pad_mode: Literal["constant", "edge", "linear_ramp", "reflect", "symmetric", "nearest_neighbors"] = "constant"
                 ):
        super().__init__()
        self.__sample_rate = sample_rate
        self.__duration = duration
        self.__frames = self.count_frames(sample_rate, duration)
        self.__mono = mono
        self.__pad_mode = self.__check_pade_mode(pad_mode)
        self.__extractor = self.__get_extractor(
            extractor, sample_rate, 
            **extractor_kwargs if extractor_kwargs else {}
        )

    @property
    def pad_mode(self) -> str:
        return self.__pad_mode

    @property
    def duration(self) -> int:
        return self.__duration
    
    @property
    def frames(self) -> int:
        return self.__frames

    @property
    def sample_rate(self) -> int:
        return self.__sample_rate
    
    @classmethod
    def __check_pade_mode(cls, pad_mode: str) -> str:
        if pad_mode not in cls.__padding:
            raise ValueError(
                f"Expected pad_mode argument must be "
                f"one of the list {cls.__padding}, but got {pad_mode}"
            )
        return pad_mode

    @classmethod
    def __get_extractor(cls, 
                        extractor: Literal["MFCC"],
                        sample_rate: int,
                        **extractor_kwargs
                        ) -> Module:
        return cls.__extractors[extractor](
            sample_rate=sample_rate,
            **extractor_kwargs
        )

    @staticmethod
    def __get_merge_side() -> Literal["right", "left"]:
        return np.random.choice(["right", "left"], p=[0.5, 0.5])
    
    def _load(self, filepath: str | Path) -> np.ndarray:
        return load(filepath, sr=self.__sample_rate, mono=self.__mono)[0]

    def _get_pad_duration(self, waveform: np.ndarray) -> float:
        return self.__duration - get_duration(y=waveform, sr=self.__sample_rate)

    def _is_proportional(self, waveform: np.ndarray, gate: float = 0.5) -> bool:
        duration = get_duration(y=waveform, sr=self.__sample_rate)
        if duration / self.__duration >= gate:
            return True
        return False

    def _merge_samples(self, waveform: np.ndarray, to_merge: np.ndarray) -> np.ndarray: 
        side = self.__get_merge_side()
        to_concat = [waveform, to_merge] if side == "right" else [to_merge, waveform]
        waveform = np.concatenate(to_concat, axis=0)
        return self.clip(waveform, side) 

    @staticmethod
    def count_frames(sample_rate: int, duration: int) -> int:
        if isinstance(sample_rate, int) and isinstance(duration, int):
            return sample_rate * duration
        
        if not isinstance(sample_rate, int):
            raise ValueError(
                f"Expected sample_rate to be int, "
                f"but got {type(sample_rate)}"
            )
        
        if not isinstance(duration, int):
            raise ValueError(
                f"Expected duration to be int, "
                f"but got {type(duration)}"
            )

    @staticmethod
    def to_tensor_waveform(waveform: np.ndarray) -> torch.FloatTensor:
        return torch.tensor(np.expand_dims(waveform, axis=0), dtype=torch.float32)
    
    def extract_features(self, waveform: torch.FloatTensor) -> torch.FloatTensor:
        return self.__extractor(waveform)
    
    def get_duration(self, waveform: np.ndarray) -> float:
        return get_duration(y=waveform, sr=self.__sample_rate)

    def clip(self, array: np.ndarray, side: Optional[Literal["right", "left"]] = None) -> np.ndarray:
        if len(array) == self.__frames:
            return array
        if side == "right" or not side:
            return array[:self.__frames]
        return array[-self.__frames:]

    def load_safely(self, filepath: str | Path) -> np.ndarray:
        waveform = self._load(filepath)
        duration = self.get_duration(waveform)
        if duration >= self.__duration:
            return self.clip(waveform)
        if self.__pad_mode == "nearest_neighbors":
            return self.merge_neighbors(filepath)
        return fix_length(waveform, size=self.__frames, mode=self.__pad_mode)   

    def pitch_shift(self, waveform: np.ndarray, p: float = 0.5) -> np.ndarray:
        return self._pitch_shift(waveform, sr=self.__sample_rate, p=p)

    def merge_neighbors(self, filepath: str | Path) -> np.ndarray:
        raise NotImplementedError(
            f"You must implement method which will load audio sample, "
            f"find his neighbors by class label and duration, "
            f"merge them and then clip the result "
            f"if it need to keep required duration of audio sample"
        )  

    def augment(self,
                waveform: np.ndarray,
                time_stretch: float, 
                value_augment: float,
                pitch_shift: float, 
                noise_by: float, 
                gaussian_noise: float, 
                hpss: float
                ) -> np.ndarray:
        raise NotImplementedError(
            "You must implement method which "
            "will augment the waveform all available ways. "
            "Use for this methods from AudioAugmentations class"
        )
    
    def preprocess_audio(self, filepath: str | Path) -> torch.FloatTensor:
        raise NotImplementedError(
            f"You must implement method which will "
            f"load, preprocess, augment and extract features from audio samples"
        )

