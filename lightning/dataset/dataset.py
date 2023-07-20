"""
Docstring
"""

__all__ = ["CardioDataset"]

from typing import Literal, Union, Any
from pathlib import Path
import os

import torch
from torch.utils.data import Dataset
from torch.nn import Module, Sequential, ModuleDict
from torch.utils.data import random_split

import torchaudio
from torchaudio import sox_effects
import torchaudio.transforms as T
import torchaudio.functional as F

import pandas as pd


class CardioDataset(Dataset):
    """
    Docstring
    """
    def __init__(self,
                 stage: Literal["train", "val", "test"],
                 data_dirpath: str | Path,
                 target_filepath: str | Path,
                 transforms: Union[Module, Sequential, Any]
                 ):

        self.stage = stage
        self.data_dirpath = data_dirpath
        self.target_df = pd.read_csv(target_filepath)
        self.transforms = transforms

        self.classes_dict = {
            "normal": 0,
            "murmur": 1,
            "extrahls": 2,
            "extrastole": 3,
            "artifact": 4,
        }

    def __len__(self) -> int:
        return len(os.listdir(self.data_dirpath))

    def __getitem__(self, idx: int) -> tuple:
        row = self.target_df.iloc[idx]
        path = os.path.join(self.data_dirpath, row.filename)
        wave = self.transforms(path)
        label = self.classes_dict[row.label]
        return wave, label
