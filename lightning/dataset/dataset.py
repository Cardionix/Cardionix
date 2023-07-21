"""
Docstring
"""

__all__ = ["CardioDataset"]

from typing import Literal, Union, Any, Optional
from pathlib import Path
import os

import pandas as pd

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic import FilePath, DirectoryPath

import torch
from torch.utils.data import Dataset
from torch.nn import Module, Sequential, ModuleDict
from torch.utils.data import random_split

import torchaudio
from torchaudio import sox_effects
import torchaudio.transforms as T
import torchaudio.functional as F


class DatasetParams(BaseModel):
    """
    Docstring
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stage: Literal["train", "val", "test"]
    audio_dirpath: DirectoryPath
    labels_filepath: FilePath
    split_ratio: list[float]
    transforms: Union[Module, Sequential]
    random_seed: Optional[int] = 42

    @classmethod
    @field_validator("labels_filepath")
    def labels_filepath_validator(cls, value):
        extension = str(value).rsplit(".", maxsplit=1)[-1]
        if extension != "csv":
            raise ValueError(
                f"The file with an annotation of audio file classes "
                f"must have the extension .csv, but received {extension}"
            )
        return value

    @classmethod
    @field_validator("split_ratio")
    def split_ratio_validator(cls, value):
        if len(value) == 0 or len(value) > 3:
            raise ValueError(
                f"The number of dataset splits should be 1-3, "
                f"but received {len(value)}"
            )

        if sum(value) != 1.0:
            raise ValueError(
                f"The sum of all parts of the dataset partitions "
                f"should be equal to 1.0, but the result is {sum(value)}"
            )


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
