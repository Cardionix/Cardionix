"""
Docstring
"""

__all__ = ["CardioDataset"]

from typing import Literal, Union, Any, Optional
from pathlib import Path
import os

import pandas as pd

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic import FilePath, DirectoryPath, Field

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

    audio_dirpath: DirectoryPath
    labels_filepath: FilePath
    split_ratio: list[float] = Field(default=[0.75, 0.25], max_items=3, min_items=2)
    random_seed: Optional[int] = 42

    @field_validator("labels_filepath")
    def labels_filepath_validator(cls, value):
        extension = str(value).rsplit(".", maxsplit=1)[-1]
        if extension != "csv":
            raise ValueError(
                f"The file with an annotation of audio file classes "
                f"must have the extension .csv, but received {extension}"
            )
        return value

    @field_validator("split_ratio")
    def split_ratio_validator(cls, value):
        if sum(value) != 1.0:
            raise ValueError(
                f"The sum of all parts of the dataset partitions "
                f"should be equal to 1.0, but the result is {sum(value)}"
            )
        return value


class CardioDataset(Dataset):
    """
    Docstring
    """
    def __init__(self,
                 dataset_params: DatasetParams,
                 transform_params: TransformParams,
                 stage: Literal["train", "val", "test"]
                 ):

        self.__stage = self.check_stage(stage)
        self.__split_ratio = dataset_params.split_ratio
        self.__generator = torch.Generator().manual_seed(dataset_params.random_seed)
        self.__audio_dirpath = dataset_params.audio_dirpath
        self.__labels_df = pd.read_csv(dataset_params.labels_filepath)
        self.__transforms = ETLPipeline(transform_params, stage)
        self.__dataset = self.split_dataset()

        self.__classes_dict = {
            "normal": 0,
            "murmur": 1,
            "extrahls": 2,
            "extrastole": 3,
            "artifact": 4,
        }

    def __len__(self) -> int:
        return len(self.__dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename, label = self.__dataset[idx]
        filepath = os.path.join(self.__audio_dirpath, filename)
        waveform = self.__transforms(filepath)
        label = self.__classes_dict[label]
        return waveform, torch.tensor(label, dtype=torch.int32)

    def split_dataset(self) -> Subset:
        dataframe = self.__labels_df[["filename", "label"]]
        data = list(zip(list(dataframe.filename), list(dataframe.label)))
        split_datasets = random_split(data, self.__split_ratio, self.__generator)

        if self.__stage == "train":
            return split_datasets[0]
        elif self.__stage == "val":
            return split_datasets[1]
        elif self.__stage == "test" and len(self.__split_ratio) == 3:
            return split_datasets[2]
        else:
            raise ValueError(
                f"Expected split ratio for the val stage should be 2 or 3 "
                f"and for the test only 3, "
                f"but got split ratio {self.__split_ratio} and stage {self.__stage}"
            )

    @staticmethod
    def check_stage(stage: str) -> str:
        if type(stage) is not str:
            raise TypeError(f"Stage must be a string, but got {stage}")
        if stage not in ["train", "val", "test"]:
            raise ValueError(f"Expected stage to be train or val or test, but got {stage}")
        return stage
