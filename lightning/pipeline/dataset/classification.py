"""
Docstring
"""

__all__ = ["CardioAnomalyDataset"]

from typing import Literal
import os

import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import random_split

import pandas as pd

from ..etl_pipeline import ETLPipeline
from lightning.validate import DatasetParams, ETLPipelineParams


class CardioAnomalyDataset(Dataset):
    """
    Docstring
    """
    def __init__(self,
                 dataset_params: DatasetParams,
                 etl_pipeline_params: ETLPipelineParams,
                 stage: Literal["train", "val", "test"]
                 ):

        self.__stage = self.check_stage(stage)
        self.__split_ratio = dataset_params.split_ratio
        self.__generator = torch.Generator().manual_seed(dataset_params.random_seed)
        self.__audio_dirpath = dataset_params.audio_dirpath
        self.__labels_df = pd.read_csv(dataset_params.labels_filepath)
        self.__transforms = ETLPipeline(etl_pipeline_params, stage)
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
        """
        Docstring
        """
        dataframe = self.__labels_df[["filename", "label"]]
        data = list(zip(list(dataframe.filename), list(dataframe.label)))
        split_datasets = random_split(data, self.__split_ratio, self.__generator)

        if self.__stage == "train":
            dataset = split_datasets[0]
        elif self.__stage == "val":
            dataset = split_datasets[1]
        elif self.__stage == "test" and len(self.__split_ratio) == 3:
            dataset = split_datasets[2]
        else:
            raise ValueError(
                f"Expected split ratio for the val stage should be 2 or 3 "
                f"and for the test only 3, "
                f"but got split ratio {self.__split_ratio} and stage {self.__stage}"
            )
        return dataset

    @staticmethod
    def check_stage(stage: str) -> str:
        """
        Docstring
        """
        if isinstance(type(stage), str):
            raise TypeError(f"Stage must be a string, but got {type(stage)}")
        if stage not in ["train", "val", "test"]:
            raise ValueError(f"Expected stage to be train or val or test, but got {stage}")
        return stage
