"""
Description:
    The module contains a dataset subclass
    for classifying heart rate deviations by sound.
    The class contains methods that allow it to flexibly share data for different stages,
    as well as determine which features will be issued
    thanks to the ETLPipeline class that is integrated into it.
"""

__all__ = ["CardioAnomalyDataset"]

from typing import Literal
import os

import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import random_split

import pandas as pd

from lightning.config import DatasetParams, ETLPipelineParams
from ..etl_pipeline import ETLPipeline


class CardioAnomalyDataset(Dataset):
    """
    Description:
        Dataset subclass for classifying heart rate deviations by sound.

    Args:
        dataset_params: (DatasetParams) subclass of ``BaseModel``
            containing parameters (configuration) for ``CardioAnomalyDataset`` initialization.

        etl_pipeline_params: (ETLPipelineParams) subclass of ``BaseModel``
            containing parameters (configuration) for ``ETLPipeline`` initialization.

        stage: (Literal["train", "val", "test"]) the data is broken down into chunks specific to each ``stage``.
            The ``stage`` argument passed will determine which part of the dataset to output at the given moment.
            Stages: ``training``, ``validation``, ``testing``.

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
        Splitting data into parts for different stages:
            1) training,
            2) validation
            3) testing.
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
        """Checking the stage argument against type and value."""
        if isinstance(type(stage), str):
            raise TypeError(f"Stage must be a string, but got {type(stage)}")
        if stage not in ["train", "val", "test"]:
            raise ValueError(f"Expected stage to be train or val or test, but got {stage}")
        return stage
