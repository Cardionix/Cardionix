"""
The module contains a dataset subclass
for classifying heart rate deviations by sound.
The class contains methods that allow it to flexibly share data for different stages,
as well as determine which features will be issued
thanks to the ETLPipeline class that is integrated into it.
"""

__all__ = ["CardioAnomalyDataset"]

from typing import Literal, Optional
from pathlib import Path
import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, Subset

from .utils import DatasetPartitioner
from cardiosonix.configs import ClassifyDatasetParams, ETLPipelineParams
from ..preprocessing import ETLPipeline


class Builder(DatasetPartitioner):
    """
    Build dataset, perform initialization classes for current loop stage:
        - Create instance of ETLPipeline for feature extraction
        - Load and get path of dataset for current loop stage by seed
        - Create merge map for merging audio files
        - etc.
    """
    def __init__(self,
                 dataset_params: ClassifyDatasetParams,
                 etl_pipeline_params: ETLPipelineParams,
                 stage: Literal["train", "val", "test"]
                 ):

        self.__split_ratio = dataset_params.split_ratio
        self.__stage = stage
        self.__audio_dirpath = dataset_params.audio_dirpath
        self.__metadata_filepath = dataset_params.metadata_filepath
        self.__labels_filepath = dataset_params.labels_filepath
        self.__labels_dict = self.__init_classes(dataset_params.merge_classes)
        self.__rules_merge_classes = dataset_params.merge_classes

        self.__pipeline = ETLPipeline(
            tabular=self.__load(dataset_params.extra_filepath, exclude="label"),
            merge_map=self.__build_merge_map(),
            **etl_pipeline_params.model_dump()
        )

        self.__dataset = self.__build_dataset(
            dataset_params.labels_filepath,
            keep_instance=False,
            columns=["filename", "label"]
        )

        self.__extra_dataset = self.__build_dataset(
            dataset_params.extra_filepath,
            keep_instance=True
        )

    @property
    def labels(self) -> dict:
        return self.__labels_dict

    @property
    def dataset(self) -> Subset:
        return self.__dataset

    @property
    def extra(self) -> pd.DataFrame | None:
        return self.__extra_dataset

    @property
    def pipeline(self) -> ETLPipeline:
        return self.__pipeline

    @staticmethod
    def __init_classes(classes: dict) -> dict:
        return {
            key: value
            for key, value in zip(classes.keys(), range(len(classes)))
        }

    def _get_filepath(self, filename: str | Path) -> str:
        filename = self.this_file_in_dir(filename, self.__audio_dirpath)
        return os.path.join(self.__audio_dirpath, filename)

    def __build_fullpath(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["filename"] = dataframe["filename"].apply(self._get_filepath)
        return dataframe.rename(columns={"filename": "filepath"})

    def __merge_classes(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(dataset, pd.DataFrame):
            return None
        for merge_into, from_classes in self.__rules_merge_classes.items():
            dataset["label"] = dataset["label"].replace(from_classes, merge_into)
        return dataset

    def __build_merge_map(self) -> pd.DataFrame | None:
        if not self.__metadata_filepath:
            return None
        duration = self.__load(self.__metadata_filepath, "duration")
        merge_map = self.__build_dataset(self.__labels_filepath, True, ["filename", "label"])
        merge_map = self.is_all_unique(merge_map, "filename")
        merge_map["duration"] = duration.iloc[merge_map.index]
        return self.__build_fullpath(merge_map)

    def __build_dataset(self,
                        filepath: str | Path,
                        keep_instance: bool,
                        columns: Optional[list[str, ...]] = None
                        ) -> None | pd.DataFrame | pd.Series:

        dataset = self.__load(filepath, columns)
        if not isinstance(dataset, pd.DataFrame):
            return None
        dataset = self.__merge_classes(dataset)
        return self._get_section(
            dataset, self.__split_ratio, self.__stage, keep_instance
        )

    @classmethod
    def __load(cls,
               filepath: str | Path,
               include: Optional[list[str, ...]] = None,
               exclude: Optional[list[str, ...]] = None
               ) -> None | pd.DataFrame | pd.Series:

        if not filepath:
            return None
        data = pd.read_csv(filepath)

        if include:
            include = cls.columns_exists(data.columns, include, filepath)
            data = data[include]

        if exclude:
            exclude = cls.columns_exists(data.columns, exclude, filepath)
            data = data.drop(columns=exclude)
        return data


class CardioAnomalyDataset(Builder, Dataset):
    """
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

    def label_encoding(self, label: str) -> torch.IntTensor:
        return torch.tensor(self.labels[label], dtype=torch.uint8)

    def extra_sample(self, label: str) -> np.ndarray:
        extra = self.extra[self.extra["label"] == label]
        extra = extra.sample().drop(columns="label")
        return self.pipeline.preprocess_tabular(extra)

    def __len__(self) -> int:
        # Return length of dataset
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.FloatTensor, torch.uint8] | tuple[torch.FloatTensor, torch.uint8, torch.FloatTensor]:
        # Get label & filename
        filename, label = self.dataset[idx]
        filepath = self._get_filepath(filename)
        features = self.pipeline.preprocess_audio(filepath)
        target = self.label_encoding(label)
        # Return features with label and extra data if available
        if not isinstance(self.extra, pd.DataFrame):
            return features, target
        extra = self.extra_sample(label)
        return features, target, extra
