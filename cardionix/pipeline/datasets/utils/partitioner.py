"""
This module implements the interface for datasets parthings.
"""

__all__ = ["DatasetPartitioner"]

from typing import Literal
import pytorch_lightning as pl
import torch
from torch.utils.data import Subset
from torch.utils.data import random_split
import numpy as np
import pandas as pd
from .health_check import HealthChecker


class DatasetPartitioner(HealthChecker):
    """
    This class is a high-level API for datasets parthings by global seed for deterministic.
    Note:
        If seed was not found in env variables
        then it will be generated and set as 'PL_GLOBAL_SEED'.
    """

    def __init__(self):
        self.__seed = pl.seed_everything()
        self.__stages: dict = {
            "train": 0,
            "val": 1,
            "test": 2
        }

    @property
    def stages(self):
        return self.__stages

    @property
    def seed(self) -> int:
        return self.__seed

    @property
    def generator(self) -> torch.Generator:
        return torch.Generator().manual_seed(self.seed)

    def __random_split(self,
                       data: list | np.ndarray | pd.DataFrame,
                       split_ratio: list[float]
                       ) -> list[Subset]:

        return random_split(
            data, split_ratio,
            generator=self.generator
        )

    def __split_array(self,
                      data: list | np.ndarray,
                      split_ratio: list[float],
                      keep_instance: bool
                      ) -> list[Subset | list | np.ndarray, ...]:

        if not keep_instance:
            return self.__random_split(data, split_ratio)
        subsets = self.__random_split(data, split_ratio)

        return [
            list(subset)
            if isinstance(data, list)
            else np.array(list(subset))
            for subset in subsets
        ]

    def __split_dataframe(self,
                          data: pd.DataFrame,
                          split_ratio: list[float],
                          keep_instance: bool
                          ) -> list[Subset | pd.DataFrame | list | np.ndarray, ...]:

        if not keep_instance:
            return self.__random_split(data.values, split_ratio)
        index_subsets = self.__random_split(data.index, split_ratio)
        return [data.iloc[list(index)] for index in index_subsets]

    def __to_subsets(self,
                     dataset: list | np.ndarray | pd.DataFrame,
                     split_ratio: list[float, ...],
                     keep_instance: bool
                     ) -> list[Subset | pd.DataFrame | list | np.ndarray, ...]:

        if isinstance(dataset, (list, np.ndarray)):
            return self.__split_array(dataset, split_ratio, keep_instance)
        if isinstance(dataset, pd.DataFrame):
            return self.__split_dataframe(dataset, split_ratio, keep_instance)
        raise TypeError(
            f"Expected dataset must be a list, "
            f"ndarray or DataFrame, but got {type(dataset)}"
        )

    def _get_section(self,
                     dataset: list | np.ndarray | pd.DataFrame,
                     split_ratio: list[float, ...],
                     stage: Literal["train", "val", "test"],
                     keep_instance: bool = False
                     ) -> Subset | pd.DataFrame | list | np.ndarray:

        split_ratio = self.check_split_ratio(split_ratio)
        stage = self.check_stage(stage, self.stages)
        subsets = self.__to_subsets(dataset, split_ratio, keep_instance)
        subset_index = self.stages[stage]
        return subsets[subset_index]
