"""
This module contains classes and functions
for checking correctness of the building and parthing datasets.
"""

__all__ = ["HealthChecker"]

import os
from typing import Optional
from pathlib import Path
import pandas as pd


class HealthChecker:
    """
    This class provides methods for checking correctness:
        - building datasets
        - parthing datasets
        - etc.
    """

    @staticmethod
    def check_stage(stage: str, stages: list) -> str:
        """Checking the stage argument against type and value."""
        if not isinstance(stage, str):
            raise TypeError(
                f"Dataset building error! "
                f"stage must be a str, "
                f"but got {type(stage)}"
            )

        if stage not in stages:
            raise ValueError(
                f"Dataset building error! "
                f"Expected stage to be 'train' "
                f"or 'val' or 'test', but got '{stage}'"
            )
        return stage

    @staticmethod
    def check_split_ratio(split_ratio: list[float, ...]) -> list[float, ...]:
        if not isinstance(split_ratio, list):
            raise TypeError(
                f"Dataset building error! "
                f"split_ratio must be a list, "
                f"but got {type(split_ratio)}"
            )

        if sum(split_ratio) != 1:
            raise ValueError(
                f"Dataset building error! "
                f"Sum of values split_ratio must be 1, "
                f"but got {sum(split_ratio)}"
            )
        return split_ratio

    @staticmethod
    def columns_exists(columns: list | tuple,
                       must_exist: list | tuple,
                       dataframe_path: Optional[str | Path] = None
                       ) -> list | tuple:

        if not isinstance(must_exist, (list, tuple)):
            must_exist = [must_exist]
        if not isinstance(dataframe_path, str):
            dataframe_path = "Dataframe"

        for column in must_exist:
            if column not in columns:
                raise ValueError(
                    f"Dataset building error! "
                    f"Column with name '{column}' "
                    f"does not exists in {dataframe_path}"
                )
        return must_exist

    @staticmethod
    def is_all_unique(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
        must_have = len(dataframe)
        have = len(dataframe[column].unique())
        if have != must_have:
            raise ValueError(
                f"Dataset building error! "
                f"Column with name '{column}' "
                f"must have only unique values, "
                f"but only {have} values of {must_have} is unique"
            )
        return dataframe

    @staticmethod
    def this_file_in_dir(filename: str, dirpath: str | Path) -> str:
        if filename not in os.listdir(dirpath):
            raise FileExistsError(
                f"Dataset building error! "
                f"File with name '{filename}' does not exist "
                f"in directory '{dirpath}'!"
            )
        return filename
