"""
This module contains class for preprocessing tabular data:
- Encoding categorical features.
- Standardization, normalization numerical features.
"""

__all__ = ["TabularPreprocessor"]

import os
from pathlib import Path
from typing import Union, Literal, Any
import joblib

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    Normalizer,
    MinMaxScaler
)


class TabularPreprocessor:
    """
    This class contains methods for:
    - Splitting features by type (categorical or numerical)
    - Encoding categorical features.
    - Standardization, normalization numerical features.
    - And unite encoded categorical features with normalized numerical features.

    Note:
        This class only for use with ETLPipeline.
        You should call _define_preprocessors() method before other methods.
        It will search folder 'preprocessors' in root project directory and create it if not found.
        All preprocessors defined first time will save in 'preprocessors' directory and load at second call.
    """

    __encoders: dict = {
        "OneHotEncoder": OneHotEncoder,
    }

    __scalers: dict = {
        "StandardScaler": StandardScaler,
        "Normalizer": Normalizer,
        "MinMaxScaler": MinMaxScaler
    }

    def __init__(self):
        self.__preprocessors_dirpath = self.__define_root()
        self.__encoder = None
        self.__scaler = None

    @staticmethod
    def __define_root() -> str:
        preprocessor_dir = os.path.join(os.getcwd(), "preprocessors")
        if not os.path.exists(preprocessor_dir):
            os.mkdir(preprocessor_dir)
        return preprocessor_dir

    def __get_path(self, path: str | Path) -> str:
        return os.path.join(self.__preprocessors_dirpath, path)

    def __select_preprocessing(self, dataframe: pd.DataFrame, preprocessor: str) -> tuple[
        pd.DataFrame, Union[TransformerMixin, BaseEstimator]]:
        if self.__encoders.get(preprocessor, None):
            preprocessor = self.__encoders[preprocessor]
            data = dataframe.select_dtypes(include=["int"])
            return data, preprocessor()
        preprocessor = self.__scalers[preprocessor]
        data = dataframe.select_dtypes(include=["float"])
        return data, preprocessor()

    def __get_preprocessor(self,
                           dataset: pd.DataFrame,
                           preprocessor: Literal["StandardScaler", "Normalizer", "MinMaxScaler", "OneHotEncoder"]
                           ) -> Union[TransformerMixin, BaseEstimator]:
        path = self.__get_path(f"{preprocessor}.joblib")
        if not os.path.exists(path):
            dataset, preprocessor = self.__select_preprocessing(dataset, preprocessor)
            preprocessor = preprocessor.fit(dataset)
            joblib.dump(preprocessor, path)
            return preprocessor
        return joblib.load(path)

    def __scaling(self, data: pd.DataFrame) -> np.ndarray:
        numerical = data.select_dtypes(include=["float"])
        return self.__scaler.transform(numerical)

    def __encoding(self, data: pd.DataFrame) -> np.ndarray:
        categorical = data.select_dtypes(include=["int"])
        return self.__encoder.transform(categorical).toarray()

    def __check_preprocessors(self) -> None:
        if not all([self.__scaler, self.__encoder]):
            raise AttributeError(
                "Preprocessing error! "
                "You must define preprocessors calling method "
                "define_preprocessors before preprocessing"
            )

    @staticmethod
    def __check_sample(sample: Any) -> None:
        if not isinstance(sample, pd.DataFrame):
            raise TypeError(
                f"Preprocessing error! "
                f"Input data must be a DataFrame, "
                f"but got {type(sample)}"
            )

    def _define_preprocessors(self,
                              dataset: pd.DataFrame,
                              scaler: Literal["StandardScaler", "Normalizer", "MinMaxScaler"],
                              encoder: Literal["OneHotEncoder"]
                              ) -> None:
        self.__check_sample(dataset)
        self.__scaler = self.__get_preprocessor(dataset, scaler)
        self.__encoder = self.__get_preprocessor(dataset, encoder)

    def preprocess_tabular(self, sample: pd.DataFrame) -> np.ndarray:
        self.__check_preprocessors()
        self.__check_sample(sample)
        categorical = self.__encoding(sample)
        numerical = self.__scaling(sample)
        return np.concatenate([numerical, categorical], axis=1)
