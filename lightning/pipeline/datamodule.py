"""
Contains the ``CardioDataModule`` class,
which initializes the ``CardioAnomalyDataset`` and issues a Dataloader depending on the learning stage.
"""

__all__ = ["CardioDataModule"]

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets import CardioAnomalyDataset
from ..config import DatasetParams, ETLPipelineParams, DataModuleParams


class CardioDataModule(pl.LightningDataModule):

    """
    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self, stage):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)
            def teardown(self):
                # clean up after fit or test
                # called on every process in DDP

    Args:
        dataset_params: (DataModuleParams) subclass of ``BaseModel``
            containing parameters (configuration) for ``CardioDataModule`` initialization.

        etl_pipeline_params: (DatasetParams) subclass of ``BaseModel``
            containing parameters (configuration) for ``CardioAnomalyDataset`` initialization.

        datamodule_params: (ETLPipelineParams) subclass of ``BaseModel``
            containing parameters (configuration) for ``ETLPipeline`` initialization.
    """
    def __init__(self,
                 dataset_params: DatasetParams,
                 etl_pipeline_params: ETLPipelineParams,
                 datamodule_params: DataModuleParams,
                 ):
        super().__init__()
        self.__dataset_params = dataset_params
        self.__datamodule_params = datamodule_params
        self.__etl_pipeline_params = etl_pipeline_params

        self.__num_subsets = len(self.__dataset_params.split_ratio)
        self.__data_source = "https://www.kaggle.com/datasets/mersico/dangerous-heartbeat-dataset-dhd"

        self.__data_train = None
        self.__data_val = None
        self.__data_test = None

    def prepare_data(self) -> None:
        print(
            f"Warning! This method does not load the data to process it. "
            f"You have to download the data yourself. Data source: {self.__data_source}"
        )

    def setup(self, stage: str = None) -> None:

        self.__data_train = CardioAnomalyDataset(
            self.__dataset_params,
            self.__etl_pipeline_params,
            stage="train"
        )

        if self.__num_subsets >= 2:
            self.__data_val = CardioAnomalyDataset(
                self.__dataset_params,
                self.__etl_pipeline_params,
                stage="val"
            )

        if self.__num_subsets == 3:
            self.__data_test = CardioAnomalyDataset(
                self.__dataset_params,
                self.__etl_pipeline_params,
                stage="test"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.__data_train,
            batch_size=self.__datamodule_params.batch_size,
            num_workers=self.__datamodule_params.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.__data_val,
            batch_size=self.__datamodule_params.batch_size,
            num_workers=self.__datamodule_params.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.__data_test,
            batch_size=self.__datamodule_params.batch_size,
            num_workers=self.__datamodule_params.num_workers,
            shuffle=False,
            pin_memory=True
        )