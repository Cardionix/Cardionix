"""
Docstring
"""

__all__ = ["CardioDataModule"]


import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import CardioAnomalyDataset
from ..validate import DatasetParams, ETLPipelineParams, DataModuleParams


class CardioDataModule(pl.LightningDataModule):
    """
    Docstring
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
        if stage == "train" or stage is None:
            self.__data_train = CardioAnomalyDataset(
                self.__dataset_params,
                self.__etl_pipeline_params,
                stage="train"
            )

        if stage == "val" or stage is None:
            self.__data_val = CardioAnomalyDataset(
                self.__dataset_params,
                self.__etl_pipeline_params,
                stage="val"
            )

        if self.__num_subsets == 3 and (stage == "test" or stage is None):
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
