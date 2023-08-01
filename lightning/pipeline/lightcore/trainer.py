"""
Docstring
"""

__all__ = ["LightTrainer"]

from typing import Union, Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import torchinfo

from .callbacks import hooks
from lightning.config import DatasetParams, ETLPipelineParams
from lightning.config import DataModuleParams, LightningModuleParams
from lightning.pipeline.datamodule import CardioDataModule
from lightning.pipeline.lightmodule import CardioLightningModule


class LightTrainer:
    """
    Docstring
    """
    def __init__(self,
                 datamodule_config: DataModuleParams,
                 dataset_config: DatasetParams,
                 etl_pipeline_config: ETLPipelineParams,
                 lightmodule_config: LightningModuleParams,
                 job_type: str,
                 name: str,
                 project: Optional[str] = "CardioSonix",
                 tags: Optional[Union[list, tuple]] = None,
                 seed: Optional[int] = 42,
                 **kwargs
                 ):

        self.model = lightmodule_config.model
        self.on_startup(seed)

        self.config = self.define_config(
            datamodule_config, lightmodule_config,
            etl_pipeline_config, dataset_config,
            seed, kwargs
        )

        self.logger = WandbLogger(
            name=name, project=project,
            config=self.config, job_type=job_type, tags=tags
        )

        self.datamodule = CardioDataModule(dataset_config, etl_pipeline_config, datamodule_config)
        self.lightmodule = CardioLightningModule(lightmodule_config)
        self.callbacks = hooks
        self.trainer = self.get_trainer(kwargs)

    @staticmethod
    def define_config(datamodule_config,
                      lightmodule_config,
                      etl_pipeline_config,
                      dataset_config,
                      seed, kwargs) -> dict:
        return {
            "datamodule_config": datamodule_config.model_dump(),
            "etl_pipeline_config": etl_pipeline_config.model_dump(),
            "dataset_config": dataset_config.model_dump(),
            "lightmodule_config": lightmodule_config.model_dump(),
            "random_seed": seed,
            "trainer_config": kwargs
        }

    def get_trainer(self, kwargs):
        """
        Docstring
        """
        return pl.Trainer(
            logger=self.logger,
            callbacks=self.callbacks,
            **kwargs
        )

    def fit(self) -> None:
        """
        Docstring
        """
        self.trainer.fit(
            model=self.lightmodule,
            datamodule=self.datamodule
        )

    def on_startup(self, seed: int) -> None:
        """
        Docstring
        """
        pl.seed_everything(seed)
        torch.set_float32_matmul_precision("medium")
        torch.cuda.empty_cache()
        model_stats = torchinfo.summary(
            model=self.model,
            input_size=self.model.example_input_array.shape
        )
        print(model_stats)
