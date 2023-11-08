"""
The module trainer contains the ``LightTrainer`` class,
which is responsible for initializing training,
building and logging the configurations of all modules and project packages.
"""

__all__ = ["LightTrainer"]

from typing import Union, Optional, Any
import warnings

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.nn import Module

from .configs import ClassifyDatasetParams, ETLPipelineParams
from .configs import DataModuleParams, LightningModuleParams
from .engine import CardioDataModule
from .engine import CardioLightningModule


class LightTrainer:
    r"""
    LightTrainer is a wrapper for the ``Trainer`` class adapted to the current project for better usability
    and it also high-level API for initializing and configuring the training process and model validation.
    This class accepts parameters encapsulated in BaseModel subclasses,
    which are then used to initialize the ``CardioLightningModule`` and ``CardioDataModule``
    and then log the passed parameters as a common pipeline launch configuration.
    There is also a group of arguments intended only for session logging
    and another group of arguments intended only for initializing the ``Trainer`` class from ''pytorch_lightning''

    Args:
        datamodule_config: (DataModuleParams) subclass of ``BaseModel``
            containing parameters (configuration) for ``CardioDataModule`` initialization.

        dataset_config: (DatasetParams) subclass of ``BaseModel``
            containing parameters (configuration) for ``CardioAnomalyDataset`` initialization.

        etl_pipeline_config: (ETLPipelineParams) subclass of ``BaseModel``
            containing parameters (configuration) for ``ETLPipeline`` initialization.

        lightmodule_config: (LightningModuleParams) subclass of ``BaseModel``
            containing parameters (configuration) for ``CardioLightningModule`` initialization.

        job_type: (str, optional) Specify the type of run,
            which is useful when you're grouping runs together into larger experiments using group.
            For example, you might have multiple jobs in a group, with job types like train and eval.
            Setting this makes it easy to filter and
            group similar runs together in the UI so you can compare apples to apples.

        name: (str) display name for the run.

        project: (str, optional) The name of the project where you're sending the new run.
            If the project is not specified, the run is put in an "CardioSonix" project.

        tags: (list, optional) A list of strings, which will populate the list of tags on this run in the UI.
            Tags are useful for organizing runs together, or applying temporary labels like "baseline" or "production".
            It's easy to add and remove tags in the UI, or filter down to just runs with a specific tag.

        seed: (int, optional) the integer value seed for global random state in Lightning.
            If not defined the default value is 42.
            Set seed for pseudo-random number generators in: ``pytorch``, ``numpy``, ``python.random``

        **kwargs: (Any) any named arguments passed to the ``LightTrainer``
            class will be passed to the ``Trainer`` class.
            That is, you can pass all the key arguments
            that can be passed to the ``Trainer`` class in ``pytorch_lightning``
    """
    def __init__(self,
                 datamodule_config: DataModuleParams,
                 dataset_config: Union[ClassifyDatasetParams, Any],
                 etl_pipeline_config: ETLPipelineParams,
                 lightmodule_config: LightningModuleParams,
                 model: Module,
                 name: str,
                 job_type: Optional[str] = None,
                 project: Optional[str] = "Cardio Sonix",
                 tags: Optional[Union[list, tuple]] = None,
                 seed: Optional[int] = 42,
                 **kwargs: Any
                 ):

        self.on_startup(seed)

        self.config = self.define_config(
            datamodule_config,
            lightmodule_config,
            etl_pipeline_config,
            dataset_config,
            seed,
            kwargs
        )

        self.__logger = WandbLogger(
            name=name,
            project=project,
            log_model=True,
            config=self.config,
            job_type=job_type,
            tags=tags
        )

        self.__datamodule = CardioDataModule(
            dataset_config,
            etl_pipeline_config,
            datamodule_config,
            seed=seed
        )

        self.__lightmodule = CardioLightningModule(
            lightmodule_config,
            model,
            dataset_config.classes
        )

        self.__trainer = self.get_trainer(kwargs)

    @staticmethod
    def define_config(
            datamodule_config,
            lightmodule_config,
            etl_pipeline_config,
            dataset_config,
            seed,
            kwargs
    ) -> dict:
        """
        The method accepts arguments encapsulated in ``BaseModel`` subclasses
        that define module configuration and initialization.
        A dictionary is created with the global configuration of all modules.
        Subclasses are converted to dictionaries and passed with keys specific to the module they are responsible for.
        Third-party parameters are also transmitted,
        which are also written to the global configuration dictionary (for example, ``seed``)
        """
        return {
            "datamodule_config": datamodule_config.model_dump(),
            "etl_pipeline_config": etl_pipeline_config.model_dump(),
            "dataset_config": dataset_config.model_dump(),
            "lightmodule_config": lightmodule_config.model_dump(),
            "random_seed": seed,
            "trainer_config": kwargs
        }

    def get_trainer(self, kwargs: dict) -> Trainer:
        """
        The method initializes the ``Trainer`` class from ``pytorch_lightning``,
        which takes hooks from callbacks module and also
        accepts any named arguments that were passed when initializing the ``LightTrainer`` class.
        """
        return Trainer(
            logger=self.__logger,
            **kwargs
        )

    def fit(self) -> None:
        """
        The method initializes training.
        Takes an instance of the ``CardioDataModule`` and ``CardioLightningModule`` class.
        """
        self.__trainer.fit(
            model=self.__lightmodule,
            datamodule=self.__datamodule
        )

    def predict(self):
        self.__trainer.predict(
            model=self.__lightmodule,
            datamodule=self.__datamodule
        )

    @staticmethod
    def on_startup(seed: int) -> None:
        """
        Method that sets seed for pseudo-random number generators in: pytorch, numpy, python.random.
        In addition, sets the following environment variables:

            * PL_GLOBAL_SEED: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
            * PL_SEED_WORKERS: (optional) is set to 1 if workers=True.

        Sets the internal precision of float32 matrix multiplications.
        Running float32 matrix multiplications in lower precision may significantly increase performance,
        and in some programs the loss of precision has a negligible impact.

        And Releases all unoccupied cached memory currently held by the caching allocator
        so that those can be used in other GPU application and visible in nvidia-smi.
        """

        pl.seed_everything(seed)
        torch.set_float32_matmul_precision("medium")
        torch.cuda.empty_cache()
        warnings.filterwarnings("ignore")
