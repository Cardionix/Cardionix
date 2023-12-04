"""
The module trainer contains the ``LightTrainer`` class,
which is responsible for initializing training,
building and logging the configurations of all modules and project packages.
"""

__all__ = ["CardioTrainer"]

from typing import Union, Optional, Any, List
import warnings

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import torch
from torch.utils.data import DataLoader
from torch.nn import Module

from .engine import CardioDataModule, CardioLightningModule
from .configs import (
    ClassifyDatasetParams,
    ETLPipelineParams,
    DataModuleParams,
    LightningModuleParams
)


class CardioTrainer(Trainer):
    r"""
    CardioTrainer is a wrapper for the ``Trainer`` class adapted to the current project for better usability
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
                 name: str,
                 job_type: Optional[str] = None,
                 project: Optional[str] = "Cardio Sonix",
                 tags: Optional[Union[list, tuple]] = None,
                 seed: Optional[int] = 42,
                 log_modules: Optional[bool] = True,
                 **kwargs
                 ):
        if log_modules:
            log_modules = self._runs_config(
                datamodule_config,
                dataset_config,
                etl_pipeline_config,
                lightmodule_config,
                seed, **kwargs
            )
        logger = WandbLogger(
            name=name,
            project=project,
            log_model=True,
            config=log_modules,
            job_type=job_type,
            tags=tags
        )

        super().__init__(logger=logger, **kwargs)
        self._on_startup(seed)
        self.__lightmodule = None
        self.__lightmodule_config = lightmodule_config
        self.__classes = tuple(dataset_config.merge_classes.keys())
        self.__datamodule = CardioDataModule(dataset_config, etl_pipeline_config, datamodule_config)

    @staticmethod
    def _runs_config(
            datamodule_config: DataModuleParams,
            dataset_config: Union[ClassifyDatasetParams, Any],
            etl_pipeline_config: ETLPipelineParams,
            lightmodule_config: LightningModuleParams,
            seed: int,
            **kwargs
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
            "datamodule": datamodule_config.model_dump(),
            "transforms": etl_pipeline_config.model_dump(),
            "dataset": dataset_config.model_dump(),
            "optimization": lightmodule_config.model_dump(),
            "seed": seed,
            "trainer": kwargs
        }

    def __get_lightning_module(self, model: Module) -> LightningModule:
        return CardioLightningModule(self.__lightmodule_config, model, self.__classes)

    def fit(self,
            model: Module | LightningModule,
            train_dataloaders: DataLoader | None = None,
            val_dataloaders: DataLoader | None = None,
            datamodule: LightningDataModule | None = None,
            ckpt_path: str | None = None
            ) -> None:
        r"""Runs the full optimization routine.

        Args:
            model: Model to fit.

            train_dataloaders: An iterable or collection of iterables specifying training samples.
                Alternatively, a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` that defines
                the :class:`~pytorch_lightning.core.hooks.DataHooks.train_dataloader` hook.

            val_dataloaders: An iterable or collection of iterables specifying validation samples.

            datamodule: A :class:`~pytorch_lightning.core.datamodule.LightningDataModule` that defines
                the :class:`~pytorch_lightning.core.hooks.DataHooks.train_dataloader` hook.

            ckpt_path: Path/URL of the checkpoint from which training is resumed. Could also be one of two special
                keywords ``"last"`` and ``"hpc"``. If there is no checkpoint file at the path, an exception is raised.

        Raises:
            TypeError:
                If ``model`` is not :class:`~pytorch_lightning.core.module.LightningModule` for torch version less than
                2.0.0 and if ``model`` is not :class:`~pytorch_lightning.core.module.LightningModule` or
                :class:`torch._dynamo.OptimizedModule` for torch versions greater than or equal to 2.0.0 .

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.
        """
        if isinstance(model, Module):
            self.__lightmodule = self.__get_lightning_module(model)
        if isinstance(model, LightningModule):
            self.__lightmodule = model
        if not isinstance(datamodule, LightningDataModule) and not isinstance(train_dataloaders, DataLoader):
            datamodule = self.__datamodule
        return super().fit(
            self.__lightmodule,
            train_dataloaders,
            val_dataloaders,
            datamodule,
            ckpt_path
        )

    def predict(self,
                model: LightningModule | None = None,
                dataloaders: DataLoader | None = None,
                datamodule: LightningDataModule | None = None,
                return_predictions: Optional[bool] = None,
                ckpt_path: Optional[str] = None
                ) -> Union[List[Any], List[List[Any]]]:
        r"""Run inference on your data. This will call the model forward function to compute predictions. Useful to
        perform distributed and batched predictions. Logging is disabled in the predict hooks.

        Args:
            model: The model to predict with.

            dataloaders: An iterable or collection of iterables specifying predict samples.
                Alternatively, a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` that defines
                the :class:`~pytorch_lightning.core.hooks.DataHooks.predict_dataloader` hook.

            datamodule: A :class:`~pytorch_lightning.core.datamodule.LightningDataModule` that defines
                the :class:`~pytorch_lightning.core.hooks.DataHooks.predict_dataloader` hook.

            return_predictions: Whether to return predictions.
                ``True`` by default except when an accelerator that spawns processes is used (not supported).

            ckpt_path: Either ``"best"``, ``"last"``, ``"hpc"`` or path to the checkpoint you wish to predict.
                If ``None`` and the model instance was passed, use the current weights.
                Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
                if a checkpoint callback is configured.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        Returns:
            Returns a list of dictionaries, one for each provided dataloader containing their respective predictions.

        Raises:
            TypeError:
                If no ``model`` is passed and there was no ``LightningModule`` passed in the previous run.
                If ``model`` passed is not `LightningModule` or `torch._dynamo.OptimizedModule`.

            MisconfigurationException:
                If both ``dataloaders`` and ``datamodule`` are passed. Pass only one of these.

            RuntimeError:
                If a compiled ``model`` is passed and the strategy is not supported.

        See :ref:`Lightning inference section<deploy/production_basic:Predict step with your LightningModule>` for more.
        """
        if isinstance(model, Module):
            self.__lightmodule = self.__get_lightning_module(model)
        if isinstance(model, LightningModule):
            self.__lightmodule = model
        if not isinstance(self.__lightmodule, LightningModule):
            raise ValueError(
                f"Expected 'model' must be a Module subclass or "
                f"LightningModule subclass if method fit was never called, "
                f"but got {model=} and {self.__lightmodule=}"
            )
        if not isinstance(datamodule, LightningDataModule) and not isinstance(dataloaders, DataLoader):
            datamodule = self.__datamodule
        return super().predict(
            self.__lightmodule,
            dataloaders,
            datamodule,
            return_predictions,
            ckpt_path
        )

    @staticmethod
    def _on_startup(seed: int) -> None:
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
