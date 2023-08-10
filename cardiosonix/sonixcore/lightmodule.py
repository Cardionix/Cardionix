"""
Contains a ``LightningModule`` subclass ``CardioLightningModule``.
It is a high-level API for learning cycle management, validation and testing.
This class also logs metrics and changes its behavior depending on callbaks.
"""

__all__ = ["CardioLightningModule"]

from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .metrics import CardioMetrics
from ..configs import LightningModuleParams


class CardioLightningModule(pl.LightningModule):
    """
    ``CardioLightningModule`` it is a high-level API for learning cycle management, validation and testing.
    This class also logs metrics and changes its behavior depending on callbaks.

    Args:
        lightning_module_params: (LightningModuleParams) subclass of ``BaseModel``
            containing parameters (configuration) for ``CardioLightningModule`` initialization.
    """
    def __init__(self,
                 lightning_module_params: LightningModuleParams,
                 model: nn.Module,
                 classes: dict
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.example_input_array = self.model.example_input_array
        self._optimizer = lightning_module_params.optimizer
        self.optimizer_kwargs = lightning_module_params.optimizer_kwargs
        self._lr_scheduler = lightning_module_params.lr_scheduler
        self.lr_scheduler_kwargs = lightning_module_params.lr_scheduler_kwargs
        self.lr_scheduler_dict_kwargs = lightning_module_params.lr_scheduler_dict_kwargs
        self.criterion = lightning_module_params.criterion(**lightning_module_params.criterion_kwargs)
        self.classes = list(classes.keys())
        self.step_outputs = {
            "train": CardioMetrics(classes=self.classes, stage="train", external_metrics=["loss"]),
            "val": CardioMetrics(classes=self.classes, stage="val", external_metrics=["loss"]),
            "predict": CardioMetrics(classes=self.classes, stage="predict")
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        x, y = batch
        logites = self.forward(x.to(torch.float32))
        loss = self.criterion(logites, y.to(torch.int64))
        self.step_outputs[stage].accumulate(logites, y)
        self.step_outputs[stage].add(loss=loss)
        return loss

    def shared_epoch_end(self, stage: str) -> None:
        metrics = self.step_outputs[stage].compute_metrics()
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        return self.shared_step(batch=batch, stage="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        return self.shared_step(batch=batch, stage="val")

    def on_train_epoch_end(self) -> None:
        return self.shared_epoch_end(stage="train")

    def on_validation_epoch_end(self) -> None:
        return self.shared_epoch_end(stage="val")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> dict:
        x, y = batch
        logites = self.forward(x.to(torch.float32))
        self.step_outputs["predict"].accumulate(logites, y)

    def on_predict_epoch_end(self) -> None:
        self.step_outputs["predict"].make_report()

    def configure_optimizers(self):
        optimizer = self._optimizer(
            self.parameters(),
            **self.optimizer_kwargs
        )

        scheduler_dict = {
            "scheduler": self._lr_scheduler(
                optimizer=optimizer,
                **self.lr_scheduler_kwargs
            ),
            **self.lr_scheduler_dict_kwargs
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict
        }
