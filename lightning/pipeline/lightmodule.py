"""
Contains a ``LightningModule`` subclass ``CardioLightningModule``.
It is a high-level API for learning cycle management, validation and testing.
This class also logs metrics and changes its behavior depending on callbaks.
"""

__all__ = ["CardioLightningModule"]

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from .metrics import LightMetrics
from lightning.config import LightningModuleParams


class CardioLightningModule(pl.LightningModule):
    """
    ``CardioLightningModule`` it is a high-level API for learning cycle management, validation and testing.
    This class also logs metrics and changes its behavior depending on callbaks.

    Args:
        lightning_module_params: (LightningModuleParams) subclass of ``BaseModel``
            containing parameters (configuration) for ``CardioLightningModule`` initialization.
    """
    def __init__(self, lightning_module_params: LightningModuleParams):
        super().__init__()
        self.save_hyperparameters()
        self.model = lightning_module_params.model
        self.example_input_array = self.model.example_input_array
        self.criterion = nn.CrossEntropyLoss(weight=lightning_module_params.class_weights)
        self.loss_dict = {
            "train": [],
            "val": []
        }
        self.step_outputs = {
            "train": LightMetrics("train"),
            "val": LightMetrics("val")
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        x, y = batch
        logites = self.forward(x.to(torch.float32))
        loss = self.criterion(logites, y.to(torch.int64))
        self.step_outputs[stage].accumulate(logites, y)
        self.loss_dict[stage].append(loss)
        return loss

    def shared_epoch_end(self, stage: str) -> None:
        metrics = self.step_outputs[stage].compute_metrics()
        metrics[f"{stage}/loss"] = torch.stack(self.loss_dict[stage]).mean()
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        return self.shared_step(batch=batch, stage="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        return self.shared_step(batch=batch, stage="val")

    def on_train_epoch_end(self) -> None:
        return self.shared_epoch_end(stage="train")

    def on_validation_epoch_end(self) -> None:
        return self.shared_epoch_end(stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=25e-3,
        )

        scheduler_dict = {
            "scheduler": lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=5
            ),
            "interval": "epoch",
            "monitor": "train/accuracy"
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict
        }
