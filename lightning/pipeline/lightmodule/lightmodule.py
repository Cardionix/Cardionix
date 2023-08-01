"""
Docstring
"""

__all__ = ["CardioLightningModule"]


from typing import Optional
import torch
from torch import Tensor
from torch import nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from .metrics import LightMetrics
from lightning.config import LightningModuleParams


class CardioLightningModule(pl.LightningModule):
    """
    Docstring
    """
    def __init__(self, lightning_module_params: LightningModuleParams):
        super().__init__()
        self.save_hyperparameters()
        self.model = lightning_module_params.model
        self.example_input_array = self.model.example_input_array
        self.criterion = nn.CrossEntropyLoss(weight=lightning_module_params.class_weights)
        self.metrics = LightMetrics()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        x, y = batch
        logites = self.forward(x.to(torch.float32))
        loss = self.criterion(logites, y.to(torch.int64))
        self.metrics.accumulate(logites, y)
        return loss

    def shared_epoch_end(self, stage: str) -> None:
        metrics = self.metrics.compute_metrics(stage)
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch=batch, stage="train")

    def on_train_epoch_end(self) -> None:
        return self.shared_epoch_end(stage="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch=batch, stage="val")

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
            "monitor": "val/loss"
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict
        }
