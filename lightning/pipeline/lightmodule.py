"""
Docstring
"""

__all__ = ["CardioLightningModule"]


import wandb
import torch
from torch import nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl


class CardioLightningModule(pl.LightningModule):
    """
    Docstring
    """
    def __init__(self,
                 lightmodule_params: LightModuleParams,
                 model: nn.Module,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(size=lightmodule_params.example_input)
        self.num_classes = lightmodule_params.num_classes
        self.model = model
        self.class_weights = lightmodule_params.class_weights
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = LightMetrics(classes=[])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        x, y = batch
        logites = self.forward(x.to(torch.float32))
        loss = self.criterion(logites, y.to(torch.int64))
        self.metrics.accumulate(logites, y)
        return {f"{stage}_loss": loss}

    def log_everything(self, metrics: dict) -> None:
        wandb.log(metrics)
        self.log_dict(metrics, prog_bar=True)

    def shared_epoch_end(self, stage: str) -> None:
        metrics = self.metrics.compute_metrics(stage)
        self.log_everything(metrics)

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
            "monitor": "val_loss"
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
