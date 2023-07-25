"""
Docstring
"""

__all__ = ["CardioLightningModule"]

from typing import Any, Union

import wandb
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import FBetaScore
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from ..validate import ETLPipelineParams


class CardioLightningModule(pl.LightningModule):
    """
    Docstring
    """
    def __init__(self,
                 num_classes: int,
                 model: nn.Module,
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(size=[1, 3, 512, 512])
        self.num_classes = num_classes
        self.model = model
        self._device = "cuda"
        self.empty_target = torch.tensor(0, dtype=torch.float32)
        self.step_outputs = self.get_metrics_dict()

        self.metrics = {
            "accuracy": Accuracy(task="multiclass",
                                 threshold=0.5,
                                 num_classes=num_classes,
                                 validate_args=True,
                                 ignore_index=None,
                                 average="micro").to(self._device),

            "fbeta_score": FBetaScore(task="multiclass",
                                      beta=1.0,
                                      threshold=0.5,
                                      num_classes=num_classes,
                                      average="micro",
                                      ignore_index=None,
                                      validate_args=True).to(self._device)
        }

    @staticmethod
    def get_metrics_dict():
        return {}.fromkeys(
            {"val", "train"},
            {
                "loss": [],
                "accuracy": [],
                "jaccard_index": [],
                "fbeta_score": []
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def compute_loss(self, logites, y) -> torch.Tensor:
        if len(y.unique()) == 1:
            weights = None
            self.empty_target += 1

        else:
            weights = compute_class_weight(
                class_weight="balanced",
                classes=y.unique().cpu().numpy(),
                y=y.flatten().cpu().numpy()
            )

            weights = torch.tensor(
                weights,
                dtype=torch.float,
                device=self._device
            )

        loss = F.cross_entropy(
            input=logites,
            target=y.to(torch.int64),
            weight=weights
        )
        return loss

    @staticmethod
    def sanity_check(x: torch.Tensor, y: torch.Tensor) -> None:
        assert x.ndim == 4
        assert x.max() <= 3 and x.min() >= -3
        assert y.ndim == 3
        assert y.max() <= 22 and y.min() >= 0

    def compute_metrics(self,
                        predictions: torch.Tensor,
                        y: torch.Tensor
                        ) -> dict:

        accuracy = self.metrics["accuracy"](predictions, y)
        jaccard_index = self.metrics["jaccard_index"](predictions, y)
        fbeta_score = self.metrics["fbeta_score"](predictions, y)
        return {"accuracy": accuracy,
                "jaccard_index": jaccard_index,
                "fbeta_score": fbeta_score}

    def update_metrics(self,
                       loss: torch.Tensor,
                       accuracy: torch.Tensor,
                       jaccard_index: torch.Tensor,
                       fbeta_score: torch.Tensor,
                       stage: str
                       ) -> None:

        self.step_outputs[stage]["loss"].append(loss)
        self.step_outputs[stage]["accuracy"].append(accuracy)
        self.step_outputs[stage]["jaccard_index"].append(jaccard_index)
        self.step_outputs[stage]["fbeta_score"].append(fbeta_score)

    def shared_step(self,
                    batch: torch.Tensor,
                    stage: str
                    ) -> torch.Tensor:

        x, y = batch
        self.sanity_check(x, y)

        logites = self.forward(x.to(torch.float32))
        loss = self.compute_loss(logites, y)
        activated = F.softmax(input=logites, dim=1)
        predictions = torch.argmax(activated, dim=1)

        metrics = self.compute_metrics(predictions, y)
        metrics.update({"loss": loss})
        self.update_metrics(stage=stage, **metrics)
        return loss

    def epoch_end_metrics(self, stage: str) -> dict:
        loss = torch.mean(torch.tensor(self.step_outputs[stage]["loss"]))
        accuracy = torch.mean(torch.tensor(self.step_outputs[stage]["accuracy"]))
        jaccard_index = torch.mean(torch.tensor(self.step_outputs[stage]["jaccard_index"]))
        fbeta_score = torch.mean(torch.tensor([self.step_outputs[stage]["fbeta_score"]]))

        return {
            f"{stage}_loss": loss,
            f"{stage}_accuracy": accuracy,
            f"{stage}_jaccard_index": jaccard_index,
            f"{stage}_fbeta_score": fbeta_score,
            f"{stage}_empty_target": self.empty_target
        }

    def empty_metrics(self, stage) -> None:
        for key in self.step_outputs[stage].keys():
            self.step_outputs[stage][key].clear()

    def log_everything(self, metrics: dict) -> None:
        wandb.log(metrics)
        self.log_dict(metrics, prog_bar=True)

    def shared_epoch_end(self, stage: str) -> None:
        metrics = self.epoch_end_metrics(stage)
        self.empty_metrics(stage)
        self.log_everything(metrics)

    def training_step(self,
                      batch: torch.Tensor,
                      batch_idx: int
                      ) -> torch.Tensor:

        return self.shared_step(batch=batch, stage="train")

    def on_train_epoch_end(self) -> None:
        return self.shared_epoch_end(stage="train")

    def validation_step(self,
                        batch: torch.Tensor,
                        batch_idx: int
                        ) -> torch.Tensor:

        return self.shared_step(batch=batch, stage="val")

    def on_validation_epoch_end(self) -> None:
        return self.shared_epoch_end(stage="val")

    def configure_optimizers(self) -> dict:
        optimizer = optim.Adam(
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
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict
        }
