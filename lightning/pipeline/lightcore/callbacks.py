"""
Docstring
"""

__all__ = ["hooks"]

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor


hooks = [
    ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}/{val_loss:.2f}_{val/accuracy:.2f}",
        save_top_k=5,
        monitor="val_loss",
        mode="min"
    ),

    EarlyStopping(
        monitor="val/loss",
        min_delta=2e-2,
        patience=5,
        verbose=False,
        mode="min"
    ),

    LearningRateMonitor(
        logging_interval="step"
    )
]
