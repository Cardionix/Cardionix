"""
Containing learning callbacks
for the CardioTrainer class as variables.
This callbacks can be used to control training (early stopping, etc.)
"""

__all__ = ["callbacks"]

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping
)


callbacks = [
    ModelCheckpoint(
        dirpath="./checkpoints",
        filename="epoch={epoch}"
                 "-precision={val/macro avg/precision:.2f}"
                 "-recall={val/macro avg/recall:.2f}"
                 "-f1-score={val/macro avg/f1-score:.2f}",
        monitor="val/macro avg/f1-score",
        mode="max",
        save_top_k=10,
        auto_insert_metric_name=False,
        save_weights_only=True
    ),

    EarlyStopping(
        monitor="val/macro avg/f1-score",
        mode="max",
        patience=8,
        min_delta=1e-6
    )
]
