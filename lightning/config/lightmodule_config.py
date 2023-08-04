"""
Contains a ``LightningModuleParams`` class
with parameters for the configuration of the ``CardioLightningModule`` class.
"""

__all__ = ["LightningModuleParams"]

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from torch.nn import Module


class LightningModuleParams(BaseModel):
    """
    The class contains parameters for the configuration of the ``CardioLightningModule`` class.

    Attributes:
        class_weights: (list[float], optional) weights for each class in the dataset,
            which will be used when calculating the loss function.

        model: (Module) neural network model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    class_weights: Optional[list[float]] = Field(default=None, max_items=5, min_items=2)
    model: Module
