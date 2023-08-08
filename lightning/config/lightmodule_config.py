"""
Contains a ``LightningModuleParams`` class
with parameters for the configuration of the ``CardioLightningModule`` class.
"""

__all__ = ["LightningModuleParams"]

from typing import Type, Any
from pydantic import BaseModel, ConfigDict
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.modules.loss import _Loss


class LightningModuleParams(BaseModel):
    """
    The class contains parameters for the configuration of the ``CardioLightningModule`` class.

    Attributes:
        class_weights: (list[float], optional) weights for each class in the dataset,
            which will be used when calculating the loss function.

        model: (Module) neural network model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    optimizer: Type[Optimizer]
    optimizer_kwargs: dict
    lr_scheduler: Type[LRScheduler] | Any
    lr_scheduler_kwargs: dict
    lr_scheduler_dict_kwargs: dict
    criterion: Type[_Loss]
    criterion_kwargs: dict
