"""
Docstring
"""

__all__ = ["LightningModuleParams"]

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from torch.nn import Module


class LightningModuleParams(BaseModel):
    """
    Docstring
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    class_weights: Optional[list[float]] = Field(default=None, max_items=5, min_items=2)
    model: Module
