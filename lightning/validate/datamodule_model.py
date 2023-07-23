"""
Docstring
"""

__all__ = ["DataModuleParams"]

from typing import Optional
from pydantic import BaseModel, Field


class DataModuleParams(BaseModel):
    """
    Docstring
    """
    batch_size: Optional[int] = Field(default=4, gt=0)
    num_workers: Optional[int] = Field(default=2, gt=0)
