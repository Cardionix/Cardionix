"""
Contains a ``DataModuleParams`` class with parameters for ``CardioDataModule`` configuration.
"""

__all__ = ["DataModuleParams"]

from typing import Optional
from pydantic import BaseModel, Field


class DataModuleParams(BaseModel):
    """
    Сlass with parameters for ``CardioDataModule`` configuration.

    Attributes:
        batch_size: (int, optional) how many samples per batch to load (default: 2).

        num_workers: how many subprocesses to use for data loading (default: 2).
            0 means that the data will be loaded in the main process.
    """
    batch_size: Optional[int] = Field(default=2, gt=0)
    num_workers: Optional[int] = Field(default=2, gt=0)
