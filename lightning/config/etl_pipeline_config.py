"""
Docstring
"""

__all__ = ["ETLPipelineParams"]

from typing import Optional
from pydantic import BaseModel, Field


class ETLPipelineParams(BaseModel):
    """
    Docstring
    """
    sample_rate: int = Field(default=22050, ge=4000, le=44100)
    duration: int = Field(default=10, ge=1, le=15)
    mono: Optional[bool] = True
    n_mfcc: int = Field(default=52, ge=10)
