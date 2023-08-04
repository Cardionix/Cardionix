"""
Contains a ``ETLPipelineParams``
with parameters for the configuration of the ETLPipeline class.
"""

__all__ = ["ETLPipelineParams"]

from typing import Optional
from pydantic import BaseModel, Field


class ETLPipelineParams(BaseModel):
    """
    The class contains parameters for the configuration of the ETLPipeline class.

    Attributes:
        sample_rate: audio sample rate.

        duration: duration of the audio.

        mono: if true then all audio files will be loaded with one channel.
            Otherwise, if there are two channels, they will both be loaded.

        n_mfcc: (int) number of MFCCs to return.
    """
    sample_rate: int = Field(default=22050, ge=4000, le=44100)
    duration: int = Field(default=10, ge=1, le=15)
    mono: Optional[bool] = True
    n_mfcc: int = Field(default=52, ge=10)
