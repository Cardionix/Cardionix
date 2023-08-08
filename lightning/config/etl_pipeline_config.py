"""
Contains a ``ETLPipelineParams``
with parameters for the configuration of the ETLPipeline class.
"""

__all__ = ["ETLPipelineParams"]

from typing import Optional, Literal
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

    extractor: Literal["MFCC"]
    extractor_kwargs: dict
    sample_rate: Optional[int] = 22050
    duration: Optional[int] = 10
    mono: Optional[bool] = True
