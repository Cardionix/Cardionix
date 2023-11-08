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
    merge_proba: Optional[float] = 0.0
    merge_rules: Optional[dict] = None
    sample_rate: Optional[int] = 22050
    duration: Optional[int] = 10
    mono: Optional[bool] = True
    pad_mode: Literal["constant", "edge", "linear_ramp", "reflect", "symmetric", "nearest_neighbors"] = "constant"
