"""
Docstring
"""

__all__ = ["MFCCExtractor"]

from typing import Literal, Optional
import torch
import torchaudio.transforms as T
from torch.nn import Module


class MFCCExtractor(Module):
    """Extractor class for MFCC features."""
    def __init__(self,
                 sample_rate: Optional[int] = 22050,
                 n_fft: Optional[int] = 2048,
                 win_length: Optional[int] = 2048,
                 hop_length: Optional[int] = 1024,
                 n_mels: Optional[int] = 52,
                 n_mfcc: Optional[int] = 52,
                 average_by: Optional[Literal["time", "mfcc"]] = None
                 ):

        super().__init__()
        self.__average_by = average_by
        self.average_dict = {
            "mfcc": 0,
            "time": 1,
        }

        self.mfcc_extractor = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "win_length": win_length,
                "hop_length": hop_length,
                "mel_scale": "htk"}
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract MFCC coefficients and average them.

        :param waveform: audio sample represented as an ``Tensor`` with a set of amplitudes.
        """
        mfcc = self.mfcc_extractor(waveform).squeeze()
        if self.__average_by:
            mfcc = mfcc.mean(self.average_dict[self.__average_by])
            mfcc = mfcc.unsqueeze(0)
        return mfcc
