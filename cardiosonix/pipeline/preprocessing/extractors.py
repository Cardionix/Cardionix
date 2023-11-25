"""
This module contain extractors for extracting any features from audio samples.
"""

__all__ = ["MFCCExtractor"]

from typing import Literal, Optional
import torch
import torchaudio.transforms as T
from torch.nn import Module


class MFCCExtractor(Module):
    r"""Create the Mel-frequency cepstrum coefficients from an audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    By default, this calculates the MFCC on the DB-scaled Mel spectrogram.
    This is not the textbook implementation, but is implemented here to
    give consistency with librosa.

    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``22050``)
        n_mfcc (int, optional): Number of mfc coefficients to retain. (Default: ``52``)
        n_mels (int, optional): Number of mell filterbanks. (Default: ``52``)
        n_fft (bool, optional): Size of FFT, creates n_fft // 2 + 1 bins. (Default: 400)
        win_length (int, optional): Window size. (Default: n_fft)
        hop_length (int, optional): Length of hop between STFT windows. (Default: win_length // 2)
        average_by (string, optional): Average mffcs by frequency or time axis. (Default: None)
    Example
        >>> extractor = MFCCExtractor()
        >>> waveform, sample_rate = torcwhaudio.load("test.wav", normalize=True)
        >>> mfcc = extractor(waveform)

    See also:
        :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
        generate the filter banks.
    """
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
        self.average_by = average_by
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
                "mel_scale": "htk"
            }
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract MFCC coefficients and average them.

        :param waveform: audio sample represented as an ``Tensor`` with a set of amplitudes.
        """
        mfcc = self.mfcc_extractor(waveform).squeeze().T
        if self.average_by:
            mfcc = mfcc.mean(self.average_dict[self.average_by])
            mfcc = mfcc.unsqueeze(0)
        return mfcc
