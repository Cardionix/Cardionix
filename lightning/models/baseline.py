"""
Docstring
"""

__all__ = [""]

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 encoder_depth: list,
                 padding: int = 2,
                 stride: int = 1,
                 kernel_size: int = 5
                 ):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size

        self.encoder = nn.Sequential(*[
            self.encoder_block(in_channels, out_channels)
            for in_channels, out_channels in zip(encoder_depth, encoder_depth[1:])
        ])

    def encoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=self.stride,
                kernel_size=self.kernel_size,
                padding=self.padding,
                padding_mode="reflect"
            ),

            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LazyBatchNorm1d()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
