"""
Docstring
"""

__all__ = [""]

from typing import Literal
import torch
from torch import nn


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


class Decoder(nn.Module):
    def __init__(self,
                 decoder_depth: list,
                 dropout: float = 0.3,
                 activation: Literal["relu", "leaky_relu", "selu", "softmax"] = "relu"
                 ):
        super().__init__()
        self.dropout = dropout
        self.head_block_index = len(decoder_depth) - 2

        self.activations = nn.ModuleDict([
            ["relu", nn.ReLU()],
            ["leaky_relu", nn.LeakyReLU()],
            ["selu", nn.SELU()],
            ["softmax", nn.Softmax(0)]
        ])

        self.decoder = nn.Sequential(*[
            self.decoder_block(i, in_features, out_features, activation, dropout)
            for i, (in_features, out_features) in enumerate(zip(decoder_depth, decoder_depth[1:]))
        ])

    def decoder_block(self,
                      block_index: int,
                      in_features: int,
                      out_features: int,
                      activation: str,
                      dropout: float
                      ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=out_features
            ),
            self.activations["softmax" if block_index == self.head_block_index else activation],
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
