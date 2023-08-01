"""
Docstring
"""

__all__ = ["BaselineRNNModel"]

from typing import Literal
import torch
from torch import nn


class Encoder(nn.Module):
    """
    Docstring
    """
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
        """
        Docstring
        """
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
            ["softmax", nn.Softmax(1)]
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
        """
        Docstring
        """
        layers = list([])

        layers.append(
            nn.Linear(
                in_features=in_features,
                out_features=out_features
            )
        )

        if block_index != self.head_block_index:
            layers.append(self.activations[activation])
            layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class BaselineRNNModel(nn.Module):
    """
    Docstring
    """
    def __init__(self,
                 example_input_array: tuple[int, ...],
                 encoder_depth: list,
                 decoder_depth: list,
                 rnn_input_size: int = 6,
                 rnn_hidden_size: int = 256,
                 activation: Literal["relu", "leaky_relu", "selu", "softmax"] = "relu",
                 dropout: float = 0.3
                 ):
        super().__init__()
        self.example_input_array = torch.zeros(size=example_input_array)

        self.encoder = Encoder(
            encoder_depth=encoder_depth,
            padding=2,
            stride=1,
            kernel_size=5
        )

        self.rnn1 = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            batch_first=True,
        )

        self.rnn2 = nn.LSTM(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size // 2,
            batch_first=True,
        )

        self.decoder = Decoder(
            decoder_depth=decoder_depth,
            dropout=dropout,
            activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Docstring
        """
        x = self.encoder(x)
        output, _ = self.rnn1(x)
        output, (hidden, _) = self.rnn2(output)
        x = self.decoder(hidden)
        return x
