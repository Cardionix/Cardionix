"""
Contains classes for assembling the baseline architecture of the solution,
which should provide a starting point for research.
The model is assembled with the help of the ``Encoder``, ``LSTMModule`` and ``Decoder`` classes,
which are used in the main class ``BaselineLSTMModel``.
"""

__all__ = ["BaselineRNNModel"]

from typing import Literal, Optional
import torch
from torch import nn


class Encoder(nn.Module):
    """
    The ``Encoder`` class is the stem of a neural network.
    The goal of the encoder is to extract more useful information
    from the incoming features, before feeding them into recurrent blocks (LSTM or GRU).
    Consists of combinations of blocks::

        1) Conv1d()
        2) ReLU(),
        3) MaxPool1d(),
        4) BatchNorm1d()
        5) Dropout (Optional)

    Args:
        input_shape: (tuple[int]) input shape of the encoder block.

        encoder_depth: (list[int, ...]) encoder depth is a list of numbers,
            where each number means the number of kernels (filters) in one encoder block.

        stride: (int) controls the stride for the cross-correlation,
            a single number or a tuple. Default is ``1``.

        kernel_size: (int) size of the convolving kernel. Default is ``5``.

        activation: (Literal["relu", "leaky_relu", "selu"]) activation function to use.
            If you don't specify anything, ``relu`` is applied.

        dropout: (float, optional) the probability (0 to 1) of a neuron falling out
            (zeroing the value of a neuron at the output).
    """

    def __init__(self,
                 encoder_depth: list[int, ...],
                 input_shape: tuple[int, ...],
                 stride: int,
                 kernel_size: int,
                 activation: Literal["relu", "leaky_relu", "selu"] = "relu",
                 dropout: Optional[float] = None
                 ):

        super().__init__()
        self.padding = "same"
        self.stride = stride
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout = dropout

        self.encoder_depth = [input_shape[0], *encoder_depth]
        self.head_block_index = len(self.encoder_depth) - 2

        self.activations = nn.ModuleDict([
            ["relu", nn.ReLU()],
            ["leaky_relu", nn.LeakyReLU()],
            ["selu", nn.SELU()],
        ])

        self.encoder = nn.Sequential(*[
            self.encoder_block(index, in_channels, out_channels)
            for index, (in_channels, out_channels) in enumerate(zip(self.encoder_depth, self.encoder_depth[1:]))
        ])

        self.output_shape = self.__get_output_shape(input_shape)

    def __get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        with torch.no_grad():
            example_input = torch.zeros((1, *input_shape))
            return self.encoder(example_input).shape

    def encoder_block(self, index: int, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Generates an encoder from blocks that are assembled based on the received argument ``encoder_depth`` and ``input_shape``.

        For example::

            encoder_depth = [2048, 1024, 512]
            input_shape = (1, 52)

              (encoder): Sequential(
                (0): Sequential(
                  (0): Conv1d(1, 2048, kernel_size=(5,), stride=(1,), padding=same)
                  (1): ReLU()
                  (2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                  (3): BatchNorm1d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (4): Dropout(p=0.3, inplace=False)
                )
                (1): Sequential(
                  (0): Conv1d(2048, 1024, kernel_size=(5,), stride=(1,), padding=same)
                  (1): ReLU()
                  (2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                  (3): BatchNorm1d(13, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (4): Dropout(p=0.3, inplace=False)
                )
                (2): Sequential(
                  (0): Conv1d(1024, 512, kernel_size=(5,), stride=(1,), padding=same)
                  (1): ReLU()
                  (2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                  (3): BatchNorm1d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )

        Note:
            About the MaxPool layer arguments.
            With ``kernel 2*2``, ``stride 2``, and default ``padding 0``,
            the output value from the MaxPool layer is equivalent to ``padding="same"`` in keras.
        """

        block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=self.stride,
                kernel_size=self.kernel_size,
                padding="same",
            ),

            self.activations[self.activation],
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(out_channels)
        )

        if self.dropout and self.head_block_index != index:
            block.append(nn.Dropout(self.dropout))
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LSTMModule(nn.Module):
    """
    The class is a module consisting of recurrent blocks, namely LSTM.

    :param input_shape: (tuple[int, ...]) input shape for first recurrent blocks.

    :param rnn_depth: (list[int, ...]) is a list of numbers,
        where each number means the size of hidden layers in each LSTM block.

    :param dropout: (float, optional) the probability (0 to 1) of a neuron falling out
            (zeroing the value of a neuron at the output).
    """

    def __init__(self,
                 input_shape: tuple[int, ...],
                 rnn_depth: list[int],
                 dropout: Optional[float] = None
                 ):

        super().__init__()
        self.rnn_depth = [input_shape[-1], *rnn_depth]
        self.dropout = dropout
        self.head_block_index = len(self.rnn_depth) - 2
        self.rnn = nn.ModuleList()
        self.rnn_block()

    def rnn_block(self) -> None:
        """
        Generates an RNN module with LSTM from blocks
        that are assembled based on the received argument ``rnn_depth`` and ``input_shape``.

        For example::

            input_shape=(512, 6)
            rnn_depth=[256, 128]

            (rnn): LSTMModule(
                (rnn): ModuleList(
                  (0): LSTM(6, 256, batch_first=True, dropout=0.3)
                  (1): LSTM(256, 128, batch_first=True)
                )
            )

        :return: None
        """

        for index, (in_features, out_features) in enumerate(zip(self.rnn_depth, self.rnn_depth[1:])):

            dropout = 0
            if self.dropout and self.head_block_index != index:
                dropout = self.dropout

            self.rnn.append(
                nn.LSTM(
                    input_size=in_features,
                    hidden_size=out_features,
                    batch_first=True,
                    dropout=dropout
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lstm_block in self.rnn:
            x, (hidden, cell) = lstm_block(x)
        return hidden


class Decoder(nn.Module):
    """
    The ``Decoder`` module takes a tensor obtained from a block with recurrent layers (``LSTMModule``)
    and decodes the results into finite membership probabilities for each class.

    Args:
        input_shape: (tuple[int, ...]) input shape for the first decoder block.

        decoder_depth: is a list of numbers,
            where each number means the size of hidden layer in one decoder block.

        dropout: (float, optional) the probability (0 to 1) of a neuron falling out
            (zeroing the value of a neuron at the output).

        activation: (Literal["relu", "leaky_relu", "selu"]) activation function to use.
            If you don't specify anything, ``relu`` is applied.
    """

    def __init__(self,
                 input_shape: tuple[int, ...],
                 decoder_depth: list[int, ...],
                 dropout: Optional[float] = None,
                 activation: Literal["relu", "leaky_relu", "selu"] = "relu"
                 ):

        super().__init__()
        self.dropout = dropout
        self.decoder_depth = [input_shape[0], *decoder_depth]
        self.head_block_index = len(self.decoder_depth) - 2
        self.activation = activation

        self.activations = nn.ModuleDict([
            ["relu", nn.ReLU()],
            ["leaky_relu", nn.LeakyReLU()],
            ["selu", nn.SELU()],
        ])

        self.decoder = nn.Sequential()
        self.decoder_block()

    def decoder_block(self) -> None:
        """
        Generates an decoder from blocks that are assembled based on the received argument ``encoder_depth`` and ``input_shape``.

        For example::

            input_shape=(128, 1)
            decoder_depth=[64, 32, 5]

            (decoder): Sequential(
                (0): Linear(in_features=128, out_features=64, bias=True)
                (1): ReLU()
                (2): Linear(in_features=64, out_features=32, bias=True)
                (3): ReLU()
                (4): Linear(in_features=32, out_features=5, bias=True)
            )
        """

        for i, (in_features, out_features) in enumerate(zip(self.decoder_depth, self.decoder_depth[1:])):
            self.decoder.append(
                nn.Linear(
                    in_features=in_features,
                    out_features=out_features
                )
            )

            if self.head_block_index != i:
                self.decoder.append(self.activations[self.activation])
                if self.dropout:
                    self.decoder.append(nn.Dropout(self.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class BaselineRNNModel(nn.Module):
    """
    The main class for the framework of the neural architecture, which consists of separate modules:
        1) Encoder
        2) LSTMModule
        3) Decoder

    Args:
        input_shape: (tuple[int, ...]) input shape of the model.

        encoder_depth: (list[int, ...]) encoder depth is a list of numbers,
            where each number means the number of kernels (filters) in each encoder block.

        rnn_depth: (list[int, ...]) is a list of numbers,
            where each number means the size of hidden layers in each LSTM block.

        decoder_depth: is a list of numbers,
            where each number means the size of hidden layer in one decoder block.

        stride: (int) controls the stride for the cross-correlation,
            a single number or a tuple. Default is ``1``.

        kernel_size: (int) size of the convolving kernel. Default is ``5``.

        dropout: (float, optional) the probability (0 to 1) of a neuron falling out
            (zeroing the value of a neuron at the output).

        activation: (Literal["relu", "leaky_relu", "selu"]) activation function to use.
            If you don't specify anything, ``relu`` is applied.
    """

    def __init__(self,
                 input_shape: tuple[int, ...],
                 encoder_depth: list[int, ...],
                 rnn_depth: list[int, ...],
                 decoder_depth: list[int, ...],
                 kernel_size: Optional[int] = 5,
                 stride: Optional[int] = 1,
                 activation: Literal["relu", "leaky_relu", "selu"] = "relu",
                 dropout: Optional[tuple[float, float, float]] = None
                 ):

        super().__init__()
        self.example_input_array = torch.zeros(size=(1, *input_shape))
        self.dropout = dropout if dropout else (None, None, None)

        self.encoder = Encoder(
            encoder_depth=encoder_depth,
            input_shape=input_shape,
            stride=stride,
            kernel_size=kernel_size,
            activation=activation,
            dropout=self.dropout[0]
        )

        self.rnn = LSTMModule(
            input_shape=self.encoder.output_shape,
            rnn_depth=rnn_depth,
            dropout=self.dropout[1]
        )

        self.decoder = Decoder(
            input_shape=(rnn_depth[-1], 1),
            decoder_depth=decoder_depth,
            dropout=self.dropout[2],
            activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.rnn(x)
        x = self.decoder(x)
        return x.squeeze()
