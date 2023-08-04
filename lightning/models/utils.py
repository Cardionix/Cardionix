"""
The module contains helper functions or classes
for more flexible and easier building of machine learning architectures.
"""

__all__ = ["_compute_conv_out_features", "_compute_same_padding", "_conv_shape_info"]

from typing import Union, Optional, Literal


def _compute_conv_out_features(
        kernel_size: int,
        stride: int,
        padding: int,
        in_features: int,
) -> int:

    """
    Calculates the size (shape) of the output after a convolution operation.
    Can also calculate size for pooling operations.

    :param kernel_size: (int) size of the convolving kernel.

    :param stride: (int) controls the stride for the cross-correlation, a single number.

    :param padding: (int) controls the amount of padding applied to the input.
        It can be either a string {‘valid’, ‘same’} or an int.

    :param in_features: (int) size of features in the input.
        If conv1d, then just the number of features in the input tensor,
        and for conv2d, the height or width of the picture,
        provided that the width is equal to the height.

    :return: (int) size of output features.
    """

    return (in_features + 2 * padding - kernel_size) // stride + 1


def _compute_same_padding(
        in_features: int,
        out_features: int,
        stride: int,
) -> int:

    """
    Calculates the padding size required for the ``same`` parameter.
    That is, what should be the padding to leave the output form of the tensor the same as it was at the input.

    :param in_features: (int) size of features in the input.
        If conv1d, then just the number of features in the input tensor,
        and for conv2d, the height or width of the picture,
        provided that the width is equal to the height.

    :param out_features: (int) the number (size) of features at the output of the convolution or pooling layer,
        which was if the padding value was 0.

    :param stride: (int) controls the stride for the cross-correlation, a single number.

    :return: (int) padding size to leave the output form of the tensor the same as it was at the input.
    """

    return stride * (in_features - out_features) // 2


def _conv_shape_info(
        kernel_size: Union[int, tuple],
        stride: int,
        padding: Union[int, Literal["same", "valid"]],
        in_features: Union[int, tuple],
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
) -> dict:

    """
    Calculates the output shape after a convolution or pooling operation.

    Example function ouptuts::

        {
            'input_shape': {
                'in_channels': 1,
                'in_features': 52
            },

            'output_shape': {
                'out_channels': 512,
                'out_features': 48
            },

            'params': {
                'padding': 0,
                'kernel_size': 5,
                'stride': 1
                }
        }

    :param kernel_size: (int or tuple) size of the convolving kernel.

    :param stride: (int) controls the stride for the cross-correlation, a single number.

    :param padding: (int or Literal["same", "valid"]) controls the amount of padding applied to the input.
        It can be either a string {‘valid’, ‘same’} or an int.

    :param in_features: (int or tuple) size of features in the input.
        If conv1d, then just the number of features in the input tensor,
        and for conv2d, the height or width of the picture,
        provided that the width is equal to the height.

    :param in_channels: (int, optional) number of channels in the input.

    :param out_channels: (int, optional) number of channels produced by the convolution.

    :return: (dict) complete information about the input and output dimensions of a tensor,
        given certain arguments, after applying a convolution or pooling operation.
    """

    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]
    if isinstance(in_features, tuple):
        in_features = in_features[-1]

    if isinstance(padding, int):
        out_features = _compute_conv_out_features(kernel_size, stride, padding, in_features)
    elif isinstance(padding, str):
        if padding == "valid":
            out_features = _compute_conv_out_features(kernel_size, stride, 0, in_features)
        elif padding == "same":
            default_out_features = _compute_conv_out_features(kernel_size, stride, 0, in_features)
            padding = _compute_same_padding(in_features, default_out_features, stride)
            out_features = _compute_conv_out_features(kernel_size, stride, padding, in_features)
        else:
            raise ValueError(
                f"Expected argument padding must be 'same' or 'valid', "
                f"but got {padding}"
            )
    else:
        raise TypeError(
            f"Expected argument padding must be int or str, "
            f"but got {type(padding)}"
        )

    return {
        "input_shape": {
            "in_channels": in_channels,
            "in_features": in_features
        },

        "output_shape": {
            "out_channels": out_channels,
            "out_features": out_features,
        },

        "params": {
            "padding": padding,
            "kernel_size": kernel_size,
            "stride": stride
        }
    }
