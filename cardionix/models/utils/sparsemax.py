"""
PyTorch implementation of Sparsemax function.
Sparsemax is an alternative function to Softmax. 
While Softmax returns probability, Sparsemax distributes sparse weights, most results are weighted equally. 
This is especially useful when you need to introduce sparsity into the data model for the output.
"""

__all_ = ["Sparsemax"]

from typing import Optional
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import torch


# Credits to Yandex https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
def _make_ix_like(
        x: torch.Tensor,
        dim: Optional[int] = 0
) -> torch.Tensor:
    d = x.size(dim)
    rho = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    view = [1] * x.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(
        x: torch.Tensor,
        dim: Optional[int] = -1
) -> torch.Tensor:
    """
    Sparsemax building block: compute the threshold.

    Parameters
    ----------
    input: torch.Tensor
        any dimension
    dim : int
        dimension along which to apply the sparsemax

    Returns
    -------
    tau : torch.Tensor
        the threshold value
    support_size : torch.Tensor
    """
    input_srt, _ = torch.sort(x, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(x, dim)
    support = rhos * input_srt > input_cumsum
    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(x.dtype)
    return tau, support_size


class SparsemaxFunction(Function):
    """
    Sparsemax is an alternative function to Softmax. 
    While Softmax returns probability, Sparsemax distributes sparse weights, most results are weighted equally. 
    This is especially useful when you need to introduce sparsity into the data model for the output.
    """

    @staticmethod
    def forward(
            ctx: torch.autograd.function.FunctionCtx,
            x: torch.Tensor,
            dim: Optional[int] = -1
    ) -> torch.Tensor:
        """
        Sparsemax: normalizing sparse transform likely softmax

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        x : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax

        Returns
        -------
        output : torch.Tensor
            same shape as input x
        """
        ctx.dim = dim
        max_val, _ = x.max(dim=dim, keepdim=True)
        x -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support(x, dim=dim)
        output = torch.clamp(x - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


class Sparsemax(nn.Module):
    """
    Sparsemax is an alternative function to Softmax. 
    While Softmax returns probability, Sparsemax distributes sparse weights, most results are weighted equally. 
    This is especially useful when you need to introduce sparsity into the data model for the output.
    """
    def __init__(self, dim: Optional[int] = -1):
        super().__init__()
        self.dim = dim
        self.__sparsemax = SparsemaxFunction().apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__sparsemax(x, self.dim)
