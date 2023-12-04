__all__ = ["GhostBatchNorm"]

from typing import Optional
import torch
from torch import nn


class GhostBatchNorm(nn.Module):
    """
    Ghost Batch Normalization (GBN) is a variant of batch normalization 
    that is used in the TabNet neural network architecture. 
    GBN aims to improve the performance of batch normalization 
    by addressing some of its limitations.

    Batch normalization is a technique commonly used in neural networks 
    to normalize the activations of each layer. 
    It helps in stabilizing the learning process and accelerating convergence. 
    However, traditional batch normalization calculates the statistics 
    (mean and variance) of the activations using only the examples in the current mini-batch. 
    This can lead to high variance in the estimated statistics, 
    especially when the mini-batch size is small.

    GBN addresses this limitation by introducing a "ghost" batch, 
    which is a larger virtual batch containing multiple mini-batches. 
    Instead of calculating the statistics based on a single mini-batch, 
    GBN computes them using both the current mini-batch and the ghost batch. 
    This way, the estimated statistics become more stable and reliable.

    The ghost batch is constructed by accumulating statistics 
    from multiple iterations during training. 
    It is typically much larger than a single mini-batch and contains 
    a representative sample of the entire training dataset. 
    By incorporating information from a larger batch, 
    GBN reduces the variance in the estimated statistics and improves their accuracy.
    """
    def __init__(self,
                 features: int,
                 virtual_batch_size: Optional[int] = 5,
                 momentum: Optional[float] = 0.01
                 ):
        super().__init__()
        self.bn = nn.BatchNorm1d(features, momentum=momentum)
        self.features = features
        self.virtual_batch_size = virtual_batch_size

    def split_chunks(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.virtual_batch_size > x.size(0):
            raise ValueError(
                f"Virtual batch size must be less than global batch size, "
                f"but 'virtual_batch_size' is {self.virtual_batch_size} "
                f"and global batch size is {x.size(0)}"
            )
        num_chunks = x.size(0) // self.virtual_batch_size
        return torch.chunk(x, num_chunks, 0)

    def batch_norm(self, chunks: tuple[torch.Tensor, ...]) -> list[torch.Tensor, ...]:
        return [
            self.bn(chunk)
            for chunk in chunks
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = self.split_chunks(x)
        chunks = self.batch_norm(chunks)
        return torch.cat(chunks, dim=0)
