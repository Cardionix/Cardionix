"""
This package implements helper functions for:
 - loading models,
 - hyper_parameters,
 - etc.
"""

__all__ = ["load_from_checkpoint"]

import os
from pathlib import Path
from torch import load
from torch.nn import Module


def load_from_checkpoint(checkpoint_path: str | Path) -> Module:
    """Load model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileExistsError(f"Checkpoint with filepath {checkpoint_path} does not exist")
    if checkpoint_path.split(".")[-1] != "ckpt":
        raise ValueError(f"Checkpoint path must have a 'ckpt' extension, but got {checkpoint_path}")
    return load(checkpoint_path)["hyper_parameters"]["model"]
