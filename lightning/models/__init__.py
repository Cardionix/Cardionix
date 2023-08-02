"""
Description:
    The package contains modules
    with model architectures for digital signal processing (audio).
    Each file contains exactly one architecture written in PyTorch.
    After creating the model and adding the file to the models package,
    it is important not to forget to specify the name of the class
    hat represents the model in the __all__ variable in order to import it in the future.

For example::

    __all__ = ["BaselineRNNModel"]

    class BaselineRNNModel(Module): # Your model
        pass
"""

from .baseline import *
