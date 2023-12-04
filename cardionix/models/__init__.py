"""
The package contains modules
with model architectures for digital signal processing (audio).
Each file contains exactly one architecture written in PyTorch.
After creating the model and adding the file to the models package,
it is important not to forget to specify the name of the class
hat represents the model in the ``__all__`` variable in order to import it in the future.

For example::

    __all__ = ["BaselineRNNModel"]

    class BaselineRNNModel(Module): # Your model
        super(BaselineRNNModel, self).__init__()
        self.example_input_array = torch.zeros((...)
Note:
    Absolutely any class with a neural network model
    that is imported from here must contain the ``example_input_array`` instance attribute.
"""

from .baseline import *
from .cardionetv2 import CardioNetV2
