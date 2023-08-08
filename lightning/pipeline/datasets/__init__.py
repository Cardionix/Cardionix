"""
The package contains implementations of subclasses of the ``Dataset`` base class.
It is used to split data into parts,
transform them and extract features
using a third-party ETLPipeline module and load into RAM.

Modules:
    * classification.py - module contains a ``Dataset`` subclass named ``CardioAnomalyDataset``
    for classifying heart rate deviations by sound.

    * segmentation.py - at the moment, the module is not implemented and is not a priority.
"""

from .classification import *

