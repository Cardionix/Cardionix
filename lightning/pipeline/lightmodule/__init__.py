"""
The package is responsible for managing the training cycle,
calculating metrics and logging them.

Modules:
    1) lightmodule - contains a ``LightningModule`` subclass ``CardioLightningModule``.
    It is a high-level API for learning cycle management, validation and testing.
    This class also logs metrics and changes its behavior depending on callbaks.

    2) metrics - contains a data structure ``MetricsStorage`` for accumulating predictions
    for a certain number of training iterations and a ``LightMetrics`` class
    that inherits the ability to accumulate predictions
    and calculate metrics for their subsequent logging.
"""

from .lightmodule import *
