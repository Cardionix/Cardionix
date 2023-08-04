"""
The core of the lightning package.
Contains ``trainer`` and ``callbacks`` modules.
The trainer is used when training is initialized and
the callbacks define events that should happen during training
under certain conditions (for example, an early stop)
"""

from .trainer import *
