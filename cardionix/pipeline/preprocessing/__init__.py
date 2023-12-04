"""
This package contains classes for preprocessing and standardization tabular and audio data:
    - Tabular data preprocessing:
        - Splitting features by type (categorical or numerical)
        - Encoding categorical features
        - Standardization, normalization numerical features
        - And unite encoded categorical features with normalized numerical features
    - Audio data preprocessing:
        - Loading audio files
        - Converting to one sample rate and duration
        - Augmentations
        - Extract any features (MFCCs)
        - etc.
"""

from .pipeline import *
