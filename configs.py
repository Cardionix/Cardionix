"""
This file should contain configurations for the Cardio Sonix Pipeline.
You also can edite this configs as you want and
import them into main.py to start training or
define configurations currently into main.py
"""

__all__ = [
    "dataset_params",
    "etl_pipeline_params",
    "datamodule_params",
    "lightmodule_params"
]

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from cardionix.configs import (
    ClassifyDatasetParams,
    ETLPipelineParams,
    DataModuleParams,
    LightningModuleParams
)


# Arguments for building and splitting dataset and unite classes.
dataset_params = ClassifyDatasetParams(
    # extra_filepath="./data/DHD/extra/CDC_survey_2020.csv",  # file .csv (label encoded CDC survey 2020)
    audio_dirpath="./data/DHD/audio",  # dirpath (DHD/audio dir)
    metadata_filepath="./data/DHD/metadata.csv",
    labels_filepath="./data/DHD/labels.csv",  # file .csv (DHD/labels.csv file)
    split_ratio=[0.80, 0.20],  # split ratio of the dataset: 80% - train and 20% - validation
    # Which classes is needed for merging into one class
    merge_classes={
        "artifact": ["artifact"],
        "healthy": ["normal"],
        "abnormal": ["murmur", "extrahls", "extrastole"]
    }
)

# Arguments for preprocessing and augment audio data, encoding and normalizing tabular data.
etl_pipeline_params = ETLPipelineParams(
    pad_mode="nearest_neighbors",
    # scaler="Normalizer",  # scaler for normalization or scaling numerical features in tabular data
    # encoder="OneHotEncoder",  # encoder for encoding categorical features in tabular data
    sample_rate=22050,  # sample rate for the resampling all audio data
    duration=20,  # duration to pad or clip all audio data
    mono=True,  # one or two audio channels load
    extractor="MFCC",  # extractor name for extracting features from audio data
    # extractor parameters for extracting features from audio data
    extractor_kwargs={
        "n_fft": 2048,
        "win_length": 2048,
        "hop_length": 1024,
        "n_mels": 128,  # number of mel filters
        "n_mfcc": 128,  # number of mfcc`s
        "average_by": None  # average time or frequency axis (non-average if None)
    },
    merge_rules={
        "artifact": "artifact",
        "healthy": "healthy",
        "abnormal": {"abnormal": 0.7, "healthy": 0.3}
    }
)

datamodule_params = DataModuleParams(batch_size=20, num_workers=12)  # Data uploading parameters

# Arguments for Optimizer, Sheduler and Loss function
lightmodule_params = LightningModuleParams(
    optimizer=optim.Adam,  # Optimizer to use for training model
    # optimizer parameters
    optimizer_kwargs={
        "lr": 1e-4
    },
    lr_scheduler=lr_scheduler.ReduceLROnPlateau,  # learning rate Scheduler
    # and his parameters
    lr_scheduler_kwargs={
        "patience": 5
    },
    lr_scheduler_dict_kwargs={
        "monitor": "val/loss",
        "interval": "epoch"
    },
    criterion=nn.CrossEntropyLoss,  # Loss function to use
    # and its parameters
    criterion_kwargs={
        "weight": torch.tensor([1.0, 1.0, 1.0])
    }
)
