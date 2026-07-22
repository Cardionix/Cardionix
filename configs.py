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
    audio_dirpath="./data/CardionixDataset/audio",  # dirpath (DHD/audio dir)

    labels_filepath="./data/CardionixDataset/annotation.csv",  # file .csv (DHD/labels.csv file)

    split_ratio=[0.70, 0.30],  # split ratio of the dataset: 80% - train and 20% - validation

    #merge_classes={
    #    "artifact": "artifact",
    #    "normal": "normal",
    #    #"murmur": "murmur",
    #    #"extrahls": "extrahls",
    #    #"extrastole": "extrastole",
    #    "abnormal": ["murmur", "extrahls", "extrastole"]
    #}
    # Which classes is needed for merging into one class
)


# Arguments for preprocessing and augment audio data, encoding and normalizing tabular data.
etl_pipeline_params = ETLPipelineParams(
    pad_mode="constant",
    sample_rate=2_000,  # sample rate for the resampling all audio data
    duration=10,  # duration to pad or clip all audio data
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


)


datamodule_params = DataModuleParams(
    batch_size=32,
    num_workers=14
)  # Data uploading parameters


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
        "patience": 3
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
