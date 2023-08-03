"""
lightning is the main engine of the project, containing nested packages with:
    1) implemented models,
    2) a regular machine learning pipeline,
    3) logging,
    4) metrics,
    5) callbacks,
    6) ETL pipeline.

It initializes the start of training and automatically logs all experiment results
through the high-level LightTrainer API.
It is necessary to import the trainer class, models and classes for configurations,
and then pass the configuration classes to the trainer class and call the ``fit()`` method.

For example::

    # Importing our libraries

    from lightning import LightTrainer
    from lightning import BaselineRNNModel
    from lightning import DatasetParams, ETLPipelineParams
    from lightning import DataModuleParams, LightningModuleParams

    # Initializing model

    model = BaselineRNNModel(
            example_input_array=(1, 52),
            encoder_depth=[1, 2048, 1024, 512],
            decoder_depth=[128, 64, 32, 5],
            rnn_input_size=6,
            rnn_hidden_size=256,
            activation="relu",
            dropout=0.3
        )

    # Defining dataset parameters

    dataset_params = DatasetParams(
        audio_dirpath="./data/audio",
        labels_filepath="./data/labels.csv",
        split_ratio=[0.80, 0.20],
    )

    # Defining transforms parameters

    etl_pipeline_params = ETLPipelineParams(
        sample_rate=22050,
        duration=10,
        mono=True,
        n_mfcc=52
    )

    # Defining datamodule parameters

    datamodule_params = DataModuleParams(
        batch_size=4,
        num_workers=2
    )

    # Defining LightningModule parameters

    lightmodule_params = LightningModuleParams(
        model=model
    )

    # Initializing LightTrainer

    trainer = LightTrainer(
        datamodule_config=datamodule_params,
        dataset_config=dataset_params,
        etl_pipeline_config=etl_pipeline_params,
        lightmodule_config=lightmodule_params,
        job_type="training",
        name="test_run",
        project="CardioSonix",
        tags=None,
        seed=42
    )

    # And start training

    trainer.fit()

Packages:
    config - package contains modules with classes representing configuration models for various parts of the project.

    models - package contains modules with model architectures for digital signal processing.

    pipeline - machine learning pipeline with metrics, datasets, feature extraction, and experiment logging.
"""

from .pipeline import *
from .models import *
from .config import *
