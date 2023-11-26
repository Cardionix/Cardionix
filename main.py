"""
Entry point for the Cardio Sonix pipeline.
Training, validation and testing start here.
This module must include configurations
for the building any pipeline components.
Also is necessary to define callbacks to control training.
Before starting the pipeline you should
install the required libraries and tools.
"""

from torch import nn
from cardiosonix import CardioTrainer
from cardiosonix.models import CardioNetV2
from callbacks import callbacks
from configs import (
    dataset_params,
    etl_pipeline_params,
    datamodule_params,
    lightmodule_params
)


def main(model: nn.Module) -> None:
    # (*) YOU SHOULD DEFINE TRAINER HERE (*)
    trainer = CardioTrainer(
        model=model,  # your Model (subclass nn.Module)
        # Module configurations
        datamodule_config=datamodule_params,  # Datamodule configurations
        dataset_config=dataset_config,  # Dataset configurations
        etl_pipeline_config=etl_pipeline_params,  # ETL-pipeline configurations
        lightmodule_config=lightmodule_config,  # Lightmodule configurations
        # Logging configuration
        job_type="feat test",  # WHAT TYPE JOB YOU DO? (maybe research or just validation)
        name="experimental run",  # WHAT YOU DO? (maybe you just test mew features)
        tags=["any Optim", "any lr"],  # WHAT CAN YOU ASK ABOUT RUN? ('SGD', 'lr 1e-4', 'etc.')
        seed=42,  # global SEED
        # pl.Trainer kwargs
        log_every_n_steps=20,  # log metrics every N steps
        callbacks=callbacks,  # define callbacks
        accelerator="auto",  # define accelerator
        devices="auto",  # define devices
        enable_model_summary=True,  # enable model summary
        enable_progress_bar=True,  # activate progress bar
        fast_dev_run=True,  # init testing on one epoch
        max_epochs=100,  # maximum epochs
        min_epochs=10,  # minimum epochs
        num_nodes=1,  # choose nodes
        strategy="auto"  # define strategy
    )

    trainer.fit()  # start training
    trainer.predict()  # start prediction


if __name__ == "__main__":
    # (~) YOU SHOULD DEFINE YOUR MODEL HERE (~)
    model = CardioNetV2(
        num_classes=3,
        audio_features_shape=(235, 52),
        tabular_features=50,
        rnn_layers=1,
        stem_channels=256,
        resnet_backbone={
            256: 2,
            512: 2,
            1024: 2,
        },
        mixer_depth={
            "tabular": [128, 256, 512],
            "mixer": [2048]
        }
    )
    main(model)
