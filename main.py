"""
Docstring
"""

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from cardiosonix import CardioTrainer
from cardiosonix.models import BaselineRNNModel
from cardiosonix.configs import ClassifyDatasetParams, ETLPipelineParams
from cardiosonix.configs import DataModuleParams, LightningModuleParams
from callbacks import callbacks


def main(
        dataset_config: ClassifyDatasetParams,
        datamodule_config: DataModuleParams,
        etl_pipeline_config: ETLPipelineParams,
        lightmodule_config: LightningModuleParams
):

    model = BaselineRNNModel(
        input_shape=(1, 52),
        encoder_depth=[1024, 512, 256],
        rnn_depth=[128, 64],
        decoder_depth=[32, 16, 3],
        activation="relu"
    )

    trainer = CardioTrainer(
        # Module configurations
        model=model,
        datamodule_config=datamodule_config,
        dataset_config=dataset_config,
        etl_pipeline_config=etl_pipeline_config,
        lightmodule_config=lightmodule_config,
        # Logging configuration
        job_type="training",
        name="low params model",
        tags=["custom classes", "low params"],
        # Global seed
        seed=42,
        # pl.Trainer kwargs
        log_every_n_steps=20,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        enable_model_summary=False,
        enable_progress_bar=True,
        fast_dev_run=False,
        max_epochs=100,
        min_epochs=10,
        num_nodes=1,
        strategy="auto"
    )

    trainer.fit()
    trainer.predict()


if __name__ == "__main__":
    dataset_params = ClassifyDatasetParams(
        #extra_filepath="./data/DHD/extra/CDC_survey_2020.csv",
        audio_dirpath="./data/DHD/audio",
        labels_filepath="./data/DHD/labels.csv",
        split_ratio=[0.80, 0.20],
        merge_classes={
            "artifact": ["artifact"],
            "healthy": ["normal"],
            "abnormal": ["murmur", "extrahls", "extrastole"]
        }
    )

    etl_pipeline_params = ETLPipelineParams(
        sample_rate=22050,
        duration=10,
        mono=True,
        extractor="MFCC",
        extractor_kwargs={
            "n_fft": 2048,
            "win_length": 2048,
            "hop_length": 1024,
            "n_mels": 52,
            "n_mfcc": 52,
            "average_by": "time"
        }
    )

    datamodule_params = DataModuleParams(batch_size=20, num_workers=12)

    lightmodule_params = LightningModuleParams(
        optimizer=optim.Adam,
        optimizer_kwargs={
            "lr": 1e-4
        },
        lr_scheduler=lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_kwargs={
            "patience": 3
        },
        lr_scheduler_dict_kwargs={
            "monitor": "val/loss",
            "interval": "epoch"
        },
        criterion=nn.CrossEntropyLoss,
        criterion_kwargs={
            "weight": torch.tensor([1.0, 1.0, 1.0])
        }
    )

    main(
        dataset_params,
        datamodule_params,
        etl_pipeline_params,
        lightmodule_params
    )
