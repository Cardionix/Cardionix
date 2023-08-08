"""
Docstring
"""

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning import LightTrainer
from lightning.models import BaselineRNNModel
from lightning.config import DatasetParams, ETLPipelineParams
from lightning.config import DataModuleParams, LightningModuleParams


def main(
        dataset_config: DatasetParams,
        datamodule_config: DataModuleParams,
        etl_pipeline_config: ETLPipelineParams,
        lightmodule_config: LightningModuleParams
):

    callbacks = [
        ModelCheckpoint(
            dirpath="./checkpoints",
            filename="epoch={epoch}-val_los={val/loss:.2f}",
            monitor="val/loss",
            mode="min",
            save_top_k=5,
            auto_insert_metric_name=False
        ),

        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=5,
            min_delta=1e-5,
        )
    ]

    trainer = LightTrainer(
        # Module configurations
        datamodule_config=datamodule_config,
        dataset_config=dataset_config,
        etl_pipeline_config=etl_pipeline_config,
        lightmodule_config=lightmodule_config,
        # Logging configuration
        job_type="training",
        name="test_run",
        tags=None,
        # Global seed
        seed=42,
        # pl.Trainer kwargs
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        enable_model_summary=True,
        enable_progress_bar=True,
        fast_dev_run=False,
        max_epochs=2,
        min_epochs=1,
        num_nodes=1,
        strategy="auto"
    )

    trainer.fit()


if __name__ == "__main__":
    model = BaselineRNNModel(
        input_shape=(52, 216),
        encoder_depth=[2048, 1024, 512],
        rnn_depth=[256, 128],
        decoder_depth=[64, 32, 5],
        activation="relu",
    )

    dataset_params = DatasetParams(
        audio_dirpath="./data/audio",
        labels_filepath="./data/labels.csv",
        split_ratio=[0.80, 0.20],
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
            "average_by": None
        }
    )

    datamodule_params = DataModuleParams(
        batch_size=32,
        num_workers=12
    )

    lightmodule_params = LightningModuleParams(
        model=model,
        class_weights=None
    )

    main(
        dataset_params,
        datamodule_params,
        etl_pipeline_params,
        lightmodule_params
    )
