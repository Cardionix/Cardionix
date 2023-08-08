"""
Docstring
"""

from lightning import LightTrainer
from lightning import BaselineRNNModel
from lightning.config import DatasetParams, ETLPipelineParams
from lightning.config import DataModuleParams, LightningModuleParams


def main(
        dataset_config: DatasetParams,
        datamodule_config: DataModuleParams,
        etl_pipeline_config: ETLPipelineParams,
        lightmodule_config: LightningModuleParams
):
    trainer = LightTrainer(
        datamodule_config=datamodule_config,
        dataset_config=dataset_config,
        etl_pipeline_config=etl_pipeline_config,
        lightmodule_config=lightmodule_config,
        job_type="training",
        name="test_run",
        project="CardioSonix",
        tags=None,
        seed=42,
        accelerator="auto",
        devices="auto",
        enable_model_summary=True,
        enable_progress_bar=True,
        fast_dev_run=False,
        max_epochs=5,
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
            "n_mfcc": 52
        }
    )

    datamodule_params = DataModuleParams(
        batch_size=32,
        num_workers=12
    )

    lightmodule_params = LightningModuleParams(model=model)

    main(
        dataset_params,
        datamodule_params,
        etl_pipeline_params,
        lightmodule_params
    )
