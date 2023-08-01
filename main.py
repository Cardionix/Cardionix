"""
Docstring
"""

from lightning import LightTrainer
from lightning import BaselineRNNModel
from lightning import DatasetParams, ETLPipelineParams
from lightning import DataModuleParams, LightningModuleParams


def main():
    pass


if __name__ == "__main__":
    model = BaselineRNNModel(
        example_input_array=(1, 52),
        encoder_depth=[1, 2048, 1024, 512],
        decoder_depth=[128, 64, 32, 5],
        rnn_input_size=6,
        rnn_hidden_size=256,
        activation="relu",
        dropout=0.3
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
        n_mfcc=52
    )

    datamodule_params = DataModuleParams(
        batch_size=4,
        num_workers=2
    )

    lightmodule_params = LightningModuleParams(
        model=model
    )

    LightTrainer(
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
