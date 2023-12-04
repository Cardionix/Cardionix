"""
Contains a ``DatasetParams`` class
with parameters for ``CardioAnomalyDataset`` configuration.
"""

__all__ = ["ClassifyDatasetParams"]

from typing import Optional
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic import FilePath, DirectoryPath, Field


class ClassifyDatasetParams(BaseModel):
    """
    Сlass with parameters for ``CardioAnomalyDataset`` configuration.

    Attributes:
        audio_dirpath: (DirectoryPath) path to the directory with audio recordings of heartbeats.

        labels_filepath: (FilePath) path to ``csv`` file with class tags for each heartbeat audio recording.

        split_ratio: dataset partitioning factor.
            If there are two numbers in the list, then the dataset will be divided into training and validation parts,
            where the list element with index 0 will be the part of the training part of the dataset,
            and with index 1 the part of the validation part.
            If there are 3 values in the collection,
            then the dataset will be divided into training, validation and test parts,
            where the element with index 2 will determine the share of the test part.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_dirpath: DirectoryPath
    extra_filepath: Optional[FilePath] = None
    labels_filepath: FilePath
    metadata_filepath: Optional[FilePath] = None
    split_ratio: list[float] = Field(default=[0.80, 0.20], max_items=3, min_items=2)
    merge_classes: Optional[dict] = {
        "artifact": "artifact",
        "healthy": "normal",
        "abnormal": ["murmur", "extrahls", "extrastole"]
    }

    @field_validator("extra_filepath", "labels_filepath", "metadata_filepath")
    def filepath_validator(cls, value):
        if not value:
            return None
        extension = str(value).rsplit(".", maxsplit=1)[-1]
        if extension != "csv":
            raise ValueError(
                f"The file with an annotation of audio file classes "
                f"must have the extension .csv, but received {extension}"
            )
        return value

    @field_validator("split_ratio")
    def split_ratio_validator(cls, value):
        if sum(value) != 1.0:
            raise ValueError(
                f"The sum of all parts of the dataset partitions "
                f"should be equal to 1.0, but the result is {sum(value)}"
            )
        return value
