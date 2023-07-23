"""
Docstring
"""

__all__ = ["DatasetParams"]

from typing import Optional
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic import FilePath, DirectoryPath, Field


class DatasetParams(BaseModel):
    """
    Docstring
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_dirpath: DirectoryPath
    labels_filepath: FilePath
    split_ratio: list[float] = Field(default=[0.75, 0.25], max_items=3, min_items=2)
    random_seed: Optional[int] = 42

    @field_validator("labels_filepath")
    def labels_filepath_validator(cls, value):
        """
        Docstring
        """
        extension = str(value).rsplit(".", maxsplit=1)[-1]
        if extension != "csv":
            raise ValueError(
                f"The file with an annotation of audio file classes "
                f"must have the extension .csv, but received {extension}"
            )
        return value

    @field_validator("split_ratio")
    def split_ratio_validator(cls, value):
        """
        Docstring
        """
        if sum(value) != 1.0:
            raise ValueError(
                f"The sum of all parts of the dataset partitions "
                f"should be equal to 1.0, but the result is {sum(value)}"
            )
        return value
