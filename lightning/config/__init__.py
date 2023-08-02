"""
Description:
    The package contains modules with classes
    representing configuration models for various parts of the project.
    Classes with configurations receive parameters for each module,
    validate input arguments, and then pass them to the target module.

For example::

    # This class is a datamodule parameter configuration model

    class DataModuleParams(BaseModel):
        batch_size: Optional[int] = Field(default=4, gt=0)
        num_workers: Optional[int] = Field(default=2, gt=0)

Where:
    batch_size -- size of the mini-batch that is issued during the iteration.

    num_workers -- number of threads used when loading data.
"""

from .dataset_config import *
from .datamodule_config import *
from .etl_pipeline_config import *
from .lightmodule_config import *
