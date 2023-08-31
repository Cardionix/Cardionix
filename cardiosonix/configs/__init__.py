"""
The package contains modules with classes
representing configuration models for various parts of the project.
Classes with configurations receive parameters for each module,
validate input arguments, and then pass them to the target module.
If any arguments appear in the pipeline modules
that can directly or indirectly affect the metrics and the learning process,
these arguments should be added to the appropriate class
with the module configuration (its parameters), as well as documented.

For example::

    # This class is a datamodule parameter configuration model

    class DataModuleParams(BaseModel):
        batch_size: Optional[int] = Field(default=4, gt=0)
        num_workers: Optional[int] = Field(default=2, gt=0)

Where:
    ``batch_size`` -- size of the mini-batch that is issued during the iteration.

    ``num_workers`` -- number of threads used when loading data.
"""

from .datasets_config import *
from .datamodule_config import *
from .transforms_config import *
from .lightmodule_config import *
