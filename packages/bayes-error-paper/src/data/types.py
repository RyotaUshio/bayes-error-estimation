import dataclasses
from typing import TypedDict
import numpy as np
import numpy.typing as npt


@dataclasses.dataclass
class Dataset(TypedDict):
    soft_labels: npt.NDArray[np.float64]
    labels: npt.NDArray[np.float64]


type Datasets = dict[str, Dataset]
