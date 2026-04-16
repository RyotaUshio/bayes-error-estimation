from typing import Literal, TypedDict
import bestperf


class Dataset(TypedDict):
    soft_labels: bestperf.SoftLabels
    labels: bestperf.Labels


type Datasets[T: str] = dict[Literal['corrupted'] | T, Dataset]
