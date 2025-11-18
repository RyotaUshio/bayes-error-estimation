from typing import Callable

import numpy as np
import numpy.typing as npt

type SoftLabels = npt.NDArray[np.floating]
type Labels = npt.NDArray[np.integer]
type TrainedCalibrator = Callable[[SoftLabels], SoftLabels]
type Calibrator = Callable[[SoftLabels, Labels], SoftLabels]
