from typing import Literal
from calibration import HistogramCalibrator
from betacal import BetaCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration

from .types import Labels, SoftLabels, TrainedCalibrator


def get_isotonic_calibrator(
    soft_labels: SoftLabels, labels: Labels
) -> TrainedCalibrator:
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(soft_labels, labels)
    return lambda z: iso.predict(z.reshape(-1))


def get_histogram_binning_calibrator(
    soft_labels: SoftLabels, labels: Labels, n_bin: int
) -> TrainedCalibrator:
    calibrator = HistogramCalibrator(len(soft_labels), n_bin)
    calibrator.train_calibration(soft_labels, labels)
    return lambda z: calibrator.calibrate(z)


type BetaCalibrationParameters = Literal['abm', 'am', 'ab', 'a']


def get_beta_calibrator(
    soft_labels: SoftLabels,
    labels: Labels,
    parameters: BetaCalibrationParameters = 'abm',
) -> TrainedCalibrator:
    bc = BetaCalibration(parameters=parameters)
    bc.fit(soft_labels, labels)
    return lambda z: bc.predict(z.reshape(-1))


def get_platt_scaling_calibrator(
    soft_labels: SoftLabels, labels: Labels
) -> TrainedCalibrator:
    calibrator = _SigmoidCalibration()
    calibrator.fit(soft_labels, labels)
    return lambda z: calibrator.predict(z.reshape(-1))


def calibrate_isotonic(soft_labels: SoftLabels, labels: Labels) -> SoftLabels:
    return get_isotonic_calibrator(soft_labels, labels)(soft_labels)


def calibrate_histogram_binning(
    soft_labels: SoftLabels, labels: Labels, n_bin: int
) -> SoftLabels:
    return get_histogram_binning_calibrator(soft_labels, labels, n_bin)(
        soft_labels
    )


def calibrate_beta(
    soft_labels: SoftLabels,
    labels: Labels,
    parameters: BetaCalibrationParameters = 'abm',
) -> SoftLabels:
    return get_beta_calibrator(soft_labels, labels, parameters)(soft_labels)


def calibrate_platt_scaling(
    soft_labels: SoftLabels, labels: Labels
) -> SoftLabels:
    return get_platt_scaling_calibrator(soft_labels, labels)(soft_labels)
