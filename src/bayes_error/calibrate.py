from sklearn.isotonic import IsotonicRegression
import calibration as cal
from betacal import BetaCalibration


def get_isotonic_calibrator(soft_labels, labels):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(soft_labels, labels)
    return lambda z: iso.predict(z.reshape(-1))

def get_histogram_binning_calibrator(soft_labels, labels, n_bin):
    calibrator = cal.HistogramCalibrator(len(soft_labels), n_bin)
    calibrator.train_calibration(soft_labels, labels)
    return lambda z: calibrator.calibrate(z)

def get_beta_calibrator(soft_labels, labels):
    bc = BetaCalibration(parameters="abm")
    bc.fit(soft_labels.reshape(-1, 1), labels)
    return lambda z: bc.predict(z.reshape(-1))


def calibrate_isotonic(soft_labels, labels):
    return get_isotonic_calibrator(soft_labels, labels)(soft_labels)

def calibrate_histogram_binning(soft_labels, labels, n_bin):
    return get_histogram_binning_calibrator(soft_labels, labels, n_bin)(soft_labels)

def calibrate_beta(soft_labels, labels):
    return get_beta_calibrator(soft_labels, labels)(soft_labels)
