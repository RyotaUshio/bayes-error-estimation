import bestperf


calibrators: dict[str, bestperf.Calibrator | None] = {
    'corrupted': None,
    'isotonic': lambda soft_labels, labels: bestperf.calibrate_isotonic(
        soft_labels, labels
    ),
    'hist10': lambda soft_labels, labels: bestperf.calibrate_histogram_binning(
        soft_labels, labels, 10
    ),
    'hist25': lambda soft_labels, labels: bestperf.calibrate_histogram_binning(
        soft_labels, labels, 25
    ),
    'hist50': lambda soft_labels, labels: bestperf.calibrate_histogram_binning(
        soft_labels, labels, 50
    ),
    'hist100': lambda soft_labels, labels: bestperf.calibrate_histogram_binning(
        soft_labels, labels, 100
    ),
    'beta': lambda soft_labels, labels: bestperf.calibrate_beta(
        soft_labels, labels
    ),
    'beta-am': lambda soft_labels, labels: bestperf.calibrate_beta(
        soft_labels, labels
    ),
    'beta-ab': lambda soft_labels, labels: bestperf.calibrate_beta(
        soft_labels, labels
    ),
    'beta-a': lambda soft_labels, labels: bestperf.calibrate_beta(
        soft_labels, labels
    ),
    'platt': lambda soft_labels, labels: bestperf.calibrate_platt_scaling(
        soft_labels, labels
    ),
}
