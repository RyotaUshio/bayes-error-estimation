import bestperf
import scipy.stats

from .experiment_config import BootstrapOptions
from .experiment_result import ConfidenceInterval, ExperimentResult


def run_experiment(
    soft_labels: bestperf.SoftLabels,
    labels: bestperf.Labels,
    calibrator: bestperf.Calibrator | None,
    bootstrap_options: BootstrapOptions,
) -> ExperimentResult:
    if calibrator is None:
        calibrator = lambda soft_labels, labels: soft_labels  # noqa: E731

    point_estimate = bestperf.bayes_error(calibrator(soft_labels, labels)) * 100

    confidence_interval = scipy.stats.bootstrap(
        (soft_labels, labels),
        lambda soft_labels, labels: bestperf.bayes_error(
            calibrator(soft_labels, labels)
        )
        * 100,
        confidence_level=0.95,
        paired=True,
        n_resamples=bootstrap_options['n_resamples'],
        method=bootstrap_options['method'],
    ).confidence_interval

    return ExperimentResult(
        point_estimate=point_estimate,
        confidence_interval=ConfidenceInterval(
            low=confidence_interval.low,
            high=confidence_interval.high,
        ),
    )
