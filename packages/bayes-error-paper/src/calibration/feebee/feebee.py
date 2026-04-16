import bestperf
import bestperf.feebee

from ...data.load import load
from ..calibrators import calibrators
from .config import FeeBeeConfig
from .data import FeeBeeData


def feebee(config: FeeBeeConfig) -> FeeBeeData:
    dataset = load(config.dataset)['corrupted']

    def estimator(
        noisy_labels: bestperf.Labels,
        calibrator: bestperf.Calibrator | None,
    ) -> float:
        soft_labels = dataset['soft_labels']
        calibrated_soft_labels = (
            calibrator(soft_labels, noisy_labels) if calibrator else soft_labels
        )
        bayes_error = bestperf.bayes_error(calibrated_soft_labels)
        return bayes_error

    results = {
        name: bestperf.feebee.feebee(
            estimator=lambda noisy_labels: estimator(noisy_labels, calibrator),
            labels=dataset['labels'],
            n_points=config.n_points,
            sota=config.sota,
        )
        for name, calibrator in {
            **calibrators,
            **{
                f'hist{n_bin}': lambda soft_labels,
                labels: bestperf.calibrate_histogram_binning(
                    soft_labels, labels, n_bin
                )
                for n_bin in range(10, dataset['labels'].size // 2 + 1, 5)
            },
        }.items()
    }
    data = FeeBeeData(config=config, results=results)
    data.save()
    return data
