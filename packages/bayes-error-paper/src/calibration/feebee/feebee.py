import bestperf
import numpy as np

from ...data.load import load
from ...data.types import Dataset
from ..calibrators import calibrators
from .result import FeeBeeResult
from .config import FeeBeeConfig
from .data import FeeBeeData


def inject_label_noise(labels: bestperf.Labels, noise_level: float):
    mask = np.random.binomial(1, noise_level, size=labels.size)
    noisy_labels = np.where(mask == 1, 1 - labels, labels)
    return noisy_labels


def create_noisy_dataset(dataset: Dataset, noise_level: float) -> Dataset:
    noisy_dataset: Dataset = {
        'soft_labels': dataset['soft_labels'],
        'labels': inject_label_noise(
            dataset['labels'], noise_level=noise_level
        ),
    }
    return noisy_dataset


def estimate_bayes_error(
    *,
    dataset: Dataset,
    noise_level: float,
    calibrator: bestperf.Calibrator | None,
) -> float:
    noisy_dataset = create_noisy_dataset(dataset, noise_level=noise_level)
    calibrated_soft_labels = (
        calibrator(noisy_dataset['soft_labels'], noisy_dataset['labels'])
        if calibrator
        else noisy_dataset['soft_labels']
    )
    bayes_error = bestperf.bayes_error(calibrated_soft_labels)
    return bayes_error


def feebee_evaluate_calibrator(
    *,
    dataset: Dataset,
    n_points: int,
    sota: float,
    calibrator: bestperf.Calibrator | None,
) -> FeeBeeResult:
    noise_levels = np.linspace(0, 1, n_points)
    estimates = [
        estimate_bayes_error(
            dataset=dataset,
            noise_level=noise_level,
            calibrator=calibrator,
        )
        for noise_level in noise_levels
    ]
    lower_bounds = 0.5 * noise_levels
    upper_bounds = 0.5 * noise_levels + (1 - noise_levels) * sota

    score_lower = np.maximum(0, estimates - upper_bounds).mean()
    score_upper = np.maximum(0, lower_bounds - estimates).mean()

    return FeeBeeResult(
        estimates=estimates,
        score_lower=score_lower,
        score_upper=score_upper,
    )


def feebee(config: FeeBeeConfig) -> FeeBeeData:
    dataset = load(config.dataset)['corrupted']
    results = {
        name: feebee_evaluate_calibrator(
            dataset=dataset,
            n_points=config.n_points,
            sota=config.sota,
            calibrator=calibrator,
        )
        for name, calibrator in calibrators.items()
    }
    data = FeeBeeData(config=config, results=results)
    data.save()
    return data
