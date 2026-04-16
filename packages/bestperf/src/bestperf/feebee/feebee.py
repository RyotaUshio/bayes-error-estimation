from collections.abc import Callable
from typing import TypedDict

import numpy as np

from ..types import Labels


type BayesErrorEstimator = Callable[[Labels], float]


class FeeBeeResult(TypedDict):
    score_lower: float
    score_upper: float
    estimates: list[float]


def inject_label_noise(labels: Labels, noise_level: float) -> Labels:
    mask = np.random.binomial(1, noise_level, size=labels.size)
    noisy_labels = np.where(mask == 1, 1 - labels, labels)
    return noisy_labels


def estimate_bayes_error(
    *,
    estimator: BayesErrorEstimator,
    labels: Labels,
    noise_level: float,
) -> float:
    noisy_labels = inject_label_noise(labels, noise_level)
    bayes_error = estimator(noisy_labels)
    return bayes_error


def feebee(
    *,
    estimator: BayesErrorEstimator,
    labels: Labels,
    n_points: int,
    sota: float,
) -> FeeBeeResult:
    """Compute FeeBee scores of the given Bayes error estimator.

    FeeBee is a real-world evaluation framework for Bayes error estimators
    proposed by:

    Renggli, C., Rimanic, L., Hollenstein, N., & Zhang, C. (2021).
    Evaluating Bayes Error Estimators on Read-World Datasets with FeeBee.
    arXiv preprint arXiv:2108.13034.

    Args:
        estimator: A function that takes noise-injected labels and returns
            estimated Bayes error.
        labels: Clean labels of the dataset.
        n_points: Number of noise levels to evaluate.
        sota: A known upper bound on the Bayes error (for the clean data).
            Typically, this is the state-of-the-art error rate.

    Returns:
        FeeBeeResult containing FeeBee scores as well as estimates at each noise level.
    """
    noise_levels = np.linspace(0, 1, n_points)
    estimates = [
        estimate_bayes_error(
            estimator=estimator,
            labels=labels,
            noise_level=noise_level,
        )
        for noise_level in noise_levels
    ]
    lower_bounds = 0.5 * noise_levels
    upper_bounds = 0.5 * noise_levels + (1 - noise_levels) * sota

    score_lower = np.maximum(0, estimates - upper_bounds).mean()
    score_upper = np.maximum(0, lower_bounds - estimates).mean()

    return FeeBeeResult(
        score_lower=score_lower,
        score_upper=score_upper,
        estimates=estimates,
    )
