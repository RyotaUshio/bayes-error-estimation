from __future__ import annotations

from typing import Annotated, Literal, TypedDict, Union

import bestperf
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field
from scipy.special import expit, logit
from scipy.stats import multivariate_normal

from .types import Datasets
from .utils import generate_approximate_soft_labels

rng = np.random.default_rng(42)


class CleanSyntheticData(TypedDict):
    soft_labels_clean: npt.NDArray[np.float64]
    soft_labels_hard: npt.NDArray[np.float64]
    labels: npt.NDArray[np.int64]


def generate_synthetic_data(
    *, n_samples: int, n_hard_labels: int, means, covs, weights
) -> CleanSyntheticData:
    assert len(means) == len(covs) == len(weights) == 2
    # Create multivariate normal distributions
    dists = [multivariate_normal(mean, cov) for mean, cov in zip(means, covs)]

    # Sample from the distributions
    labels = rng.choice(len(means), size=n_samples, p=weights)
    instances = np.array([dists[i].rvs(random_state=rng) for i in labels])
    # (n_samples, 2), where 2 is the number of classes
    soft_labels_for_each_class = np.array(
        [
            [
                dist.pdf(instance) * weight
                for dist, weight in zip(dists, weights)
            ]
            for instance in instances
        ]
    )
    # normalize
    soft_labels_for_each_class = soft_labels_for_each_class / np.sum(
        soft_labels_for_each_class, axis=1, keepdims=True
    )

    soft_labels_clean = soft_labels_for_each_class[:, 1]
    soft_labels_hard = np.array(
        [
            np.random.multinomial(n_hard_labels, soft_label) / n_hard_labels
            for soft_label in soft_labels_for_each_class
        ]
    )[:, 1]
    return {
        'soft_labels_clean': soft_labels_clean,
        'soft_labels_hard': soft_labels_hard,
        'labels': labels,
    }


def corrupt_soft_labels(
    soft_labels: bestperf.SoftLabels,
    a: float,
    b: float,
    shuffle_fraction: float,
    perturbation: PerturbationOptions | None = None,
) -> bestperf.SoftLabels:
    def f(p, a, b):
        assert a >= 0
        assert b > 0 and b < 1
        return 1 / (1 + ((1 - p) / p) ** (1 / a) * ((1 - b) / b))

    def perturb(
        soft_labels: bestperf.SoftLabels, perturbation: PerturbationOptions
    ) -> bestperf.SoftLabels:
        match perturbation['type']:
            case 'logit_gaussian':
                sigma = perturbation['sigma']
                logits = logit(soft_labels)
                noisy_logits = logits + rng.normal(0, sigma, size=logits.shape)
                noisy_soft_labels = expit(noisy_logits)
                return noisy_soft_labels
            case _:
                raise ValueError(
                    f'Unknown perturbation type: {perturbation["type"]}'
                )

    corrupted_soft_labels = f(soft_labels, a, b)
    if shuffle_fraction > 0:
        corrupted_soft_labels = shuffle_partial(
            corrupted_soft_labels, shuffle_fraction
        )
    if perturbation is not None:
        corrupted_soft_labels = perturb(corrupted_soft_labels, perturbation)
    return corrupted_soft_labels


def shuffle_partial(arr, fraction):
    result = np.copy(arr)
    if fraction == 0:
        return result

    first_axis_size = arr.shape[0]
    n_shuffle = int(first_axis_size * fraction)
    indices_to_shuffle = rng.choice(
        first_axis_size, size=n_shuffle, replace=False
    )
    selected_rows = result[indices_to_shuffle].copy()
    rng.shuffle(indices_to_shuffle)

    for i, idx in enumerate(indices_to_shuffle):
        result[idx] = selected_rows[i]

    return result


class LogitGaussianPerturbationOptions(TypedDict):
    type: Literal['logit_gaussian']
    sigma: float


type PerturbationOptions = Annotated[
    Union[LogitGaussianPerturbationOptions], Field(discriminator='type')
]


class SyntheticOptions(BaseModel):
    dataset: Literal['synthetic'] = 'synthetic'
    shuffle_fraction: float
    a: float
    b: float
    n_hard_labels: int
    binom_noise: bool
    perturbation: PerturbationOptions | None = None


def load_synthetic(
    options: SyntheticOptions,
) -> Datasets[Literal['clean', 'hard']]:
    data = generate_synthetic_data(
        n_samples=10000,
        n_hard_labels=options.n_hard_labels,
        means=[
            np.array([0, 0]),
            np.array([2, 2]),
        ],
        covs=[
            np.eye(2),
            np.eye(2),
        ],
        weights=[0.6, 0.4],
    )
    corrupted = corrupt_soft_labels(
        data['soft_labels_clean'],
        a=options.a,
        b=options.b,
        shuffle_fraction=options.shuffle_fraction,
        perturbation=options.perturbation,
    )

    return {
        'corrupted': {
            'soft_labels': (
                generate_approximate_soft_labels(
                    rng,
                    corrupted,
                    n_hard_labels=options.n_hard_labels,
                )
                if options.binom_noise
                else corrupted
            ),
            'labels': data['labels'],
        },
        'clean': {
            'soft_labels': data['soft_labels_clean'],
            'labels': data['labels'],
        },
        'hard': {
            'soft_labels': data['soft_labels_hard'],
            'labels': data['labels'],
        },
    }
