import argparse
import json
from pathlib import Path
from typing import Literal
import numpy as np
from tqdm import tqdm
from data.synthetic import generate_synthetic_data
from data.uniform import generate_uniform_data
from bayes_error import bayes_error


type Distribution = Literal['a', 'b', 'c']


def generate_data(distribution: Distribution, m):
    if distribution == 'a':
        return generate_synthetic_data(
            n_samples=2000,
            n_hard_labels=m,
            means=[
                [0, 0],
                [2, 2],
            ],
            covs=[
                np.eye(2),
                np.eye(2),
            ],
            weights=[0.5, 0.5],
        )
    elif distribution == 'b':
        return generate_synthetic_data(
            n_samples=2000,
            n_hard_labels=m,
            means=[
                [0, 0],
                [0, 0],
            ],
            covs=[
                np.eye(2),
                np.eye(2),
            ],
            weights=[0.5, 0.5],
        )
    elif distribution == 'c':
        return generate_uniform_data(
            n_samples=2000,
            n_hard_labels=m,
            label_flip_rate=0.1,
            weights=[0.5, 0.5],
        )


def theoretical_bound(m, soft_labels):
    return np.mean(
        np.minimum(
            (1 / m)
            * soft_labels
            * (1 - soft_labels)
            / np.abs(2 * soft_labels - 1),
            np.sqrt(np.pi / (2 * m)),
        )
    )


def estimate_bayes_error(distribution: Distribution, m: int):
    data = generate_data(distribution, m)
    return (
        bayes_error(data['soft_labels_clean']),
        bayes_error(data['soft_labels_hard']),
        theoretical_bound(m, data['soft_labels_clean']),
    )


def get_bias_for_m(m: int, distribution: Distribution):
    print(f'Computing bias for distribution={distribution}, m={m}')
    expectation_of_estimator, true_bayes_error, theoretical = np.mean(
        [estimate_bayes_error(distribution, m) for _ in tqdm(range(1000))],
        axis=0,
    )
    bias = np.abs(expectation_of_estimator - true_bayes_error)
    return {
        'm': m,
        'bias': bias,
        'theoretical': theoretical,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'distribution', type=str, default='a', choices=['a', 'b', 'c']
    )
    args = parser.parse_args()

    results = {
        m: get_bias_for_m(m, args.distribution)
        for m in [10, 25, 50, 100, 250, 500, 1000]
    }
    outfile = Path(f'results/bias_{args.distribution}.json')
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=4)
