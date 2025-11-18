import numpy as np
from .synthetic import SyntheticOptions, load_synthetic


def test():
    np.random.seed(0)
    datasets = load_synthetic(
        SyntheticOptions(
            shuffle_fraction=0.1,
            a=2,
            b=0.7,
            n_hard_labels=50,
            binom_noise=False,
        )
    )
    n_coincide = np.count_nonzero(
        datasets['corrupted']['labels']
        == np.where(datasets['corrupted']['soft_labels'] > 0.5, 1, 0)
    )
    n_total = datasets['corrupted']['labels'].shape[0]
    assert n_total == 10000
    assert n_coincide == 8401
    for data in datasets.values():
        assert data['soft_labels'].shape == (10000,)
        assert data['labels'].shape == (10000,)
        assert np.all(data['soft_labels'] >= 0) and np.all(
            data['soft_labels'] <= 1
        )
