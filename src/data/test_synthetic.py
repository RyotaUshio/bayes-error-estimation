import numpy as np
from .synthetic import load_synthetic


def test():
    np.random.seed(0)
    data = load_synthetic(0.1, 2, 0.7, 50)
    n_coincide = np.count_nonzero(
        data['labels'] == np.where(data['soft_labels_corrupted'] > 0.5, 1, 0)
    )
    n_total = data['labels'].shape[0]
    assert n_total == 10000
    assert n_coincide == 8401
    assert data['soft_labels_clean'].shape == (10000,)
    assert data['soft_labels_corrupted'].shape == (10000,)
    assert data['labels'].shape == (10000,)
    assert np.all(data['soft_labels_clean'] >= 0) and np.all(
        data['soft_labels_clean'] <= 1
    )
    assert np.all(data['soft_labels_corrupted'] >= 0) and np.all(
        data['soft_labels_corrupted'] <= 1
    )
