import numpy as np
from .fashion_mnist import (
    load_fashion_mnist_labels,
    load_fashion_mnist_soft_labels,
)


def test():
    labels = load_fashion_mnist_labels()
    soft_labels = load_fashion_mnist_soft_labels()
    n_coincide = np.count_nonzero(labels == np.where(soft_labels > 0.5, 1, 0))
    n_total = labels.shape[0]
    assert n_total == 10000
    assert n_coincide == 9955
    assert np.all(soft_labels >= 0) and np.all(soft_labels <= 1)
    assert labels.shape == (10000,)
    assert soft_labels.shape == (10000,)
