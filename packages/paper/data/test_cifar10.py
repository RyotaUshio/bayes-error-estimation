import numpy as np
from .cifar10 import load_cifar10_labels, load_cifar10h_soft_labels


def test():
    labels = load_cifar10_labels()
    soft_labels = load_cifar10h_soft_labels()
    n_coincide = np.count_nonzero(labels == np.where(soft_labels > 0.5, 1, 0))
    n_total = labels.shape[0]
    assert n_total == 10000
    assert n_coincide == 9994
    assert np.all(soft_labels >= 0) and np.all(soft_labels <= 1)
    assert labels.shape == (10000,)
    assert soft_labels.shape == (10000,)
