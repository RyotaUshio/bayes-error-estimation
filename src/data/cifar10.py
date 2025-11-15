import pickle
import tarfile
import numpy as np

from .utils import (
    download_if_not_exists,
    binarize_hard_labels,
    binarize_soft_labels,
)

# the indices of the animal classes (bird, cat, deer, dog, frog, and horse)
positive_classe_indices = [2, 3, 4, 5, 6, 7]


def load_cifar10_labels():
    path = './data/cifar-10/cifar-10-python.tar.gz'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    # download the dataset if it is not already downloaded
    download_if_not_exists(url, path)

    # extract the original labels
    with tarfile.open(path, 'r:gz') as f:
        with f.extractfile('cifar-10-batches-py/test_batch') as fo:
            original = np.array(pickle.load(fo, encoding='bytes')[b'labels'])
    # row each element of data, if it is in positive_classe_indices, set it to 1, otherwise set it to 0
    binarized = binarize_hard_labels(original, positive_classe_indices)
    return binarized


def load_cifar10h_soft_labels():
    path = './data/cifar-10h/cifar10h-counts.npy'
    url = 'https://github.com/jcpeterson/cifar-10h/raw/refs/heads/master/data/cifar10h-counts.npy'

    # download the dataset if it is not already downloaded
    download_if_not_exists(url, path)

    # use the counts data, not the probs data, to avoid overflow
    original = np.load(path)
    binarized = binarize_soft_labels(original, positive_classe_indices)
    return binarized


def load_cifar10():
    return {
        'soft_labels_corrupted': load_cifar10h_soft_labels(),
        'labels': load_cifar10_labels(),
    }


def test():
    labels = load_cifar10_labels()
    soft_labels = load_cifar10h_soft_labels()
    n_coincide = np.count_nonzero(labels == np.where(soft_labels > 0.5, 1, 0))
    n_total = labels.shape[0]
    print(f'{n_coincide}/{n_total} coincide')
    assert np.all(soft_labels >= 0) and np.all(soft_labels <= 1)
    assert labels.shape == (10000,)
    assert soft_labels.shape == (10000,)


if __name__ == '__main__':
    test()
