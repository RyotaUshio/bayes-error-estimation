import gzip
import numpy as np

from .utils import (
    download_if_not_exists,
    binarize_hard_labels,
    binarize_soft_labels,
)

# the indices of the tops classes (Shirt, Dress, Coat, Pullover, and T-shirt/top)
positive_classe_indices = [6, 3, 4, 2, 0]


def load_fashion_mnist_labels():
    path = './data/fashion-mnist/t10k-labels-idx1-ubyte.gz'
    url = 'https://github.com/zalandoresearch/fashion-mnist/raw/b2617bb6d3ffa2e429640350f613e3291e10b141/data/fashion/t10k-labels-idx1-ubyte.gz'

    # download the dataset if it is not already downloaded
    download_if_not_exists(url, path)

    # extract the original labels
    #
    # The following two lines were taken and modified from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    #
    # The MIT License (MIT) Copyright © 2017 Zalando SE, https://tech.zalando.com
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    with gzip.open(path, 'rb') as f:
        original = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    # row each element of data, if it is in positive_classe_indices, set it to 1, otherwise set it to 0
    binarized = binarize_hard_labels(original, positive_classe_indices)
    return binarized


def load_fashion_mnist_soft_labels():
    path = './data/fashion-mnist-h/fmh_counts.csv'
    url = 'https://raw.githubusercontent.com/takashiishida/irreducible/refs/heads/main/data/fmh_counts.csv'

    # download the dataset if it is not already downloaded
    download_if_not_exists(url, path)

    # use the counts data, not the probs data, to avoid overflow
    original = np.loadtxt(path, delimiter=',')
    binarized = binarize_soft_labels(original, positive_classe_indices)
    return binarized


def load_fashion_mnist():
    return {
        'soft_labels_corrupted': load_fashion_mnist_soft_labels(),
        'labels': load_fashion_mnist_labels(),
    }


def test():
    labels = load_fashion_mnist_labels()
    soft_labels = load_fashion_mnist_soft_labels()
    n_coincide = np.count_nonzero(labels == np.where(soft_labels > 0.5, 1, 0))
    n_total = labels.shape[0]
    print(f'{n_coincide}/{n_total} coincide')
    assert np.all(soft_labels >= 0) and np.all(soft_labels <= 1)
    assert labels.shape == (10000,)
    assert soft_labels.shape == (10000,)


if __name__ == '__main__':
    test()
