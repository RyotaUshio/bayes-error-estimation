import numpy as np
from .utils import generate_approximate_soft_labels

rng = np.random.default_rng(42)

def generate_uniform_data(n_samples, n_hard_labels, label_flip_rate, weights):
    # Sample from the distributions
    original_labels = rng.choice(2, size=n_samples, p=weights)
    flip_indices = rng.choice(
        2, size=n_samples, p=[1 - label_flip_rate, label_flip_rate]
    )
    # flip labels
    labels = (original_labels + flip_indices) % 2
    soft_labels_clean = np.where(original_labels == 1, 1 - label_flip_rate, label_flip_rate)
    soft_labels_hard = generate_approximate_soft_labels(rng, soft_labels_clean, n_hard_labels)
    return {
        "soft_labels_clean": soft_labels_clean,
        "soft_labels_hard": soft_labels_hard,
        "labels": labels,
    }

def test():
    data = generate_uniform_data(1000, 10, 0.1, [0.5, 0.5])
    n_coincide = np.count_nonzero(data['labels'] == np.where(data['soft_labels_hard'] > 0.5, 1, 0))
    n_total = data['labels'].shape[0]
    print(f'{n_coincide}/{n_total} coincide')
    assert data['soft_labels_clean'].shape == (1000,)
    assert data['soft_labels_hard'].shape == (1000,)
    assert data['labels'].shape == (1000,)
    assert np.all(data['soft_labels_clean'] >= 0) and np.all(data['soft_labels_clean'] <= 1)
    assert np.all(data['soft_labels_hard'] >= 0) and np.all(data['soft_labels_hard'] <= 1)

if __name__ == '__main__':
    test()
