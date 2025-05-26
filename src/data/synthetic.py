import numpy as np
from scipy.stats import multivariate_normal
from .utils import generate_approximate_soft_labels

rng = np.random.default_rng(42)

def generate_synthetic_data(*, n_samples: int, n_hard_labels: int, means, covs, weights):
    assert len(means) == len(covs) == len(weights) == 2
    # Create multivariate normal distributions
    dists = [multivariate_normal(mean, cov) for mean, cov in zip(means, covs)]

    # Sample from the distributions
    labels = rng.choice(len(means), size=n_samples, p=weights)
    instances = np.array([dists[i].rvs(random_state=rng) for i in labels])
    # (n_samples, 2), where 2 is the number of classes
    soft_labels_for_each_class = np.array(
        [
            [dist.pdf(instance) * weight for dist, weight in zip(dists, weights)]
            for instance in instances
        ]
    )
    # normalize
    soft_labels_for_each_class = soft_labels_for_each_class / np.sum(soft_labels_for_each_class, axis=1, keepdims=True)

    soft_labels_clean = soft_labels_for_each_class[:, 1]
    soft_labels_hard = np.array([np.random.multinomial(n_hard_labels, soft_label) / n_hard_labels for soft_label in soft_labels_for_each_class])[:, 1]
    return { 
        "soft_labels_clean": soft_labels_clean,
        "soft_labels_hard": soft_labels_hard,
        "labels": labels,
    }

def corrupt_soft_labels(soft_labels, a, b, shuffle_fraction):
    def f(p, a, b):
        assert a >= 0
        assert b > 0 and b < 1
        return 1 / (
            1 + ( (1 - p) / p )**(1/a) * ( (1 - b) / b )
        )
    
    return shuffle_partial(f(soft_labels, a, b), shuffle_fraction)

def shuffle_partial(arr, fraction):
    result = np.copy(arr)
    if fraction == 0:
        return result

    first_axis_size = arr.shape[0]
    n_shuffle = int(first_axis_size * fraction)
    indices_to_shuffle = rng.choice(first_axis_size, size=n_shuffle, replace=False)
    selected_rows = result[indices_to_shuffle].copy()
    rng.shuffle(indices_to_shuffle)
    
    for i, idx in enumerate(indices_to_shuffle):
        result[idx] = selected_rows[i]
    
    return result

def load_synthetic(shuffle_fraction, a, b, n_hard_labels):
    result = generate_synthetic_data(
        n_samples=10000,
        n_hard_labels=n_hard_labels,
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
    result["soft_labels_corrupted"] = corrupt_soft_labels(result["soft_labels_clean"], a=a, b=b, shuffle_fraction=shuffle_fraction)
    result["soft_labels_corrupted_hard"] = generate_approximate_soft_labels(rng, result["soft_labels_corrupted"], n_hard_labels=n_hard_labels)
    return result

def test():
    data = load_synthetic(0.1, 2, 0.7, 50)
    n_coincide = np.count_nonzero(data['labels'] == np.where(data['soft_labels_corrupted'] > 0.5, 1, 0))
    n_total = data['labels'].shape[0]
    print(f'{n_coincide}/{n_total} coincide')
    assert data['soft_labels_clean'].shape == (10000,)
    assert data['soft_labels_corrupted'].shape == (10000,)
    assert data['labels'].shape == (10000,)
    assert np.all(data['soft_labels_clean'] >= 0) and np.all(data['soft_labels_clean'] <= 1)
    assert np.all(data['soft_labels_corrupted'] >= 0) and np.all(data['soft_labels_corrupted'] <= 1)

if __name__ == '__main__':
    test()
