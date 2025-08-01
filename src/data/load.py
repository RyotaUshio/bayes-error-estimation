from .cifar10 import load_cifar10
from .fashion_mnist import load_fashion_mnist
from .synthetic import load_synthetic
from .chaos_nli import load_chaos_nli

def load(dataset_name: str, *, a: float | None = None, b: float | None = None, shuffle_fraction: float | None = None, n_hard_labels: int | None = None):
    match dataset_name:
        case 'cifar10':
            return load_cifar10()
        case 'fashion_mnist':
            return load_fashion_mnist()
        case 'snli' | 'mnli' | 'abduptive_nli':
            return load_chaos_nli(dataset_name)
        case 'synthetic':
            return load_synthetic(shuffle_fraction=shuffle_fraction, a=a, b=b, n_hard_labels=n_hard_labels)
        case _:
            raise ValueError(f'Invalid dataset name: {dataset_name}')
