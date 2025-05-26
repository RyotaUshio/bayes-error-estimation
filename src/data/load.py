from .cifar10 import load_cifar10
from .fashion_mnist import load_fashion_mnist
from .synthetic import load_synthetic

def load(dataset_name: str, *, a: float = None, b: float = None, shuffle_fraction: float = None, n_hard_labels: int = None):
    if dataset_name == 'cifar10':
        return load_cifar10()
    if dataset_name == 'fashion_mnist':
        return load_fashion_mnist()
    if dataset_name == 'synthetic':
        return load_synthetic(shuffle_fraction=shuffle_fraction, a=a, b=b, n_hard_labels=n_hard_labels)
    
    raise ValueError(f'Invalid dataset name: {dataset_name}')
