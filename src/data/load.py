from typing import Annotated

from pydantic import Field
from .cifar10 import Cifar10Options, load_cifar10
from .fashion_mnist import FashionMnistOptions, load_fashion_mnist
from .synthetic import SyntheticOptions, load_synthetic
from .chaos_nli import ChaosNliOptions, load_chaos_nli
from .iclr import ICLROptions, load_iclr


type DatasetOptions = Annotated[
    Cifar10Options
    | FashionMnistOptions
    | ICLROptions
    | ChaosNliOptions
    | SyntheticOptions,
    Field(discriminator='dataset'),
]


def load(options: DatasetOptions):
    match options.dataset:
        case 'cifar10':
            return load_cifar10()
        case 'fashion_mnist':
            return load_fashion_mnist()
        case 'iclr':
            return load_iclr(options)
        case 'snli' | 'mnli' | 'abduptive_nli':
            return load_chaos_nli(options)
        case 'synthetic':
            return load_synthetic(options)
        case _:
            raise ValueError(f'Invalid dataset name: {options.dataset}')
