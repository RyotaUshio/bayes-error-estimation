from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import BaseModel, field_validator

from ...data.load import DatasetOptions

type BootstrapMethod = Literal['BCa', 'basic', 'percentile']


class BootstrapOptions(TypedDict):
    n_resamples: int
    method: BootstrapMethod


class ExperimentConfig(BaseModel):
    dataset: DatasetOptions
    bootstrap: BootstrapOptions

    @field_validator('dataset', mode='before')
    @classmethod
    def validate_dataset(cls, val):
        if not isinstance(val, str):
            return val
        
        # if a string is given, assume it's a path to a dataset config file
        with open(val, 'r') as f:
            data = json.load(f)

        return data

    @staticmethod
    def from_commandline() -> ExperimentConfig:
        parser = argparse.ArgumentParser()
        # dataset options
        parser.add_argument(
            '--dataset',
            type=str,
            default='cifar10',
            choices=[
                'cifar10',
                'fashion_mnist',
                'iclr',
                'snli',
                'mnli',
                'abduptive_nli',
                'synthetic',
            ],
        )
        # synthetic dataset options
        parser.add_argument(
            '--a',
            type=float,
            default=2.0,
            help='Used only for --dataset synthetic',
        )
        parser.add_argument(
            '--b',
            type=float,
            default=0.7,
            help='Used only for --dataset synthetic',
        )
        parser.add_argument(
            '--shuffle_fraction',
            type=float,
            default=0.0,
            help='Used only for --dataset synthetic',
        )
        parser.add_argument(
            '--n_hard_labels',
            type=int,
            default=50,
            help='Used only for --dataset synthetic',
        )
        parser.add_argument('--calibrate_hard', action='store_true')
        # iclr options
        parser.add_argument(
            '--years',
            default='2017-2025',
            help='Used only for --dataset iclr. E.g., "2017,2019-2021,2024-"',
        )

        # bootstrap options
        parser.add_argument(
            '--bootstrap',
            default='BCa',
            choices=['BCa', 'basic', 'percentile'],
            help='Bootstrap method to use for confidence intervals. BCa is most accurate but BY FAR the most memory-intensive and the slowest.',
        )
        parser.add_argument('--n_resamples', type=int, default=1000)

        # config file option
        parser.add_argument(
            '--config',
            type=Path,
            help='Path to JSON config file. Overrides other arguments if provided.',
        )
        args = parser.parse_args()
        if args.config:
            return ExperimentConfig.from_file(args.config)

        dataset_options = vars(args)
        dataset_options['binom_noise'] = args.calibrate_hard

        return ExperimentConfig(
            dataset=dataset_options,  # type: ignore
            bootstrap=BootstrapOptions(
                method=args.bootstrap, n_resamples=args.n_resamples
            ),
        )

    @staticmethod
    def from_file(filepath: Path) -> ExperimentConfig:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return ExperimentConfig.model_validate(data)

    def hash(self) -> str:
        return sha256(self.model_dump_json().encode()).hexdigest()
