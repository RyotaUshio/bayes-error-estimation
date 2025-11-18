from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import shutil
from typing import Callable, ClassVar, Literal, TypedDict

import numpy as np
import numpy.typing as npt
import scipy.stats
from pydantic import BaseModel

import bayes_error as be
from data.load import DatasetOptions, load

type BootstrapMethod = Literal['BCa', 'basic', 'percentile']


class BootstrapOptions(TypedDict):
    n_resamples: int
    method: BootstrapMethod


class ConfidenceInterval(TypedDict):
    low: float
    high: float


class ExperimentResult(TypedDict):
    point_estimate: float
    confidence_interval: ConfidenceInterval


def run_experiment(
    soft_labels: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    calibrator: Calibrator | None,
    bootstrap_options: BootstrapOptions,
) -> ExperimentResult:
    if calibrator is None:
        calibrator = lambda soft_labels, labels: soft_labels  # noqa: E731

    point_estimate = be.bayes_error(calibrator(soft_labels, labels)) * 100

    confidence_interval = scipy.stats.bootstrap(
        (soft_labels, labels),
        lambda soft_labels, labels: be.bayes_error(
            calibrator(soft_labels, labels)
        )
        * 100,
        confidence_level=0.95,
        paired=True,
        n_resamples=bootstrap_options['n_resamples'],
        method=bootstrap_options['method'],
    ).confidence_interval

    return ExperimentResult(
        point_estimate=point_estimate,
        confidence_interval=ConfidenceInterval(
            low=confidence_interval.low,
            high=confidence_interval.high,
        ),
    )


class ExperimentMetadata(TypedDict):
    spearman_corr: float | None


class ExperimentData(BaseModel):
    outdir: ClassVar[Path] = Path('results')

    config: ExperimentConfig
    results: dict[str, ExperimentResult]
    metadata: ExperimentMetadata

    def save(self) -> None:
        # dataset = self.config.dataset
        # bootstrap = self.config.bootstrap

        # suffix = (
        #     f'_{dataset.a:.1f}_{dataset.b:.1f}_{dataset.shuffle_fraction:.1f}_{dataset.n_hard_labels}{"_hard" if dataset.binom_noise else ""}'
        #     if dataset.dataset == 'synthetic'
        #     else ''
        # )
        # outfile = (
        #     outdir
        #     / f'{dataset.dataset}_{bootstrap["n_resamples"]}_{bootstrap["method"]}{suffix}.json'
        # )

        outfile = ExperimentData.outdir / f'{self.config.hash()}.json'
        outfile.parent.mkdir(parents=True, exist_ok=True)

        with open(outfile, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)

        print(f'Results saved to {outfile}')
        print(
            f'To visualize the result, run: \nuv run scripts/calibration/plot.py {outfile}'
        )


class ExperimentConfig(BaseModel):
    dataset: DatasetOptions
    bootstrap: BootstrapOptions

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


type Calibrator = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.int64]], npt.NDArray[np.float64]
]

calibrators: dict[str, Calibrator | None] = {
    'corrupted': None,
    'isotonic': lambda soft_labels, labels: be.calibrate_isotonic(
        soft_labels, labels
    ),
    'hist10': lambda soft_labels, labels: be.calibrate_histogram_binning(
        soft_labels, labels, 10
    ),
    'hist25': lambda soft_labels, labels: be.calibrate_histogram_binning(
        soft_labels, labels, 25
    ),
    'hist50': lambda soft_labels, labels: be.calibrate_histogram_binning(
        soft_labels, labels, 50
    ),
    'hist100': lambda soft_labels, labels: be.calibrate_histogram_binning(
        soft_labels, labels, 100
    ),
    'beta': lambda soft_labels, labels: be.calibrate_beta(soft_labels, labels),
}


def get_metadata(datasets: dict) -> ExperimentMetadata:
    metadata: ExperimentMetadata = {'spearman_corr': None}
    if 'corrupted' in datasets and 'clean' in datasets:
        spearman_corr: float = scipy.stats.spearmanr(
            datasets['clean']['soft_labels'],
            datasets['corrupted']['soft_labels'],
        ).statistic  # type: ignore
        metadata['spearman_corr'] = spearman_corr
    return metadata


def main():
    config = ExperimentConfig.from_commandline()
    print('Experiment configuration:')
    print(config.model_dump_json(indent=2))
    print('=' * shutil.get_terminal_size().columns)

    datasets = load(config.dataset)
    results: dict[str, ExperimentResult] = {}

    for name, dataset in datasets.items():
        soft_labels_corrupted = dataset['soft_labels']
        labels = dataset['labels']

        print(f'Running experiment for "{name}"...')

        if name == 'corrupted':
            for calibrator_name, calibrator in calibrators.items():
                print(f'  Using calibrator "{calibrator_name}"...')
                results[calibrator_name] = run_experiment(
                    soft_labels_corrupted,
                    labels,
                    calibrator,
                    config.bootstrap,
                )
        else:
            results[name] = run_experiment(
                soft_labels_corrupted,
                labels,
                None,
                config.bootstrap,
            )

    metadata = get_metadata(datasets)

    ExperimentData(
        config=config,
        results=results,
        metadata=metadata,
    ).save()


if __name__ == '__main__':
    main()
