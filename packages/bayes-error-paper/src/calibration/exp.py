from __future__ import annotations

import shutil
from pathlib import Path

import bestperf
import scipy.stats

from ..data.load import load
from .utils.experiment_config import ExperimentConfig
from .utils.experiment_data import (
    ExperimentData,
    ExperimentMetadata,
    ExperimentResult,
)
from .utils.run_experiment import run_experiment

calibrators: dict[str, bestperf.Calibrator | None] = {
    'corrupted': None,
    'isotonic': lambda soft_labels, labels: bestperf.calibrate_isotonic(
        soft_labels, labels
    ),
    'hist10': lambda soft_labels, labels: bestperf.calibrate_histogram_binning(
        soft_labels, labels, 10
    ),
    'hist25': lambda soft_labels, labels: bestperf.calibrate_histogram_binning(
        soft_labels, labels, 25
    ),
    'hist50': lambda soft_labels, labels: bestperf.calibrate_histogram_binning(
        soft_labels, labels, 50
    ),
    'hist100': lambda soft_labels, labels: bestperf.calibrate_histogram_binning(
        soft_labels, labels, 100
    ),
    'beta': lambda soft_labels, labels: bestperf.calibrate_beta(
        soft_labels, labels
    ),
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


def run(config: ExperimentConfig) -> tuple[ExperimentData, Path]:
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

    data = ExperimentData(
        config=config,
        results=results,
        metadata=metadata,
    )
    outfile = data.save()
    return data, outfile


def main():
    config = ExperimentConfig.from_commandline()
    _, outfile = run(config)
    print(f'Results saved to {outfile}')
    print(
        f'To visualize the result, run: \nuv run -m src.calibration.plot {outfile}'
    )


if __name__ == '__main__':
    main()
