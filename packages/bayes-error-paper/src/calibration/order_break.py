import argparse
import json
from pathlib import Path
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np

from .utils.experiment_data import ExperimentData
from .utils.experiment_config import ExperimentConfig

type XType = Literal['spearman', 'kendall', 'sigma']


def get_x(type: XType, file_data: dict) -> float:
    match type:
        case 'spearman':
            return file_data['metadata']['spearman_corr']
        case 'kendall':
            return file_data['metadata']['kendall_corr']
        case 'sigma':
            return file_data['config']['dataset']['perturbation']['sigma']
        case _:
            raise ValueError(f'Unknown x type: {type}')


def get_xlabel(type: XType) -> str:
    match type:
        case 'spearman':
            return "Spearman's rank correlation"
        case 'kendall':
            return "Kendall's rank correlation"
        case 'sigma':
            return 'Perturbation $\\sigma$'
        case _:
            raise ValueError(f'Unknown x type: {type}')


def load_data(args):
    data = {'results': {}}

    order = [
        'clean',
        'hard',
        'corrupted',
        'isotonic',
        'hist10',
        'hist25',
        'hist50',
        'hist100',
        'beta',
        'beta-am',
        'beta-ab',
        'beta-a',
        'platt',
    ]

    for config_file_path in args.input:
        config = ExperimentConfig.from_file(config_file_path)
        results_json_path = ExperimentData.get_outfile_path(config)
        with open(results_json_path, 'r') as f:
            file_data = json.load(f)

        stem = Path(config_file_path).stem
        suffix = '_binom_noise'
        noiseless_results = None
        if stem.endswith(suffix):
            noiseless_config_file_path = Path(config_file_path).with_stem(
                stem[: -len(suffix)]
            )
            noiseless_config = ExperimentConfig.from_file(
                noiseless_config_file_path
            )
            noiseless_results_json_path = ExperimentData.get_outfile_path(
                noiseless_config
            )
            with open(noiseless_results_json_path, 'r') as f:
                noiseless_results = json.load(f)

        x = get_x(args.x, noiseless_results or file_data)

        methods = [key for key in order if key in file_data['results']]

        for method in methods:
            method_data = file_data['results'][method]

            if method not in data['results']:
                data['results'][method] = {
                    'point_estimates': [],
                    'low_ci': [],
                    'high_ci': [],
                    'x': [],
                }

            data['results'][method]['point_estimates'].append(
                method_data['point_estimate']
            )
            data['results'][method]['low_ci'].append(
                method_data['confidence_interval']['low']
            )
            data['results'][method]['high_ci'].append(
                method_data['confidence_interval']['high']
            )
            data['results'][method]['x'].append(x)

    for method in data['results']:
        sort_indices = np.argsort(data['results'][method]['x'])
        data['results'][method]['x'] = np.array(data['results'][method]['x'])[
            sort_indices
        ]
        data['results'][method]['point_estimates'] = np.array(
            data['results'][method]['point_estimates']
        )[sort_indices]
        data['results'][method]['low_ci'] = np.array(
            data['results'][method]['low_ci']
        )[sort_indices]
        data['results'][method]['high_ci'] = np.array(
            data['results'][method]['high_ci']
        )[sort_indices]

    return data


def plot_results(data, args):
    omit = set(['clean', 'hard'] + args.omit.split(','))

    plt.rcParams['font.size'] = 10
    fig, ax = plt.subplots()

    cmap = plt.get_cmap('Set2')
    methods = [key for key in data['results'] if key not in omit]
    methods_calibrated = [method for method in methods if method != 'corrupted']
    colors = [
        'black'
        if method == 'corrupted'
        else cmap((methods_calibrated.index(method) + 1) / (len(methods_calibrated) + 1))
        for method in methods
    ]
    for method, color in zip(methods, colors):
        method_data = data['results'][method]
        zorder = 1 if method == 'isotonic' else 0

        if args.fancy_errorbar:
            ax.plot(
                method_data['x'],
                method_data['point_estimates'],
                marker='o',
                linestyle='-',
                color=color,
                label=method,
                zorder=zorder,
            )
            ax.fill_between(
                method_data['x'],
                method_data['low_ci'],
                method_data['high_ci'],
                color=color,
                alpha=0.2,
                zorder=zorder,
            )
        else:
            ax.errorbar(
                method_data['x'],
                method_data['point_estimates'],
                yerr=[
                    method_data['point_estimates'] - method_data['low_ci'],
                    method_data['high_ci'] - method_data['point_estimates'],
                ],
                fmt='o-',
                color=color,
                label=method,
                capsize=3,
                elinewidth=1,
                markersize=5,
                zorder=zorder,
            )

    clean_value = float(np.mean(data['results']['clean']['point_estimates']))
    ax.axhline(
        y=clean_value,
        color='black',
        linestyle='--',
        linewidth=0.8,
        xmin=0,
        xmax=1,
        zorder=-1,
    )

    ylim = ax.get_ylim()
    ax.set_ylim(0, ylim[1])

    if args.x == 'sigma':
        ax.set_xscale('log')
    ax.set_xlabel(get_xlabel(args.x), fontsize=12)
    ax.set_ylabel('Estimated Bayes error (%)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))

    fig.tight_layout()

    return fig, ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input files', nargs='+')
    parser.add_argument(
        '-o', '--output', type=Path, help='Output file', required=True
    )
    parser.add_argument(
        '-x', choices=['spearman', 'kendall', 'sigma'], default='kendall'
    )
    parser.add_argument('--omit', type=str, default='')
    parser.add_argument('--fancy_errorbar', action='store_true')
    args = parser.parse_args()

    data = load_data(args)
    fig, ax = plot_results(data, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')
    print(f'Saved to {args.output}')
