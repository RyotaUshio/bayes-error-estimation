import argparse
import json
from pathlib import Path
from scipy.stats import bootstrap, spearmanr

import bayes_error as be
from data.load import load


def run_experiment(
    soft_labels, labels, calibrator, n_resamples, bootstrap_method
):
    if calibrator is None:
        calibrator = lambda soft_labels, labels: soft_labels

    point_estimate = be.bayes_error(calibrator(soft_labels, labels)) * 100

    confidence_interval = bootstrap(
        (soft_labels, labels),
        lambda soft_labels, labels: be.bayes_error(
            calibrator(soft_labels, labels)
        )
        * 100,
        confidence_level=0.95,
        n_resamples=n_resamples,
        paired=True,
        method=bootstrap_method,
    ).confidence_interval

    return {
        'point_estimate': point_estimate,
        'confidence_interval': {
            'low': confidence_interval.low,
            'high': confidence_interval.high,
        },
        'n_resamples': n_resamples,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--n_resamples', type=int, default=1000)
    parser.add_argument(
        '--bootstrap',
        type=str,
        default='BCa',
        choices=['BCa', 'basic', 'percentile'],
    )
    parser.add_argument(
        '--a', type=float, default=2.0, help='Used only for --dataset synthetic'
    )
    parser.add_argument(
        '--b', type=float, default=0.7, help='Used only for --dataset synthetic'
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
    args = parser.parse_args()

    # load data
    dataset = load(
        args.dataset,
        a=args.a,
        b=args.b,
        shuffle_fraction=args.shuffle_fraction,
        n_hard_labels=args.n_hard_labels,
    )
    soft_labels_corrupted = (
        dataset['soft_labels_corrupted']
        if args.dataset != 'synthetic' or not args.calibrate_hard
        else dataset['soft_labels_corrupted_hard']
    )
    labels = dataset['labels']

    calibrators = {
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
        'beta': lambda soft_labels, labels: be.calibrate_beta(
            soft_labels, labels
        ),
    }

    results = {}
    data = {'results': results, 'metadata': {'args': vars(args)}}

    for name, calibrator in calibrators.items():
        print(f'Running experiment for "{name}"...')
        results[name] = run_experiment(
            soft_labels_corrupted,
            labels,
            calibrator,
            args.n_resamples,
            args.bootstrap,
        )

    if args.dataset == 'synthetic':
        print('Running experiment for "clean"...')
        results['clean'] = run_experiment(
            dataset['soft_labels_clean'],
            labels,
            None,
            args.n_resamples,
            args.bootstrap,
        )

        print('Running experiment for "hard"...')
        results['hard'] = run_experiment(
            dataset['soft_labels_hard'],
            labels,
            None,
            args.n_resamples,
            args.bootstrap,
        )

        spearman_corr = spearmanr(
            dataset['soft_labels_clean'], soft_labels_corrupted
        ).statistic
        data['metadata']['spearman_corr'] = spearman_corr

    outdir = Path('results')
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = (
        f'_{args.a:.1f}_{args.b:.1f}_{args.shuffle_fraction:.1f}_{args.n_hard_labels}{"_hard" if args.calibrate_hard else ""}'
        if args.dataset == 'synthetic'
        else ''
    )
    outfile = (
        outdir
        / f'{args.dataset}_{args.n_resamples}_{args.bootstrap}{suffix}.json'
    )
    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'Results saved to {outfile}')
    print(
        f'To visualize the result, run: \nuv run scripts/calibration/plot.py {outfile}'
    )
