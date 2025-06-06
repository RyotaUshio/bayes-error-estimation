from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('-s', '--show', action='store_true')
    parser.add_argument('--hline', type=float, default=None)
    parser.add_argument('--omit', type=str, default='')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)

    if 'results' in data:
        data = data['results']

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
    ]

    omit = args.omit.split(',')
    labels = [key for key in order if key in data and key not in omit]
    point_estimates = [data[label]['point_estimate'] for label in labels]
    error_low = [data[label]['point_estimate'] - data[label]['confidence_interval']['low'] for label in labels]
    error_high = [data[label]['confidence_interval']['high'] - data[label]['point_estimate'] for label in labels]

    errors = [error_low, error_high]

    plt.figure()
    plt.style.use('ggplot')
    plt.rcParams.update({
        'font.size': 14,
    })

    x_pos = np.arange(len(labels))

    bars = plt.bar(x_pos, point_estimates, align='center', alpha=0.7, color='skyblue', edgecolor='black', linewidth=1)
    plt.errorbar(x_pos, point_estimates, yerr=errors, fmt='none', ecolor='black', capsize=5, capthick=1, elinewidth=1)

    plt.ylabel('Estimated Bayes error (%)', fontweight='bold')
    plt.xticks(x_pos, labels, rotation=45, ha='right')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if args.hline is not None:
        xlim = plt.xlim()
        plt.hlines(args.hline, xlim[0], xlim[1], colors='black', linestyles='dashed', linewidth=0.8, alpha=0.8)
        plt.xlim(xlim)

    plt.tight_layout()

    outfile = args.input.with_suffix('.pdf')
    print(f'Saving to {outfile}')
    plt.savefig(outfile, bbox_inches='tight')

    if args.show:
        plt.show()
