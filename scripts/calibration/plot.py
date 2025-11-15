from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('-s', '--show', action='store_true')
    parser.add_argument('--hline', type=float, default=None)
    parser.add_argument('--omit', type=str, default='')
    parser.add_argument(
        '--orientation',
        choices=['vertical', 'horizontal'],
        default='vertical',
        help='Bar orientation for the plot. Default is vertical bars; use horizontal for barh.',
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        default=None,
        help='Figure size in inches as WIDTH HEIGHT. Defaults to Matplotlib settings when omitted.',
    )
    parser.add_argument(
        '--tick-step',
        type=float,
        default=None,
        help='Major tick spacing for the Estimated Bayes error axis. Defaults to Matplotlib settings when omitted.',
    )
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
    error_low = [
        data[label]['point_estimate']
        - data[label]['confidence_interval']['low']
        for label in labels
    ]
    error_high = [
        data[label]['confidence_interval']['high']
        - data[label]['point_estimate']
        for label in labels
    ]

    errors = [error_low, error_high]

    fig_kwargs = {'figsize': tuple(args.figsize)} if args.figsize else {}

    plt.figure(**fig_kwargs)
    plt.style.use('ggplot')
    plt.rcParams.update(
        {
            'font.size': 14,
        }
    )

    metric_label = 'Estimated Bayes error (%)'

    if args.orientation == 'vertical':
        x_pos = np.arange(len(labels))
        plt.bar(
            x_pos,
            point_estimates,
            align='center',
            alpha=0.7,
            color='skyblue',
            edgecolor='black',
            linewidth=1,
        )
        plt.errorbar(
            x_pos,
            point_estimates,
            yerr=errors,
            fmt='none',
            ecolor='black',
            capsize=5,
            capthick=1,
            elinewidth=1,
        )

        plt.ylabel(metric_label, fontweight='bold')
        plt.xticks(x_pos, labels, rotation=45, ha='right')

        if args.tick_step:
            plt.gca().yaxis.set_major_locator(
                ticker.MultipleLocator(args.tick_step)
            )

        if args.hline is not None:
            xlim = plt.xlim()
            plt.hlines(
                args.hline,
                xlim[0],
                xlim[1],
                colors='black',
                linestyles='dashed',
                linewidth=0.8,
                alpha=0.8,
            )
            plt.xlim(xlim)
    else:
        y_pos = np.arange(len(labels))
        plt.barh(
            y_pos,
            point_estimates,
            align='center',
            alpha=0.7,
            color='skyblue',
            edgecolor='black',
            linewidth=1,
        )
        plt.errorbar(
            point_estimates,
            y_pos,
            xerr=errors,
            fmt='none',
            ecolor='black',
            capsize=5,
            capthick=1,
            elinewidth=1,
        )

        plt.xlabel(metric_label, fontweight='bold')
        plt.yticks(y_pos, labels)

        if args.tick_step:
            plt.gca().xaxis.set_major_locator(
                ticker.MultipleLocator(args.tick_step)
            )

        if args.hline is not None:
            ylim = plt.ylim()
            plt.vlines(
                args.hline,
                ylim[0],
                ylim[1],
                colors='black',
                linestyles='dashed',
                linewidth=0.8,
                alpha=0.8,
            )
            plt.ylim(ylim)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()

    outfile = args.input.with_suffix('.pdf')
    print(f'Saving to {outfile}')
    plt.savefig(outfile, bbox_inches='tight')

    if args.show:
        plt.show()
