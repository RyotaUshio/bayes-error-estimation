import json
from pathlib import Path
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from pydantic import Field, TypeAdapter
from pydantic_settings import BaseSettings, CliPositionalArg, SettingsConfigDict

from .utils.experiment_result import ExperimentResult


class PlotOptions(BaseSettings):
    orientation: Annotated[
        Literal['vertical', 'horizontal'],
        Field(
            description='Bar orientation for the plot. Default is vertical bars; use horizontal for barh.'
        ),
    ] = 'vertical'
    omit: list[str] = []
    hline: float | None = None
    figsize: Annotated[
        tuple[float, float] | None,
        Field(
            description='Figure size in inches as WIDTH HEIGHT. Defaults to Matplotlib settings when omitted.'
        ),
    ] = None
    tick_step: Annotated[
        float | None,
        Field(
            description='Major tick spacing for the Estimated Bayes error axis. Defaults to Matplotlib settings when omitted.'
        ),
    ] = None
    ymax: float | None = None


class PlotCliArgs(PlotOptions):
    model_config = SettingsConfigDict(
        cli_parse_args=True, cli_implicit_flags=True
    )
    input: CliPositionalArg[Path]
    show: bool = False


def parse_args():
    return PlotCliArgs()  # type: ignore


def load_results(input_path: Path) -> dict[str, ExperimentResult]:
    with open(input_path, 'r') as f:
        data = json.load(f)
    if 'results' in data:
        data = data['results']

    return TypeAdapter(dict[str, ExperimentResult]).validate_python(data)


def plot(results: dict[str, ExperimentResult], options: PlotOptions):
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

    labels = [
        key for key in order if key in results and key not in options.omit
    ]
    point_estimates = [results[label]['point_estimate'] for label in labels]
    error_low = [
        results[label]['point_estimate']
        - results[label]['confidence_interval']['low']
        for label in labels
    ]
    error_high = [
        results[label]['confidence_interval']['high']
        - results[label]['point_estimate']
        for label in labels
    ]

    errors = [error_low, error_high]

    plt.figure(figsize=options.figsize)
    plt.style.use('ggplot')
    plt.rcParams.update(
        {
            'font.size': 14,
        }
    )

    metric_label = 'Estimated Bayes error (%)'

    if options.orientation == 'vertical':
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

        if options.tick_step:
            plt.gca().yaxis.set_major_locator(
                ticker.MultipleLocator(options.tick_step)
            )

        if options.hline is not None:
            xlim = plt.xlim()
            plt.hlines(
                options.hline,
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

        if options.tick_step:
            plt.gca().xaxis.set_major_locator(
                ticker.MultipleLocator(options.tick_step)
            )

        if options.hline is not None:
            ylim = plt.ylim()
            plt.vlines(
                options.hline,
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

    plt.ylim(bottom=0)
    if options.ymax is not None:
        plt.ylim(top=options.ymax)

    plt.tight_layout()


def main():
    args = parse_args()

    data = load_results(args.input)

    plot(results=data, **args.model_dump())

    outfile = args.input.with_suffix('.pdf')
    print(f'Saving to {outfile}')
    plt.savefig(outfile, bbox_inches='tight')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
