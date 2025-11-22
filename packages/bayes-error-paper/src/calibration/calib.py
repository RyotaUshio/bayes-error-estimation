from pathlib import Path
import subprocess
from typing import Annotated

import matplotlib.pyplot as plt
from pydantic import AliasChoices, Field
from pydantic_settings import SettingsConfigDict, CliPositionalArg

from .exp import run
from .plot import load_results, plot, PlotOptions
from .utils.experiment_config import ExperimentConfig
from .utils.experiment_data import ExperimentData


class CliArgs(PlotOptions):
    model_config = SettingsConfigDict(
        cli_parse_args=True, cli_implicit_flags=True
    )

    config: CliPositionalArg[Path]
    output: Annotated[
        Path | None,
        Field(
            validation_alias=AliasChoices('o'),
            description='plot output path',
        ),
    ] = None
    force: Annotated[
        bool,
        Field(
            validation_alias=AliasChoices('f'),
            description='rerun the experiment even if a result corresponding to the same config is found',
        ),
    ] = False
    open: bool = False


def parse_args():
    return CliArgs()  # type: ignore


def main():
    args = parse_args()
    config = ExperimentConfig.from_file(args.config)

    results_json_path = ExperimentData.get_outfile_path(config)

    results = not args.force and load_results(results_json_path)
    if not results:
        # need to run the experiment before plotting
        data, results_json_path = run(config)
        results = data.results

    print(f'Results: {results_json_path}')
    plot(results, args)
    plot_path = args.output or results_json_path.with_suffix('.pdf')
    print(f'Saving to {plot_path}')
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')

    if args.open:
        subprocess.run(['open', plot_path])


if __name__ == '__main__':
    main()
