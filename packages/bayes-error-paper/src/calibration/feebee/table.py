from __future__ import annotations

from pathlib import Path
from typing import Annotated
import re

import pandas as pd
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, CliPositionalArg, SettingsConfigDict

from .data import FeeBeeData


class CliArgs(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True, cli_implicit_flags=True
    )

    configs: CliPositionalArg[list[Path]]
    output: Annotated[
        Path | None,
        Field(
            validation_alias=AliasChoices('o'),
            description='table .tex output path',
        ),
    ] = None

    @field_validator('output')
    @classmethod
    def validate_output(cls, val: Path | None) -> Path | None:
        if isinstance(val, Path) and val.suffix != '.tex':
            raise ValueError('Output file must have .tex extension')
        return val

    @staticmethod
    def parse() -> CliArgs:
        return CliArgs()  # type: ignore


def dataset_display_name(dataset_name: str) -> str:
    match dataset_name:
        case 'cifar10':
            return 'CIFAR-10'
        case 'fashion_mnist':
            return 'Fashion-MNIST'
        case 'mnli':
            return 'MNLI'
        case 'snli':
            return 'SNLI'
        case 'abduptive_nli':
            return 'AbductiveNLI'
        case 'iclr':
            return 'ICLR2017-2025'
        case val if val.startswith('iclr_'):
            return val.replace('_', '').upper()
        case _:
            return dataset_name


def tabular_to_tabularstar(tabular: str) -> str:
    """Convert a LaTeX tabular to tabular* that spans the full line width."""
    return re.sub(
        r'\\end\{tabular\}$',
        r'\\end{tabular*}',
        re.sub(
            r'^\\begin\{tabular\}\{(.*)\}(\s)',
            r'\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}\1}\2',
            tabular,
        ),
    )


def add_index_name(tabular: str, name: str) -> str:
    def replace(m: re.Match) -> str:
        header: str = m.groups()[0]
        lines = header.splitlines()
        lines[1] = lines[1] + ' \\cmidrule{3-7}'
        lines[2] = re.sub(
            r'^ &  & ', r'\\multicolumn{2}{c}{' + name + '} & ', lines[2]
        )
        return '\\toprule' + '\n'.join(lines) + '\n\\midrule'

    return re.sub(
        r'\\toprule([\s\S]*)\\midrule',
        replace,
        tabular,
    )


def generate_table(config_paths: list[Path]) -> str:
    data = {
        config_path.stem: {
            name: result['score_lower'] + result['score_upper']
            for name, result in FeeBeeData.for_config(
                config_path
            ).results.items()
            if name != 'corrupted'
        }
        for config_path in config_paths
    }

    raw_df = pd.DataFrame.from_dict(data)

    hist_index = raw_df.index.str.startswith('hist')

    isotonic = raw_df.loc[['isotonic']].copy()
    isotonic.index = pd.MultiIndex.from_tuples([('isotonic', '')])

    hist = pd.DataFrame(
        [raw_df.loc[hist_index].min(), raw_df.loc[hist_index].max()],
        index=pd.MultiIndex.from_tuples([('hist', 'min.'), ('hist', 'max.')]),
    )

    rest = pd.DataFrame(
        raw_df.loc[~hist_index & (raw_df.index != 'isotonic')],
    )
    rest.index = pd.MultiIndex.from_product([rest.index, ['']])

    df = pd.concat([isotonic, hist, rest])
    df.columns = pd.MultiIndex.from_product([['Dataset'], df.columns])

    dataset_groups = [
        ('cifar10', 'fashion_mnist', 'snli', 'mnli', 'abduptive_nli'),
        ('iclr', 'iclr_2017', 'iclr_2018', 'iclr_2019', 'iclr_2020'),
        ('iclr_2021', 'iclr_2022', 'iclr_2023', 'iclr_2024', 'iclr_2025'),
    ]
    latex = ''
    for dataset_group in dataset_groups:
        if latex:
            latex += '\n\\medskip\n'
        latex += '\\begin{subtable}{\\linewidth}\n\\centering\n'
        tabular = (
            df.loc[:, ('Dataset', dataset_group)]
            .style.format(precision=4)
            .format_index(
                lambda v: '{\\small \\texttt{' + v + '}}', axis=0, level=0
            )
            .format_index(dataset_display_name, axis=1)
            .highlight_min(props='bfseries: ;')
            .to_latex(
                hrules=True,
                column_format=f'll *{len(dataset_group)}{{c}}',
                multicol_align='c',
            )
        )
        tabular = tabular_to_tabularstar(tabular)
        tabular = add_index_name(tabular, 'Calibration algorithm')
        latex += tabular
        latex += '\\end{subtable}\n'

    return latex


def main():
    args = CliArgs.parse()

    latex = generate_table(args.configs)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(latex)
    else:
        print(latex)


if __name__ == '__main__':
    main()
