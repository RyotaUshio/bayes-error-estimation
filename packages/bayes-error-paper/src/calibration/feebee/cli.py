from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, ValidationError
from pydantic_settings import BaseSettings, CliPositionalArg, SettingsConfigDict

from .config import FeeBeeConfig
from .data import FeeBeeData
from .feebee import feebee
from .plot import plot


class CliArgs(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True, cli_implicit_flags=True
    )

    config: CliPositionalArg[Path]
    output: Annotated[
        Path | None, Field(validation_alias=AliasChoices('o'))
    ] = None

    @staticmethod
    def parse() -> CliArgs:
        return CliArgs()  # type: ignore


def main():
    args = CliArgs.parse()
    config = FeeBeeConfig.from_file(args.config)

    try:
        data = FeeBeeData.load(config)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError):
        # need to run the experiment before plotting
        data = feebee(config)

    fig = plot(data)
    outfile = args.output or data.get_outfile_path(data.config).with_suffix('.pdf')
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches='tight')
    print(outfile)


if __name__ == '__main__':
    main()
