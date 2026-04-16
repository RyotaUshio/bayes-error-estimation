from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from bestperf.feebee import FeeBeeResult
from pydantic import BaseModel, ValidationError

from .config import FeeBeeConfig


class FeeBeeData(BaseModel):
    outdir: ClassVar[Path] = Path('results/feebee')

    config: FeeBeeConfig
    results: dict[str, FeeBeeResult]

    @staticmethod
    def get_outfile_path(config: FeeBeeConfig) -> Path:
        return FeeBeeData.outdir / f'{config.hash()}.json'

    @staticmethod
    def load(config: FeeBeeConfig) -> FeeBeeData | None:
        try:
            outfile = FeeBeeData.get_outfile_path(config)
            with open(outfile, 'r') as f:
                return FeeBeeData.model_validate(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError, ValidationError):
            return None

    @staticmethod
    def for_config(
        config: FeeBeeConfig | str | Path, *, force_rerun: bool = False
    ) -> FeeBeeData:
        config = (
            config
            if isinstance(config, FeeBeeConfig)
            else FeeBeeConfig.from_file(Path(config))
        )

        data = not force_rerun and FeeBeeData.load(config)
        if not data:
            from .feebee import feebee

            data = feebee(config)

        return data

    def save(self) -> Path:
        outfile = FeeBeeData.get_outfile_path(self.config)
        outfile.parent.mkdir(parents=True, exist_ok=True)

        with open(outfile, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)

        return outfile
