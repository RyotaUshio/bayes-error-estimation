from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from .config import FeeBeeConfig
from .result import FeeBeeResult


class FeeBeeData(BaseModel):
    outdir: ClassVar[Path] = Path('results')

    config: FeeBeeConfig
    results: dict[str, FeeBeeResult]

    @staticmethod
    def get_outfile_path(config: FeeBeeConfig) -> Path:
        return FeeBeeData.outdir / f'{config.hash()}.json'

    @staticmethod
    def load(config: FeeBeeConfig) -> FeeBeeData:
        outfile = FeeBeeData.get_outfile_path(config)
        with open(outfile, 'r') as f:
            return FeeBeeData.model_validate(json.load(f))

    def save(self) -> Path:
        outfile = FeeBeeData.get_outfile_path(self.config)
        outfile.parent.mkdir(parents=True, exist_ok=True)

        with open(outfile, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)

        return outfile
