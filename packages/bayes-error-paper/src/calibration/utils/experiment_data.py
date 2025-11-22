import json
from pathlib import Path
from typing import ClassVar, TypedDict

from pydantic import BaseModel

from .experiment_config import ExperimentConfig
from .experiment_result import ExperimentResult


class ExperimentMetadata(TypedDict):
    spearman_corr: float | None
    kendall_corr: float | None


class ExperimentData(BaseModel):
    outdir: ClassVar[Path] = Path('results/calib')

    config: ExperimentConfig
    results: dict[str, ExperimentResult]
    metadata: ExperimentMetadata

    @staticmethod
    def get_outfile_path(config: ExperimentConfig) -> Path:
        return ExperimentData.outdir / f'{config.hash()}.json'

    def save(self) -> Path:
        # dataset = self.config.dataset
        # bootstrap = self.config.bootstrap

        # suffix = (
        #     f'_{dataset.a:.1f}_{dataset.b:.1f}_{dataset.shuffle_fraction:.1f}_{dataset.n_hard_labels}{"_hard" if dataset.binom_noise else ""}'
        #     if dataset.dataset == 'synthetic'
        #     else ''
        # )
        # outfile = (
        #     outdir
        #     / f'{dataset.dataset}_{bootstrap["n_resamples"]}_{bootstrap["method"]}{suffix}.json'
        # )

        outfile = ExperimentData.get_outfile_path(self.config)
        outfile.parent.mkdir(parents=True, exist_ok=True)

        with open(outfile, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)

        return outfile
