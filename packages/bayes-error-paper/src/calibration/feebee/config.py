from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, field_validator

from ...data.load import DatasetOptions


class FeeBeeConfig(BaseModel):
    dataset: DatasetOptions
    n_points: int
    sota: float

    @field_validator('dataset', mode='before')
    @classmethod
    def validate_dataset(cls, val):
        if not isinstance(val, str):
            return val
        
        # if a string is given, assume it's a path to a dataset config file
        with open(val, 'r') as f:
            data = json.load(f)

        return data


    @staticmethod
    def from_file(filepath: Path) -> FeeBeeConfig:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return FeeBeeConfig.model_validate(data)

    def hash(self) -> str:
        return sha256(self.model_dump_json().encode()).hexdigest()
