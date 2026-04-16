from __future__ import annotations

from typing import TypedDict


class ExperimentResult(TypedDict):
    point_estimate: float
    confidence_interval: ConfidenceInterval


class ConfidenceInterval(TypedDict):
    low: float
    high: float
