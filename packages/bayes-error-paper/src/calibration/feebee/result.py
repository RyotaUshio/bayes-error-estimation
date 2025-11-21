from typing import TypedDict


class FeeBeeResult(TypedDict):
    estimates: list[float]
    score_lower: float
    score_upper: float
