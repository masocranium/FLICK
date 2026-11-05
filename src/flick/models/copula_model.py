"""The default model: a NORTA Gaussian copula, conditioned in closed form."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from ..copula import GaussianCopula
from ..schema import DISTANCE_COLUMN, FEATURE_BOUNDS
from .base import MovementModel, register_model, require_distance


@register_model
class CopulaModel(MovementModel):
    """NORTA Gaussian copula, conditioned on distance in closed form (the default).

    Options for `fit` are those of `GaussianCopula.fit`, such as `shrinkage`,
    `calibrate` and `min_attenuation`. `self.copula` is the fitted model."""

    name = "copula"

    def __init__(self, copula: GaussianCopula):
        require_distance(copula.columns)
        self.copula = copula
        self.columns = list(copula.columns)

    @classmethod
    def fit(cls, data, columns: Sequence[str],
            bounds: Optional[Mapping[str, tuple[float, float]]] = FEATURE_BOUNDS,
            rng: Optional[np.random.Generator] = None, **options) -> "CopulaModel":
        require_distance(columns)
        return cls(GaussianCopula.fit(data, columns, bounds=bounds, rng=rng, **options))

    def sample(self, distance_px: float, n: int = 1,
               rng: Optional[np.random.Generator] = None) -> dict[str, np.ndarray]:
        return self.copula.sample_conditional(
            {DISTANCE_COLUMN: float(distance_px)}, n=n, rng=rng)

    def to_json(self) -> dict:
        return self.copula.to_json()

    @classmethod
    def from_json(cls, data: dict) -> "CopulaModel":
        return cls(GaussianCopula.from_json(data))

    def describe(self) -> list[tuple[str, str]]:
        return self.copula.describe()
