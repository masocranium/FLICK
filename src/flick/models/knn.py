"""A nonparametric baseline: resample recorded movements of a similar distance."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np

from ..schema import DISTANCE_COLUMN, FEATURE_BOUNDS
from .base import MovementModel, register_model, require_distance

KNN_FORMAT_VERSION = 1


@register_model
class KNNModel(MovementModel):
    """Resamples recorded rows of a similar distance. A simple baseline.

    Returns one of the `k` recorded rows nearest in log distance, unchanged except
    that `distance_px` becomes the value asked for.

    Useful as a baseline, because it assumes almost nothing, and it keeps dependence
    between features because whole rows stay together. It can only repeat recorded
    values, though, and it does not rescale features to the requested distance.
    Asked for 1000 px when the longest recording is 500 px, it returns 500 px moves
    labelled 1000 px.

    `k` defaults to the square root of the row count."""

    name = "knn"

    def __init__(self, columns: Sequence[str], rows: np.ndarray, k: int):
        require_distance(columns)
        self.columns = list(columns)
        self.rows = np.asarray(rows, dtype=float)
        if self.rows.ndim != 2 or self.rows.shape[0] == 0:
            raise ValueError("KNNModel needs at least one complete row")
        self.k = max(1, min(int(k), self.rows.shape[0]))
        self._i_distance = self.columns.index(DISTANCE_COLUMN)
        self._log_distance = np.log(np.maximum(self.rows[:, self._i_distance], 1e-9))

    @classmethod
    def fit(cls, data, columns: Sequence[str],
            bounds: Optional[Mapping[str, tuple[float, float]]] = FEATURE_BOUNDS,
            rng: Optional[np.random.Generator] = None,
            k: Optional[int] = None) -> "KNNModel":
        """`bounds` and `rng` are unused: nothing here is random or can leave the
        range of the data."""
        require_distance(columns)
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data).all(axis=1)]
        chosen = k if k is not None else int(round(math.sqrt(data.shape[0])))
        return cls(columns, data, chosen)

    def sample(self, distance_px: float, n: int = 1,
               rng: Optional[np.random.Generator] = None) -> dict[str, np.ndarray]:
        rng = rng if rng is not None else np.random.default_rng()
        gaps = np.abs(self._log_distance - math.log(max(float(distance_px), 1e-9)))
        nearest = np.argpartition(gaps, self.k - 1)[:self.k]
        picked = self.rows[rng.choice(nearest, size=int(n))]
        out = {c: picked[:, j].copy() for j, c in enumerate(self.columns)}
        out[DISTANCE_COLUMN] = np.full(int(n), float(distance_px))
        return out

    def to_json(self) -> dict:
        return {"version": KNN_FORMAT_VERSION, "columns": self.columns,
                "k": self.k, "rows": self.rows.tolist()}

    @classmethod
    def from_json(cls, data: dict) -> "KNNModel":
        version = int(data.get("version", 0))
        if version != KNN_FORMAT_VERSION:
            raise ValueError(f"knn format version {version}, this build reads "
                             f"{KNN_FORMAT_VERSION}")
        return cls(data["columns"], np.asarray(data["rows"], dtype=float), data["k"])

    def describe(self) -> list[tuple[str, str]]:
        return [("rows", str(self.rows.shape[0])), ("k", str(self.k))]
