"""
The mouse-movement schema, feature-table IO, and a distance-conditioned sampler.

`copula.py` knows nothing about what its columns mean. This module supplies the
meaning for one specific use: tables with one row per pointer movement, as
defined in docs/feature-schema.md.

    cop = fit_mouse_copula(matrix)
    spec = MovementSampler(cop).draw(distance_px=250)
    spec.duration_s, spec.get("straightness")

The output is a FEATURE VECTOR describing one movement, not a trajectory of
(x, y, t) points. Turning a vector into a path is out of scope for this package.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .copula import GaussianCopula

#: The conditioning feature: the one thing known before a movement starts.
DISTANCE_COLUMN = "distance_px"

#: Default columns for a fit.
#:
#: `path_length_px` and `v_mean` are deliberately excluded: they are exact
#: functions of columns already present (path_length = distance / straightness,
#: v_mean = path_length / duration), so including them makes the correlation
#: matrix singular by construction. Recover them from a drawn row if needed.
DEFAULT_COLUMNS: tuple[str, ...] = (
    "distance_px",
    "duration_s",
    "straightness",
    "max_dev_px",
    "max_dev_frac",
    "v_peak",
    "t_peak_frac",
    "accel_peak",
    "decel_peak",
    "jerk_rms",
    "n_submovements",
    "overshoot_px",
    "settle_s",
    "n_direction_changes",
)

#: Physical bounds per feature: facts about what the number MEANS, not
#: properties of any sample.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "distance_px": (0.0, math.inf),
    "duration_s": (0.0, math.inf),
    "path_length_px": (0.0, math.inf),
    "straightness": (0.0, 1.0),
    "max_dev_px": (-math.inf, math.inf),
    "max_dev_frac": (0.0, 1.0),
    "v_peak": (0.0, math.inf),
    "v_mean": (0.0, math.inf),
    "t_peak_frac": (0.0, 1.0),
    "accel_peak": (0.0, math.inf),
    "decel_peak": (-math.inf, 0.0),
    "jerk_rms": (0.0, math.inf),
    "n_submovements": (0.0, math.inf),
    "overshoot_px": (0.0, math.inf),
    "settle_s": (0.0, math.inf),
    "n_direction_changes": (0.0, math.inf),
    "pause_before_s": (0.0, math.inf),
    "dwell_s": (0.0, math.inf),
}

#: Guard rails on a drawn duration, in seconds. A copula extrapolates past the
#: observed range by design, which is right for a tail and wrong for a value a
#: caller is about to use as a real time span. Wide enough never to bind on a
#: plausible draw.
MIN_DURATION_S = 0.02
MAX_DURATION_S = 8.0

#: The one feature-file version this package reads.
FEATURES_FORMAT_VERSION = 1


def fit_mouse_copula(data, columns: Sequence[str] = DEFAULT_COLUMNS,
                     **kwargs) -> GaussianCopula:
    """`GaussianCopula.fit` with this schema's physical bounds applied.

    Prefer this to calling `fit` directly on mouse features: without the bounds
    every column is treated as unbounded, and sampled values can be impossible."""
    return GaussianCopula.fit(data, columns, bounds=FEATURE_BOUNDS, **kwargs)


# --------------------------------------------------------------------------- #
# Feature tables
# --------------------------------------------------------------------------- #

def read_rows(path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a feature table. Returns (rows, header).

    `.csv` is read with its first line as column names, and has no header
    record. Anything else is read as JSONL: one JSON object per line, with an
    optional `{"kind_record": "header", "version": 1, ...}` record."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh)), {}

    rows: list[dict[str, Any]] = []
    header: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind_record") == "header":
                version = int(rec.get("version", 0))
                if version != FEATURES_FORMAT_VERSION:
                    raise ValueError(
                        f"{path}: features format version {version}, this build "
                        f"reads {FEATURES_FORMAT_VERSION}")
                header = rec
                continue
            rec.pop("kind_record", None)
            rows.append(rec)
    return rows, header


def select_rows(rows: Iterable[dict], activity: Optional[str] = None,
                kind: Optional[str] = None) -> list[dict]:
    """Rows matching a stratum. `None` means do not filter on that field."""
    return [r for r in rows
            if (activity is None or r.get("activity") == activity)
            and (kind is None or r.get("kind") == kind)]


def _cell(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def to_matrix(rows: Iterable[dict],
              columns: Sequence[str] = DEFAULT_COLUMNS) -> np.ndarray:
    """Rows -> a finite-only 2D float array over `columns`.

    A row with a missing or non-finite value in any selected column is dropped,
    so select columns deliberately: asking for one that only some movements have
    silently discards all the others."""
    cols = list(columns)
    data = np.array([[_cell(r.get(c)) for c in cols] for r in rows], dtype=float)
    if data.size == 0:
        return data.reshape(0, len(cols))
    return data[np.isfinite(data).all(axis=1)]


# --------------------------------------------------------------------------- #
# Sampling one coherent movement
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MovementSpec:
    """One movement's features, drawn jointly rather than field by field."""

    distance_px: float
    duration_s: float
    #: Everything the copula drew, by column name, unclamped.
    features: dict

    def get(self, column: str, default: float = float("nan")) -> float:
        return float(self.features.get(column, default))


class MovementSampler:
    """Draws movement feature vectors from a fitted copula, given a distance."""

    def __init__(self, copula: GaussianCopula,
                 rng: Optional[np.random.Generator] = None):
        if DISTANCE_COLUMN not in copula.columns:
            raise ValueError(
                f"copula has no {DISTANCE_COLUMN!r} column, so there is nothing "
                "to condition on. Refit including it.")
        self.copula = copula
        self._rng = rng if rng is not None else np.random.default_rng()

    @classmethod
    def from_file(cls, path, rng: Optional[np.random.Generator] = None
                  ) -> "MovementSampler":
        """Load a saved copula. Raises if the file is missing or corrupt."""
        return cls(GaussianCopula.load(path), rng=rng)

    def draw(self, distance_px: float) -> Optional[MovementSpec]:
        """One movement consistent with travelling `distance_px`.

        None for a non-finite or non-positive distance, which has no movement to
        describe, or if the model has no usable `duration_s`. `duration_s` is
        clamped to [MIN_DURATION_S, MAX_DURATION_S]."""
        distance = float(distance_px)
        if not np.isfinite(distance) or distance <= 0.0:
            return None

        drawn = self.copula.sample_conditional(
            {DISTANCE_COLUMN: distance}, n=1, rng=self._rng)
        features = {k: float(np.asarray(v).reshape(-1)[0])
                    for k, v in drawn.items()}

        duration = features.get("duration_s", float("nan"))
        if not np.isfinite(duration):
            return None
        duration = min(max(duration, MIN_DURATION_S), MAX_DURATION_S)
        return MovementSpec(distance_px=distance, duration_s=duration,
                            features=features)
