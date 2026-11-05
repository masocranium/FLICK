"""
Is a model any good? Held-out scoring and physical-bounds checks, for any model.

A fit scored against the rows it was fitted on always looks better than it is, so
the number to trust is cross-validated: fit on k-1 folds, generate, and measure the
energy distance to the held-out fold, next to the distance between two REAL samples.
See docs/evaluation.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from .metrics import Standardizer, energy_distance
from .models.base import MovementModel, get_model
from .schema import DISTANCE_COLUMN, FEATURE_BOUNDS

#: Below this many held-out rows an energy distance is more noise than signal.
MIN_FOLD_TEST_ROWS = 8

#: Generated rows per fold, at minimum.
MIN_FOLD_DRAWS = 256


def kfold(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """`k` disjoint sorted index arrays that together cover range(n)."""
    order = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(order, k)]


def sample_like(model: MovementModel, train: np.ndarray, columns: Sequence[str],
                n: int, rng: np.random.Generator) -> np.ndarray:
    """Generated rows whose distances are drawn from the training rows.

    This is how a model that only predicts features GIVEN a distance gets compared
    to real data: the distances come from what it was trained on, and the rest from
    the model. Returned columns are in `columns` order."""
    i = list(columns).index(DISTANCE_COLUMN)
    distances = rng.choice(np.asarray(train, dtype=float)[:, i], size=int(n))
    rows = model.sample_rows(distances, rng)
    order = [model.columns.index(c) for c in columns]
    return rows[:, order]


def cross_validate(data: np.ndarray, columns: Sequence[str],
                   model: Union[str, type[MovementModel]] = "copula",
                   folds: int = 5, seed: int = 0,
                   bounds: Optional[Mapping[str, tuple[float, float]]] = FEATURE_BOUNDS,
                   **options) -> list[dict]:
    """Fit on k-1 folds, generate, measure distance to the held-out fold.

    Each result has `fold`, `n_train`, `n_test`, `score` and `floor`. The floor is
    the energy distance between the TRAIN rows and the TEST rows: two real samples.
    A model at or below it is as close to held-out data as more real data would be.

    Folds too small to score, or that cannot be fitted, are skipped, so the result
    can be shorter than `folds`, or empty."""
    cls = get_model(model) if isinstance(model, str) else model
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    results = []
    for i, test_idx in enumerate(kfold(data.shape[0], folds, rng)):
        train_idx = np.setdiff1d(np.arange(data.shape[0]), test_idx)
        if test_idx.size < MIN_FOLD_TEST_ROWS or train_idx.size < 3:
            continue
        train, test = data[train_idx], data[test_idx]

        # Standardize on TRAIN only: using the test fold's spread to scale the
        # metric leaks the thing being measured.
        std = Standardizer.fit(train, list(columns))
        try:
            fitted = cls.fit(train, columns, bounds=bounds,
                             rng=np.random.default_rng(seed + i), **options)
        except ValueError:
            continue
        drawn = sample_like(fitted, train, columns, max(test.shape[0], MIN_FOLD_DRAWS),
                            np.random.default_rng(seed + 1000 + i))
        results.append({
            "fold": i + 1,
            "n_train": int(train.shape[0]),
            "n_test": int(test.shape[0]),
            "score": energy_distance(std.transform(test), std.transform(drawn)),
            "floor": energy_distance(std.transform(test), std.transform(train)),
        })
    return results


@dataclass(frozen=True)
class BoundsReport:
    """Values outside their declared bounds in a batch of samples."""

    total_values: int
    #: Column name -> count, only for columns with at least one violation.
    violations: dict
    nonfinite: int

    @property
    def n_impossible(self) -> int:
        return sum(self.violations.values())

    @property
    def rate(self) -> float:
        return self.n_impossible / self.total_values if self.total_values else 0.0


def check_bounds(samples: np.ndarray, columns: Sequence[str],
                 bounds: Mapping[str, tuple[float, float]] = FEATURE_BOUNDS,
                 tol: float = 1e-9) -> BoundsReport:
    """Count values outside `bounds` by more than `tol`. A column it does not name is
    unbounded. The tolerance is for floating-point dust such as -1e-14, which derived
    columns hold in real recordings and which isn't a physically impossible value."""
    samples = np.asarray(samples, dtype=float)
    violations: dict[str, int] = {}
    for j, name in enumerate(columns):
        lo, hi = bounds.get(name, (-math.inf, math.inf))
        bad = int(((samples[:, j] < lo - tol) | (samples[:, j] > hi + tol)).sum())
        if bad:
            violations[name] = bad
    return BoundsReport(total_values=int(samples.size), violations=violations,
                        nonfinite=int((~np.isfinite(samples)).sum()))


def gaussian_baseline(data: np.ndarray, n: int,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Samples from a multivariate Gaussian fitted to `data`, as a comparison.

    A singular sample covariance is sampled through a pseudo-square-root, which a
    strict fit could not do at all, so this flatters the baseline if anything."""
    data = np.asarray(data, dtype=float)
    rng = rng or np.random.default_rng()
    return rng.multivariate_normal(data.mean(axis=0), np.cov(data, rowvar=False),
                                   size=int(n), check_valid="ignore")
