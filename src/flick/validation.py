"""
Is the fit any good? Held-out scoring and physical-bounds checks.

A fit scored against the rows it was fitted on always looks better than it is,
so the number to trust is cross-validated: fit on k-1 folds, sample, and measure
the energy distance to the held-out fold, next to the distance between two REAL
samples of the same sizes. See docs/evaluation.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from .copula import GaussianCopula
from .metrics import Standardizer, energy_distance

#: Below this many held-out rows an energy distance is more noise than signal,
#: so a fold that small is not worth scoring.
MIN_FOLD_TEST_ROWS = 8


def kfold(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """`k` disjoint sorted index arrays that together cover range(n)."""
    order = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(order, k)]


def cross_validate(data: np.ndarray, columns: Sequence[str], folds: int = 5,
                   seed: int = 0, shrinkage: Optional[float] = None,
                   bounds: Optional[Mapping[str, tuple[float, float]]] = None
                   ) -> list[dict]:
    """Fit on k-1 folds, sample, measure distance to the held-out fold.

    Each result has `fold`, `n_train`, `n_test`, `copula` and `floor`. The floor
    is the energy distance between the TRAIN rows and the TEST rows: two real
    samples of exactly these sizes. A copula that scores at or below it is as
    close to the held-out data as more real data would be, which is the only
    "good enough" that means anything at a few hundred rows.

    Folds too small to score, or that cannot be fitted, are skipped, so the
    result can be shorter than `folds`, or empty."""
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
            cop = GaussianCopula.fit(train, columns, shrinkage=shrinkage,
                                     bounds=bounds,
                                     rng=np.random.default_rng(seed + i))
        except ValueError:
            continue
        drawn = cop.sample(max(test.shape[0], 256),
                           np.random.default_rng(seed + 1000 + i))
        results.append({
            "fold": i + 1,
            "n_train": int(train.shape[0]),
            "n_test": int(test.shape[0]),
            "copula": energy_distance(std.transform(test), std.transform(drawn)),
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
                 bounds: Mapping[str, tuple[float, float]]) -> BoundsReport:
    """Count values outside `bounds`. A column it does not name is unbounded."""
    samples = np.asarray(samples, dtype=float)
    violations: dict[str, int] = {}
    for j, name in enumerate(columns):
        lo, hi = bounds.get(name, (-math.inf, math.inf))
        bad = int(((samples[:, j] < lo) | (samples[:, j] > hi)).sum())
        if bad:
            violations[name] = bad
    return BoundsReport(total_values=int(samples.size), violations=violations,
                        nonfinite=int((~np.isfinite(samples)).sum()))


def gaussian_baseline(data: np.ndarray, n: int,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Samples from a multivariate Gaussian fitted to `data`.

    The model a copula is meant to beat, for use with `check_bounds` and
    `energy_distance`. A singular sample covariance (more columns than rows, or
    a constant column) is sampled through a pseudo-square-root, which a strict
    MVN fit could not do at all, so this flatters the baseline if anything."""
    data = np.asarray(data, dtype=float)
    rng = rng or np.random.default_rng()
    return rng.multivariate_normal(data.mean(axis=0),
                                   np.cov(data, rowvar=False),
                                   size=int(n), check_valid="ignore")
