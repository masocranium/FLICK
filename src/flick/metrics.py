"""
Two-sample distances: how far is a generated batch from the real one?

Deliberately separate from whatever is being fitted, because the same question
("do these two point clouds come from the same distribution?") is asked of
every candidate model.

WHY NOT KL DIVERGENCE

KL is infinite wherever the model assigns zero probability to something the
real data does. So it is +inf precisely where a model is WORST: it cannot rank
two bad models against each other, and an optimizer cannot descend it.
Estimating a density in 13 dimensions from a few hundred rows is hopeless
independently of that.

ENERGY DISTANCE INSTEAD

    D(X, Y) = 2 E|X - Y| - E|X - X'| - E|Y - Y'|

Zero if and only if the two distributions match, always finite, needs no
density estimate or binning, and behaves at small sample sizes. It is a distance
between two point clouds and nothing more, which is why it survives
non-overlapping support where KL does not.

THE NOISE FLOOR IS PART OF THE MEASUREMENT

An energy distance of 0.4 means nothing on its own. `noise_floor()` splits the
REAL data against itself repeatedly, which gives the distance one expects from
sampling noise alone at this n. Anything at or below that floor is
indistinguishable from a perfect fit given the data available, and tuning past
it is fitting noise. Report the two numbers together, always.

SCALE MATTERS, SO STANDARDIZE FIRST

Energy distance is computed on raw Euclidean distance, so a feature measured in
the hundreds of thousands (`jerk_rms`) would drown one measured in fractions
(`straightness`): the metric would silently become "how well does jerk match".
`Standardizer` rescales every column by the REAL data's spread so each
contributes comparably, using a robust center and scale because heavy-tailed and
zero-inflated columns would drag a mean around.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def _as_2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _pairwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distances between every row of A and every row of B.

    Via |a-b|^2 = |a|^2 + |b|^2 - 2a.b rather than an (n, m, d) broadcast:
    the broadcast allocates n*m*d floats, which for a few thousand rows is
    gigabytes, while this is n*m."""
    aa = (A * A).sum(axis=1)[:, None]
    bb = (B * B).sum(axis=1)[None, :]
    d2 = aa + bb - 2.0 * (A @ B.T)
    # Floating-point error can push a squared distance a hair below zero.
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def energy_distance(x, y) -> float:
    """Energy distance between two samples. 0 iff they match; always finite.

    Accepts 1D (one feature) or 2D (rows = observations, columns =
    features) arrays, which need not have the same number of rows."""
    X, Y = _as_2d(x), _as_2d(y)
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return float("nan")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"feature-count mismatch: {X.shape[1]} vs {Y.shape[1]}")
    return float(
        2.0 * _pairwise(X, Y).mean()
        - _pairwise(X, X).mean()
        - _pairwise(Y, Y).mean()
    )


def noise_floor(x, repeats: int = 25, rng: Optional[np.random.Generator] = None
                ) -> tuple[float, float]:
    """Energy distance of the real data against ITSELF, split in half.

    Returns (median, p95). The p95 is the useful threshold: a candidate
    scoring below it is not distinguishable from the real data at this
    sample size, so further tuning is fitting noise rather than signal.

    Both halves are genuinely from the same distribution by construction, so
    whatever this returns is pure finite-sample noise."""
    X = _as_2d(x)
    n = X.shape[0]
    if n < 8:
        return (float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    half = n // 2
    out = []
    for _ in range(repeats):
        order = rng.permutation(n)
        out.append(energy_distance(X[order[:half]], X[order[half:2 * half]]))
    arr = np.array(out, dtype=float)
    return float(np.median(arr)), float(np.percentile(arr, 95))


def wasserstein1(a, b, quantiles: int = 256) -> float:
    """1D Wasserstein-1 ("earth mover") distance between two samples.

    Used for the PER-FEATURE report rather than the objective: it is in the
    feature's own units, so "straightness is 0.08 apart" is directly
    readable in a way a standardized multivariate number is not. The
    quantile formulation handles samples of different lengths and, unlike
    KL, is finite when the supports do not overlap."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    q = (np.arange(quantiles) + 0.5) / quantiles
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


@dataclass
class Standardizer:
    """Per-column rescaling, fitted on the REAL data and applied to both.

    Fitted on real rather than on the pooled pair deliberately: the question
    is "how far is the candidate from the recording", measured in units of
    the recording's own spread. Rescaling by a pooled spread would let a
    wildly-dispersed candidate shrink its own error term."""

    center: np.ndarray
    scale: np.ndarray
    columns: tuple[str, ...] = ()

    @classmethod
    def fit(cls, x, columns: Sequence[str] = ()) -> "Standardizer":
        """Center on the median, scale by mean absolute deviation from it.

        NOT the IQR, which fails badly on zero-inflated columns. A column that
        is ~73% floating-point dust around zero with a genuine tail out to 150
        had an IQR of 0.04 and an sd of 18.4, a 460x disagreement. Dividing by
        0.04 multiplied that column by ~25x relative to every other one. In one
        182-row table it took up 100.0% of the variance in the standardized
        space, so the energy distance measured that column and nothing else.
        Guarding on `scale > 0` does not catch that, because 0.04 is not zero;
        the scale has to be one that cannot go degenerate in the first place.

        Mean absolute deviation from the median can't: it is only zero for a
        genuinely constant column, it grows with a tail instead of ignoring
        it, and being linear in the outliers it is not dominated by the single
        largest one the way sd is. For a Gaussian it is 0.798σ against the
        IQR's 1.349σ — a constant factor apart, which is all that matters when
        every column is being put on a comparable footing."""
        X = _as_2d(x)
        center = np.nanmedian(X, axis=0)
        scale = np.nanmean(np.abs(X - center), axis=0)
        # A genuinely constant column has no spread by any measure. Fall
        # back to sd, then to 1.0 — a scale of zero would divide the whole
        # comparison by nothing.
        sd = np.nanstd(X, axis=0)
        scale = np.where(scale > 1e-12, scale, sd)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(center=center, scale=scale, columns=tuple(columns))

    def transform(self, x) -> np.ndarray:
        return (_as_2d(x) - self.center) / self.scale


@dataclass
class FeatureGap:
    """One feature's disagreement between real and candidate, in its own units."""

    feature: str
    real_median: float
    sim_median: float
    distance: float          # wasserstein-1, in the feature's own units
    # The same, divided by the real data's spread for that column (see
    # Standardizer). Report BOTH: a scaled number alone hides whether a
    # large ratio is a large gap or a small denominator, which is exactly
    # how a degenerate scale can go unnoticed on a zero-inflated column.
    scaled_distance: float

    @property
    def ratio(self) -> float:
        if abs(self.real_median) < 1e-12:
            return float("nan")
        return self.sim_median / self.real_median


def feature_gaps(real, sim, columns: Sequence[str],
                 standardizer: Optional[Standardizer] = None) -> list[FeatureGap]:
    """Per-feature Wasserstein-1 gaps, worst-scaled-first.

    The ordering is the point: it says which features a candidate is failing
    on, which is what decides whether the next move is more tuning or a
    different generative model."""
    R, S = _as_2d(real), _as_2d(sim)
    std = standardizer or Standardizer.fit(R, columns)
    out: list[FeatureGap] = []
    for i, name in enumerate(columns):
        r, s = R[:, i], S[:, i]
        d = wasserstein1(r, s)
        out.append(FeatureGap(
            feature=name,
            real_median=float(np.nanmedian(r)),
            sim_median=float(np.nanmedian(s)),
            distance=d,
            scaled_distance=d / float(std.scale[i]),
        ))
    out.sort(key=lambda g: -g.scaled_distance)
    return out
