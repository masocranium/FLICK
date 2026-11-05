"""The measuring stick for every generated batch."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flick import metrics


def test_energy_distance_is_zero_for_identical_samples():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(120, 4))
    assert metrics.energy_distance(a, a) == pytest.approx(0.0, abs=1e-9)


def test_energy_distance_grows_with_separation():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(150, 3))
    near = rng.normal(size=(150, 3)) + 0.5
    far = rng.normal(size=(150, 3)) + 4.0
    assert metrics.energy_distance(a, near) < metrics.energy_distance(a, far)


def test_energy_distance_is_symmetric_and_accepts_1d():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=200), rng.normal(size=180) + 1.0
    assert metrics.energy_distance(a, b) == pytest.approx(
        metrics.energy_distance(b, a), abs=1e-9)


def test_energy_distance_stays_finite_where_kl_diverges():
    """The reason this module exists rather than a KL estimator: a model that
    puts zero mass on values the real data has gets KL = +inf. This must not."""
    real = np.array([0, 1, 2, 3, 4, 5] * 20, dtype=float)
    degenerate = np.zeros(120)          # no overlap in the upper support
    d = metrics.energy_distance(real, degenerate)
    assert math.isfinite(d) and d > 0


def test_noise_floor_is_small_but_nonzero():
    """Real-vs-itself is the yardstick a candidate is judged against, so it
    must be well under a genuine difference yet not zero."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(200, 3))
    median, p95 = metrics.noise_floor(a, rng=np.random.default_rng(4))
    assert 0.0 < median < p95
    shifted = rng.normal(size=(200, 3)) + 1.0
    assert p95 < metrics.energy_distance(a, shifted)


def test_standardizer_survives_a_constant_column():
    a = np.column_stack([np.ones(50), np.arange(50, dtype=float)])
    std = metrics.Standardizer.fit(a, ("constant", "ramp"))
    assert np.all(std.scale > 0)
    assert np.isfinite(std.transform(a)).all()


def test_standardizer_is_not_fooled_by_a_zero_inflated_column():
    """The bug this guards is silent and total: a column that is ~73% dust
    around zero with a genuine tail has a tiny IQR. Scaling by it multiplies the
    column far past every other one, so it takes all the variance in the
    standardized space and the energy distance measures nothing else, while
    every guard passes because the IQR is not zero."""
    rng = np.random.default_rng(0)
    # 133 dust values around zero and 49 real values. The dust fraction sitting
    # just under 75% is what
    # makes it pathological -- p75 lands right at the dust boundary, so the
    # IQR comes out tiny but NOT zero, sailing past any `scale > 0` guard.
    dust = rng.normal(0.0, 1e-14, 133)
    tail = rng.lognormal(1.5, 1.5, 49)
    zero_inflated = np.concatenate([dust, tail])
    well_behaved = rng.normal(0.5, 0.15, zero_inflated.size)
    data = np.column_stack([zero_inflated, well_behaved])

    q75, q25 = np.percentile(zero_inflated, 75), np.percentile(zero_inflated, 25)
    iqr = q75 - q25
    spread = np.mean(np.abs(zero_inflated - np.median(zero_inflated)))
    assert 0.0 < iqr < 0.2 * spread, "premise: a tiny but nonzero IQR"

    std = metrics.Standardizer.fit(data, ("zero_inflated", "well_behaved"))
    assert std.scale[0] > 0.5 * spread, "scale must reflect the real spread"

    # No column may dominate the Euclidean space the energy distance uses.
    z = std.transform(data)
    share = (z ** 2).sum(axis=0) / (z ** 2).sum()
    assert share.max() < 0.9, f"one column takes {share.max():.1%} of the variance"


def test_feature_gaps_rank_the_worst_feature_first():
    rng = np.random.default_rng(5)
    real = np.column_stack([rng.normal(size=200), rng.normal(size=200)])
    sim = np.column_stack([rng.normal(size=200), rng.normal(size=200) + 5.0])
    gaps = metrics.feature_gaps(real, sim, ("fine", "broken"))
    assert gaps[0].feature == "broken"
    assert gaps[0].scaled_distance > gaps[1].scaled_distance
