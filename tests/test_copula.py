"""
Copula tests.

The headline property, and the reason this model was chosen over a multivariate
Gaussian, is that a sampled row is always PHYSICALLY POSSIBLE: no negative
durations, no fractions above 1.0, no fractional counts.

The fixtures are drawn from a KNOWN Gaussian copula with known marginals, so
recovery can be asserted against ground truth rather than against "it ran".
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from norta_copula import copula as cop
from norta_copula.copula import (
    MIN_ATOM_COUNT,
    ConstantMarginal,
    DiscreteMarginal,
    HurdleMarginal,
    KDEMarginal,
    _norm_cdf,
    _norm_ppf,
    choose_marginal as _choose_marginal,
    ledoit_wolf_intensity,
    marginal_from_json,
    nearest_psd,
    shrink_correlation,
)
from norta_copula.copula import GaussianCopula as _GaussianCopula
from norta_copula.mouse import FEATURE_BOUNDS


def choose_marginal(v, name="", **kw):
    """choose_marginal with the mouse schema's bounds, looked up by name."""
    kw.setdefault("bounds", FEATURE_BOUNDS.get(name))
    return _choose_marginal(v, name=name, **kw)


class GaussianCopula(_GaussianCopula):
    """GaussianCopula.fit with the mouse schema's bounds applied."""

    @classmethod
    def fit(cls, data, columns, **kw):
        kw.setdefault("bounds", FEATURE_BOUNDS)
        return _GaussianCopula.fit(data, columns, **kw)


# --------------------------------------------------------------------------- #
# The two special functions (no scipy)
# --------------------------------------------------------------------------- #

def test_norm_ppf_matches_known_quantiles():
    got = _norm_ppf(np.array([0.025, 0.5, 0.975, 0.001, 0.999]))
    want = np.array([-1.959964, 0.0, 1.959964, -3.090232, 3.090232])
    assert np.allclose(got, want, atol=1e-5)


def test_norm_cdf_and_ppf_are_inverses():
    u = np.linspace(1e-6, 1 - 1e-6, 5000)
    assert np.allclose(_norm_cdf(_norm_ppf(u)), u, atol=1e-7)


def test_norm_ppf_clips_rather_than_returning_infinity():
    """One infinity anywhere in the latent matrix makes the whole correlation
    estimate NaN, so the boundary must be clipped, not propagated."""
    out = _norm_ppf(np.array([0.0, 1.0]))
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Marginal selection
# --------------------------------------------------------------------------- #

def test_constant_column_is_recognized():
    m = choose_marginal(np.zeros(50), name="n_direction_changes")
    assert isinstance(m, ConstantMarginal)
    assert np.all(m.ppf(np.random.random(20)) == 0.0)


def test_integer_counts_become_a_discrete_marginal():
    rng = np.random.default_rng(0)
    v = rng.choice([0, 1, 2, 3], size=400, p=[0.5, 0.3, 0.15, 0.05]).astype(float)
    m = choose_marginal(v, name="n_submovements")
    assert isinstance(m, DiscreteMarginal)

    drawn = m.ppf(rng.random(20000))
    assert set(np.unique(drawn)) <= {0.0, 1.0, 2.0, 3.0}
    # Against the EMPIRICAL frequency, not the generating p: an empirical PMF
    # is supposed to reproduce the sample it was given, and this 400-row draw
    # sits at 0.4575 rather than the 0.5 it was generated from.
    assert np.mean(drawn == 0.0) == pytest.approx(np.mean(v == 0.0), abs=0.02)


def test_positive_skewed_column_uses_a_log_kde():
    rng = np.random.default_rng(1)
    v = rng.lognormal(-0.7, 0.5, 300)
    m = choose_marginal(v, name="duration_s")
    assert isinstance(m, KDEMarginal) and m.transform == "log"
    assert (m.ppf(rng.random(20000)) > 0).all(), "a duration can never be <= 0"


def test_bounded_column_uses_a_logit_kde_and_never_escapes():
    rng = np.random.default_rng(2)
    v = np.clip(rng.beta(9, 1.2, 300), 1e-6, 1 - 1e-6)
    m = choose_marginal(v, name="straightness")
    assert isinstance(m, KDEMarginal) and m.transform == "logit"

    drawn = m.ppf(rng.random(50000))
    assert drawn.min() >= 0.0 and drawn.max() <= 1.0


def test_all_negative_column_uses_neglog():
    rng = np.random.default_rng(3)
    m = choose_marginal(-rng.lognormal(9.0, 0.6, 300), name="decel_peak")
    assert isinstance(m, KDEMarginal) and m.transform == "neglog"
    assert (m.ppf(rng.random(10000)) <= 0).all()


def test_signed_unbounded_column_stays_in_identity_space():
    rng = np.random.default_rng(4)
    m = choose_marginal(rng.normal(0, 40, 300), name="max_dev_px")
    assert isinstance(m, KDEMarginal) and m.transform == "identity"


# --------------------------------------------------------------------------- #
# Zero inflation — the shape that broke the fitting metric once already
# --------------------------------------------------------------------------- #

def _zero_inflated(rng, n=400, share=0.6):
    """Mostly floating-point dust at zero, with a genuine continuous tail. The
    dust is deliberately NOT all the same float (0.0, -0.0 and 7e-15 all occur
    in derived columns), so exact-equality detection cannot find it."""
    v = rng.lognormal(1.2, 1.0, n)
    mask = rng.random(n) < share
    dust = rng.normal(0.0, 1e-14, n)
    return np.where(mask, dust, v)


def test_zero_inflated_column_becomes_a_hurdle():
    rng = np.random.default_rng(5)
    v = _zero_inflated(rng, share=0.6)
    m = choose_marginal(v, name="overshoot_px")
    assert isinstance(m, HurdleMarginal)
    assert m.atom == 0.0
    assert m.atom_prob == pytest.approx(0.6, abs=0.06)


def test_hurdle_reproduces_the_atom_rate_and_the_tail():
    rng = np.random.default_rng(6)
    v = _zero_inflated(rng, n=600, share=0.73)
    m = choose_marginal(v, name="overshoot_px")

    drawn = m.ppf(rng.random(40000))
    assert np.mean(drawn == 0.0) == pytest.approx(0.73, abs=0.03)
    tail = drawn[drawn > 0]
    assert tail.size > 0
    assert np.median(tail) == pytest.approx(np.median(v[v > 1e-9]), rel=0.4)


def test_a_value_seen_once_is_not_an_atom():
    """Regression. At n=17 a single observation is 5.9% of the sample, so an
    `atom_frac` of 0.05 alone declared the modal value of every continuous
    column a point mass — measured: 13 of 14 columns came back as spurious
    hurdles at p=1/17, each stealing a real observation out of its own KDE.
    An atom has to REPEAT."""
    rng = np.random.default_rng(7)
    v = rng.lognormal(0.0, 0.7, 17)                 # all distinct, no atom
    assert np.unique(v).size == 17, "premise: every value is distinct"

    m = choose_marginal(v, name="duration_s")
    assert not isinstance(m, HurdleMarginal)
    assert isinstance(m, KDEMarginal) and m.samples.size == 17


def test_repeated_value_below_the_count_floor_is_not_an_atom():
    v = np.concatenate([[2.0, 2.0], np.linspace(3.0, 9.0, 30)])
    assert MIN_ATOM_COUNT == 3
    assert not isinstance(choose_marginal(v, name="duration_s"), HurdleMarginal)


# --------------------------------------------------------------------------- #
# KDE mechanics
# --------------------------------------------------------------------------- #

def test_kde_cdf_and_ppf_are_mutual_inverses():
    """They interpolate one shared grid precisely so this holds. A copula
    whose forward and inverse marginal transforms disagree accumulates that
    error straight into the correlation estimate."""
    rng = np.random.default_rng(8)
    m = choose_marginal(rng.lognormal(0, 0.5, 200), name="duration_s")
    u = np.linspace(0.01, 0.99, 500)
    assert np.allclose(m.cdf(m.ppf(u)), u, atol=1e-3)


def test_kde_extrapolates_past_the_observed_range():
    """The reason a KDE is used rather than a bare ECDF: inverting an
    empirical CDF can only ever return values already observed, so the tail —
    the interesting part — would be hard-truncated at the sample maximum."""
    rng = np.random.default_rng(9)
    v = rng.lognormal(0, 0.6, 120)
    m = choose_marginal(v, name="pause_before_s")
    assert m.ppf(np.array([0.999]))[0] > v.max()


# --------------------------------------------------------------------------- #
# Correlation handling
# --------------------------------------------------------------------------- #

def test_fallback_shrinkage_scales_with_how_little_data_there_is():
    """The p/(p+n) fallback, used only when there is no latent matrix to
    estimate Ledoit-Wolf from."""
    R = np.eye(14)
    _, lam_small = shrink_correlation(R, n=20)
    _, lam_large = shrink_correlation(R, n=2000)
    assert lam_small > 0.35 and lam_large < 0.02


def _latent(n, p, rho, seed):
    R = np.full((p, p), rho)
    np.fill_diagonal(R, 1.0)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)) @ np.linalg.cholesky(R).T


def test_ledoit_wolf_shrinks_completely_when_there_is_nothing_to_preserve():
    """Independent columns: the identity IS the truth, so the optimal weight
    on it is ~1. This is the case a fixed heuristic gets most wrong — it
    would keep most of a correlation matrix that is entirely noise."""
    assert ledoit_wolf_intensity(_latent(50, 14, 0.0, 0)) > 0.9


def test_ledoit_wolf_barely_shrinks_real_structure():
    assert ledoit_wolf_intensity(_latent(182, 14, 0.7, 1)) < 0.1


def test_ledoit_wolf_separates_signal_from_noise_at_the_same_n():
    """The whole point: p/(p+n) returns one number regardless of what the
    data says, so it cannot tell these two apart."""
    noise = ledoit_wolf_intensity(_latent(100, 14, 0.0, 2))
    signal = ledoit_wolf_intensity(_latent(100, 14, 0.6, 2))
    assert noise > signal * 5


def test_ledoit_wolf_decreases_as_rows_accumulate():
    vals = [ledoit_wolf_intensity(_latent(n, 14, 0.5, 3))
            for n in (25, 100, 400, 2000)]
    assert all(a > b for a, b in zip(vals, vals[1:])), vals


def test_ledoit_wolf_stays_a_weight_in_the_unit_interval():
    for n, p, rho, seed in [(5, 3, 0.9, 4), (2000, 4, 0.0, 5), (20, 19, 0.3, 6)]:
        lam = ledoit_wolf_intensity(_latent(n, p, rho, seed))
        assert 0.0 <= lam <= 1.0


def test_ledoit_wolf_handles_a_degenerate_column():
    Z = np.column_stack([_latent(60, 2, 0.4, 7), np.zeros(60)])
    assert ledoit_wolf_intensity(Z) == 1.0


def test_shrink_correlation_prefers_ledoit_wolf_when_given_the_latent():
    Z = _latent(100, 6, 0.0, 8)
    R = np.corrcoef(Z, rowvar=False)
    _, lam_lw = shrink_correlation(R, n=100, latent=Z)
    _, lam_fallback = shrink_correlation(R, n=100)
    assert lam_lw > lam_fallback, "pure noise should shrink harder than 6/106"


def test_fit_uses_ledoit_wolf_by_default_and_honours_an_override():
    rng = np.random.default_rng(20)
    data = _draw_known(400, 0.6, rng)
    auto = GaussianCopula.fit(data, TEST_COLUMNS, rng=np.random.default_rng(0))
    forced = GaussianCopula.fit(data, TEST_COLUMNS, shrinkage=0.5,
                                rng=np.random.default_rng(0))
    assert forced.shrinkage == 0.5
    assert auto.shrinkage != 0.5
    # Real structure at 400 rows should be largely preserved.
    assert auto.shrinkage < 0.15


def test_nearest_psd_rescues_a_singular_matrix():
    R = np.array([[1.0, 1.0], [1.0, 1.0]])          # perfectly collinear
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(R)
    fixed = nearest_psd(R)
    np.linalg.cholesky(fixed)                        # must not raise
    assert np.allclose(np.diag(fixed), 1.0)


# --------------------------------------------------------------------------- #
# End-to-end fitting against a known ground truth
# --------------------------------------------------------------------------- #

TEST_COLUMNS = ("distance_px", "duration_s", "straightness", "n_submovements")


def _draw_known(n, rho, rng):
    """Sample from a copula with a KNOWN correlation and known marginals, so
    the fit can be checked against ground truth rather than against itself."""
    R = np.full((4, 4), rho)
    np.fill_diagonal(R, 1.0)
    Z = rng.standard_normal((n, 4)) @ np.linalg.cholesky(R).T
    U = _norm_cdf(Z)
    return np.column_stack([
        20.0 + 600.0 * U[:, 0],                       # distance_px
        np.exp(-1.0 + 0.5 * _norm_ppf(U[:, 1])),      # duration_s, lognormal
        np.clip(U[:, 2] ** 0.25, 1e-6, 1 - 1e-6),     # straightness, in (0,1)
        np.floor(U[:, 3] * 5.0),                      # n_submovements, discrete
    ])


def test_fit_recovers_a_known_correlation():
    rng = np.random.default_rng(10)
    data = _draw_known(4000, 0.6, rng)
    fitted = GaussianCopula.fit(data, TEST_COLUMNS, shrinkage=0.0,
                             rng=np.random.default_rng(0))
    off = fitted.correlation[np.triu_indices(4, k=1)]
    assert np.allclose(off, 0.6, atol=0.1), off


def test_sampled_values_are_always_physically_possible():
    """The property that rules out a multivariate Gaussian: an MVN emits
    negative durations and straightness above 1.0. This must be exactly zero."""
    rng = np.random.default_rng(11)
    data = _draw_known(600, 0.5, rng)
    fitted = GaussianCopula.fit(data, TEST_COLUMNS, rng=np.random.default_rng(0))
    drawn = fitted.sample(30000, np.random.default_rng(1))

    assert np.isfinite(drawn).all()
    for j, name in enumerate(TEST_COLUMNS):
        lo, hi = FEATURE_BOUNDS[name]
        assert drawn[:, j].min() >= lo, f"{name} below {lo}"
        assert drawn[:, j].max() <= hi, f"{name} above {hi}"

    counts = drawn[:, TEST_COLUMNS.index("n_submovements")]
    assert np.allclose(counts, np.round(counts)), "counts must stay integers"


def test_marginals_are_recovered():
    rng = np.random.default_rng(12)
    data = _draw_known(3000, 0.4, rng)
    fitted = GaussianCopula.fit(data, TEST_COLUMNS, rng=np.random.default_rng(0))
    drawn = fitted.sample(30000, np.random.default_rng(2))
    for j, name in enumerate(TEST_COLUMNS):
        assert np.median(drawn[:, j]) == pytest.approx(
            np.median(data[:, j]), rel=0.2), name


def test_constant_column_does_not_poison_the_fit():
    """A column that is always 0 is a finding, not missing data, and a naive
    correlation over it is all NaN."""
    rng = np.random.default_rng(13)
    data = np.column_stack([_draw_known(300, 0.5, rng),
                            np.zeros(300)])
    cols = (*TEST_COLUMNS, "n_direction_changes")
    fitted = GaussianCopula.fit(data, cols, rng=np.random.default_rng(0))

    assert isinstance(fitted.marginals[-1], ConstantMarginal)
    assert np.isfinite(fitted.correlation).all()
    assert np.allclose(fitted.correlation[-1, :-1], 0.0)
    assert np.all(fitted.sample(500, np.random.default_rng(3))[:, -1] == 0.0)


def test_fit_refuses_too_few_rows():
    with pytest.raises(ValueError, match="complete rows"):
        GaussianCopula.fit(np.zeros((2, 4)), TEST_COLUMNS)


# --------------------------------------------------------------------------- #
# Conditional sampling — the runtime call
# --------------------------------------------------------------------------- #

def test_conditioning_shifts_the_other_features_in_the_right_direction():
    """Given a positively correlated distance, a longer move must come back
    slower. The ordering is not imposed anywhere — it falls out of the fitted
    correlation, which is the whole reason to model the joint at all."""
    rng = np.random.default_rng(14)
    fitted = GaussianCopula.fit(_draw_known(3000, 0.7, rng), TEST_COLUMNS,
                             rng=np.random.default_rng(0))

    near = fitted.sample_conditional({"distance_px": 60.0}, 4000,
                                  np.random.default_rng(1))
    far = fitted.sample_conditional({"distance_px": 560.0}, 4000,
                                 np.random.default_rng(1))
    assert np.median(far["duration_s"]) > np.median(near["duration_s"])


def test_conditional_echoes_the_known_column_and_keeps_bounds():
    rng = np.random.default_rng(15)
    fitted = GaussianCopula.fit(_draw_known(600, 0.5, rng), TEST_COLUMNS,
                             rng=np.random.default_rng(0))
    out = fitted.sample_conditional({"distance_px": 250.0}, 2000,
                                 np.random.default_rng(1))
    assert set(out) == set(TEST_COLUMNS)
    assert np.all(out["distance_px"] == 250.0)
    assert out["duration_s"].min() > 0
    assert out["straightness"].max() <= 1.0


def test_conditional_is_deterministic_in_what_it_conditions_on():
    """Two calls with the same known value and the same rng must agree. The
    midpoint of the cdf interval is used rather than a jittered draw for
    exactly this reason."""
    rng = np.random.default_rng(16)
    fitted = GaussianCopula.fit(_draw_known(400, 0.5, rng), TEST_COLUMNS,
                             rng=np.random.default_rng(0))
    a = fitted.sample_conditional({"distance_px": 300.0}, 50, np.random.default_rng(4))
    b = fitted.sample_conditional({"distance_px": 300.0}, 50, np.random.default_rng(4))
    assert np.allclose(a["duration_s"], b["duration_s"])


def test_conditional_rejects_an_unknown_column():
    rng = np.random.default_rng(17)
    fitted = GaussianCopula.fit(_draw_known(400, 0.5, rng), TEST_COLUMNS,
                             rng=np.random.default_rng(0))
    with pytest.raises(KeyError, match="not fitted columns"):
        fitted.sample_conditional({"nope_px": 1.0}, 5)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def test_json_roundtrip_reproduces_the_same_samples(tmp_path):
    rng = np.random.default_rng(18)
    data = np.column_stack([_draw_known(400, 0.5, rng),
                            _zero_inflated(rng, 400, 0.5)])
    cols = (*TEST_COLUMNS, "overshoot_px")
    fitted = GaussianCopula.fit(data, cols, rng=np.random.default_rng(0),
                                activity="test activity")

    path = fitted.save(tmp_path / "c.json")
    back = _GaussianCopula.load(path)

    assert back.columns == fitted.columns
    assert back.activity == "test activity"
    assert np.allclose(back.correlation, fitted.correlation)
    assert np.allclose(fitted.sample(200, np.random.default_rng(9)),
                       back.sample(200, np.random.default_rng(9)))


def test_load_rejects_an_unknown_version(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"version": 999, "columns": [], "marginals": [],
                             "correlation": []}))
    with pytest.raises(ValueError, match="version"):
        _GaussianCopula.load(p)


def test_every_marginal_type_survives_a_json_roundtrip():
    rng = np.random.default_rng(19)
    for m in (ConstantMarginal(value=3.0, name="x"),
              choose_marginal(rng.choice([0, 1, 2], 200).astype(float),
                              name="n_submovements"),
              choose_marginal(rng.lognormal(0, 0.5, 200), name="duration_s"),
              choose_marginal(_zero_inflated(rng, 300, 0.5), name="overshoot_px")):
        back = marginal_from_json(json.loads(json.dumps(m.to_json())))
        u = np.linspace(0.02, 0.98, 200)
        assert np.allclose(m.ppf(u), back.ppf(u)), m.describe()


# --------------------------------------------------------------------------- #
# NORTA correlation matching
# --------------------------------------------------------------------------- #

def test_spearman_averages_ties():
    """An atomic column is mostly ties, so a rank statistic that breaks them
    arbitrarily is measuring the tie-breaking, not the feature."""
    a = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    b = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    assert cop.spearman(a, a) == pytest.approx(1.0)
    averaged_ranks = np.array([2.0, 2.0, 2.0, 4.0, 5.0])
    assert cop.spearman(a, b) == pytest.approx(
        np.corrcoef(averaged_ranks, np.arange(1.0, 6.0))[0, 1])
    # A constant column cannot correlate with anything.
    assert cop.spearman(np.ones(5), b) == 0.0


def test_atomless_marginals_need_no_calibration():
    """The theoretical premise for only calibrating atom-touching pairs: a
    strictly monotone ppf preserves ranks, so the sampled Spearman already
    equals the latent correlation and there is nothing to correct."""
    rng = np.random.default_rng(0)
    data = np.abs(rng.lognormal(0.0, 1.0, 400))
    m = cop.choose_marginal(data, name="continuous")
    assert not m.has_atoms

    z0, z1 = rng.standard_normal(20000), rng.standard_normal(20000)
    for rho in (-0.6, 0.3, 0.8):
        got = cop._sampled_spearman(m, m, rho, z0, z1)
        # Spearman of a Gaussian copula is (6/pi)*arcsin(rho/2), the arcsin law.
        expected = 6.0 / math.pi * math.asin(rho / 2.0)
        assert abs(got - expected) < 0.03, f"rho={rho}: {got} vs {expected}"


def test_atoms_attenuate_the_estimate_and_calibration_recovers_it():
    """The whole point, on data with a KNOWN answer.

    Built from a Gaussian copula at latent rho = 0.80 with one hurdle
    marginal, so the right answer is not a matter of opinion. Estimating rho
    the way `fit` does — distributional transform, then correlate — returns
    ~0.53, because spreading the ties injects independent noise and noise
    attenuates a correlation toward zero. Calibration recovers ~0.80."""
    rng = np.random.default_rng(1)
    n = 600
    true_rho = 0.8
    latent = rng.multivariate_normal([0, 0], [[1, true_rho], [true_rho, 1]], n)
    continuous = np.exp(latent[:, 0])
    hurdle = np.where(latent[:, 1] > 0.62, np.exp(latent[:, 1]), 0.0)

    mc = cop.choose_marginal(continuous, name="cont")
    mh = cop.choose_marginal(hurdle, name="hurdle")
    assert mh.has_atoms and not mc.has_atoms

    # Estimate exactly as GaussianCopula.fit does.
    g = np.random.default_rng(10)
    Z = cop._norm_ppf(np.column_stack([
        cop._to_uniform(mc, continuous, g), cop._to_uniform(mh, hurdle, g)]))
    rho_estimated = np.corrcoef(Z, rowvar=False)[0, 1]
    assert rho_estimated < 0.65, "premise: the naive estimate is attenuated"

    target = cop.spearman(continuous, hurdle)
    rho_calibrated = cop.calibrate_pair(mc, mh, target, np.random.default_rng(2))

    # Calibration recovers the rho the data was actually generated with.
    assert abs(rho_calibrated - true_rho) < 0.08
    assert abs(rho_calibrated - true_rho) < abs(rho_estimated - true_rho)

    # And therefore reproduces the observed rank correlation.
    z0, z1 = rng.standard_normal(20000), rng.standard_normal(20000)
    naive = cop._sampled_spearman(mc, mh, rho_estimated, z0, z1)
    calibrated = cop._sampled_spearman(mc, mh, rho_calibrated, z0, z1)
    assert abs(calibrated - target) < abs(naive - target)
    assert abs(calibrated - target) < 0.05


def test_calibrate_pair_returns_a_bracket_end_when_target_is_unreachable():
    """Two atoms can genuinely cap the attainable rank correlation. Returning
    the closest attainable end is the honest answer; raising is not."""
    rng = np.random.default_rng(3)
    counts = rng.integers(0, 3, 400).astype(float)
    m = cop.choose_marginal(counts, name="counts")
    assert m.has_atoms
    assert cop.calibrate_pair(m, m, 0.999, rng) == pytest.approx(0.999, abs=1e-3)
    assert cop.calibrate_pair(m, m, -0.999, rng) == pytest.approx(-0.999, abs=1e-3)


def test_calibration_only_touches_badly_attenuated_pairs():
    """Calibrating every atom-touching pair makes each individually right but
    the matrix jointly unrealizable, and the positive-definiteness repair then
    costs more elsewhere than it buys. The threshold is what keeps it a net
    win, so a fit must not be calibrating everything in sight."""
    rng = np.random.default_rng(4)
    n = 300
    base = rng.normal(size=n)
    data = np.column_stack([
        np.exp(base + 0.3 * rng.normal(size=n)),
        np.abs(base * 2 + rng.normal(size=n)),
        np.where(base > 0.5, np.abs(base), 0.0),
        rng.integers(0, 4, n).astype(float),
    ])
    cols = ["a", "b", "hurdle", "counts"]

    fit = GaussianCopula.fit(data, cols, rng=np.random.default_rng(5))
    n_pairs = len(cols) * (len(cols) - 1) // 2
    assert 0 <= fit.n_calibrated < n_pairs
    assert np.all(np.linalg.eigvalsh(fit.correlation) > 0), "must stay PD"


def test_nearest_correlation_repairs_an_indefinite_matrix():
    """Higham's projection, used because calibration reliably produces a
    matrix that is not positive semidefinite."""
    bad = np.array([
        [1.0, 0.9, -0.9],
        [0.9, 1.0, 0.9],
        [-0.9, 0.9, 1.0],
    ])
    assert np.linalg.eigvalsh(bad).min() < 0, "premise: indefinite"

    fixed = cop.nearest_correlation(bad)
    assert np.linalg.eigvalsh(fixed).min() > 0
    assert np.allclose(np.diag(fixed), 1.0)
    assert np.allclose(fixed, fixed.T)
    # It must stay near the input -- a projection that just returns the
    # identity would satisfy everything above and be useless.
    assert np.max(np.abs(fixed - bad)) < 0.75


def test_calibration_can_be_switched_off_and_round_trips():
    rng = np.random.default_rng(6)
    n = 200
    base = rng.normal(size=n)
    data = np.column_stack([
        np.exp(base), np.where(base > 0.4, np.abs(base), 0.0),
    ])
    cols = ["cont", "hurdle"]

    off = GaussianCopula.fit(data, cols, rng=np.random.default_rng(7),
                                 calibrate=False)
    assert off.n_calibrated == 0

    on = GaussianCopula.fit(data, cols, rng=np.random.default_rng(7),
                                min_attenuation=0.0)
    assert on.n_calibrated == 1
    assert _GaussianCopula.from_json(on.to_json()).n_calibrated == 1


def test_every_marginal_transform_preserves_ORDER():
    """A quantile function must be non-decreasing, and a CDF must be too.

    `neglog` failed this: log(hi - x) DECREASES as x rises, so the CDF grid
    built on it ran from 1 down to 0. It stayed a valid inverse pair, so
    marginals still sampled correctly and the defect was invisible in every
    marginal-recovery check — but every latent correlation involving such a
    column came out sign-flipped, and NORTA's bisection assumes rho ->
    Spearman is increasing, so it searched the wrong direction.
    """
    rng = np.random.default_rng(0)
    samples = {
        "log": np.exp(rng.normal(0, 1, 300)),                  # (0, inf)
        "neglog": -np.exp(rng.normal(9, 1, 300)),              # (-inf, 0)
        "logit": rng.beta(2, 3, 300),                          # (0, 1)
        "identity": rng.normal(0, 50, 300),                    # unbounded
    }
    bounds = {
        "log": (0.0, math.inf), "neglog": (-math.inf, 0.0),
        "logit": (0.0, 1.0), "identity": (-math.inf, math.inf),
    }

    u = np.linspace(0.005, 0.995, 200)
    for name, data in samples.items():
        m = choose_marginal(data, name=name, bounds=bounds[name])
        assert isinstance(m, KDEMarginal), f"{name}: expected a KDE"
        assert m.transform == name, f"{name}: got transform {m.transform!r}"

        x = m.ppf(u)
        assert np.all(np.diff(x) >= -1e-9), f"{name}: ppf must not decrease"

        ordered = np.sort(data)
        c = m.cdf(ordered)
        assert np.all(np.diff(c) >= -1e-12), f"{name}: cdf must not decrease"
        assert c[0] < c[-1], f"{name}: cdf runs backwards"

        # ...and the two must still be exact inverses of each other.
        assert np.max(np.abs(m.ppf(m.cdf(ordered)) - ordered)) < 1e-6


def test_a_negative_column_keeps_the_sign_of_its_correlations():
    """The observable symptom of the neglog bug: decel_peak-style columns
    (always negative) reported latent correlations opposite in sign to the
    relationship they actually reproduced."""
    rng = np.random.default_rng(1)
    n = 500
    base = rng.normal(size=n)
    data = np.column_stack([
        np.exp(base),                          # positive, log transform
        -np.exp(base + 0.2 * rng.normal(size=n)),   # negative, neglog
    ])
    cols = ["positive", "negative"]

    fit = GaussianCopula.fit(data, cols, rng=np.random.default_rng(2))
    drawn = fit.sample(20000, np.random.default_rng(3))

    recorded = cop.spearman(data[:, 0], data[:, 1])
    sampled = cop.spearman(drawn[:, 0], drawn[:, 1])
    latent = fit.correlation[0, 1]

    assert recorded < 0, "premise: the two move in opposite directions"
    assert np.sign(sampled) == np.sign(recorded)
    assert np.sign(latent) == np.sign(recorded), "latent sign must not flip"


# --------------------------------------------------------------------------- #
# Bounds are the caller's to declare
# --------------------------------------------------------------------------- #

def test_fit_without_bounds_treats_every_column_as_unbounded():
    rng = np.random.default_rng(30)
    data = np.column_stack([rng.lognormal(0, 0.5, 300), rng.lognormal(0, 0.5, 300)])
    fitted = _GaussianCopula.fit(data, ["a", "b"], rng=np.random.default_rng(0))
    assert all(m.transform == "identity" for m in fitted.marginals)


def test_fit_applies_bounds_by_column_name():
    rng = np.random.default_rng(31)
    data = np.column_stack([rng.lognormal(0, 0.5, 300), rng.lognormal(0, 0.5, 300)])
    fitted = _GaussianCopula.fit(data, ["a", "b"], rng=np.random.default_rng(0),
                                 bounds={"a": (0.0, math.inf)})
    assert fitted.marginals[0].transform == "log"
    assert fitted.marginals[1].transform == "identity"
