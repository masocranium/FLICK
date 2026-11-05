"""
A Gaussian copula for small, awkwardly-shaped tables, sampled by NORTA.

The model: each column keeps its own marginal distribution, and one correlation
matrix in latent normal space couples them. Sampling is NORTA ("NORmal To
Anything", Cario and Nelson, 1997): draw correlated standard normals, map them
to uniforms, push each through its column's inverse CDF.

Nothing in this module knows what the columns mean. Physical bounds are passed
to `GaussianCopula.fit`; the mouse-movement schema lives in `schema.py`. The
derivations are in docs/theory.md.

WHY NOT A MULTIVARIATE GAUSSIAN

A multivariate Gaussian forces every marginal to be Gaussian, so a positive,
skewed column gets negative draws and a bounded one escapes its bounds. A
Gaussian copula with Gaussian marginals IS a multivariate Gaussian, so this is
a strict generalization that gives up nothing and buys correct marginals.

THE MARGINALS ARE THE HARD PART, NOT THE COPULA

Three shapes defeat the obvious choices:

  ZERO-INFLATED.  A column that is mostly exact zeros with a continuous tail.
  No continuous density can put mass on an atom. `HurdleMarginal` models it as
  what it is: a Bernoulli for "is it the atom", times a continuous distribution
  for "if not, how much".

  DISCRETE.  A count on {0, 1, 2, ...}. A KDE over a lattice is simply wrong,
  and a discrete marginal breaks the uniqueness in Sklar's theorem, so the
  copula is not identified without the DISTRIBUTIONAL TRANSFORM (`_to_uniform`
  below).

  BOUNDED.  A fraction cannot exceed 1. Bounds are facts about what a number
  means, not properties of a sample, so they are declared by the caller rather
  than inferred. Inferring them from a small sample is how a fit ends up
  sampling a fraction of 1.02.

WHY NOT A PURE EMPIRICAL (ECDF) MARGINAL

It cannot extrapolate. Inverting a 17-step ECDF resamples only the 17 values
observed, which hard-truncates the tail. A KDE in TRANSFORMED space keeps the
empirical shape while staying continuous and extending past the observed range,
and doing it in log space keeps a positive quantity positive.

CONDITIONAL SAMPLING

`sample_conditional({"distance_px": 250})` draws the remaining columns from
their distribution GIVEN that value. A Gaussian copula supports this exactly,
in closed form, which is a large part of why it is the right family here rather
than a black-box density model.

NUMPY ONLY

No scipy: the two special functions needed (the normal CDF and its inverse) are
twenty lines. `_norm_ppf` is Acklam's rational approximation, accurate to
~1.15e-9 in absolute value, which is far below the noise in any correlation
estimated from a few hundred rows.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

COPULA_FORMAT_VERSION = 1

_SQRT2 = math.sqrt(2.0)
_erf_vec = np.vectorize(math.erf, otypes=[float])


# --------------------------------------------------------------------------- #
# The two special functions, without scipy
# --------------------------------------------------------------------------- #

def _norm_cdf(z) -> np.ndarray:
    return 0.5 * (1.0 + _erf_vec(np.asarray(z, dtype=float) / _SQRT2))


# Acklam's rational approximation to the inverse normal CDF.
_ACK_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_ACK_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
_ACK_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_ACK_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00)
_ACK_PLOW = 0.02425


def _norm_ppf(u) -> np.ndarray:
    """Inverse standard-normal CDF, vectorized.

    Inputs are clipped strictly inside (0, 1): a u of exactly 0 or 1 maps to
    -inf/+inf, and one infinity anywhere in the z-matrix makes the whole
    correlation estimate NaN. Clipping at 1e-12 caps |z| near 7, which is
    further into the tail than a few hundred rows can resolve anyway."""
    p = np.clip(np.asarray(u, dtype=float), 1e-12, 1.0 - 1e-12)
    out = np.empty_like(p)

    lower = p < _ACK_PLOW
    upper = p > 1.0 - _ACK_PLOW
    central = ~(lower | upper)

    if central.any():
        q = p[central] - 0.5
        r = q * q
        num = (((((_ACK_A[0] * r + _ACK_A[1]) * r + _ACK_A[2]) * r + _ACK_A[3]) * r
                + _ACK_A[4]) * r + _ACK_A[5]) * q
        den = ((((_ACK_B[0] * r + _ACK_B[1]) * r + _ACK_B[2]) * r + _ACK_B[3]) * r
               + _ACK_B[4]) * r + 1.0
        out[central] = num / den

    for mask, flip in ((lower, False), (upper, True)):
        if not mask.any():
            continue
        pp = 1.0 - p[mask] if flip else p[mask]
        q = np.sqrt(-2.0 * np.log(pp))
        num = ((((_ACK_C[0] * q + _ACK_C[1]) * q + _ACK_C[2]) * q + _ACK_C[3]) * q
               + _ACK_C[4]) * q + _ACK_C[5]
        den = (((_ACK_D[0] * q + _ACK_D[1]) * q + _ACK_D[2]) * q + _ACK_D[3]) * q + 1.0
        val = num / den
        out[mask] = -val if flip else val

    return out


def _mad(v: np.ndarray) -> float:
    """Mean absolute deviation from the median. The same statistic as
    metrics.Standardizer, and for the same reason: it cannot go
    degenerate on a zero-inflated column the way an IQR can."""
    v = np.asarray(v, dtype=float)
    return float(np.mean(np.abs(v - np.median(v)))) if v.size else 0.0


# --------------------------------------------------------------------------- #
# Marginals
# --------------------------------------------------------------------------- #

class Marginal:
    """One feature's own distribution.

    Three operations, and the third is the one that is easy to miss:

      ppf(u)          sample: uniform in, feature value out
      cdf(x)          the ordinary CDF
      cdf_interval(x) (F(x-), F(x)) — the JUMP at x, which is nonzero only
                      at an atom. Continuous marginals return the same value
                      twice; discrete and hurdle marginals do not, and that
                      gap is exactly what the distributional transform needs
                      to make u uniform. Without it, a discrete column's u
                      values pile up on a few points and the correlation
                      estimated from them is badly biased toward zero.
    """

    name: str = ""

    def ppf(self, u) -> np.ndarray:      # pragma: no cover - interface
        raise NotImplementedError

    def cdf(self, x) -> np.ndarray:      # pragma: no cover - interface
        raise NotImplementedError

    def cdf_interval(self, x) -> tuple[np.ndarray, np.ndarray]:
        c = self.cdf(x)
        return c, c

    def to_json(self) -> dict:           # pragma: no cover - interface
        raise NotImplementedError

    @property
    def kind(self) -> str:
        return type(self).__name__

    @property
    def has_atoms(self) -> bool:
        """Whether this marginal puts positive probability on single points.

        The distinction drives NORTA calibration below: a pair of atomless
        marginals needs none, because `ppf` is then strictly monotone and
        preserves ranks exactly, so the sampled Spearman correlation equals
        the latent one by construction. An atom breaks that — it maps a whole
        interval of u onto one value, creating ties that no rank statistic
        can see through."""
        return False

    def describe(self) -> str:
        return self.kind


@dataclass
class ConstantMarginal(Marginal):
    """A column with no variation at all.

    Not a degenerate case to be rejected: a count that is always 0 in some
    stratum is a finding, not missing data. It gets a correlation row of pure
    identity (it cannot correlate with anything — it never moves) and always
    samples its one value."""

    value: float
    name: str = ""

    def ppf(self, u) -> np.ndarray:
        return np.full(np.shape(u), self.value, dtype=float)

    def cdf(self, x) -> np.ndarray:
        # Everything at or above the value is "all the mass".
        return np.where(np.asarray(x, dtype=float) >= self.value, 1.0, 0.0)

    def cdf_interval(self, x):
        x = np.asarray(x, dtype=float)
        at = np.isclose(x, self.value)
        left = np.where(at, 0.0, self.cdf(x))
        right = np.where(at, 1.0, self.cdf(x))
        return left, right

    @property
    def has_atoms(self) -> bool:
        return True

    def to_json(self) -> dict:
        return {"kind": "constant", "name": self.name, "value": self.value}

    def describe(self) -> str:
        return f"constant {self.value:g}"


@dataclass
class DiscreteMarginal(Marginal):
    """Empirical PMF over the observed values.

    For a lattice variable this IS the right answer — there is nothing to
    smooth between 2 and 3 submovements. The cost is that it cannot produce a
    value never observed, which for a bounded count is correct behaviour
    rather than the truncation problem an ECDF has on a continuous tail."""

    values: np.ndarray
    probs: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.probs = np.asarray(self.probs, dtype=float)
        self.probs = self.probs / self.probs.sum()
        self._cum = np.cumsum(self.probs)

    def ppf(self, u) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        idx = np.searchsorted(self._cum, u, side="left")
        return self.values[np.clip(idx, 0, self.values.size - 1)]

    def cdf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.values, x, side="right")
        padded = np.concatenate(([0.0], self._cum))
        return padded[np.clip(idx, 0, self.values.size)]

    def cdf_interval(self, x):
        x = np.asarray(x, dtype=float)
        padded = np.concatenate(([0.0], self._cum))
        right = padded[np.clip(np.searchsorted(self.values, x, side="right"),
                               0, self.values.size)]
        left = padded[np.clip(np.searchsorted(self.values, x, side="left"),
                              0, self.values.size)]
        return left, right

    @property
    def has_atoms(self) -> bool:
        return True

    def to_json(self) -> dict:
        return {"kind": "discrete", "name": self.name,
                "values": self.values.tolist(), "probs": self.probs.tolist()}

    def describe(self) -> str:
        return f"discrete over {self.values.size} values {self.values.min():g}..{self.values.max():g}"


@dataclass
class KDEMarginal(Marginal):
    """Gaussian KDE, fitted in a TRANSFORMED space.

    The transform is what keeps a sampled value physically possible. A raw
    Gaussian KDE has unbounded support and will happily emit a negative
    duration or a straightness of 1.4 — the exact failure that rules out an
    MVN in the first place. Fitting in log space for a positive quantity, or
    logit space for a bounded one, makes those values unreachable by
    construction rather than by clipping after the fact.

    `transform` is one of:
      identity   unbounded
      log        (lo, inf)  -> log(x - lo)
      neglog     (-inf, hi) -> -log(hi - x)
      logit      (lo, hi)   -> log(s / (1 - s)),  s = (x - lo) / (hi - lo)

    The CDF is evaluated once onto a grid at fit time and both `cdf` and
    `ppf` interpolate it. That is not just a speed trick: it guarantees the
    two are exact inverses of each other, and a copula whose forward and
    inverse marginal transforms disagree even slightly accumulates that error
    into the correlation estimate.
    """

    samples: np.ndarray           # in TRANSFORMED space
    bandwidth: float
    transform: str
    lo: float
    hi: float
    name: str = ""
    grid_points: int = 2048

    def __post_init__(self):
        self.samples = np.asarray(self.samples, dtype=float)
        self.bandwidth = float(self.bandwidth)
        self._build_grid()

    # -- the transform and its inverse ------------------------------------- #
    def _fwd(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.transform == "log":
            return np.log(np.maximum(x - self.lo, 1e-300))
        if self.transform == "neglog":
            # NEGATED, and that sign is load-bearing. `log(hi - x)` DECREASES
            # as x rises, which would make this the only transform here that
            # reverses order — and the CDF grid below is built assuming the
            # forward transform is increasing, so a reversed one yields a
            # "CDF" that runs from 1 down to 0. It stays a valid inverse pair
            # with `_inv`, so marginals still sample correctly and the error
            # hides, but every latent correlation involving the column comes
            # out sign-flipped and NORTA's bisection (which assumes rho ->
            # Spearman is increasing) searches the wrong direction.
            return -np.log(np.maximum(self.hi - x, 1e-300))
        if self.transform == "logit":
            s = (x - self.lo) / (self.hi - self.lo)
            s = np.clip(s, 1e-9, 1.0 - 1e-9)
            return np.log(s / (1.0 - s))
        return x

    def _inv(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self.transform == "log":
            return self.lo + np.exp(z)
        if self.transform == "neglog":
            return self.hi - np.exp(-z)
        if self.transform == "logit":
            s = 1.0 / (1.0 + np.exp(-z))
            return self.lo + s * (self.hi - self.lo)
        return z

    # -- fit-time grid ------------------------------------------------------ #
    def _build_grid(self) -> None:
        h = max(self.bandwidth, 1e-12)
        lo = float(self.samples.min()) - 4.0 * h
        hi = float(self.samples.max()) + 4.0 * h
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = -1.0, 1.0
        self._grid = np.linspace(lo, hi, self.grid_points)
        # (grid, n) matrix of standardized offsets -> one vectorized erf call.
        zz = (self._grid[:, None] - self.samples[None, :]) / h
        self._grid_cdf = _norm_cdf(zz).mean(axis=1)
        # Force strict monotonicity so np.interp inverts cleanly.
        self._grid_cdf = np.maximum.accumulate(self._grid_cdf)
        span = self._grid_cdf[-1] - self._grid_cdf[0]
        if span > 0:
            self._grid_cdf = (self._grid_cdf - self._grid_cdf[0]) / span
        eps = np.linspace(0.0, 1e-9, self._grid_cdf.size)
        self._grid_cdf = np.clip(self._grid_cdf + eps, 0.0, 1.0)

    # -- interface ---------------------------------------------------------- #
    def cdf(self, x) -> np.ndarray:
        return np.interp(self._fwd(x), self._grid, self._grid_cdf)

    def ppf(self, u) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        z = np.interp(u, self._grid_cdf, self._grid)
        out = self._inv(z)
        # Numerical belt and braces: the transform already makes out-of-bounds
        # values unreachable, but a logit round-trip at u ~ 1 can land a hair
        # outside on the last bit.
        return np.clip(out, self.lo, self.hi)

    def to_json(self) -> dict:
        return {"kind": "kde", "name": self.name,
                "samples": self.samples.tolist(),
                "bandwidth": self.bandwidth, "transform": self.transform,
                "lo": _json_num(self.lo), "hi": _json_num(self.hi),
                "grid_points": self.grid_points}

    def describe(self) -> str:
        return (f"KDE[{self.transform}] n={self.samples.size} "
                f"h={self.bandwidth:.4g}")


@dataclass
class HurdleMarginal(Marginal):
    """A point mass at one value, plus a continuous body for the rest.

    The shape of, say, cursor overshoot: usually none at all (an exact zero),
    and otherwise some continuous amount. Nothing that models it as a single
    continuous density can represent that, and the failure is not subtle: it
    shows up as a fitted median of "0.0000000000000071" standing in for a
    genuine atom.

    The atom is placed at its true position in the ordering rather than
    assumed to be at the bottom, so this also works for, say, a value that
    piles up at an upper bound."""

    atom: float
    atom_prob: float
    body: Marginal
    tol: float = 0.0
    name: str = ""

    def __post_init__(self):
        self.atom_prob = float(min(max(self.atom_prob, 0.0), 1.0))
        # Where the atom sits within the body's own distribution, so the
        # ordering is right whether the atom is at the bottom, middle or top.
        self._below = float(np.asarray(self.body.cdf(np.array([self.atom])))[0])
        self._lo_u = (1.0 - self.atom_prob) * self._below
        self._hi_u = self._lo_u + self.atom_prob

    def ppf(self, u) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        out = np.empty_like(u)
        in_atom = (u >= self._lo_u) & (u <= self._hi_u)
        out[in_atom] = self.atom

        rest = ~in_atom
        if rest.any():
            ur = u[rest]
            # Squeeze the atom's probability back out, then invert the body.
            adjusted = np.where(ur < self._lo_u, ur, ur - self.atom_prob)
            denom = max(1.0 - self.atom_prob, 1e-12)
            out[rest] = self.body.ppf(np.clip(adjusted / denom, 0.0, 1.0))
        return out

    def cdf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        body_part = (1.0 - self.atom_prob) * np.asarray(self.body.cdf(x))
        return body_part + self.atom_prob * (x >= self.atom - self.tol)

    def cdf_interval(self, x):
        x = np.asarray(x, dtype=float)
        at = np.abs(x - self.atom) <= max(self.tol, 0.0)
        body_part = (1.0 - self.atom_prob) * np.asarray(self.body.cdf(x))
        right = body_part + self.atom_prob * (x >= self.atom - self.tol)
        left = np.where(at, body_part, right)
        return left, right

    @property
    def has_atoms(self) -> bool:
        return True

    def to_json(self) -> dict:
        return {"kind": "hurdle", "name": self.name, "atom": self.atom,
                "atom_prob": self.atom_prob, "tol": self.tol,
                "body": self.body.to_json()}

    def describe(self) -> str:
        return (f"hurdle at {self.atom:g} (p={self.atom_prob:.3f}) "
                f"+ {self.body.describe()}")


def _json_num(v: float):
    """JSON has no infinity. Bounds are routinely +/-inf, so they travel as
    null and come back as inf."""
    return None if not math.isfinite(v) else float(v)


def _read_num(v, default: float) -> float:
    return default if v is None else float(v)


def marginal_from_json(data: dict) -> Marginal:
    kind = data.get("kind")
    if kind == "constant":
        return ConstantMarginal(value=float(data["value"]), name=data.get("name", ""))
    if kind == "discrete":
        return DiscreteMarginal(values=np.asarray(data["values"], dtype=float),
                                probs=np.asarray(data["probs"], dtype=float),
                                name=data.get("name", ""))
    if kind == "kde":
        return KDEMarginal(
            samples=np.asarray(data["samples"], dtype=float),
            bandwidth=float(data["bandwidth"]), transform=data["transform"],
            lo=_read_num(data.get("lo"), -math.inf),
            hi=_read_num(data.get("hi"), math.inf),
            name=data.get("name", ""),
            grid_points=int(data.get("grid_points", 2048)))
    if kind == "hurdle":
        return HurdleMarginal(
            atom=float(data["atom"]), atom_prob=float(data["atom_prob"]),
            body=marginal_from_json(data["body"]),
            tol=float(data.get("tol", 0.0)), name=data.get("name", ""))
    raise ValueError(f"unknown marginal kind {kind!r}")


# --------------------------------------------------------------------------- #
# Choosing a marginal
# --------------------------------------------------------------------------- #

def _silverman(z: np.ndarray) -> float:
    """Silverman's rule on the TRANSFORMED sample.

    Applied post-transform on purpose: the rule assumes something roughly
    Gaussian, which a log-duration is and a raw duration is not. `min(sd,
    IQR/1.349)` is the standard robustification, and the fallbacks below
    matter for the small bodies a hurdle can leave behind (a handful of
    overshoots), where the IQR can legitimately be zero."""
    n = z.size
    if n < 2:
        return 1.0
    sd = float(np.std(z, ddof=1))
    iqr = float(np.percentile(z, 75) - np.percentile(z, 25))
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    if spread <= 0:
        spread = _mad(z) or 1.0
    return float(0.9 * spread * n ** (-0.2))


#: An atom has to REPEAT, not merely hold a large share of a small sample.
#: At n=17 a single observation is 5.9% of the data, so an `atom_frac` of
#: 0.05 alone declared the modal value of every continuous column a point
#: mass — measured: 13 of 14 columns came back as spurious hurdles at
#: p=1/17. A value seen once is a sample; a value seen three times in
#: continuous data is a quantization or a genuine floor.
MIN_ATOM_COUNT = 3


def _detect_atom(v: np.ndarray, atom_frac: float) -> tuple[Optional[float], float]:
    """A value carrying at least `atom_frac` of the mass, and its tolerance.

    Zero is tested FIRST and with a tolerance, because the atom that actually
    occurs is usually floating-point dust: a derived column can hold 0.0, -0.0
    and 7.1e-15, which are three distinct floats and one physical fact. Exact
    equality finds none of it.

    Both tests require MIN_ATOM_COUNT occurrences as well as the share —
    see the note above."""
    scale = _mad(v)
    tol = max(1e-12, 1e-9 * max(scale, 1.0))
    n_zero = int(np.count_nonzero(np.abs(v) <= tol))
    if n_zero >= MIN_ATOM_COUNT and n_zero / v.size >= atom_frac:
        return 0.0, tol

    vals, counts = np.unique(v, return_counts=True)
    i = int(np.argmax(counts))
    if counts[i] >= MIN_ATOM_COUNT and counts[i] / v.size >= atom_frac:
        return float(vals[i]), 0.0
    return None, 0.0


def _choose_transform(v: np.ndarray, lo: float, hi: float) -> str:
    """Which space to fit the KDE in, from the feature's physical bounds."""
    finite_lo, finite_hi = math.isfinite(lo), math.isfinite(hi)
    if finite_lo and finite_hi and hi > lo:
        return "logit"
    if finite_lo and float(v.min()) > lo:
        return "log"
    if finite_hi and float(v.max()) < hi:
        return "neglog"
    return "identity"


def choose_marginal(
    v: np.ndarray,
    name: str = "",
    bounds: Optional[tuple[float, float]] = None,
    atom_frac: float = 0.05,
    max_discrete: int = 12,
) -> Marginal:
    """Pick and fit the marginal that matches this column's actual shape.

    `bounds` is the (lo, hi) the value can physically take. Omitted means
    unbounded, which also means no log or logit transform: a positive column
    fitted without its bound can sample negative values.

    Order matters and is not arbitrary:

      1. constant   — no variation; must be caught before anything divides
                      by a spread of zero.
      2. discrete   — integer lattice with few distinct values. Checked
                      before the atom test, or a count variable's modal value
                      would be mistaken for a point mass.
      3. hurdle     — a genuine atom sitting in an otherwise continuous
                      column.
      4. KDE        — everything else, in whichever transformed space the
                      feature's bounds call for.
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError(f"{name or 'column'}: no finite values to fit")

    lo, hi = bounds if bounds is not None else (-math.inf, math.inf)

    if float(np.ptp(v)) <= 0.0:
        return ConstantMarginal(value=float(v[0]), name=name)

    uniq = np.unique(v)
    if uniq.size <= max_discrete and np.allclose(uniq, np.round(uniq)):
        counts = np.array([np.count_nonzero(v == u) for u in uniq], dtype=float)
        return DiscreteMarginal(values=uniq, probs=counts / counts.sum(), name=name)

    atom, tol = _detect_atom(v, atom_frac)
    if atom is not None:
        body_vals = v[np.abs(v - atom) > tol]
        atom_prob = 1.0 - body_vals.size / v.size
        if body_vals.size >= 3 and float(np.ptp(body_vals)) > 0:
            body = choose_marginal(body_vals, name=name, bounds=(lo, hi),
                                   atom_frac=1.1,          # no nested hurdle
                                   max_discrete=max_discrete)
        else:
            # Too little left to model. A near-constant column with a couple
            # of stragglers is honestly reported as constant rather than
            # given a KDE fitted to two points.
            body = ConstantMarginal(
                value=float(np.median(body_vals)) if body_vals.size else atom,
                name=name)
        return HurdleMarginal(atom=atom, atom_prob=atom_prob, body=body,
                              tol=tol, name=name)

    transform = _choose_transform(v, lo, hi)
    dummy = KDEMarginal(samples=np.zeros(1), bandwidth=1.0, transform=transform,
                        lo=lo, hi=hi, name=name)
    z = dummy._fwd(v)
    z = z[np.isfinite(z)]
    if z.size < 2:
        return ConstantMarginal(value=float(np.median(v)), name=name)
    return KDEMarginal(samples=z, bandwidth=_silverman(z), transform=transform,
                       lo=lo, hi=hi, name=name)


# --------------------------------------------------------------------------- #
# Correlation estimation
# --------------------------------------------------------------------------- #

def ledoit_wolf_intensity(Z: np.ndarray) -> float:
    """Optimal shrinkage intensity toward the identity, Ledoit-Wolf (2004).

    The intensity that minimizes expected squared error to the TRUE
    correlation matrix, in closed form, estimated from the data itself. The
    ratio it computes is

        delta = b^2 / d^2 = (how noisily R is estimated)
                            (how far R is from the identity)

    which is the question that actually matters and which no fixed heuristic
    can answer: two datasets with identical n and p deserve different
    shrinkage depending on whether their correlations are real. Weak true
    correlation (small d^2) shrinks hard, because what is there is mostly
    noise; strong true correlation (large d^2) barely shrinks, because there
    is real structure to preserve. `p/(p+n)`, what this replaces, sees only
    the shape of the matrix and never the data in it.

    `Z` is the latent normal matrix (rows = observations). Its columns are
    standardized here so the sample covariance IS the sample correlation,
    which makes the paper's mu*I target exactly the identity.

    Both norms carry the paper's 1/p normalization; it cancels in the ratio,
    and is kept only so the intermediate quantities match the source."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("latent matrix must be 2D")
    n, p = Z.shape
    if n < 2 or p < 2:
        return 1.0

    sd = Z.std(axis=0)
    if not np.all(sd > 0):
        # A column with no latent variation cannot inform the estimate; shrink
        # all the way rather than dividing by zero.
        return 1.0
    Zs = (Z - Z.mean(axis=0)) / sd
    S = (Zs.T @ Zs) / n

    mu = float(np.trace(S)) / p                     # 1.0, by standardization
    d2 = float(np.sum((S - mu * np.eye(p)) ** 2)) / p
    if d2 <= 0:
        # S is already exactly the identity: nothing to preserve.
        return 1.0

    # b_bar^2 = (1/n^2) sum_k || z_k z_k' - S ||^2, expanded so the (n, p, p)
    # outer products never have to be materialized:
    #     ||z z' - S||^2 = ||z||^4 - 2 z'Sz + ||S||^2
    norms4 = (Zs ** 2).sum(axis=1) ** 2
    quad = np.einsum("ij,jk,ik->i", Zs, S, Zs)
    ss = float(np.sum(S ** 2))
    b2_bar = float(np.sum(norms4 - 2.0 * quad + ss)) / (p * n * n)

    # b^2 is capped at d^2, which is what keeps delta a genuine weight in
    # [0, 1] rather than an extrapolation past the target.
    return float(min(max(min(b2_bar, d2) / d2, 0.0), 1.0))


# --------------------------------------------------------------------------- #
# NORTA correlation matching
# --------------------------------------------------------------------------- #
# The generative recipe this module uses -- draw correlated standard normals,
# push each through its own marginal's inverse CDF -- is NORTA, "NORmal To
# Anything" (Cario and Nelson, 1997). It carries one subtlety that has a name
# in that literature: the CORRELATION-MATCHING PROBLEM.
#
# There are two different correlation matrices in play and they are not the
# same number. There is the LATENT correlation, the one fed to the Cholesky
# factor and used to draw the normals; and there is the correlation you
# actually observe in the data, measured on the output side. Fitting
# means answering: what latent correlation makes the SAMPLED output reproduce
# the correlation I measured?
#
# For two atomless marginals the answer is "the same one", and no work is
# needed: `ppf` is then strictly monotone, so it preserves ranks exactly and
# the sampled Spearman correlation equals the latent one by construction.
# Measured here, those pairs agree to a median of 0.029 without any
# correction, which is why only atom-touching pairs are calibrated below.
#
# An atom breaks that guarantee. `_to_uniform` has to spread tied observations
# across the probability jump they share, using independent random draws, or
# the u values are not uniform and Sklar's theorem does not identify a copula
# at all. But adding independent noise to a variable before correlating it is
# textbook attenuation -- the estimate is biased toward zero. In one recorded
# 182-row table, `straightness ~ n_direction_changes` came out at -0.519 when
# reproducing the observed Spearman of -0.508 actually required a latent -0.825.
#
# There is no closed form once ties are involved, so the latent value is found
# numerically: propose a rho, simulate through the same fitted marginals,
# measure the Spearman correlation, and bisect. That mapping is monotone in
# rho, which is what makes bisection valid.

def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks, so tied values share a rank rather than getting an
    arbitrary order. Spearman on an atomic column is meaningless otherwise —
    the ties ARE the feature."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    values, inverse = np.unique(a, return_inverse=True)
    if values.size < a.size:
        sums = np.bincount(inverse, weights=ranks)
        counts = np.bincount(inverse)
        ranks = (sums / counts)[inverse]
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, tie-aware."""
    ra, rb = _rank(a), _rank(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _sampled_spearman(ma: Marginal, mb: Marginal, rho: float,
                      z0: np.ndarray, z1: np.ndarray) -> float:
    """Spearman correlation of a NORTA sample at latent correlation `rho`.

    `z0` and `z1` are two INDEPENDENT standard-normal vectors, drawn once by
    the caller and reused for every rho in a bisection. Building the
    correlated pair as rho*z0 + sqrt(1-rho^2)*z1 rather than redrawing gives
    common random numbers across the search, which makes the rho -> Spearman
    curve smooth instead of jittery — bisection on a Monte-Carlo estimate
    that jumps around each evaluation would not converge."""
    rho = float(np.clip(rho, -0.999999, 0.999999))
    za = z0
    zb = rho * z0 + math.sqrt(max(0.0, 1.0 - rho * rho)) * z1
    xa = ma.ppf(_norm_cdf(za))
    xb = mb.ppf(_norm_cdf(zb))
    return spearman(xa, xb)


def calibrate_pair(ma: Marginal, mb: Marginal, target: float,
                   rng: np.random.Generator, samples: int = 20000,
                   tol: float = 5e-3, max_iter: int = 24) -> float:
    """Latent rho whose NORTA sample reproduces `target` Spearman correlation.

    Returns the rho to put in the correlation matrix. If `target` is outside
    what these marginals can express — two atoms can genuinely cap the
    attainable rank correlation well below 1 — the closest attainable end of
    the range is returned rather than an error, since a capped correlation is
    the honest answer and refusing to fit is not."""
    if not math.isfinite(target) or target == 0.0:
        return 0.0

    z0 = rng.standard_normal(samples)
    z1 = rng.standard_normal(samples)

    lo, hi = -0.999, 0.999
    s_lo = _sampled_spearman(ma, mb, lo, z0, z1)
    s_hi = _sampled_spearman(ma, mb, hi, z0, z1)
    # Attainability check, before spending iterations bisecting toward
    # something outside the range.
    if target <= s_lo:
        return lo
    if target >= s_hi:
        return hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s_mid = _sampled_spearman(ma, mb, mid, z0, z1)
        if abs(s_mid - target) < tol:
            return mid
        if s_mid < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_correlation(R: np.ndarray, data: np.ndarray,
                          marginals: Sequence[Marginal],
                          rng: np.random.Generator,
                          samples: int = 20000,
                          min_attenuation: float = 0.15
                          ) -> tuple[np.ndarray, tuple]:
    """Replace the badly-attenuated entries of `R` with NORTA-matched ones.

    Only pairs where at least one marginal has atoms are considered — see the
    section comment above for why atomless pairs need no correction.

    `min_attenuation` then narrows it further, and the reason is measured
    rather than theoretical. Calibrating every atom-touching pair (36 of 91
    here) makes each pair individually right but the matrix jointly
    unrealizable, and the repair that positive-definiteness then forces costs
    more across the other 87 pairs than the 36 gain: held-out energy distance
    went from 0.2997 uncalibrated to 0.3091 fully calibrated, i.e. worse.
    Correcting only the pairs that are off by more than this threshold — 4 of
    91, in practice — keeps the repair small enough to be worth it, at 0.2948.

    The attenuation is measured, not assumed: each candidate pair is
    simulated at its CURRENT latent rho and the resulting Spearman
    correlation compared against the recorded one. Returns the corrected
    matrix and the (i, j) index pairs that changed."""
    R = np.array(R, dtype=float, copy=True)
    p = R.shape[0]
    changed: list[tuple[int, int]] = []

    z0 = rng.standard_normal(samples)
    z1 = rng.standard_normal(samples)

    for i in range(p):
        for j in range(i + 1, p):
            if not (marginals[i].has_atoms or marginals[j].has_atoms):
                continue
            if isinstance(marginals[i], ConstantMarginal) or \
                    isinstance(marginals[j], ConstantMarginal):
                continue        # cannot correlate with anything; leave at 0

            target = spearman(data[:, i], data[:, j])
            achieved = _sampled_spearman(marginals[i], marginals[j],
                                         R[i, j], z0, z1)
            if abs(target - achieved) <= min_attenuation:
                continue

            R[i, j] = R[j, i] = calibrate_pair(
                marginals[i], marginals[j], target, rng, samples=samples)
            changed.append((i, j))
    return R, tuple(changed)


def nearest_correlation(A: np.ndarray, max_iter: int = 100,
                        tol: float = 1e-8) -> np.ndarray:
    """Nearest positive-semidefinite CORRELATION matrix, by Higham (2002).

    Needed because NORTA calibration solves each pair in isolation, and a set
    of individually-correct pairwise correlations need not be jointly
    realizable. This is not a defect in the calibration — it is a documented
    property of NORTA as dimension grows (Ghosh and Henderson, 2003) — but it
    does mean the calibrated matrix routinely has negative eigenvalues and
    something has to repair it. Measured here at 14 columns: min eigenvalue
    -1.38 straight out of calibration.

    Dykstra's alternating projections, bouncing between "positive
    semidefinite" and "unit diagonal" until both hold. This matters versus
    the cruder `nearest_psd` (clip the eigenvalues, rescale): clipping
    concentrates the whole repair on whichever entries the smallest
    eigenvector loads on, so one pair can absorb a 0.50 correction while its
    neighbours are untouched. Higham minimizes the total Frobenius distance
    instead, which spreads the repair across every entry that has to give.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    Y = A.copy()
    dS = np.zeros_like(A)
    prev = Y.copy()

    for _ in range(max_iter):
        R = Y - dS
        w, V = np.linalg.eigh(R)
        w = np.maximum(w, 0.0)
        X = (V * w) @ V.T
        dS = X - R
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        if np.max(np.abs(Y - prev)) < tol:
            break
        prev = Y.copy()

    # A last eigenvalue floor: alternating projections converge to the PSD
    # boundary, so the result can sit at exactly zero and still fail Cholesky.
    return nearest_psd(Y)


def shrink_correlation(R: np.ndarray, n: int, lam: Optional[float] = None,
                       latent: Optional[np.ndarray] = None
                       ) -> tuple[np.ndarray, float]:
    """Pull a sample correlation matrix toward the identity.

    At the sample sizes one person can record, the off-diagonal entries are
    genuinely noisy — with p columns there are p(p-1)/2 of them estimated
    from n rows, and at n=182, p=14 that is 91 parameters from 182
    observations. Left alone the matrix over-fits, and if n <= p it is
    singular outright and Cholesky fails.

    `lam` forces an intensity. Otherwise it is estimated by Ledoit-Wolf from
    `latent`; without a latent matrix to estimate from, it falls back to the
    p/(p+n) heuristic, which is monotone in the right quantity but blind to
    whether the correlations it is shrinking are real."""
    p = R.shape[0]
    if lam is None:
        if latent is not None and np.asarray(latent).ndim == 2 \
                and np.asarray(latent).shape[1] >= 2:
            lam = ledoit_wolf_intensity(latent)
        else:
            lam = p / float(p + max(n, 1))
    lam = float(min(max(lam, 0.0), 1.0))
    out = (1.0 - lam) * R + lam * np.eye(p)
    return out, lam


def nearest_psd(R: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    """Closest positive-definite correlation matrix, so Cholesky cannot fail.

    Shrinkage usually suffices, but a constant column or a pair of perfectly
    rank-correlated features can still leave a zero eigenvalue. Clip the
    spectrum, rebuild, then restore a unit diagonal — the renormalization
    matters, because clipping alone changes the variances and would silently
    rescale every marginal transform."""
    R = np.asarray(R, dtype=float)
    R = 0.5 * (R + R.T)
    vals, vecs = np.linalg.eigh(R)
    if vals.min() >= min_eig:
        return R
    vals = np.maximum(vals, min_eig)
    out = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(out), min_eig, None))
    out = out / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return 0.5 * (out + out.T)


# --------------------------------------------------------------------------- #
# The copula
# --------------------------------------------------------------------------- #

@dataclass
class GaussianCopula:
    """Marginals + one correlation matrix in latent normal space."""

    columns: list[str]
    marginals: list[Marginal]
    correlation: np.ndarray
    n_rows: int = 0
    shrinkage: float = 0.0
    #: Index pairs whose correlation NORTA calibration replaced. These entries
    #: are NOT comparable with the rest: a calibrated latent rho is
    #: deliberately inflated so the SAMPLED rank correlation matches the
    #: recorded one, so reading it as a rank correlation overstates the
    #: relationship. Anything displaying the matrix has to say which is which.
    calibrated_pairs: tuple = ()
    activity: Optional[str] = None
    kind: str = "click"
    fitted_at: str = ""
    extract_params: dict = None            # type: ignore[assignment]

    def __post_init__(self):
        self.correlation = np.asarray(self.correlation, dtype=float)
        if self.extract_params is None:
            self.extract_params = {}
        self._chol = np.linalg.cholesky(nearest_psd(self.correlation))

    @property
    def n_features(self) -> int:
        return len(self.columns)

    @property
    def n_calibrated(self) -> int:
        return len(self.calibrated_pairs)

    def was_calibrated(self, i: int, j: int) -> bool:
        a, b = (i, j) if i < j else (j, i)
        return (a, b) in {(int(x), int(y)) for x, y in self.calibrated_pairs}

    # -- fitting ------------------------------------------------------------ #
    @classmethod
    def fit(
        cls,
        data: np.ndarray,
        columns: Sequence[str],
        rng: Optional[np.random.Generator] = None,
        shrinkage: Optional[float] = None,
        atom_frac: float = 0.05,
        calibrate: bool = True,
        calibration_samples: int = 20000,
        min_attenuation: float = 0.15,
        bounds: Optional[Mapping[str, tuple[float, float]]] = None,
        **meta,
    ) -> "GaussianCopula":
        """Fit marginals and the latent correlation from a rows x columns array.

        `bounds` maps a column name to its physical (lo, hi). A column it does
        not name is treated as unbounded. Remaining keyword arguments are stored
        as metadata (`activity`, `kind`, `fitted_at`, `extract_params`).

        `calibrate` runs NORTA correlation matching on the atom-touching
        pairs — see the section comment above `calibrate_pair`. It is on by
        default because without it those pairs are measurably attenuated;
        pass False to reproduce the uncalibrated fit."""
        data = np.asarray(data, dtype=float)
        cols = list(columns)
        if data.ndim != 2 or data.shape[1] != len(cols):
            raise ValueError(
                f"data shape {data.shape} does not match {len(cols)} columns")
        data = data[np.isfinite(data).all(axis=1)]
        n = data.shape[0]
        if n < 3:
            raise ValueError(
                f"only {n} complete rows — nothing to fit. Collect more, or "
                f"drop columns that are mostly missing.")

        rng = rng or np.random.default_rng(0)
        bounds = bounds or {}
        marginals = [choose_marginal(data[:, j], name=cols[j], atom_frac=atom_frac,
                                     bounds=bounds.get(cols[j]))
                     for j in range(len(cols))]

        U = np.empty_like(data)
        for j, m in enumerate(marginals):
            U[:, j] = _to_uniform(m, data[:, j], rng)
        Z = _norm_ppf(U)

        # A constant column has no latent variation; corrcoef would return
        # NaN for its row. Give it an identity row instead: it cannot
        # correlate with anything, which is the truth, not a workaround.
        varying = np.array([not isinstance(m, ConstantMarginal) for m in marginals])
        R = np.eye(len(cols))
        latent = None
        if varying.sum() >= 2:
            latent = Z[:, varying]
            sub = np.corrcoef(latent, rowvar=False)
            sub = np.nan_to_num(sub, nan=0.0)
            np.fill_diagonal(sub, 1.0)
            idx = np.flatnonzero(varying)
            R[np.ix_(idx, idx)] = sub

        # NORTA correlation matching, BEFORE shrinkage. The two corrections
        # address different problems and compose in this order: calibration
        # removes a BIAS (ties attenuate the estimate toward zero), shrinkage
        # handles VARIANCE (91 correlations from a few hundred rows are
        # noisy). Shrinking first and calibrating after would have the
        # calibration undo the shrinkage on exactly the entries it touched.
        calibrated_pairs: tuple = ()
        if calibrate:
            R, calibrated_pairs = calibrate_correlation(
                R, data, marginals, rng, samples=calibration_samples,
                min_attenuation=min_attenuation)

        # Ledoit-Wolf is estimated from the VARYING columns only — a constant
        # column has no latent variance to standardize by, and its identity
        # row is unaffected by shrinking toward the identity anyway.
        R, lam = shrink_correlation(R, n, shrinkage, latent=latent)
        # Higham when calibration has been applied (it reliably breaks
        # positive-definiteness and the repair needs to be spread evenly);
        # the cheap eigenvalue floor otherwise, where R is a genuine sample
        # correlation matrix and already all but PSD.
        R = nearest_correlation(R) if calibrated_pairs else nearest_psd(R)

        return cls(columns=cols, marginals=marginals, correlation=R,
                   n_rows=n, shrinkage=lam,
                   calibrated_pairs=tuple(calibrated_pairs), **meta)

    # -- sampling ----------------------------------------------------------- #
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw `n` joint rows, one per generated observation."""
        rng = rng or np.random.default_rng()
        Z = rng.standard_normal((int(n), self.n_features)) @ self._chol.T
        return self._from_latent(Z)

    def sample_conditional(
        self,
        known: dict[str, float],
        n: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> dict[str, np.ndarray]:
        """Draw the remaining features GIVEN the ones you already know.

        Given a known `distance_px`, what you want is not a random movement but
        one consistent with that distance. A Gaussian copula gives this in
        closed form: condition the latent normal on the known coordinates,
        sample the rest, push back through the marginals.

            row = cop.sample_conditional({"distance_px": 250.0})
            duration = float(row["duration_s"][0])

        Returns every column, the known ones echoed back unchanged so a
        caller can treat the result as one complete row."""
        rng = rng or np.random.default_rng()
        unknown_names = [c for c in self.columns if c not in known]
        if not known:
            drawn = self.sample(n, rng)
            return {c: drawn[:, j] for j, c in enumerate(self.columns)}
        missing = [k for k in known if k not in self.columns]
        if missing:
            raise KeyError(f"not fitted columns: {missing}. "
                           f"Available: {self.columns}")
        if not unknown_names:
            return {c: np.full(n, float(known[c])) for c in self.columns}

        a_idx = np.array([self.columns.index(c) for c in known], dtype=int)
        b_idx = np.array([self.columns.index(c) for c in unknown_names], dtype=int)

        # Known values -> latent z. The MIDPOINT of the cdf interval is used
        # rather than a jittered draw: conditioning is supposed to be a
        # deterministic function of what you know, and jitter here would make
        # two calls with the same distance condition on different points.
        z_a = np.empty(a_idx.size)
        for k, j in enumerate(a_idx):
            m = self.marginals[j]
            x = np.array([float(known[self.columns[j]])])
            left, right = m.cdf_interval(x)
            z_a[k] = _norm_ppf(np.array([0.5 * (float(left[0]) + float(right[0]))]))[0]

        R = nearest_psd(self.correlation)
        R_aa = R[np.ix_(a_idx, a_idx)]
        R_ba = R[np.ix_(b_idx, a_idx)]
        R_bb = R[np.ix_(b_idx, b_idx)]

        solved = np.linalg.solve(R_aa, z_a)
        mean_b = R_ba @ solved
        cov_b = R_bb - R_ba @ np.linalg.solve(R_aa, R_ba.T)
        cov_b = nearest_psd_cov(cov_b)

        L = np.linalg.cholesky(cov_b)
        Z_b = mean_b[None, :] + rng.standard_normal((int(n), b_idx.size)) @ L.T

        out: dict[str, np.ndarray] = {}
        U_b = _norm_cdf(Z_b)
        for k, j in enumerate(b_idx):
            out[self.columns[j]] = self.marginals[j].ppf(U_b[:, k])
        for c in known:
            out[c] = np.full(int(n), float(known[c]))
        return out

    def _from_latent(self, Z: np.ndarray) -> np.ndarray:
        U = _norm_cdf(Z)
        out = np.empty_like(U)
        for j, m in enumerate(self.marginals):
            out[:, j] = m.ppf(U[:, j])
        return out

    # -- reporting ---------------------------------------------------------- #
    def describe(self) -> list[tuple[str, str]]:
        return [(c, m.describe()) for c, m in zip(self.columns, self.marginals)]

    def strongest_pairs(self, top: int = 12
                        ) -> list[tuple[str, str, float, bool]]:
        """Largest latent correlations, with whether each was calibrated.

        The flag is not decoration: a calibrated entry is an inflated latent
        value, not a rank correlation, so displaying the two side by side
        without saying which is which invites reading a -0.81 as a rank
        correlation when the relationship it reproduces is -0.63."""
        R = self.correlation
        pairs = [(self.columns[i], self.columns[j], float(R[i, j]),
                  self.was_calibrated(i, j))
                 for i in range(self.n_features)
                 for j in range(i + 1, self.n_features)]
        pairs.sort(key=lambda t: -abs(t[2]))
        return pairs[:top]

    # -- persistence -------------------------------------------------------- #
    def to_json(self) -> dict:
        return {
            "version": COPULA_FORMAT_VERSION,
            "columns": list(self.columns),
            "marginals": [m.to_json() for m in self.marginals],
            "correlation": self.correlation.tolist(),
            "n_rows": int(self.n_rows),
            "shrinkage": float(self.shrinkage),
            "calibrated_pairs": [[int(i), int(j)] for i, j in self.calibrated_pairs],
            "activity": self.activity,
            "kind": self.kind,
            "fitted_at": self.fitted_at,
            "extract_params": self.extract_params,
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n",
                        encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, data: dict) -> "GaussianCopula":
        version = int(data.get("version", 0))
        if version != COPULA_FORMAT_VERSION:
            raise ValueError(
                f"copula format version {version}, this build reads "
                f"{COPULA_FORMAT_VERSION}")
        return cls(
            columns=list(data["columns"]),
            marginals=[marginal_from_json(m) for m in data["marginals"]],
            correlation=np.asarray(data["correlation"], dtype=float),
            n_rows=int(data.get("n_rows", 0)),
            shrinkage=float(data.get("shrinkage", 0.0)),
            calibrated_pairs=tuple(
                (int(i), int(j)) for i, j in data.get("calibrated_pairs", ())),
            activity=data.get("activity"),
            kind=data.get("kind", "click"),
            fitted_at=data.get("fitted_at", ""),
            extract_params=data.get("extract_params") or {},
        )

    @classmethod
    def load(cls, path) -> "GaussianCopula":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def nearest_psd_cov(C: np.ndarray, min_eig: float = 1e-12) -> np.ndarray:
    """PSD projection for a COVARIANCE (no unit-diagonal renormalization).

    Separate from `nearest_psd` on purpose: a conditional covariance has
    genuinely non-unit variances, and forcing its diagonal to 1 — which is
    correct for a correlation matrix — would inflate every conditional draw."""
    C = 0.5 * (np.asarray(C, dtype=float) + np.asarray(C, dtype=float).T)
    vals, vecs = np.linalg.eigh(C)
    if vals.min() >= min_eig:
        return C
    vals = np.maximum(vals, min_eig)
    return vecs @ np.diag(vals) @ vecs.T


def _to_uniform(m: Marginal, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Marginal value -> uniform, via the distributional transform.

    For a continuous marginal this is just F(x). For one with atoms it is
    NOT: F(x) piles every tied observation onto the same u, and a correlation
    estimated from those ties is biased toward zero. Spreading each tie
    uniformly across the jump it sits on restores uniformity, which is what
    Sklar's theorem needs for the copula to be identified at all.

        u = F(x-) + V * (F(x) - F(x-)),   V ~ Uniform(0, 1)
    """
    left, right = m.cdf_interval(np.asarray(x, dtype=float))
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    v = rng.random(left.shape)
    return np.clip(left + v * (right - left), 1e-12, 1.0 - 1e-12)
