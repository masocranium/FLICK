"""
Fabricated movement features, so the package can be tried without any recording.

THIS IS NOT HUMAN DATA. Every value comes from the formulas below, chosen only
to have the same awkward SHAPES real movement features have: skewed, bounded,
all-negative, discrete and zero-inflated columns, with dependence between them.
Numbers measured on it say the code works. They say nothing about how well the
model fits real people.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .mouse import DEFAULT_COLUMNS


def make_demo_features(n: int = 400, seed: int = 0) -> np.ndarray:
    """An `n` x len(DEFAULT_COLUMNS) array, in DEFAULT_COLUMNS order."""
    rng = np.random.default_rng(seed)
    # Shared latent factors are what make the columns dependent.
    hurry = rng.normal(size=n)
    care = rng.normal(size=n)

    distance = np.clip(rng.lognormal(5.0, 0.8, n), 5.0, 1800.0)
    duration = (0.10 + 0.08 * np.log2(distance + 1.0)) \
        * np.exp(-0.25 * hurry + 0.15 * care + rng.normal(0.0, 0.12, n))
    v_peak = 1.8 * distance / duration * np.exp(rng.normal(0.0, 0.15, n))
    accel_peak = v_peak / (0.3 * duration) * np.exp(rng.normal(0.0, 0.2, n))
    decel_peak = -v_peak / (0.4 * duration) * np.exp(rng.normal(0.0, 0.2, n))
    jerk_rms = accel_peak / (0.2 * duration) * np.exp(rng.normal(0.0, 0.25, n))

    straightness = 1.0 / (1.0 + np.exp(-(2.8 + 0.5 * care - 0.3 * hurry
                                         + rng.normal(0.0, 0.5, n))))
    bow = rng.normal(0.0, 1.0, n) * 0.5 * distance * (1.0 - straightness)
    t_peak_frac = np.clip(rng.beta(2.5, 4.0, n), 1e-3, 1.0 - 1e-3)
    max_dev_frac = np.clip(rng.beta(3.0, 3.0, n), 1e-3, 1.0 - 1e-3)

    n_submovements = rng.poisson(np.exp(0.2 + 0.5 * care))
    n_direction_changes = rng.binomial(n_submovements, 0.25)

    # Zero-inflated: most movements never pass the end point. The small noise
    # around zero is deliberate, because a derived column holds floating-point
    # dust rather than exact zeros.
    overshoots = (hurry + rng.normal(0.0, 0.7, n)) > 0.8
    overshoot_px = np.where(
        overshoots,
        0.05 * distance * np.exp(rng.normal(0.0, 0.7, n)),
        rng.normal(0.0, 1e-14, n),
    )
    settle_s = 0.1 * duration * np.exp(0.6 * care + rng.normal(0.0, 0.4, n))

    by_name = {
        "distance_px": distance,
        "duration_s": duration,
        "straightness": straightness,
        "max_dev_px": bow,
        "max_dev_frac": max_dev_frac,
        "v_peak": v_peak,
        "t_peak_frac": t_peak_frac,
        "accel_peak": accel_peak,
        "decel_peak": decel_peak,
        "jerk_rms": jerk_rms,
        "n_submovements": n_submovements.astype(float),
        "overshoot_px": overshoot_px,
        "settle_s": settle_s,
        "n_direction_changes": n_direction_changes.astype(float),
    }
    return np.column_stack([by_name[c] for c in DEFAULT_COLUMNS])


def write_demo_csv(path, n: int = 400, seed: int = 0) -> Path:
    """Write `make_demo_features` as a CSV `norta-copula fit` can read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = make_demo_features(n, seed)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(DEFAULT_COLUMNS)
        writer.writerows(data.tolist())
    return path
