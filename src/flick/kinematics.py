"""
Signal processing shared by the segmenter and the feature extractor.

You cannot differentiate raw pointer samples. The pointer reports INTEGER pixels
on a jittery clock: at 200 Hz a hand moving 250 px/s advances 1.25 px per sample,
so consecutive samples differ by 1 or 2 px essentially at random. A naive `dx/dt`
alternates between 200 and 400 px/s, and each further derivative multiplies the
noise. So: put the samples on a uniform time grid, smooth the POSITION with a
Gaussian, and only then differentiate.

  resample -> smooth -> d/dt -> d/dt -> d/dt

`jerk_rms`, `accel_peak` and the submovement count are functions of
`smooth_sigma_s` as much as of the hand: halve the smoothing and jerk roughly
octuples. Two feature rows are only comparable if extracted with the SAME
parameters, which is why they are stamped into every features file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def resample_uniform(
    t: Sequence[float],
    x: Sequence[float],
    y: Sequence[float],
    hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put irregularly-timed samples on a uniform grid by linear interpolation.

    Polling jitter means the recorder's timestamps are near-uniform but not
    exactly uniform, and every derivative below assumes a constant `dt`.
    Interpolating position (rather than pretending the jitter is not there) is
    the conservative choice: it slightly under-states high-frequency detail
    instead of inventing spurious velocity spikes where a sample ran late."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2:
        return t.copy(), x.copy(), y.copy()

    # Guard against a non-monotonic clock; np.interp requires sorted xp.
    order = np.argsort(t, kind="stable")
    t, x, y = t[order], x[order], y[order]

    duration = float(t[-1] - t[0])
    n = max(2, int(round(duration * hz)) + 1)
    grid = t[0] + np.arange(n) / hz
    grid = np.clip(grid, t[0], t[-1])
    return grid, np.interp(grid, t, x), np.interp(grid, t, y)


def gaussian_smooth(a: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Convolve with a normalized Gaussian, reflecting at the edges.

    Reflect rather than zero-pad: zero-padding a position signal would drag
    the first and last samples toward the origin, which shows up as a large
    fictitious velocity exactly at movement onset and at the click — the two
    moments the features care most about."""
    a = np.asarray(a, dtype=float)
    if sigma_samples <= 0 or a.size < 3:
        return a.copy()

    radius = int(max(1, round(3.0 * sigma_samples)))
    radius = min(radius, a.size - 1)
    if radius < 1:
        return a.copy()

    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    kernel /= kernel.sum()

    padded = np.pad(a, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


@dataclass(frozen=True)
class Kinematics:
    """Position and its first three derivatives on a uniform time grid.

    `speed`, `accel` and `jerk` are the tangential (scalar) magnitudes, which
    is what the movement literature means by a "velocity profile" and what
    every feature here consumes."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    speed: np.ndarray
    accel: np.ndarray
    jerk: np.ndarray
    dt: float

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size else 0.0

    def __len__(self) -> int:
        return int(self.t.size)


def compute(
    t: Sequence[float],
    x: Sequence[float],
    y: Sequence[float],
    hz: float = 200.0,
    smooth_sigma_s: float = 0.012,
) -> Kinematics:
    """Full resample -> smooth -> differentiate pipeline. See module docstring."""
    grid, xu, yu = resample_uniform(t, x, y, hz)
    dt = 1.0 / hz

    if grid.size < 2:
        z = np.zeros_like(grid)
        return Kinematics(grid, xu, yu, z, z.copy(), z.copy(), z.copy(), z.copy(), dt)

    sigma_samples = smooth_sigma_s * hz
    xs = gaussian_smooth(xu, sigma_samples)
    ys = gaussian_smooth(yu, sigma_samples)

    vx = np.gradient(xs, dt)
    vy = np.gradient(ys, dt)
    speed = np.hypot(vx, vy)

    # Tangential acceleration and jerk: derivatives OF THE SPEED PROFILE, not
    # the magnitude of the acceleration vector. The two differ whenever the
    # path curves (centripetal component), and it is the speed profile —
    # the bell curve and its corrective bumps — that the features describe.
    accel = np.gradient(speed, dt)
    jerk = np.gradient(accel, dt)

    return Kinematics(grid, xs, ys, vx, vy, speed, accel, jerk, dt)


def find_peaks(
    y: np.ndarray,
    min_prominence: float,
    min_height: float = 0.0,
    min_separation: int = 1,
) -> list[int]:
    """Indices of local maxima with at least `min_prominence` prominence.

    Prominence, not bare local-maximum count: a smoothed velocity profile
    still carries small ripples, and counting those as submovements would
    report a dozen corrections on a movement that plainly had one. Prominence
    is the standard definition — how far the peak rises above the highest
    saddle separating it from any taller peak — so a ripple riding on the
    main ballistic hump scores near zero and a genuine corrective bump does
    not.

    Hand-rolled rather than `scipy.signal.find_peaks` so that recording and
    extraction stay numpy-only."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return []

    candidates = [
        i for i in range(1, n - 1)
        if y[i] >= y[i - 1] and y[i] > y[i + 1] and y[i] >= min_height
    ]

    peaks: list[tuple[int, float]] = []
    for i in candidates:
        # Walk left/right to the first sample STRICTLY taller than this peak,
        # tracking the minimum crossed on the way; the deeper of the two
        # valleys sets the prominence.
        #
        # The comparison must be `<=`, not `<`. A smoothed signal reaches its
        # maximum on a flat top of two or more equal samples, and stopping at
        # the first sample merely EQUAL to the peak would end the walk on the
        # plateau itself — reporting prominence 0 for every peak that has one,
        # which is every peak in practice. Traversing equal samples and
        # stopping only at a strictly greater one is the standard definition.
        j = i - 1
        left_min = y[i]
        while j >= 0 and y[j] <= y[i]:
            left_min = min(left_min, y[j])
            j -= 1

        k = i + 1
        right_min = y[i]
        while k < n and y[k] <= y[i]:
            right_min = min(right_min, y[k])
            k += 1

        prominence = y[i] - max(left_min, right_min)
        if prominence >= min_prominence:
            peaks.append((i, prominence))

    if min_separation > 1 and peaks:
        # Greedy by prominence: keep the strongest peak, drop anything within
        # the refractory window, repeat. Prevents one broad correction from
        # being counted as two.
        peaks.sort(key=lambda p: -p[1])
        kept: list[int] = []
        for idx, _ in peaks:
            if all(abs(idx - k) >= min_separation for k in kept):
                kept.append(idx)
        return sorted(kept)

    return sorted(i for i, _ in peaks)
