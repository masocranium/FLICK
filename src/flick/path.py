"""
Turn a generated feature vector into an (x, y, t) trajectory.

This module produces DATA: arrays of points and times. Nothing here moves a cursor
or sends input to anything.

Shape comes from chaining walks through waypoints, and timing from a separate pass:

  build_waypoints  turns `overshoot_px` and `n_submovements` into a chain of targets.
  wind_walk        walks each leg. It restarts from zero velocity on every call, so
                   each leg adds a fresh velocity peak, which is what a submovement is.
  time_schedule    decides how long each already-fixed step takes, with speed shaped
                   as a bell peaking at `t_peak_frac`. It never moves a point.

Corrective stops sit ON the approach axis. With no overshoot the movement stops short
of the target and resumes in the same direction; with an overshoot it reaches past
once and walks back. That is what recordings show (a median of 2 submovements with a
median of 0 reversals). A lateral zigzag gave the right submovement count, but dropped
`straightness` to 0.796 against a recorded 0.954 and pushed `n_direction_changes` to
1.5 against a recorded 0.

WHAT THE MODEL DOES NOT CONTROL

`duration_s`, `overshoot_px`, `n_submovements` and `t_peak_frac` are honoured. The
other columns (`straightness`, `max_dev_px`, `jerk_rms` and so on) emerge from the walk
and are not steered to the drawn values. `measure_path` shows what a path really has.

The walk is the wind-and-gravity model published by Benjamin Land as WindMouse,
implemented here from its description. Pass your own `leg_fn` to replace it.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

Point = tuple[float, float]

#: Takes (start_x, start_y, dest_x, dest_y, rng) and yields (x, y) points, excluding
#: the start. Must restart from rest on every call, or legs won't read as submovements.
LegFn = Callable[[float, float, float, float, np.random.Generator], Iterable[Point]]

#: Below this, an overshoot is not worth a dedicated reach-past leg.
OVERSHOOT_THRESHOLD_PX = 1.5

#: How corrective legs are sized. A correction only counts as a submovement if it
#: clears a prominence threshold set by the MAIN leg's peak speed, and each leg's
#: duration is pinned near MIN_LEG_DURATION_S, which leaves distance as the free
#: variable. Measured against three failures:
#:
#:  1. Tapering loses the last corrections, so legs are equal length.
#:  2. Dividing a fixed span fails as the count rises: legs shrink while the main leg
#:     speeds up and the threshold rises. So this is the length of ONE leg.
#:  3. There is an absolute floor. A short leg is low and narrow, and the 12 ms
#:     smoothing fills the valley between two of them. 18 px legs failed on a 300 px
#:     move while 48 px legs on an 800 px move were exact.
CORRECTION_LEG_FRAC = 0.06
CORRECTION_MIN_PX = 25.0
#: Past this share of the distance, the main leg stops being the main leg.
MAX_CORRECTION_SPAN_FRAC = 0.55

#: Sideways wobble on a corrective stop, in pixels. Small on purpose: see the module
#: docstring on what a lateral zigzag does to other features.
LATERAL_JITTER_PX = 2.0

#: Peak timing when none is supplied.
DEFAULT_T_PEAK_FRAC = 0.35

#: u^p * (1-u)^q with p+q held at this value. Larger makes a narrower bell.
ENVELOPE_SHARPNESS = 5.0

#: Pause at the seam between two legs. A walk stops within 1 px of its target while
#: still moving, so without this the speed never returns near zero at the seams, and
#: requested 2, 3 or 4 submovements all measured 1.
INTER_LEG_DWELL_S = 0.045

#: Protected minimum time for each corrective leg. Timing purely by step length gave a
#: reach-and-correct sequence 3-30 ms of a 0.6 s movement, which the 12 ms smoothing
#: erased: a requested 60 px overshoot measured as little as 1.99 px.
MIN_LEG_DURATION_S = 0.085

#: Corrections may not take more than this share of the duration between them, so a
#: fast movement's main phase isn't squeezed to nothing.
MAX_CORRECTION_SHARE = 0.55

#: Safety limit on a single leg's steps.
MAX_LEG_STEPS = 20000

_SQRT3 = math.sqrt(3.0)
_SQRT5 = math.sqrt(5.0)


def wind_walk(start_x: float, start_y: float, dest_x: float, dest_y: float,
              rng: np.random.Generator, gravity: float = 9.0, wind: float = 3.0,
              max_step: float = 15.0, damped_distance: float = 12.0
              ) -> Iterable[tuple[int, int]]:
    """Walk toward a destination under gravity and random wind, yielding integer
    pixels. All randomness comes from `rng`, so a seed fixes the exact path."""
    x, y = float(start_x), float(start_y)
    vx = vy = wx = wy = 0.0
    step_cap = float(max_step)
    last = (int(round(x)), int(round(y)))

    for _ in range(MAX_LEG_STEPS):
        dist = math.hypot(dest_x - x, dest_y - y)
        if dist < 1.0:
            return
        w = min(wind, dist)
        if dist >= damped_distance:
            wx = wx / _SQRT3 + (2.0 * rng.random() - 1.0) * w / _SQRT5
            wy = wy / _SQRT3 + (2.0 * rng.random() - 1.0) * w / _SQRT5
        else:
            wx /= _SQRT3
            wy /= _SQRT3
            if step_cap < 3.0:
                step_cap = 3.0 + rng.random() * 3.0
            else:
                step_cap /= _SQRT5
        vx += wx + gravity * (dest_x - x) / dist
        vy += wy + gravity * (dest_y - y) / dist
        speed = math.hypot(vx, vy)
        if speed > step_cap:
            clipped = step_cap / 2.0 + rng.random() * step_cap / 2.0
            vx, vy = vx / speed * clipped, vy / speed * clipped
        x += vx
        y += vy
        here = (int(round(x)), int(round(y)))
        if here != last:
            last = here
            yield here


def _bell_envelope(u: np.ndarray, peak_frac: float,
                   sharpness: float = ENVELOPE_SHARPNESS) -> np.ndarray:
    """Desired-speed weight peaking at `peak_frac` of [0, 1]: u^p * (1-u)^q, which
    peaks at p/(p+q). Computed in log space to avoid underflow at the far end."""
    f = min(max(float(peak_frac), 0.04), 0.96)
    p = max(sharpness * f, 0.2)
    q = max(sharpness * (1.0 - f), 0.2)
    uu = np.clip(u, 1e-6, 1.0 - 1e-6)
    log_e = p * np.log(uu) + q * np.log(1.0 - uu)
    return np.exp(log_e - log_e.max())


def time_schedule(step_lengths: Sequence[float], duration_s: float,
                  t_peak_frac: Optional[float] = None,
                  sharpness: float = ENVELOPE_SHARPNESS) -> np.ndarray:
    """Per-step delays summing to `duration_s`, with speed a bell peaking at
    `t_peak_frac` of the elapsed time.

    By arc-length reparametrization, not `dt = length / envelope`: a bell goes to zero
    at its ends, and dividing by it handed the final step 0.587 s of a 0.6 s movement.
    Integrating the speed profile and inverting that monotone curve cannot blow up."""
    lengths = np.asarray(step_lengths, dtype=float)
    n = lengths.size
    if n == 0:
        return np.zeros(0, dtype=float)
    duration = max(float(duration_s), 1e-3)
    if t_peak_frac is None or not math.isfinite(t_peak_frac):
        t_peak_frac = DEFAULT_T_PEAK_FRAC

    total = float(lengths.sum())
    if total <= 1e-9:
        return np.full(n, duration / n)

    tau = np.linspace(0.0, 1.0, 2048)
    speed = _bell_envelope(tau, t_peak_frac, sharpness)
    # The floor keeps the movement from being exactly stationary at either end, which
    # would make the inverse lookup flat there.
    speed = 0.02 + 0.98 * (speed / speed.max())
    covered = np.cumsum(speed)
    covered -= covered[0]
    covered /= covered[-1]

    cum = np.concatenate(([0.0], np.cumsum(lengths))) / total
    return np.diff(np.interp(cum, covered, tau) * duration)


def build_waypoints(start: Point, dest: Point, overshoot_px: Optional[float],
                    n_submovements: Optional[float],
                    rng: np.random.Generator) -> list[Point]:
    """The chain of targets a movement passes through, from `start` to exactly `dest`.
    Degrades to `[start, dest]` when neither feature asks for more."""
    sx, sy = float(start[0]), float(start[1])
    dx, dy = float(dest[0]), float(dest[1])
    total = math.hypot(dx - sx, dy - sy)
    if total < 1e-6:
        return [(sx, sy), (dx, dy)]
    ux, uy = (dx - sx) / total, (dy - sy) / total
    px, py = -uy, ux

    n_sub = 0
    if n_submovements is not None and math.isfinite(n_submovements):
        n_sub = max(0, int(round(n_submovements)))
    has_overshoot = (overshoot_px is not None and math.isfinite(overshoot_px)
                     and overshoot_px > OVERSHOOT_THRESHOLD_PX)

    def stop_at(along: float) -> Point:
        """A stop `along` px from `dest` on the approach axis: negative is short."""
        jitter = float(rng.uniform(-LATERAL_JITTER_PX, LATERAL_JITTER_PX))
        return (dx + ux * along + px * jitter, dy + uy * along + py * jitter)

    waypoints = [(sx, sy)]
    remaining = n_sub
    if has_overshoot:
        waypoints.append(stop_at(overshoot_px))
        # Reaching past and correcting back IS one adjustment, so it counts against
        # the requested total.
        remaining = max(0, n_sub - 1)

    if remaining > 0:
        if has_overshoot:
            # Walk back monotonically, so the return is one reversal, not several.
            for k in range(remaining):
                waypoints.append(stop_at(overshoot_px * (remaining - k) / (remaining + 1.0)))
        else:
            # Cap the TOTAL, so asking for more than fits degrades smoothly.
            leg = max(CORRECTION_MIN_PX, CORRECTION_LEG_FRAC * total)
            span = min(leg * remaining, MAX_CORRECTION_SPAN_FRAC * total)
            for k in range(remaining):
                waypoints.append(stop_at(-span * (remaining - k) / float(remaining)))
    waypoints.append((dx, dy))

    # Drop legs a walk would treat as already arrived.
    out = [waypoints[0]]
    for wp in waypoints[1:]:
        if math.hypot(wp[0] - out[-1][0], wp[1] - out[-1][1]) >= 1.0:
            out.append(wp)
    if out[-1] != (dx, dy):
        out.append((dx, dy))
    return out


def _leg_timing(step_lengths: np.ndarray, duration: float, rng: np.random.Generator,
                dwell_s: float = INTER_LEG_DWELL_S) -> np.ndarray:
    """Pace one corrective leg. The first entry is the seam dwell, which drives the
    speed down to a real valley; the rest get roughly constant speed with mild jitter."""
    n = step_lengths.size
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.array([duration], dtype=float)
    dwell = min(max(float(dwell_s), 0.0), 0.6 * duration)
    weights = np.maximum(step_lengths[1:], 1e-6) * rng.uniform(0.7, 1.3, size=n - 1)
    out = np.empty(n, dtype=float)
    out[0] = dwell
    out[1:] = weights * ((duration - dwell) / weights.sum())
    return out


def generate_path(start: Point, dest: Point, duration_s: float,
                  overshoot_px: Optional[float], n_submovements: Optional[float],
                  t_peak_frac: Optional[float], rng: np.random.Generator,
                  leg_fn: LegFn = wind_walk
                  ) -> tuple[list[tuple[int, int]], list[float]]:
    """Waypoints chained through `leg_fn`, then timed.

    Returns (points, delays) with `len(delays) == len(points) - 1`. The first point is
    `start` and the last is `dest`, both rounded to pixels. Each leg after the first is
    a correction and reserves its own time, with the rest going to the main leg."""
    waypoints = build_waypoints(start, dest, overshoot_px, n_submovements, rng)
    end = (int(round(dest[0])), int(round(dest[1])))

    points = [(int(round(waypoints[0][0])), int(round(waypoints[0][1])))]
    leg_starts: list[int] = []
    for (lx, ly), (nx, ny) in zip(waypoints, waypoints[1:]):
        leg = list(leg_fn(lx, ly, nx, ny, rng))
        if leg:
            leg_starts.append(len(points) - 1)
            points.extend((int(round(px)), int(round(py))) for px, py in leg)
    # A walk stops within 1 px of its target, so land on it exactly.
    if points[-1] != end or len(points) < 2:
        if not leg_starts:
            leg_starts.append(0)
        points.append(end)

    lengths = np.array([math.hypot(b[0] - a[0], b[1] - a[1])
                        for a, b in zip(points, points[1:])])
    leg_starts.append(lengths.size)
    duration = max(float(duration_s), 1e-3)

    if len(leg_starts) <= 2:
        return points, [float(d) for d in time_schedule(lengths, duration, t_peak_frac)]

    n_corrections = len(leg_starts) - 2
    floor_total = min(MIN_LEG_DURATION_S * n_corrections, MAX_CORRECTION_SHARE * duration)
    floor_each = floor_total / n_corrections

    delays = np.zeros(lengths.size, dtype=float)
    main_end = leg_starts[1]
    delays[:main_end] = time_schedule(lengths[:main_end], duration - floor_total, t_peak_frac)
    for a, b in zip(leg_starts[1:], leg_starts[2:]):
        delays[a:b] = _leg_timing(lengths[a:b], floor_each, rng)
    return points, [float(d) for d in delays]
