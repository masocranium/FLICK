"""
Path-shaping tests.

Most are round trips through the real pipeline: generate a path, feed it to the same
`features.extract` that reads recordings, and assert on what comes back. A waypoint in
the right place is worth nothing if the feature extractor cannot see it afterward.
"""

from __future__ import annotations

import csv
import io
import math

import numpy as np
import pytest

from flick import kinematics
from flick.cli import main
from flick.demo import make_demo_features
from flick.features import ExtractParams, extract
from flick.generate import ClickGenerator, measure_path
from flick.models import CopulaModel, KNNModel
from flick.path import (
    CORRECTION_MIN_PX,
    DEFAULT_T_PEAK_FRAC,
    INTER_LEG_DWELL_S,
    MIN_LEG_DURATION_S,
    build_waypoints,
    generate_path,
    time_schedule,
    wind_walk,
)
from flick.schema import DEFAULT_COLUMNS
from flick.segment import Movement

PARAMS = ExtractParams()


# --------------------------------------------------------------------------- #
# Helpers: the real round trip
# --------------------------------------------------------------------------- #

def build(dist=300.0, duration=0.6, overshoot=0.0, n_sub=0.0,
          t_peak=0.35, seed=0):
    """Generate one shaped path and measure it back through features.py."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    start = (600.0, 400.0)
    dest = (start[0] + dist * math.cos(angle), start[1] + dist * math.sin(angle))

    points, delays = generate_path(start, dest, duration, overshoot, n_sub,
                                   t_peak, rng)
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    t = np.concatenate(([0.0], np.cumsum(delays)))
    movement = Movement(kind="click", activity="test", session_id="s",
                        t=t, x=xs, y=ys, button="left",
                        press_t=float(t[-1]), release_t=float(t[-1]) + 0.05)
    movement.pause_before_s = 0.3
    return extract(movement, PARAMS)


def median_of(attr, n=12, **kwargs):
    vals = [getattr(build(seed=1000 + 7 * i, **kwargs), attr) for i in range(n)]
    vals = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    return float(np.median(vals))


# --------------------------------------------------------------------------- #
# time_schedule
# --------------------------------------------------------------------------- #

def test_delays_sum_to_exactly_the_requested_duration():
    lengths = np.full(40, 7.0)
    for duration in (0.15, 0.6, 2.4):
        assert time_schedule(lengths, duration).sum() == pytest.approx(duration)


def test_every_delay_is_positive():
    """A non-monotone time axis would make the resampled kinematics
    meaningless downstream."""
    dts = time_schedule(np.full(60, 5.0), 0.8, 0.2)
    assert (dts > 0).all()


def test_schedule_peaks_where_asked_not_at_the_final_step():
    """Regression, and the reason this is arc-length reparametrization rather
    than `dt = length / envelope`. Dividing by a bell envelope that goes to
    zero at the ends handed the FINAL step 0.587s of a 0.6s movement, pushing
    every measured peak to 0.017 regardless of what was requested."""
    lengths = np.full(80, 4.0)
    dts = time_schedule(lengths, 0.6, 0.30)
    assert dts[-1] < 0.25 * 0.6, "the last step must not swallow the movement"

    # Checked on the TIME axis, not the index axis. With equal-length steps
    # the two are not interchangeable: more arc length is covered during the
    # fast middle, so the peak sits at index fraction 0.405 while sitting at
    # time fraction 0.305 — and time is what `t_peak_frac` is defined in.
    fastest = int(np.argmin(dts))
    when = float(np.cumsum(dts)[fastest] / 0.6)
    assert when == pytest.approx(0.30, abs=0.06), when


def test_schedule_handles_a_zero_length_path():
    dts = time_schedule(np.zeros(5), 0.4)
    assert dts.sum() == pytest.approx(0.4)
    assert (dts > 0).all()


def test_schedule_of_nothing_is_empty():
    assert time_schedule([], 0.5).size == 0


def test_missing_peak_falls_back_to_the_default():
    lengths = np.full(50, 3.0)
    assert np.allclose(time_schedule(lengths, 0.5, None),
                       time_schedule(lengths, 0.5, DEFAULT_T_PEAK_FRAC))
    assert np.allclose(time_schedule(lengths, 0.5, float("nan")),
                       time_schedule(lengths, 0.5, DEFAULT_T_PEAK_FRAC))


# --------------------------------------------------------------------------- #
# build_waypoints
# --------------------------------------------------------------------------- #

def test_no_features_requested_is_a_single_plain_leg():
    """The majority case — 70 of 182 clicks in the reference recording have
    exactly zero overshoot — and it must cost nothing."""
    wps = build_waypoints((0.0, 0.0), (300.0, 0.0), 0.0, 0.0, np.random.default_rng(0))
    assert wps == [(0.0, 0.0), (300.0, 0.0)]


def test_waypoints_always_end_exactly_on_the_target():
    for overshoot, n_sub in [(0.0, 0.0), (40.0, 0.0), (0.0, 3.0), (25.0, 4.0)]:
        wps = build_waypoints((0.0, 0.0), (300.0, 0.0), overshoot, n_sub,
                              np.random.default_rng(1))
        assert wps[-1] == (300.0, 0.0)
        assert wps[0] == (0.0, 0.0)


def test_overshoot_adds_a_waypoint_past_the_target():
    wps = build_waypoints((0.0, 0.0), (300.0, 0.0), 40.0, 0.0, np.random.default_rng(2))
    assert any(w[0] > 300.0 for w in wps), "nothing reached past the target"
    assert max(w[0] for w in wps) == pytest.approx(340.0, abs=1.0)


def test_corrections_without_overshoot_stop_short_and_never_pass_the_target():
    """The shape the recording implies: a median of 2 submovements alongside
    a median of 0 direction reversals means correction is 'stop short, then
    resume', not an excursion past the target and back."""
    wps = build_waypoints((0.0, 0.0), (300.0, 0.0), 0.0, 3.0, np.random.default_rng(3))
    assert len(wps) == 5, "start + 3 corrective stops + target"
    along = [w[0] for w in wps]
    assert all(a <= 300.0 + 1e-6 for a in along), along
    assert along == sorted(along), "must advance monotonically, never reverse"


def test_corrections_stay_on_the_approach_axis():
    """Regression. Corrections used to be placed on a ray rotated 20-70
    degrees off-axis, alternating sides — a zigzag that produced the right
    submovement count while dropping `straightness` to 0.796 (real: 0.954)
    and pushing `n_direction_changes` to 1.5 (real: 0)."""
    wps = build_waypoints((0.0, 0.0), (500.0, 0.0), 0.0, 4.0, np.random.default_rng(4))
    lateral = [abs(w[1]) for w in wps]
    assert max(lateral) < 5.0, f"corrections wandered off-axis: {lateral}"


def test_correction_legs_are_equal_length_not_tapering():
    """Regression: a geometric taper made the last corrections too small to
    clear find_peaks' prominence threshold, so they were silently uncounted."""
    wps = build_waypoints((0.0, 0.0), (800.0, 0.0), 0.0, 4.0, np.random.default_rng(5))
    legs = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(wps, wps[1:])]
    corrections = legs[1:]                      # skip the main ballistic leg
    assert len(corrections) == 4
    assert max(corrections) - min(corrections) < 0.35 * max(corrections)
    assert min(corrections) >= CORRECTION_MIN_PX * 0.8


def test_overshoot_counts_as_one_of_the_requested_submovements():
    with_over = build_waypoints((0.0, 0.0), (300.0, 0.0), 30.0, 2.0,
                                np.random.default_rng(6))
    without = build_waypoints((0.0, 0.0), (300.0, 0.0), 0.0, 2.0,
                              np.random.default_rng(6))
    assert len(with_over) == len(without)


def test_zero_length_move_degenerates_safely():
    wps = build_waypoints((5.0, 5.0), (5.0, 5.0), 10.0, 3.0, np.random.default_rng(7))
    assert wps[0] == (5.0, 5.0) and wps[-1] == (5.0, 5.0)


# --------------------------------------------------------------------------- #
# generate_path structure
# --------------------------------------------------------------------------- #

def test_one_delay_per_emitted_step():
    """One delay per gap between points."""
    for n_sub in (0.0, 2.0, 4.0):
        points, delays = generate_path(
            (0.0, 0.0), (400.0, 100.0), 0.7, 0.0, n_sub, 0.35,
            np.random.default_rng(8))
        assert len(delays) == len(points) - 1
        assert len(points) >= 2


def test_generated_path_starts_at_the_origin_and_ends_on_target():
    points, _ = generate_path((10.0, 20.0), (410.0, 220.0), 0.6, 20.0, 2.0,
                              0.35, np.random.default_rng(9))
    assert points[0] == (10, 20)
    assert math.hypot(points[-1][0] - 410, points[-1][1] - 220) < 2.0


def test_total_time_matches_the_request_with_corrections():
    for n_sub in (0.0, 1.0, 3.0, 5.0):
        _, delays = generate_path((0.0, 0.0), (500.0, 0.0), 0.9, 0.0, n_sub,
                                  0.35, np.random.default_rng(10))
        assert sum(delays) == pytest.approx(0.9, rel=1e-6)


def test_each_correction_leg_gets_protected_wall_clock_time():
    """Regression, and load-bearing. Allocating time purely by step length
    gave a whole reach-and-correct sequence 3-30ms of a 0.6s movement; the
    12ms smoothing then flattened it, and a REQUESTED 60px overshoot measured
    as little as 1.99px, non-monotonically in what was asked for."""
    _, delays = generate_path((0.0, 0.0), (400.0, 0.0), 0.8, 0.0, 2.0, 0.35,
                              np.random.default_rng(11))
    tail = sum(delays[-8:])
    assert tail >= MIN_LEG_DURATION_S, tail
    assert max(delays) >= INTER_LEG_DWELL_S * 0.9, "no seam dwell present"


def test_a_single_leg_produces_exactly_one_velocity_peak():
    """Nothing is corrected, so nothing should read as a submovement — the single-leg
    case has to survive being routed through the shaping path unchanged."""
    points, delays = generate_path((0.0, 0.0), (400.0, 0.0), 0.6, 0.0, 0.0,
                                   0.35, np.random.default_rng(12))
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    t = np.concatenate(([0.0], np.cumsum(delays)))
    kin = kinematics.compute(t, xs, ys, hz=200.0, smooth_sigma_s=0.012)
    peaks = kinematics.find_peaks(kin.speed, 0.10 * kin.speed.max(),
                                  0.05 * kin.speed.max(), 8)
    assert len(peaks) == 1, f"expected one ballistic peak, found {len(peaks)}"


# --------------------------------------------------------------------------- #
# Round trips — does features.py measure what was asked for?
# --------------------------------------------------------------------------- #

def test_requested_duration_is_reproduced_exactly():
    for duration in (0.30, 0.60, 1.40):
        assert build(duration=duration, seed=13).duration_s == \
            pytest.approx(duration, rel=0.02)


@pytest.mark.parametrize("requested", [5.0, 15.0, 30.0, 60.0])
def test_requested_overshoot_is_measurable_afterward(requested):
    got = median_of("overshoot_px", overshoot=requested, n_sub=1.0)
    assert got == pytest.approx(requested, rel=0.35), f"{requested} -> {got}"


def test_no_overshoot_requested_means_none_measured():
    assert median_of("overshoot_px", overshoot=0.0, n_sub=0.0) < 2.0


@pytest.mark.parametrize("requested", [0, 1, 2, 3])
def test_requested_submovement_count_is_measurable_afterward(requested):
    """One walk has exactly one velocity ramp, so an unshaped path measures 0."""
    got = median_of("n_submovements", dist=800.0, duration=1.2,
                    n_sub=float(requested))
    assert got == pytest.approx(requested, abs=1.0), f"{requested} -> {got}"


@pytest.mark.parametrize("requested", [0.20, 0.35, 0.50])
def test_requested_peak_timing_is_measurable_afterward(requested):
    got = median_of("t_peak_frac", t_peak=requested)
    assert got == pytest.approx(requested, abs=0.06), f"{requested} -> {got}"


def test_shaping_does_not_wreck_straightness():
    """The zigzag regression measured 0.796 here against a real 0.954."""
    assert median_of("straightness", dist=500.0, duration=0.8, n_sub=3.0) > 0.90


def test_shaping_does_not_invent_direction_reversals():
    """Real data: median 2 submovements WITH median 0 reversals."""
    assert median_of("n_direction_changes", dist=500.0, duration=0.8,
                     n_sub=3.0) <= 1.0


def test_asking_for_more_corrections_than_fit_degrades_smoothly():
    """A short movement physically cannot hold many distinct corrections —
    but it must taper off, not fall to zero the way an earlier fixed-span
    design did (a 300px movement returned 0 submovements for a requested 4)."""
    got = [median_of("n_submovements", dist=80.0, duration=0.4, n_sub=float(k))
           for k in (1, 3, 5)]
    assert got[0] >= 1.0
    assert all(g >= 1.0 for g in got), got


def test_submovements_survive_the_smoothing_the_extractor_applies():
    """Directly checks the mechanism: chained legs must leave a real velocity
    VALLEY between peaks. Without the seam dwell the profile never returns
    near zero, prominence collapses, and 3 legs read as 1 submovement."""
    points, delays = generate_path((0.0, 0.0), (800.0, 0.0), 1.2, 0.0, 3.0,
                                   0.35, np.random.default_rng(14))
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    t = np.concatenate(([0.0], np.cumsum(delays)))
    kin = kinematics.compute(t, xs, ys, hz=200.0, smooth_sigma_s=0.012)
    peaks = kinematics.find_peaks(kin.speed, 0.10 * kin.speed.max(),
                                  0.05 * kin.speed.max(), 8)
    assert len(peaks) >= 4, f"expected 4 velocity peaks, found {len(peaks)}"


def test_a_seed_fixes_the_exact_path():
    a = generate_path((0.0, 0.0), (350.0, 120.0), 0.7, 20.0, 3.0, 0.3,
                      np.random.default_rng(99))
    b = generate_path((0.0, 0.0), (350.0, 120.0), 0.7, 20.0, 3.0, 0.3,
                      np.random.default_rng(99))
    assert a[0] == b[0]
    assert np.allclose(a[1], b[1])


def test_a_custom_leg_function_replaces_the_walk():
    def straight(sx, sy, dx, dy, rng):
        for i in range(1, 11):
            yield (sx + (dx - sx) * i / 10.0, sy + (dy - sy) * i / 10.0)

    points, delays = generate_path((0.0, 0.0), (100.0, 0.0), 0.5, 0.0, 0.0, 0.3,
                                   np.random.default_rng(0), leg_fn=straight)
    assert all(p[1] == 0 for p in points)
    assert len(points) == 11 and len(delays) == 10


def test_wind_walk_reaches_the_target_from_any_direction():
    for seed, (dx, dy) in enumerate([(500, 0), (-300, 200), (0, -40), (3, 3)]):
        points = list(wind_walk(0, 0, dx, dy, np.random.default_rng(seed)))
        assert math.hypot(points[-1][0] - dx, points[-1][1] - dy) <= 3.0


def test_a_tiny_move_still_gives_a_valid_path():
    points, delays = generate_path((10.0, 10.0), (11.0, 10.0), 0.2, 0.0, 0.0, 0.35,
                                   np.random.default_rng(0))
    assert points[0] == (10, 10) and points[-1] == (11, 10)
    assert len(delays) == len(points) - 1


# --------------------------------------------------------------------------- #
# From a position and a target, through a model
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def demo():
    return make_demo_features(300)


@pytest.mark.parametrize("cls", [CopulaModel, KNNModel], ids=lambda m: m.name)
def test_generator_produces_a_path_between_the_two_points(cls, demo):
    model = cls.fit(demo, DEFAULT_COLUMNS, rng=np.random.default_rng(0))
    path = ClickGenerator(model, np.random.default_rng(1)).generate_path((100, 200), (640, 360))
    assert (path.x[0], path.y[0]) == (100, 200)
    assert (path.x[-1], path.y[-1]) == (640, 360)
    assert path.t[0] == 0.0
    assert path.t[-1] == pytest.approx(path.click.duration_s, rel=1e-6)
    assert np.all(np.diff(path.t) >= 0)
    assert path.to_rows()[0] == {"t": 0.0, "x": 100, "y": 200}


def test_generator_has_no_path_to_where_it_already_is(demo):
    model = CopulaModel.fit(demo, DEFAULT_COLUMNS, rng=np.random.default_rng(0))
    assert ClickGenerator(model).generate_path((5, 5), (5, 5)) is None


def test_generator_accepts_a_custom_leg_function(demo):
    model = CopulaModel.fit(demo, DEFAULT_COLUMNS, rng=np.random.default_rng(0))
    calls = []

    def straight(sx, sy, dx, dy, rng):
        calls.append((dx, dy))
        return [(dx, dy)]

    ClickGenerator(model, np.random.default_rng(1), leg_fn=straight).generate_path((0, 0), (300, 0))
    assert calls


def test_measured_path_reflects_the_steered_features(demo):
    model = CopulaModel.fit(demo, DEFAULT_COLUMNS, rng=np.random.default_rng(0))
    gen = ClickGenerator(model, np.random.default_rng(2))
    wanted, got = [], []
    for _ in range(40):
        path = gen.generate_path((0, 0), (500, 0))
        wanted.append(path.click.duration_s)
        got.append(measure_path(path).duration_s)
    assert np.allclose(wanted, got, rtol=0.03)


def test_path_command_writes_labelled_trajectories(tmp_path, capsys, demo):
    model = tmp_path / "m.json"
    CopulaModel.fit(demo, DEFAULT_COLUMNS, rng=np.random.default_rng(0)).save(model)
    assert main(["path", str(model), "--from", "0,0", "--to", "300,400", "-n", "2",
                 "--seed", "1"]) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert {r["movement"] for r in rows} == {"0", "1"}
    first = [r for r in rows if r["movement"] == "0"]
    assert (first[0]["x"], first[0]["y"]) == ("0", "0")
    assert (first[-1]["x"], first[-1]["y"]) == ("300", "400")


def test_path_command_from_a_point_to_itself_says_so(tmp_path, demo):
    model = tmp_path / "m.json"
    CopulaModel.fit(demo, DEFAULT_COLUMNS, rng=np.random.default_rng(0)).save(model)
    with pytest.raises(SystemExit, match="same point"):
        main(["path", str(model), "--from", "1,1", "--to", "1,1"])
