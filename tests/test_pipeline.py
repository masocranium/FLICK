"""
The recording pipeline, on SYNTHETIC pointer streams: no X server, no real hand.

The synthetic movements are minimum-jerk trajectories, whose position follows
10t^3 - 15t^4 + 6t^5. Using a known-analytic trajectory means assertions can be
about VALUES (peak velocity arrives at the midpoint, straightness is 1.0 for a
straight path) rather than merely "it ran without raising".
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest

from flick import kinematics
from flick.capture import (
    BASE_CHANNELS,
    CHANNEL_WINDOW_BOUNDS,
    OUTSIDE_WINDOW_BIT,
    RAW_FORMAT_VERSION,
    BUTTON_BITS,
    RawSession,
    Recorder,
    ReplaySource,
    SessionHeader,
    iter_sessions,
    read_session,
    slugify,
)
from flick.features import CLICK_ONLY_COLUMNS, ExtractParams, extract, extract_all
from flick.segment import SegmentParams, button_events, excursion_events, segment

LEGACY_CHANNELS = BASE_CHANNELS



LEFT = BUTTON_BITS["left"]
MIDDLE = BUTTON_BITS["middle"]
BOUNDS_CHANNELS = (*LEGACY_CHANNELS, CHANNEL_WINDOW_BOUNDS)


# --------------------------------------------------------------------------- #
# Synthetic stream construction
# --------------------------------------------------------------------------- #

def min_jerk(p0: float, p1: float, tau: np.ndarray) -> np.ndarray:
    """Minimum-jerk interpolation — the classic human reaching profile."""
    return p0 + (p1 - p0) * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)


class StreamBuilder:
    """Assembles a raw session out of rests, moves and clicks."""

    def __init__(self, hz: float = 200.0, start: tuple[int, int] = (100, 100)):
        self.hz = hz
        self.dt = 1.0 / hz
        self.t: list[float] = []
        self.x: list[int] = []
        self.y: list[int] = []
        self.b: list[int] = []
        self._now = 0.0
        self._pos = (float(start[0]), float(start[1]))

    def _push(self, x: float, y: float, mask: int) -> None:
        self.t.append(self._now)
        # Integer pixels, exactly as X reports them — this quantization is
        # what kinematics.py's smoothing exists to survive.
        self.x.append(int(round(x)))
        self.y.append(int(round(y)))
        self.b.append(mask)
        self._now += self.dt

    def rest(self, seconds: float, mask: int = 0) -> "StreamBuilder":
        for _ in range(int(round(seconds * self.hz))):
            self._push(self._pos[0], self._pos[1], mask)
        return self

    def move(self, to: tuple[int, int], seconds: float,
             bow_px: float = 0.0, mask: int = 0) -> "StreamBuilder":
        """Minimum-jerk move, optionally bowed perpendicular to the straight
        line so straightness < 1 as a real hand's would be.

        `mask` is held for the whole move — a middle-drag is
        exactly this with `mask=MIDDLE`."""
        n = int(round(seconds * self.hz))
        tau = np.linspace(0.0, 1.0, n)
        x0, y0 = self._pos
        x1, y1 = float(to[0]), float(to[1])
        xs = min_jerk(x0, x1, tau)
        ys = min_jerk(y0, y1, tau)
        if bow_px:
            dx, dy = x1 - x0, y1 - y0
            dist = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / dist, dx / dist
            bow = bow_px * np.sin(np.pi * tau)
            xs = xs + nx * bow
            ys = ys + ny * bow
        for px, py in zip(xs, ys):
            self._push(px, py, mask)
        self._pos = (x1, y1)
        return self

    def click(self, dwell_s: float = 0.05) -> "StreamBuilder":
        for _ in range(max(1, int(round(dwell_s * self.hz)))):
            self._push(self._pos[0], self._pos[1], LEFT)
        return self

    def build(self, activity: str = "test activity",
              channels: tuple[str, ...] = LEGACY_CHANNELS) -> RawSession:
        """Defaults to a session with no window tracking."""
        header = SessionHeader(
            activity=activity, hz=self.hz, started_at="2026-08-13T00:00:00+00:00",
            session_id="test-session", channels=channels,
        )
        return RawSession(header=header, t=list(self.t), x=list(self.x),
                          y=list(self.y), b=list(self.b))


# --------------------------------------------------------------------------- #
# kinematics
# --------------------------------------------------------------------------- #

def test_smoothing_survives_pixel_quantization():
    """The whole reason kinematics.py exists: a slow drag across integer
    pixels must not produce an oscillating velocity."""
    hz = 200.0
    t = np.arange(0, 1.0, 1 / hz)
    # 250 px/s -> 1.25 px per sample, so raw diffs alternate 1 and 2 px.
    x = np.round(250.0 * t)
    y = np.zeros_like(x)

    raw_v = np.diff(x) * hz
    assert raw_v.std() > 80.0, "test premise: naive differencing is noisy"

    kin = kinematics.compute(t, x, y, hz=hz, smooth_sigma_s=0.012)
    interior = kin.speed[10:-10]
    assert interior.std() < 15.0
    assert 235.0 < interior.mean() < 265.0


def test_min_jerk_velocity_peaks_at_midpoint():
    hz = 200.0
    n = 100
    tau = np.linspace(0, 1, n)
    t = tau * 0.5
    x = min_jerk(0.0, 400.0, tau)
    y = np.zeros_like(x)

    kin = kinematics.compute(t, x, y, hz=hz)
    peak_frac = np.argmax(kin.speed) / (len(kin) - 1)
    assert 0.40 < peak_frac < 0.60


def test_find_peaks_ignores_ripple():
    x = np.linspace(0, 4 * np.pi, 400)
    signal = np.exp(-((x - 3) ** 2)) + 0.8 * np.exp(-((x - 9) ** 2))
    signal += 0.01 * np.sin(40 * x)          # ripple, must not count

    peaks = kinematics.find_peaks(signal, min_prominence=0.1, min_separation=5)
    assert len(peaks) == 2

    noisy = kinematics.find_peaks(signal, min_prominence=0.001, min_separation=1)
    assert len(noisy) > 10, "test premise: the ripple is there to be ignored"


def test_gaussian_smooth_preserves_level():
    a = np.full(50, 7.0)
    assert np.allclose(kinematics.gaussian_smooth(a, 3.0), 7.0)


def test_resample_handles_jittered_timestamps():
    t = np.array([0.0, 0.004, 0.011, 0.015, 0.022])
    x = np.array([0.0, 4.0, 11.0, 15.0, 22.0])
    grid, xu, _ = kinematics.resample_uniform(t, x, x * 0, hz=500.0)
    assert np.allclose(np.diff(grid), 0.002)
    assert np.allclose(xu, grid * 1000.0, atol=0.5)


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #

def test_recorder_roundtrip(tmp_path: Path):
    samples = [(10 + i, 20 + 2 * i, LEFT if 40 <= i < 50 else 0) for i in range(120)]
    clock = _FakeClock(step=0.005)

    rec = Recorder(
        source=ReplaySource(samples), activity="photo editing", hz=200.0,
        chunk=25, clock=clock, sleep=clock.sleep,
    )
    path = rec.record(tmp_path / "s.raw.jsonl", seconds=0.5)

    session = read_session(path)
    assert session.header.activity == "photo editing"
    assert session.header.session_id.startswith("photo-editing-")
    assert len(session) == rec.samples_written > 50
    assert session.x[0] == 10 and session.y[0] == 20
    assert any(m & LEFT for m in session.b)


def test_recorder_applies_origin_offset(tmp_path: Path):
    clock = _FakeClock(step=0.005)
    rec = Recorder(source=ReplaySource([(500, 400, 0)]), activity="a", hz=200.0,
                   origin=(100, 50), chunk=5, clock=clock, sleep=clock.sleep)
    session = read_session(rec.record(tmp_path / "s.raw.jsonl", seconds=0.1))
    assert session.x[0] == 400
    assert session.y[0] == 350


def test_read_session_rejects_unknown_version(tmp_path: Path):
    p = tmp_path / "bad.raw.jsonl"
    p.write_text('{"kind":"header","version":999,"activity":"a","hz":200}\n')
    with pytest.raises(ValueError, match="version"):
        read_session(p)


def test_slugify():
    assert slugify("Photo Editing") == "photo-editing"
    assert slugify("  video / calls  ") == "video-calls"
    assert slugify("!!!") == "unlabeled"


class _FakeClock:
    """Time advances only when something sleeps.

    Deliberately NOT "advance on every read": the recorder reads the clock
    several times per iteration (sample timestamp, then the deadline lag
    check), and a read-advancing clock would burn wall time on bookkeeping
    that a real `perf_counter` costs nothing for — leaving the loop with a
    third of its expected samples for reasons that have nothing to do with
    the code under test."""

    def __init__(self, step: float = 0.0):
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #

def _two_click_session() -> RawSession:
    return (StreamBuilder(start=(100, 100))
            .rest(0.30)
            .move((500, 400), 0.50, bow_px=18)
            .rest(0.08)
            .click(0.06)
            .rest(0.35)
            .move((200, 600), 0.40, bow_px=12)
            .rest(0.05)
            .click(0.05)
            .rest(0.30)
            .build("photo editing"))


def test_button_events_pairs_press_and_release():
    events = button_events(_two_click_session())
    assert len(events) == 2
    for e in events:
        assert e.button == "left"
        assert e.release_t is not None
        assert 0.03 < (e.release_t - e.press_t) < 0.09


def test_button_event_without_release_is_kept():
    session = (StreamBuilder().rest(0.2).move((400, 400), 0.4).click(0.05).build())
    session.b[-1] = LEFT          # recording stopped mid-click
    events = button_events(session)
    assert len(events) == 1
    assert events[0].release_t is None


def test_segment_finds_both_clicks():
    movements = segment(_two_click_session())
    clicks = [m for m in movements if m.kind == "click"]
    assert len(clicks) == 2
    for m in clicks:
        assert m.button == "left"
        assert 0.03 < m.dwell_s < 0.09

    first = clicks[0]
    assert first.displacement_px == pytest.approx(500.0, abs=25)
    # Onset backdating must recover the acceleration ramp, so the recorded
    # duration lands near the true 0.50s move rather than a clipped middle.
    assert 0.40 < first.duration_s < 0.70


def test_segment_labels_a_movement_with_no_click_as_drift():
    session = (StreamBuilder()
               .rest(0.3).move((400, 300), 0.35).rest(0.4)
               .move((420, 320), 0.20).rest(0.3)
               .build())
    movements = segment(session)
    assert movements
    assert all(m.kind == "drift" for m in movements)
    assert all(m.press_t is None for m in movements)


def test_segment_drops_tiny_twitches():
    session = (StreamBuilder()
               .rest(0.3).move((104, 103), 0.10).rest(0.3)
               .build())
    assert segment(session, SegmentParams(min_distance_px=12.0)) == []


def test_segment_records_pause_before():
    movements = segment(_two_click_session())
    assert math.isnan(movements[0].pause_before_s)
    assert movements[1].pause_before_s > 0.2


def test_segment_splits_a_run_containing_a_press():
    """Click while still moving, then keep moving: two movements, the first
    terminated by the press."""
    b = StreamBuilder(start=(100, 100))
    b.rest(0.3).move((400, 100), 0.35)
    b.click(0.04)
    b.move((700, 100), 0.35).rest(0.3)
    movements = segment(b.build())
    kinds = [m.kind for m in movements]
    assert kinds[0] == "click"
    assert len(movements) >= 2


def test_empty_and_tiny_sessions_are_safe():
    header = SessionHeader(activity="a", hz=200.0, started_at="")
    assert segment(RawSession(header=header)) == []
    tiny = RawSession(header=header, t=[0.0, 0.005], x=[1, 2], y=[1, 2], b=[0, 0])
    assert segment(tiny) == []


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #

def test_straight_path_scores_straightness_one():
    """A perfectly straight synthetic path — what a naive generator makes —
    scores exactly 1.0. This is the feature's defining case, so the
    feature has to reproduce it."""
    session = (StreamBuilder().rest(0.3).move((600, 100), 0.4, bow_px=0)
               .rest(0.05).click().rest(0.2).build())
    row = extract(segment(session)[0])
    assert row.straightness == pytest.approx(1.0, abs=0.02)


def test_bowed_path_scores_below_one_and_keeps_handedness():
    left = (StreamBuilder().rest(0.3).move((600, 100), 0.4, bow_px=40)
            .rest(0.05).click().rest(0.2).build())
    right = (StreamBuilder().rest(0.3).move((600, 100), 0.4, bow_px=-40)
             .rest(0.05).click().rest(0.2).build())

    row_l = extract(segment(left)[0])
    row_r = extract(segment(right)[0])

    assert row_l.straightness < 0.99
    assert abs(row_l.max_dev_px) > 25
    # Signed deviation: a hand bows consistently one way, and an unsigned
    # magnitude would average that structure away.
    assert math.copysign(1, row_l.max_dev_px) != math.copysign(1, row_r.max_dev_px)


def test_min_jerk_movement_reports_one_peak_and_no_submovements():
    session = (StreamBuilder().rest(0.3).move((600, 400), 0.5)
               .rest(0.05).click().rest(0.2).build())
    row = extract(segment(session)[0])
    assert row.n_submovements == 0
    assert 0.35 < row.t_peak_frac < 0.65
    assert row.v_peak > row.v_mean


def test_corrective_submovement_is_counted():
    """Overshoot then correct back — the classic primary + corrective pair."""
    b = StreamBuilder(start=(100, 100))
    b.rest(0.3).move((640, 100), 0.40)      # overshoot past 600
    b.move((600, 100), 0.18)                # correction back
    b.rest(0.05).click().rest(0.2)
    row = extract(segment(b.build())[0])
    assert row.n_submovements >= 1
    assert row.overshoot_px > 20


def test_drift_rows_have_no_click_only_features():
    session = (StreamBuilder().rest(0.3).move((400, 300), 0.35).rest(0.4).build())
    row = extract(segment(session)[0])
    assert row.kind == "drift"
    assert math.isnan(row.dwell_s)
    assert math.isnan(row.overshoot_px)
    assert math.isnan(row.settle_s)
    assert not math.isnan(row.duration_s)
    assert not math.isnan(row.straightness)


def test_activity_label_propagates_to_rows():
    rows = extract_all(segment(_two_click_session()))
    assert {r.activity for r in rows} == {"photo editing"}
    assert {r.session_id for r in rows} == {"test-session"}


def test_settle_time_is_shorter_than_the_movement():
    session = (StreamBuilder().rest(0.3).move((600, 400), 0.5)
               .rest(0.10).click().rest(0.2).build())
    row = extract(segment(session)[0])
    assert 0.0 < row.settle_s < row.duration_s


def test_jerk_scales_with_smoothing_width():
    """Documented sensitivity: jerk_rms is a function of the smoothing width,
    which is exactly why ExtractParams is stamped into every features file."""
    session = (StreamBuilder().rest(0.3).move((600, 400), 0.5)
               .rest(0.05).click().rest(0.2).build())
    movement = segment(session)[0]
    fine = extract(movement, ExtractParams(smooth_sigma_s=0.006)).jerk_rms
    coarse = extract(movement, ExtractParams(smooth_sigma_s=0.024)).jerk_rms
    assert fine > coarse * 2


# --------------------------------------------------------------------------- #
# feature IO
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# drags
#
# A middle-drag filed under click or drift would corrupt that stratum silently.
# --------------------------------------------------------------------------- #

def _pan_session(channels: tuple[str, ...] = LEGACY_CHANNELS) -> RawSession:
    """Approach, middle-drag, then a real left click."""
    return (StreamBuilder(start=(400, 400))
            .rest(0.30)
            .move((580, 400), 0.30)                      # approach
            .rest(0.10)
            .move((1060, 520), 0.60, mask=MIDDLE)        # middle-drag
            .rest(0.20)
            .rest(0.20)
            .move((880, 520), 0.35)                      # approach to a real click
            .rest(0.06)
            .click(0.05)
            .rest(0.20)
            .build("drag test", channels=channels))


def test_a_middle_press_is_not_an_aimed_click():
    """Nothing is aimed at when the middle button goes down, so no movement
    may be filed as a click on account of one — that would put precision
    features on a movement that never exercised precision."""
    movements = segment(_pan_session())

    assert all(m.button != "middle" for m in movements)
    assert all(m.kind != "click" for m in movements if m.button is None)


def test_a_real_click_still_segments_as_a_click():
    """Guard against the fix swallowing ordinary clicks: the left click after
    the drag must survive untouched."""
    clicks = [m for m in segment(_pan_session()) if m.kind == "click"]

    assert len(clicks) == 1
    assert clicks[0].button == "left"
    assert 0.03 < clicks[0].dwell_s < 0.09


# --------------------------------------------------------------------------- #
# excursions: the pointer leaving the tracked window
#
# Without this kind, leaving the window would be segmented as an ordinary `drift`.
# --------------------------------------------------------------------------- #

def _excursion_session(channels: tuple[str, ...] = BOUNDS_CHANNELS) -> RawSession:
    """A small in-window drift, then a long excursion off the client, then a
    real left click back inside."""
    return (StreamBuilder(start=(400, 400))
            .rest(0.30)
            .move((430, 410), 0.20)                          # small in-window drift
            .rest(0.20)
            .move((1400, 900), 0.50, mask=OUTSIDE_WINDOW_BIT)  # leaves the window
            .rest(0.30, mask=OUTSIDE_WINDOW_BIT)                # held elsewhere
            .move((600, 500), 0.40)                          # back, approach a click
            .rest(0.06)
            .click(0.05)
            .rest(0.20)
            .build("excursion test", channels=channels))


def test_small_in_window_drift_is_unaffected():
    """Guard against over-matching: the short in-bounds drift before the
    excursion must still segment as ordinary `drift`."""
    drifts = [m for m in segment(_excursion_session()) if m.kind == "drift"]
    assert any(d.displacement_px < 100 for d in drifts)


def test_a_real_click_survives_an_earlier_excursion():
    clicks = [m for m in segment(_excursion_session()) if m.kind == "click"]
    assert len(clicks) == 1
    assert clicks[0].button == "left"


def test_excursion_events_reports_the_time_outside():
    events = excursion_events(_excursion_session())
    assert len(events) == 1
    assert events[0].duration_s == pytest.approx(0.80, abs=0.03)


def test_excursion_rows_carry_no_click_only_features():
    rows = extract_all(segment(_excursion_session()))
    excursions = [r for r in rows if r.kind == "excursion"]

    assert excursions
    for row in excursions:
        for column in CLICK_ONLY_COLUMNS:
            assert math.isnan(getattr(row, column)), column
        assert row.duration_s > 0


def test_window_bounds_bit_survives_a_recording_round_trip(tmp_path: Path):
    class _BoundsSource:
        channels = BOUNDS_CHANNELS

        def __init__(self):
            self._i = 0

        def read(self):
            self._i += 1
            return (900, 900, OUTSIDE_WINDOW_BIT if self._i > 2 else 0)

        def close(self):
            pass

    clock = _FakeClock()
    recorder = Recorder(source=_BoundsSource(), activity="bounds", hz=100.0,
                        clock=clock, sleep=clock.sleep, chunk=4)
    path = recorder.record(tmp_path / "bounds.raw.jsonl", seconds=0.10)

    session = read_session(path)
    assert session.header.records_window_bounds
    assert session.header.version == RAW_FORMAT_VERSION
    assert any(state & OUTSIDE_WINDOW_BIT for state in session.b)


# --------------------------------------------------------------------------- #
# Restored from the original suite, for the generalized drag and window kinds
# --------------------------------------------------------------------------- #

def _drag_session():
    return (StreamBuilder(start=(400, 400))
            .rest(0.30)
            .move((900, 500), 0.50, mask=MIDDLE)
            .rest(0.30)
            .build("drag test"))


def test_a_middle_drag_is_its_own_kind_not_drift():
    movements = segment(_drag_session())
    drags = [m for m in movements if m.kind == "drag"]
    assert len(drags) == 1 and drags[0].displacement_px > 400
    assert not any(m.kind == "drift" and m.displacement_px > 400 for m in movements)
    assert not any(m.kind == "click" for m in movements)


def test_drag_rows_carry_no_click_only_features():
    rows = [r for r in extract_all(segment(_drag_session())) if r.kind == "drag"]
    assert rows
    for col in CLICK_ONLY_COLUMNS:
        assert math.isnan(getattr(rows[0], col)), col


def test_click_rows_populate_every_feature_column():
    from flick.features import FEATURE_COLUMNS
    clicks = [r for r in extract_all(segment(_two_click_session())) if r.kind == "click"]
    assert clicks
    for r in clicks[1:]:            # row 0 has no pause_before by definition
        for col in FEATURE_COLUMNS:
            assert math.isfinite(getattr(r, col)), f"{col} is not finite"


def test_excursion_is_its_own_kind_not_drift():
    excursions = [m for m in segment(_excursion_session()) if m.kind == "excursion"]
    assert len(excursions) == 1 and excursions[0].displacement_px > 800


def test_a_session_without_the_bounds_channel_reports_no_excursions():
    """"Never left" and "was not being tracked" must stay distinguishable."""
    untracked = _excursion_session(channels=BASE_CHANNELS)
    assert not untracked.header.records_window_bounds
    assert excursion_events(untracked) == []


def test_recorder_never_claims_a_channel_its_source_did_not_declare(tmp_path):
    rec = Recorder(ReplaySource([(0, 0, 0)]), "x", clock=lambda: 0.0, sleep=lambda _: None)
    assert rec.header.channels == BASE_CHANNELS
