"""
Cut a continuous pointer recording into individual movements.

Getting this boundary wrong contaminates every feature downstream: merge two
movements and you get a fictitious mid-flight "submovement"; split one and you
get two half-movements whose velocity profiles are each missing a limb.

THE FOUR KINDS, FITTED SEPARATELY

  click      ends in a left or right press. Has a destination, so distance,
             duration and terminal approach all mean something.
  drift      ends in stillness, with no destination: fidgeting or repositioning.
  drag       made while the middle button is held. A middle press is not aimed
             at anything, so it does not end a click movement.
  excursion  overlaps time spent outside the tracked window, if one was tracked.

THE ALGORITHM

  1. Smoothed speed over the whole session.
  2. Hysteresis threshold -> boolean "moving". Two thresholds, because a single
     one chatters in the slow tails and shatters one movement into five.
  3. Backdate each run's start to where speed last left the LOW threshold.
  4. Split runs at button presses, and attach a press that lands shortly after
     a run ends (the hand stops, then clicks).
  5. Drop fragments too short or too small to carry a velocity profile.

Every threshold is in `SegmentParams` and is stamped into the output file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import kinematics
from .capture import (
    BUTTON_BITS,
    CHANNEL_WINDOW_BOUNDS,
    DRAG_BUTTONS,
    OUTSIDE_WINDOW_BIT,
    RawSession,
)

# Buttons that aim at something, and so terminate a click movement.
CLICK_BUTTONS: tuple[str, ...] = tuple(
    name for name in BUTTON_BITS if name not in DRAG_BUTTONS)


@dataclass(frozen=True)
class SegmentParams:
    """Thresholds controlling where one movement ends and the next begins."""

    hz: float = 200.0
    smooth_sigma_s: float = 0.012

    # Hysteresis band. Enter high enough to ignore hand tremor on a resting
    # mouse; exit low enough that the deceleration tail stays attached.
    move_enter_px_s: float = 45.0
    move_exit_px_s: float = 12.0

    # A press this long after a run ends still belongs to that run. Covers
    # the normal "arrive, settle, click" pattern without swallowing a click
    # that follows an unrelated pause.
    click_grace_s: float = 0.40

    # Fragment filters. A 20 px twitch has no meaningful velocity profile and
    # would otherwise dominate the sample count.
    min_duration_s: float = 0.050
    min_distance_px: float = 12.0

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Movement:
    """One act of the hand, with its raw samples retained.

    Raw samples are kept (not just the derived features) so that a movement
    can be re-featurized under different `ExtractParams` without going back
    to the session file, and so an odd-looking row can be plotted."""

    kind: str                 # "click" | "drift" | "drag" | "excursion"
    activity: str
    session_id: str
    t: np.ndarray             # session-relative seconds
    x: np.ndarray
    y: np.ndarray
    button: Optional[str] = None
    press_t: Optional[float] = None
    release_t: Optional[float] = None
    pause_before_s: float = float("nan")
    index: int = -1

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size >= 2 else 0.0

    @property
    def dwell_s(self) -> float:
        """Button down to button up. NaN for drifts, and for a click whose
        release fell outside the recording."""
        if self.press_t is None or self.release_t is None:
            return float("nan")
        return float(self.release_t - self.press_t)

    @property
    def displacement_px(self) -> float:
        if self.x.size < 2:
            return 0.0
        return float(np.hypot(self.x[-1] - self.x[0], self.y[-1] - self.y[0]))


@dataclass
class ButtonEvent:
    button: str
    press_t: float
    release_t: Optional[float]


def button_events(session: RawSession,
                  buttons: Optional[Sequence[str]] = None) -> list[ButtonEvent]:
    """Press/release pairs recovered from the button-mask column.

    A press whose release is missing (recording stopped mid-click) keeps
    `release_t=None` rather than being dropped — the movement leading up to
    it is still a perfectly good sample; only its dwell is unknown.

    `buttons` restricts which ones are reported; the default is all of them,
    so a caller asking "what did the hand press" still gets the whole answer.
    The segmenter asks only for CLICK_BUTTONS."""
    if len(session) < 2:
        return []

    wanted = BUTTON_BITS if buttons is None else {
        name: bit for name, bit in BUTTON_BITS.items() if name in buttons}

    t = np.asarray(session.t, dtype=float)
    b = np.asarray(session.b, dtype=int)
    events: list[ButtonEvent] = []

    for name, bit in wanted.items():
        held = (b & bit) != 0
        # Prepend the initial state so a button already down at t0 is not
        # reported as a press; we never saw that press.
        changes = np.flatnonzero(held[1:] != held[:-1]) + 1
        open_press: Optional[float] = None
        for i in changes:
            if held[i]:
                open_press = float(t[i])
            elif open_press is not None:
                events.append(ButtonEvent(name, open_press, float(t[i])))
                open_press = None
        if open_press is not None:
            events.append(ButtonEvent(name, open_press, None))

    events.sort(key=lambda e: e.press_t)
    return events


def _held_spans(t: np.ndarray, held: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous True spans of a boolean series, as (start_t, end_t) pairs.

    A state already True at t[0] is not reported — the same reasoning as
    `button_events`: its actual start was never observed, so a span cannot
    honestly be given one."""
    if not held.any():
        return []
    changes = np.flatnonzero(held[1:] != held[:-1]) + 1
    spans: list[tuple[float, float]] = []
    open_at: Optional[float] = None
    for i in changes:
        if held[i]:
            open_at = float(t[i])
        elif open_at is not None:
            spans.append((open_at, float(t[i])))
            open_at = None
    if open_at is not None:
        spans.append((open_at, float(t[-1])))
    return spans


def drag_spans(session: RawSession) -> list[tuple[float, float]]:
    """Every span with a drag button held."""
    if len(session) < 2:
        return []
    t = np.asarray(session.t, dtype=float)
    b = np.asarray(session.b, dtype=int)
    spans: list[tuple[float, float]] = []
    for name in DRAG_BUTTONS:
        spans.extend(_held_spans(t, (b & BUTTON_BITS[name]) != 0))
    return sorted(spans)


@dataclass
class ExcursionEvent:
    """One stretch of the pointer being outside the tracked window."""

    start_t: float
    end_t: float

    @property
    def duration_s(self) -> float:
        return float(self.end_t - self.start_t)


def excursion_events(session: RawSession) -> list[ExcursionEvent]:
    """Every span the pointer spent outside the tracked window.

    Empty for a session recorded without the window_bounds channel. Check
    `session.header.records_window_bounds` to tell "never left" from "was not
    being tracked"."""
    if len(session) < 2 or not session.header.records_window_bounds:
        return []

    t = np.asarray(session.t, dtype=float)
    b = np.asarray(session.b, dtype=int)
    outside = (b & OUTSIDE_WINDOW_BIT) != 0
    return [ExcursionEvent(start, end) for start, end in _held_spans(t, outside)]


def _overlaps_span(t_start: float, t_end: float,
                   spans: Sequence[tuple[float, float]]) -> bool:
    """Did any of `spans` overlap [t_start, t_end] at all?

    Any overlap counts, not containment: an input that begins mid-movement
    makes the rest of that movement belong to it, and the movement cannot be
    attributed to ordinary aiming either way."""
    return any(start <= t_end and end >= t_start for start, end in spans)


def _hysteresis(speed: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    """Boolean 'is moving', latching on above `enter` and off below `exit_`."""
    moving = np.zeros(speed.size, dtype=bool)
    state = False
    for i in range(speed.size):
        if state:
            if speed[i] < exit_:
                state = False
        elif speed[i] > enter:
            state = True
        moving[i] = state
    return moving


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True spans as inclusive (start, end) index pairs."""
    if not mask.any():
        return []
    edges = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    bounds = np.concatenate(([0], edges, [mask.size]))
    out: list[tuple[int, int]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if mask[a]:
            out.append((int(a), int(b) - 1))
    return out


def segment(session: RawSession, params: Optional[SegmentParams] = None) -> list[Movement]:
    """Split one recorded session into movements. See module docstring."""
    p = params or SegmentParams()
    if len(session) < 3:
        return []

    kin = kinematics.compute(
        session.t, session.x, session.y, hz=p.hz, smooth_sigma_s=p.smooth_sigma_s
    )
    if len(kin) < 3:
        return []

    moving = _hysteresis(kin.speed, p.move_enter_px_s, p.move_exit_px_s)
    spans = _runs(moving)
    if not spans:
        return []

    # Backdate onsets: hysteresis latches on at the HIGH threshold, which sits
    # partway up the acceleration ramp. The movement began where speed left
    # the low threshold.
    extended: list[tuple[int, int]] = []
    for i0, i1 in spans:
        while i0 > 0 and kin.speed[i0 - 1] > p.move_exit_px_s:
            i0 -= 1
        while i1 < len(kin) - 1 and kin.speed[i1 + 1] > p.move_exit_px_s:
            i1 += 1
        if extended and i0 <= extended[-1][1]:
            # Backdating merged this run into the previous one.
            prev0, _ = extended.pop()
            extended.append((prev0, i1))
        else:
            extended.append((i0, i1))

    # CLICK buttons only: a middle press is a drag starting, not a target being
    # aimed at, so it must not terminate a click movement.
    events = button_events(session, buttons=CLICK_BUTTONS)
    press_times = np.array([e.press_t for e in events], dtype=float)
    # Checked in this order in `_append`: outside the window first, since a
    # button held while the pointer is elsewhere is not aiming at anything there.
    overlays: list[tuple[str, list[tuple[float, float]]]] = [
        ("excursion", [(e.start_t, e.end_t) for e in excursion_events(session)]),
        ("drag", drag_spans(session)),
    ]

    raw_t = np.asarray(session.t, dtype=float)
    raw_x = np.asarray(session.x, dtype=float)
    raw_y = np.asarray(session.y, dtype=float)

    movements: list[Movement] = []
    # End time of the last emitted movement. A two-phase approach (ballistic
    # move, brief stop, corrective move, click) produces TWO speed runs but
    # only ONE press, and the press sits within `click_grace_s` of both run
    # ends — so without this, one click yields two movements whose samples
    # overlap, double-counting the correction and inflating the click count.
    # The first run's movement already spans through the press, so any later
    # run lying entirely inside that span is skipped.
    covered_until = float("-inf")

    for i0, i1 in extended:
        t_start, t_end = float(kin.t[i0]), float(kin.t[i1])
        if t_end <= covered_until:
            continue
        cursor = max(t_start, covered_until)

        # Terminates because every `continue` advances `cursor` to a press
        # time strictly greater than the current one, and presses are finite.
        while True:
            # Presses strictly inside the remaining span terminate a click
            # movement here; the rest of the span continues as a new one.
            inside = np.flatnonzero((press_times > cursor) & (press_times <= t_end))
            if inside.size:
                ev = events[int(inside[0])]
                _append(movements, session, raw_t, raw_x, raw_y, cursor, ev.press_t,
                        "click", ev, p, overlays)
                cursor = ev.press_t
                covered_until = cursor
                if cursor >= t_end:
                    break
                continue

            # No press inside. Does one land just after the run ends?
            after = np.flatnonzero(
                (press_times > t_end) & (press_times <= t_end + p.click_grace_s)
            )
            if after.size:
                ev = events[int(after[0])]
                _append(movements, session, raw_t, raw_x, raw_y, cursor, ev.press_t,
                        "click", ev, p, overlays)
                covered_until = float(ev.press_t)
            else:
                _append(movements, session, raw_t, raw_x, raw_y, cursor, t_end,
                        "drift", None, p, overlays)
                covered_until = t_end
            break

    # Rest gap preceding each movement.
    prev_end: Optional[float] = None
    for i, m in enumerate(movements):
        m.index = i
        if prev_end is not None:
            m.pause_before_s = float(m.t[0] - prev_end)
        prev_end = float(m.press_t if m.press_t is not None else m.t[-1])

    return movements


def _append(
    out: list[Movement],
    session: RawSession,
    raw_t: np.ndarray,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    t_start: float,
    t_end: float,
    kind: str,
    ev: Optional[ButtonEvent],
    p: SegmentParams,
    overlays: Sequence[tuple[str, Sequence[tuple[float, float]]]] = (),
) -> None:
    """Slice the raw samples for one movement and keep it if it is big enough.

    A movement overlapping any span in `overlays` (checked in order, first
    match wins) is reclassified to that overlay's kind and drops its button
    association: the press that ends it is incidental to a drag or an
    excursion, not an aimed click, so keeping it would put click-only
    features on a row that never aimed."""
    if t_end <= t_start:
        return

    lo = int(np.searchsorted(raw_t, t_start, side="left"))
    hi = int(np.searchsorted(raw_t, t_end, side="right"))
    if hi - lo < 3:
        return

    t = raw_t[lo:hi]
    x = raw_x[lo:hi]
    y = raw_y[lo:hi]

    duration = float(t[-1] - t[0])
    displacement = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    if duration < p.min_duration_s or displacement < p.min_distance_px:
        return

    for overlay_kind, spans in overlays:
        if _overlaps_span(float(t[0]), float(t[-1]), spans):
            kind, ev = overlay_kind, None
            break

    out.append(Movement(
        kind=kind,
        activity=session.header.activity,
        session_id=session.header.session_id,
        t=t, x=x, y=y,
        button=ev.button if ev else None,
        press_t=ev.press_t if ev else None,
        release_t=ev.release_t if ev else None,
    ))
