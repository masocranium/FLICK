"""
Pointer capture: sample the pointer and write it to disk. Nothing else.

A sample is `(t, x, y, state)`: a timestamp, root-window coordinates, and a
bitmask holding the mouse buttons. Everything downstream is DERIVED from exactly
this, so the raw file is the asset. Feature definitions can change and be
re-derived for free; a recording cannot be regenerated.

ONLY THE POINTER AND ITS BUTTONS

The recorder never reads the keyboard. It samples whatever the pointer does
across the whole display, in any application, so only record yourself or people
who have agreed to it. Mouse dynamics are an identifying biometric.

TRACKING ONE WINDOW (OPTIONAL)

With `window="some title"`, samples taken while the pointer is outside that
window get `OUTSIDE_WINDOW_BIT` set. Without this, "left the window" and "stayed
inside" look identical downstream. `SessionHeader.channels` records whether this
was on, because a file without the channel cannot answer the question at all,
which is different from the answer being no.

WHY POLLING, NOT AN EVENT HOOK

Polling `XQueryPointer` needs only python-xlib and cannot miss a position the way
a dropped event can. The cost is timing quantization: a button transition is
located to within one poll interval, so at 200 Hz a press carries up to 5 ms of
error. Raise `hz` if click dwell matters to you.

`PointerSource` is a protocol, so another backend (other platforms, or exact
event timestamps) can be dropped in without touching the recorder, the segmenter
or the file format.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol, Sequence

# Bump when the on-disk schema changes in a way older extractors cannot read.
RAW_FORMAT_VERSION = 2
SUPPORTED_RAW_VERSIONS = frozenset({1, 2})

# X11 button mask bits, inlined so this module imports with no X server present.
BUTTON_BITS = {"left": 1 << 8, "middle": 1 << 9, "right": 1 << 10}

# Buttons that drag rather than click a target.
DRAG_BUTTONS = ("middle",)

CHANNEL_POINTER = "pointer"
CHANNEL_BUTTONS = "buttons"
#: Named `client_bounds` on disk so files from earlier recorders stay readable.
CHANNEL_WINDOW_BOUNDS = "client_bounds"

# Set when the pointer was NOT over the tracked window. X's own mask stops at
# bit 12, so this cannot collide with it.
OUTSIDE_WINDOW_BIT = 1 << 20

BASE_CHANNELS = (CHANNEL_POINTER, CHANNEL_BUTTONS)


def find_x_window(display, name_substring: str):
    """Depth-first search for a window whose WM_NAME contains the substring."""
    stack = [display.screen().root]
    while stack:
        win = stack.pop()
        try:
            name = win.get_wm_name()
        except Exception:
            name = None
        if name and name_substring in name:
            return win
        try:
            stack.extend(win.query_tree().children)
        except Exception:
            pass
    return None


class PointerSource(Protocol):
    """Anything that can report where the pointer is right now.

    `read()` returns `(x, y, state)`. Implementations must be cheap: the recorder
    calls this at up to 500 Hz on the same thread. An implementation may expose a
    `channels` tuple saying what it fills in, which is stamped into the header."""

    def read(self) -> tuple[int, int, int]: ...

    def close(self) -> None: ...


class XQueryPointerSource:
    """X11 source. python-xlib is imported lazily, so importing this module never
    requires a display or the optional dependency."""

    def __init__(self, display: Optional[str] = None, window: Optional[str] = None,
                 bounds_refresh_s: float = 1.0):
        try:
            from Xlib import display as xdisplay
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "python-xlib is required to record the pointer. "
                "Install it with:  pip install 'flick-mouse[record]'"
            ) from exc

        self._display = xdisplay.Display(display)
        self._root = self._display.screen().root
        self._window_name = window
        self._win = find_x_window(self._display, window) if window else None
        self._size: Optional[tuple[int, int]] = None
        self._bounds_refresh_s = bounds_refresh_s
        self._next_bounds_refresh = 0.0
        if self._win is not None:
            self._refresh_geometry()
        self.channels: tuple[str, ...] = BASE_CHANNELS + (
            (CHANNEL_WINDOW_BOUNDS,) if self._win is not None else ())

    def _refresh_geometry(self) -> None:
        """Size only: querying the pointer against the window recomputes
        window-relative coordinates fresh on every call."""
        try:
            geom = self._win.get_geometry()
            self._size = (int(geom.width), int(geom.height))
        except Exception:
            self._win = None
            self._size = None

    def read(self) -> tuple[int, int, int]:
        now = time.monotonic()
        if self._win is not None and now >= self._next_bounds_refresh:
            self._refresh_geometry()
            self._next_bounds_refresh = now + self._bounds_refresh_s

        # Querying the window itself, instead of root, costs nothing: the reply
        # carries both root-absolute and window-relative coordinates.
        queried = self._win
        if queried is not None:
            try:
                p = queried.query_pointer()
            except Exception:
                # The window went away: one re-find, so a relaunch under the same
                # title recovers rather than losing the channel for the session.
                self._win = find_x_window(self._display, self._window_name)
                if self._win is not None:
                    self._refresh_geometry()
                    self._next_bounds_refresh = now + self._bounds_refresh_s
                p = self._root.query_pointer()
                queried = None
        else:
            p = self._root.query_pointer()

        state = int(p.mask)
        if queried is not None and self._size is not None:
            w, h = self._size
            if not (0 <= p.win_x < w and 0 <= p.win_y < h):
                state |= OUTSIDE_WINDOW_BIT
        return int(p.root_x), int(p.root_y), state

    def close(self) -> None:
        try:
            self._display.close()
        except Exception:
            pass


class ReplaySource:
    """A canned source for tests. Yields a fixed list of samples, then repeats the
    last one forever, like a stationary pointer."""

    def __init__(self, samples: Sequence[tuple[int, int, int]]):
        if not samples:
            raise ValueError("ReplaySource needs at least one sample")
        self._samples = list(samples)
        self._i = 0

    def read(self) -> tuple[int, int, int]:
        s = self._samples[min(self._i, len(self._samples) - 1)]
        self._i += 1
        return s

    def close(self) -> None:
        pass


@dataclass
class SessionHeader:
    """The first line of every raw file. `activity` is the stratum label: different
    tasks are different distributions, and are fitted separately."""

    activity: str
    hz: float
    started_at: str
    version: int = RAW_FORMAT_VERSION
    origin: tuple[int, int] = (0, 0)
    note: str = ""
    session_id: str = ""
    channels: tuple[str, ...] = BASE_CHANNELS

    def has_channel(self, channel: str) -> bool:
        return channel in self.channels

    @property
    def records_window_bounds(self) -> bool:
        """Whether "the pointer never left the window" can be READ from this file.
        False means unanswerable, not "it never left"."""
        return self.has_channel(CHANNEL_WINDOW_BOUNDS)

    def to_json(self) -> dict:
        return {
            "kind": "header",
            "version": self.version,
            "activity": self.activity,
            "hz": self.hz,
            "started_at": self.started_at,
            "origin": list(self.origin),
            "note": self.note,
            "session_id": self.session_id,
            "channels": list(self.channels),
        }


@dataclass
class RawSession:
    """A whole recording in memory: the header plus four parallel lists."""

    header: SessionHeader
    t: list[float] = field(default_factory=list)   # seconds since session start
    x: list[int] = field(default_factory=list)
    y: list[int] = field(default_factory=list)
    b: list[int] = field(default_factory=list)     # state bitmask

    def __len__(self) -> int:
        return len(self.t)

    @property
    def duration_s(self) -> float:
        return (self.t[-1] - self.t[0]) if self.t else 0.0


def slugify(text: str) -> str:
    """"Photo Editing" -> "photo-editing". For filenames only; the header keeps the
    label exactly as typed, since that is what groups strata at fit time."""
    slug = "".join(c.lower() if c.isalnum() else "-" for c in text.strip())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "unlabeled"


class Recorder:
    """Polls a `PointerSource` on a fixed cadence and streams samples to disk.

    Deadlines are ABSOLUTE (`deadline += period`), not `sleep(period)`: sleeping a
    fixed period accumulates each iteration's overshoot, so a nominal 200 Hz
    recording drifts and its timestamps stop meaning what they say.

    Samples are flushed in chunks, so Ctrl-C (the normal way to stop) loses at most
    one chunk. `clock` and `sleep` are injected so tests run without wall time."""

    def __init__(
        self,
        source: PointerSource,
        activity: str,
        hz: float = 200.0,
        origin: tuple[int, int] = (0, 0),
        note: str = "",
        chunk: int = 1000,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if hz <= 0:
            raise ValueError("hz must be positive")
        self.source = source
        self.hz = float(hz)
        self.origin = origin
        self.chunk = max(1, int(chunk))
        self._clock = clock
        self._sleep = sleep

        started = datetime.now(timezone.utc)
        self.header = SessionHeader(
            activity=activity,
            hz=self.hz,
            started_at=started.isoformat(timespec="seconds"),
            origin=origin,
            note=note,
            session_id=f"{slugify(activity)}-{started.strftime('%Y%m%d-%H%M%S')}",
            channels=tuple(getattr(source, "channels", BASE_CHANNELS)),
        )
        self.samples_written = 0
        self._buf: list[tuple[float, int, int, int]] = []

    def record(
        self,
        path: Path,
        seconds: Optional[float] = None,
        stop: Optional[Callable[[], bool]] = None,
        on_chunk: Optional[Callable[[int, float], None]] = None,
    ) -> Path:
        """Record until `seconds` elapse, `stop()` is True, or KeyboardInterrupt.
        The file is valid at every chunk boundary."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        period = 1.0 / self.hz
        ox, oy = self.origin

        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(self.header.to_json()) + "\n")
            fh.flush()

            t0 = self._clock()
            deadline = t0
            try:
                while True:
                    now = self._clock()
                    if seconds is not None and (now - t0) >= seconds:
                        break
                    if stop is not None and stop():
                        break

                    x, y, mask = self.source.read()
                    self._buf.append((now - t0, x - ox, y - oy, mask))

                    if len(self._buf) >= self.chunk:
                        self._flush(fh)
                        if on_chunk is not None:
                            on_chunk(self.samples_written, now - t0)

                    deadline += period
                    lag = deadline - self._clock()
                    if lag > 0:
                        self._sleep(lag)
                    else:
                        # Fell behind. Resynchronize rather than bursting reads
                        # that would all carry near-identical timestamps.
                        deadline = self._clock()
            except KeyboardInterrupt:
                pass
            finally:
                self._flush(fh)
        return path

    def _flush(self, fh) -> None:
        if not self._buf:
            return
        fh.write(json.dumps({
            "kind": "samples",
            "t": [round(s[0], 6) for s in self._buf],
            "x": [s[1] for s in self._buf],
            "y": [s[2] for s in self._buf],
            "b": [s[3] for s in self._buf],
        }) + "\n")
        fh.flush()
        self.samples_written += len(self._buf)
        self._buf.clear()


def read_session(path: Path) -> RawSession:
    """Load one raw file. Rejects an unknown format version outright: mis-reading
    a future schema would produce plausible-looking features fitted from garbage."""
    path = Path(path)
    session: Optional[RawSession] = None

    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON") from exc

            kind = rec.get("kind")
            if kind == "header":
                version = int(rec.get("version", 0))
                if version not in SUPPORTED_RAW_VERSIONS:
                    raise ValueError(
                        f"{path}: raw format version {version}, this build reads "
                        f"{sorted(SUPPORTED_RAW_VERSIONS)}")
                origin = rec.get("origin") or [0, 0]
                session = RawSession(header=SessionHeader(
                    activity=rec["activity"],
                    hz=float(rec["hz"]),
                    started_at=rec.get("started_at", ""),
                    version=version,
                    origin=(int(origin[0]), int(origin[1])),
                    note=rec.get("note", ""),
                    session_id=rec.get("session_id", path.stem),
                    channels=tuple(rec.get("channels") or BASE_CHANNELS),
                ))
            elif kind == "samples":
                if session is None:
                    raise ValueError(f"{path}:{lineno}: samples before header")
                session.t.extend(float(v) for v in rec["t"])
                session.x.extend(int(v) for v in rec["x"])
                session.y.extend(int(v) for v in rec["y"])
                session.b.extend(int(v) for v in rec["b"])

    if session is None:
        raise ValueError(f"{path}: no header line, so not a raw pointer session")
    return session


def iter_sessions(paths: Sequence[Path]) -> Iterator[RawSession]:
    """Yield sessions from a mix of files and directories, in sorted order."""
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.glob("*.raw.jsonl")):
                yield read_session(f)
        else:
            yield read_session(p)
