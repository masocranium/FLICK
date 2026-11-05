"""
Display generated paths: a single self-contained HTML page, with no dependencies.

    write_view("paths.view.html", generator, start=(100, 200), target=(640, 360), n=20)

The page shows the paths on screen coordinates, replays them at real speed, plots
their speed over time, and tables what the model drew against what each path measures.

Recorded movements can be overlaid. They are aligned onto the same start and target by
rotating and scaling, so shapes can be compared directly. Their speed curves and table
rows are from the original, unscaled movement. A page with recorded movements in it is
as sensitive as the recording.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from . import kinematics
from .capture import iter_sessions
from .features import extract
from .generate import ClickGenerator, measure_path
from .segment import Movement, segment
from .view_template import PAGE_TEMPLATE

#: The same resampling and smoothing as the default feature extraction, so a curve on
#: the page is the signal `v_peak` and `t_peak_frac` were measured from.
SPEED_HZ = 200.0
SPEED_SMOOTH_S = 0.012

#: A recorded click is only a fair comparison if its distance is within this fraction
#: of the requested one. Aligning it onto the target scales it by at most this much.
REAL_DISTANCE_BAND = 0.3

_DRAWN_COLUMNS = ("duration_s", "n_submovements", "overshoot_px", "t_peak_frac")
_MEASURED_COLUMNS = ("duration_s", "v_peak", "straightness", "n_submovements",
                     "overshoot_px", "t_peak_frac")


def _num(value, digits: int = 3) -> Optional[float]:
    """A JSON-safe rounded number. JSON has no NaN, and `-0.0` would read oddly."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits) + 0.0


def _speed_curve(t, x, y) -> list[Optional[float]]:
    kin = kinematics.compute(t, x, y, hz=SPEED_HZ, smooth_sigma_s=SPEED_SMOOTH_S)
    return [_num(v, 1) for v in kin.speed]


def _entry(ident: str, kind: str, t, x, y, speed, drawn: dict, measured) -> dict:
    t = np.asarray(t, dtype=float)
    return {
        "id": ident,
        "kind": kind,
        "t": [_num(v, 4) for v in t - t[0]],
        "x": [_num(v, 1) for v in x],
        "y": [_num(v, 1) for v in y],
        "speed": speed,
        "drawn": {c: _num(drawn.get(c)) for c in _DRAWN_COLUMNS},
        "measured": {c: _num(getattr(measured, c, None)) for c in _MEASURED_COLUMNS},
    }


def select_real_movements(sessions: Iterable, distance_px: float, limit: int,
                          activity: Optional[str] = None) -> list[Movement]:
    """Up to `limit` recorded clicks nearest `distance_px`, within REAL_DISTANCE_BAND."""
    found = []
    for session in sessions:
        if activity is not None and session.header.activity != activity:
            continue
        for movement in segment(session):
            if movement.kind != "click":
                continue
            gap = abs(movement.displacement_px / distance_px - 1.0)
            if gap <= REAL_DISTANCE_BAND:
                found.append((gap, len(found), movement))
    found.sort(key=lambda item: item[:2])
    return [movement for _, _, movement in found[:limit]]


def align_to(movement: Movement, start: Sequence[float],
             target: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Rotate, scale and translate a movement so it runs from `start` to `target`."""
    z = movement.x + 1j * movement.y
    wanted = complex(target[0] - start[0], target[1] - start[1])
    have = z[-1] - z[0]
    # Multiplying by a complex ratio rotates and scales in one step.
    aligned = (z - z[0]) * (wanted / have) + complex(start[0], start[1])
    return aligned.real, aligned.imag


def build_view_data(generator: ClickGenerator, start: Sequence[float],
                    target: Sequence[float], n: int = 20,
                    real: Sequence[Movement] = (), seed: Optional[int] = None,
                    model_name: str = "") -> dict:
    """Everything the page displays, as JSON-safe data."""
    clicks = generator.generate_many(start, target, n)
    if not clicks:
        raise ValueError("Nothing to display: the start and target are the same point.")

    movements = []
    for i, click in enumerate(clicks, start=1):
        path = generator.path_for(click)
        movements.append(_entry(f"Generated {i}", "generated", path.t, path.x, path.y,
                                _speed_curve(path.t, path.x, path.y),
                                click.features, measure_path(path)))
    for i, movement in enumerate(real, start=1):
        x, y = align_to(movement, start, target)
        movements.append(_entry(f"Recorded {i}", "real", movement.t, x, y,
                                _speed_curve(movement.t, movement.x, movement.y),
                                {}, extract(movement)))
    return {
        "model": model_name or generator.model.name,
        "start": [float(start[0]), float(start[1])],
        "target": [float(target[0]), float(target[1])],
        "distance_px": _num(clicks[0].distance_px, 1),
        "seed": seed,
        "hz": SPEED_HZ,
        "movements": movements,
    }


def render_page(data: dict) -> str:
    """The page, with `data` embedded. `<` is escaped so that no string in the data,
    such as a label, can close the script element it sits in."""
    embedded = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return PAGE_TEMPLATE.replace("/*__FLICK_DATA__*/null", embedded)


def write_view(path, generator: ClickGenerator, start: Sequence[float],
               target: Sequence[float], n: int = 20, real_paths: Sequence = (),
               activity: Optional[str] = None, seed: Optional[int] = None) -> dict:
    """Generate paths, write the page, and return the data it displays."""
    distance = math.hypot(float(target[0]) - float(start[0]),
                          float(target[1]) - float(start[1]))
    real: list[Movement] = []
    if real_paths and distance > 0:
        real = select_real_movements(iter_sessions([Path(p) for p in real_paths]),
                                     distance, n, activity)
    data = build_view_data(generator, start, target, n, real, seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_page(data), encoding="utf-8")
    return data
