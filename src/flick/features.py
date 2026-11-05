"""
One movement in, one row of numbers out, plus feature-file IO.

WHAT A RECORDING CANNOT SEE

A recording shows where someone pressed, not what they aimed at, so there is no
landing-error column, and target width (and so a true Fitts index of difficulty)
is unavailable too. What IS observable is the terminal-approach structure the
same precision produces: `overshoot_px`, `settle_s`, `n_direction_changes` and
`n_submovements`.

PARAMETER SENSITIVITY

`jerk_rms`, `accel_peak` and `n_submovements` depend on `smooth_sigma_s` as much
as on the hand. `ExtractParams` is written into every features file for this
reason. Do not pool rows across differing parameters.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from . import kinematics
from .schema import DEFAULT_COLUMNS
from .segment import Movement

FEATURES_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ExtractParams:
    """Everything that changes what a feature number means."""

    hz: float = 200.0
    smooth_sigma_s: float = 0.012

    # A velocity bump counts as a submovement if it rises this fraction of
    # the movement's peak speed above the surrounding valley. 0.10 is low
    # enough to catch a real correction, high enough to ignore smoothing
    # ripple.
    submove_prominence_frac: float = 0.10
    # Corrective submovements in the literature run ~100 ms; 40 ms is a
    # refractory window well inside that, so two genuinely separate
    # corrections stay separate while one broad bump is not double-counted.
    submove_min_separation_s: float = 0.040

    # "Close enough to be aiming rather than travelling" — sets settle_s.
    settle_radius_px: float = 15.0

    # Along-axis velocity must exceed this fraction of peak speed to count as
    # a real direction, so noise around zero does not inflate the reversal
    # count.
    direction_deadband_frac: float = 0.05

    def to_json(self) -> dict:
        return asdict(self)


# The numeric columns, in a fixed order. This is the list a copula fit will
# select from — kept explicit (rather than "every float on the row") so that
# adding a diagnostic field to FeatureRow cannot silently change the model's
# dimensionality.
FEATURE_COLUMNS: tuple[str, ...] = (
    "duration_s",
    "distance_px",
    "path_length_px",
    "straightness",
    "max_dev_px",
    "max_dev_frac",
    "v_peak",
    "v_mean",
    "t_peak_frac",
    "accel_peak",
    "decel_peak",
    "jerk_rms",
    "n_submovements",
    "overshoot_px",
    "settle_s",
    "n_direction_changes",
    "pause_before_s",
    "dwell_s",
)

# Columns that only mean something for a movement that ends in a click.
CLICK_ONLY_COLUMNS: frozenset[str] = frozenset(
    {"overshoot_px", "settle_s", "n_direction_changes", "dwell_s"}
)


@dataclass
class FeatureRow:
    """One movement's feature vector plus the context needed to stratify it."""

    # --- stratification / context (not distribution dimensions) ---
    activity: str
    session_id: str
    kind: str
    index: int
    t_session_s: float
    button: Optional[str] = None
    start_x: float = float("nan")
    start_y: float = float("nan")
    end_x: float = float("nan")
    end_y: float = float("nan")
    angle_rad: float = float("nan")
    n_samples: int = 0

    # --- the joint distribution's dimensions ---
    duration_s: float = float("nan")
    distance_px: float = float("nan")
    path_length_px: float = float("nan")
    straightness: float = float("nan")
    max_dev_px: float = float("nan")
    max_dev_frac: float = float("nan")
    v_peak: float = float("nan")
    v_mean: float = float("nan")
    t_peak_frac: float = float("nan")
    accel_peak: float = float("nan")
    decel_peak: float = float("nan")
    jerk_rms: float = float("nan")
    n_submovements: float = float("nan")
    overshoot_px: float = float("nan")
    settle_s: float = float("nan")
    n_direction_changes: float = float("nan")
    pause_before_s: float = float("nan")
    dwell_s: float = float("nan")

    def to_json(self) -> dict:
        return {"kind_record": "row", **asdict(self)}

    def vector(self, columns: Iterable[str] = FEATURE_COLUMNS) -> np.ndarray:
        return np.array([getattr(self, c) for c in columns], dtype=float)


def extract(movement: Movement, params: Optional[ExtractParams] = None) -> FeatureRow:
    """Compute one feature row. See module docstring for what is and is not
    observable from a human recording."""
    p = params or ExtractParams()
    is_click = movement.kind == "click"

    kin = kinematics.compute(
        movement.t, movement.x, movement.y, hz=p.hz, smooth_sigma_s=p.smooth_sigma_s
    )

    row = FeatureRow(
        activity=movement.activity,
        session_id=movement.session_id,
        kind=movement.kind,
        index=movement.index,
        t_session_s=float(movement.t[0]),
        button=movement.button,
        n_samples=int(movement.t.size),
        start_x=float(movement.x[0]),
        start_y=float(movement.y[0]),
        end_x=float(movement.x[-1]),
        end_y=float(movement.y[-1]),
        pause_before_s=float(movement.pause_before_s),
        dwell_s=float(movement.dwell_s) if is_click else float("nan"),
    )

    if len(kin) < 3:
        return row

    x, y, speed = kin.x, kin.y, kin.speed
    dx, dy = float(x[-1] - x[0]), float(y[-1] - y[0])
    distance = float(np.hypot(dx, dy))

    steps = np.hypot(np.diff(x), np.diff(y))
    path_length = float(steps.sum())

    row.duration_s = kin.duration_s
    row.distance_px = distance
    row.path_length_px = path_length
    # Move efficiency: exactly 1.0 for a perfectly straight path.
    row.straightness = distance / path_length if path_length > 1e-9 else float("nan")
    row.angle_rad = float(np.arctan2(dy, dx))

    row.v_peak = float(speed.max())
    row.v_mean = path_length / kin.duration_s if kin.duration_s > 1e-9 else float("nan")
    # Normalized time of peak velocity.
    row.t_peak_frac = float(np.argmax(speed) / max(1, len(kin) - 1))

    row.accel_peak = float(kin.accel.max())
    row.decel_peak = float(kin.accel.min())
    row.jerk_rms = float(np.sqrt(np.mean(kin.jerk ** 2)))

    if distance > 1e-6:
        ux, uy = dx / distance, dy / distance
        nx, ny = -uy, ux
        rel_x, rel_y = x - x[0], y - y[0]

        # Signed perpendicular deviation: the arc's amplitude AND handedness.
        # Sign matters — a hand tends to bow consistently one way, which an
        # unsigned magnitude would average out of existence.
        dev = rel_x * nx + rel_y * ny
        k = int(np.argmax(np.abs(dev)))
        row.max_dev_px = float(dev[k])
        row.max_dev_frac = float(k / max(1, len(kin) - 1))

        along = rel_x * ux + rel_y * uy
        if is_click:
            # How far past the eventual click point the cursor traveled.
            row.overshoot_px = float(along.max() - distance)

            v_along = np.gradient(along, kin.dt)
            deadband = p.direction_deadband_frac * row.v_peak
            signs = np.sign(np.where(np.abs(v_along) < deadband, 0.0, v_along))
            signs = signs[signs != 0]
            if signs.size:
                row.n_direction_changes = float(np.count_nonzero(np.diff(signs) != 0))
            else:
                row.n_direction_changes = 0.0
    else:
        row.max_dev_px = 0.0
        row.max_dev_frac = float("nan")

    if is_click:
        # Homing phase: from first entry into the settle radius to the press.
        to_end = np.hypot(x - x[-1], y - y[-1])
        inside = np.flatnonzero(to_end <= p.settle_radius_px)
        if inside.size:
            row.settle_s = float(kin.t[-1] - kin.t[int(inside[0])])
        else:
            row.settle_s = 0.0

    prominence = p.submove_prominence_frac * row.v_peak
    separation = max(1, int(round(p.submove_min_separation_s * p.hz)))
    peaks = kinematics.find_peaks(
        speed,
        min_prominence=prominence,
        min_height=0.05 * row.v_peak,
        min_separation=separation,
    )
    # A clean ballistic movement has exactly one velocity peak and therefore
    # zero submovements; each additional peak is one corrective adjustment.
    row.n_submovements = float(max(0, len(peaks) - 1))

    return row


def extract_all(
    movements: Iterable[Movement], params: Optional[ExtractParams] = None
) -> list[FeatureRow]:
    p = params or ExtractParams()
    return [extract(m, p) for m in movements]


# --------------------------------------------------------------------------- #
# Feature-file IO
# --------------------------------------------------------------------------- #

def write_rows(
    path: Path,
    rows: Iterable[FeatureRow],
    params: ExtractParams,
    segment_params: Optional[dict] = None,
) -> int:
    """Write a features JSONL: one header line, then one line per movement.

    The header carries both parameter sets, because a `jerk_rms` extracted at
    6 ms smoothing and one extracted at 12 ms are not the same quantity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "kind_record": "header",
            "version": FEATURES_FORMAT_VERSION,
            "extract_params": params.to_json(),
            "segment_params": segment_params or {},
            "feature_columns": list(FEATURE_COLUMNS),
        }) + "\n")
        for row in rows:
            fh.write(json.dumps(row.to_json()) + "\n")
            n += 1
    return n


def read_rows(path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a feature table. Returns (rows, header).

    `.csv` is read with its first line as column names and has no header
    record. Anything else is read as JSONL, with an optional header record."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh)), {}

    rows: list[dict[str, Any]] = []
    header: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind_record") == "header":
                version = int(rec.get("version", 0))
                if version != FEATURES_FORMAT_VERSION:
                    raise ValueError(
                        f"{path}: features format version {version}, this build "
                        f"reads {FEATURES_FORMAT_VERSION}")
                header = rec
                continue
            rec.pop("kind_record", None)
            rows.append(rec)
    return rows, header


def select_rows(rows: Iterable[dict], activity: Optional[str] = None,
                kind: Optional[str] = None) -> list[dict]:
    """Rows matching a stratum. `None` means do not filter on that field."""
    return [r for r in rows
            if (activity is None or r.get("activity") == activity)
            and (kind is None or r.get("kind") == kind)]


def _cell(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def to_matrix(rows: Iterable[dict],
              columns: Sequence[str] = DEFAULT_COLUMNS) -> np.ndarray:
    """Rows -> a finite-only 2D float array over `columns`.

    A row with a missing or non-finite value in any selected column is dropped,
    so select columns deliberately: asking for one only some movements have
    silently discards all the others."""
    cols = list(columns)
    data = np.array([[_cell(r.get(c)) for c in cols] for r in rows], dtype=float)
    if data.size == 0:
        return data.reshape(0, len(cols))
    return data[np.isfinite(data).all(axis=1)]


def write_csv(path: Path, rows: Iterable[FeatureRow]) -> int:
    """Flat CSV of every field, for eyeballing in a spreadsheet or pandas."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return 0

    fields = list(FeatureRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    return len(rows)
