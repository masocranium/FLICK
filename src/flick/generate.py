"""
From a cursor position and a click target to synthetic movement data.

    generator = ClickGenerator(load_model("my.model.json"))

    click = generator.generate(start=(100, 200), target=(640, 360))   # features
    click.duration_s, click.distance_px, click.get("straightness")

    path = generator.generate_path(start=(100, 200), target=(640, 360))   # trajectory
    path.t, path.x, path.y

Both return data. Nothing here moves a cursor or sends input to anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .features import ExtractParams, FeatureRow, extract
from .models.base import MovementModel, load_model
from .path import LegFn, generate_path, wind_walk
from .schema import DISTANCE_COLUMN
from .segment import Movement

#: Guard rails on a drawn duration, in seconds. A model can extrapolate past the
#: observed range, which is right for a tail and wrong for a value a caller may use
#: as a real time span. Wide enough never to bind on a plausible draw.
MIN_DURATION_S = 0.02
MAX_DURATION_S = 8.0

Point = tuple[float, float]


@dataclass(frozen=True)
class SyntheticClick:
    """One generated movement from `start` to `target`."""

    start: Point
    target: Point
    distance_px: float
    #: Heading from start to target: `atan2(dy, dx)`, in the coordinates given.
    angle_rad: float
    #: Drawn duration, clamped to [MIN_DURATION_S, MAX_DURATION_S].
    duration_s: float
    #: Everything the model drew, by column name, unclamped.
    features: dict

    def get(self, column: str, default: float = float("nan")) -> float:
        return float(self.features.get(column, default))

    def to_row(self) -> dict:
        """A flat dict: positions first, then the features."""
        return {
            "start_x": self.start[0], "start_y": self.start[1],
            "target_x": self.target[0], "target_y": self.target[1],
            "angle_rad": self.angle_rad,
            **self.features,
            "duration_s": self.duration_s,
        }


@dataclass(frozen=True)
class SyntheticPath:
    """A generated trajectory: parallel arrays, with `t` in seconds from the start."""

    click: SyntheticClick
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def to_rows(self) -> list[dict]:
        return [{"t": float(t), "x": int(x), "y": int(y)}
                for t, x, y in zip(self.t, self.x, self.y)]


def measure_path(path: SyntheticPath, params: Optional[ExtractParams] = None) -> FeatureRow:
    """The features a path really has, measured by the same code as recordings.

    Compare this to `path.click.features` to see how much of what the model drew the
    path achieved. Only some columns are steered: see the `flick.path` docstring."""
    end = float(path.t[-1])
    movement = Movement(kind="click", activity="synthetic", session_id="synthetic",
                        t=np.asarray(path.t, dtype=float),
                        x=np.asarray(path.x, dtype=float),
                        y=np.asarray(path.y, dtype=float),
                        button="left", press_t=end)
    return extract(movement, params)


class ClickGenerator:
    """Generates movements with any `MovementModel`.

    `leg_fn` walks each leg of a trajectory, and defaults to `flick.path.wind_walk`."""

    def __init__(self, model: MovementModel,
                 rng: Optional[np.random.Generator] = None,
                 leg_fn: LegFn = wind_walk):
        self.model = model
        self._rng = rng if rng is not None else np.random.default_rng()
        self._leg_fn = leg_fn

    def generate_path(self, start: Point, target: Point) -> Optional[SyntheticPath]:
        """One trajectory from `start` to `target`. None under the same conditions
        as `generate`."""
        click = self.generate(start, target)
        return self.path_for(click) if click is not None else None

    def path_for(self, click: SyntheticClick) -> SyntheticPath:
        """A trajectory realizing an already-generated click."""
        points, delays = generate_path(
            click.start, click.target, click.duration_s,
            click.features.get("overshoot_px"),
            click.features.get("n_submovements"),
            click.features.get("t_peak_frac"),
            self._rng, leg_fn=self._leg_fn)
        t = np.concatenate(([0.0], np.cumsum(delays)))
        xy = np.asarray(points, dtype=int)
        return SyntheticPath(click=click, t=t, x=xy[:, 0], y=xy[:, 1])

    @classmethod
    def from_file(cls, path, rng: Optional[np.random.Generator] = None
                  ) -> "ClickGenerator":
        """Load a saved model. Raises if the file is missing or corrupt."""
        return cls(load_model(path), rng=rng)

    def generate(self, start: Point, target: Point) -> Optional[SyntheticClick]:
        """One movement from `start` to `target`.

        None if the two points coincide or aren't finite, since there's no movement
        to describe, or if the model has no usable `duration_s`."""
        found = self.generate_many(start, target, 1)
        return found[0] if found else None

    def generate_many(self, start: Point, target: Point,
                      n: int) -> list[SyntheticClick]:
        """`n` independent movements between the same two points."""
        dx, dy = float(target[0]) - float(start[0]), float(target[1]) - float(start[1])
        distance = math.hypot(dx, dy)
        if not math.isfinite(distance) or distance <= 0.0:
            return []

        drawn = self.model.sample(distance, n=int(n), rng=self._rng)
        angle = math.atan2(dy, dx)
        clicks = []
        for i in range(int(n)):
            features = {c: float(np.asarray(v)[i]) for c, v in drawn.items()}
            features[DISTANCE_COLUMN] = distance
            duration = features.get("duration_s", float("nan"))
            if not math.isfinite(duration):
                continue
            clicks.append(SyntheticClick(
                start=(float(start[0]), float(start[1])),
                target=(float(target[0]), float(target[1])),
                distance_px=distance,
                angle_rad=angle,
                duration_s=min(max(duration, MIN_DURATION_S), MAX_DURATION_S),
                features=features,
            ))
        return clicks
