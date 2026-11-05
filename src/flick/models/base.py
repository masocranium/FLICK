"""
The contract a swappable model has to meet, and the registry that finds one.

A model predicts a movement's features given its distance. That is all the rest
of FLICK knows about it: the generator, the CLI and the evaluation code only use
the methods on `MovementModel`. See docs/models.md for how to write one.

WHY A SAVED MODEL CANNOT NAME ARBITRARY CODE

A model file records a registered name, never an import path. Loading a file only
ever looks that name up. If loading imported whatever the file said, opening a
model someone sent you could run code of their choosing.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from importlib import metadata
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from ..schema import DISTANCE_COLUMN, FEATURE_BOUNDS

#: Installed packages can publish models under this entry-point group.
ENTRY_POINT_GROUP = "flick.models"

#: A file with no `model` field predates the field, and is a saved copula.
LEGACY_MODEL_NAME = "copula"


class MovementModel(ABC):
    """Predicts movement features conditional on distance.

    Subclasses set `name` and `columns`, and implement `fit`, `sample`, `to_json`
    and `from_json`. Decorate the class with `@register_model` to make it available
    by name."""

    #: The registered name, stored in saved files and passed to `flick fit --model`.
    name: str = ""
    #: Every feature this model produces. Must include `distance_px`.
    columns: list[str]

    @classmethod
    @abstractmethod
    def fit(cls, data: np.ndarray, columns: Sequence[str],
            bounds: Optional[Mapping[str, tuple[float, float]]] = FEATURE_BOUNDS,
            rng: Optional[np.random.Generator] = None,
            **options) -> "MovementModel":
        """Fit to a rows-by-columns array. `options` are model-specific."""

    @abstractmethod
    def sample(self, distance_px: float, n: int = 1,
               rng: Optional[np.random.Generator] = None) -> dict[str, np.ndarray]:
        """Draw `n` feature vectors given a distance.

        Returns column name to an array of length `n`, for every name in `columns`,
        with `distance_px` equal to the value given."""

    @abstractmethod
    def to_json(self) -> dict:
        """A JSON-serializable representation. `save` adds the `model` field."""

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict) -> "MovementModel":
        """Rebuild a model from `to_json`'s output."""

    def describe(self) -> list[tuple[str, str]]:
        """(label, description) pairs for reports. Optional."""
        return []

    def sample_rows(self, distances: Sequence[float],
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """One row per distance, as a len(distances)-by-len(columns) array."""
        rng = rng if rng is not None else np.random.default_rng()
        out = np.empty((len(distances), len(self.columns)))
        for i, distance in enumerate(distances):
            drawn = self.sample(float(distance), n=1, rng=rng)
            out[i] = [float(drawn[c][0]) for c in self.columns]
        return out

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"model": self.name, **self.to_json()}
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        return path


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, type[MovementModel]] = {}


def register_model(cls: type[MovementModel]) -> type[MovementModel]:
    """Class decorator making a model available by its `name`."""
    if not cls.name:
        raise ValueError(f"{cls.__qualname__} must set a non-empty `name`")
    existing = _REGISTRY.get(cls.name)
    if existing is not None and (existing.__module__, existing.__qualname__) != (
            cls.__module__, cls.__qualname__):
        raise ValueError(
            f"model name {cls.name!r} is already taken by "
            f"{existing.__module__}.{existing.__qualname__}")
    _REGISTRY[cls.name] = cls
    return cls


def _load_entry_points() -> None:
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        if ep.name not in _REGISTRY:
            _REGISTRY[ep.name] = ep.load()


def available_models() -> list[str]:
    """Names of every registered and installed model."""
    _load_entry_points()
    return sorted(_REGISTRY)


def get_model(name: str) -> type[MovementModel]:
    """The model class registered under `name`."""
    if name not in _REGISTRY:
        _load_entry_points()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no model named {name!r}. "
                       f"Available: {', '.join(available_models())}") from None


def load_model(path) -> MovementModel:
    """Load a saved model of any registered type."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return get_model(data.get("model", LEGACY_MODEL_NAME)).from_json(data)


def require_distance(columns: Sequence[str]) -> None:
    """Raise unless `columns` has the conditioning column. For use in `fit`."""
    if DISTANCE_COLUMN not in columns:
        raise ValueError(
            f"columns must include {DISTANCE_COLUMN!r}, the value models are "
            f"conditioned on. Got: {list(columns)}")
