"""The mouse-movement feature schema. Defined in docs/feature-schema.md."""

from __future__ import annotations

import math

#: The conditioning feature: known from the start and target before any movement.
DISTANCE_COLUMN = "distance_px"

#: Default columns for a fit.
#:
#: `path_length_px` and `v_mean` are deliberately excluded: they are exact
#: functions of columns already present (path_length = distance / straightness,
#: v_mean = path_length / duration), so a copula over them has a singular
#: correlation matrix by construction.
DEFAULT_COLUMNS: tuple[str, ...] = (
    "distance_px",
    "duration_s",
    "straightness",
    "max_dev_px",
    "max_dev_frac",
    "v_peak",
    "t_peak_frac",
    "accel_peak",
    "decel_peak",
    "jerk_rms",
    "n_submovements",
    "overshoot_px",
    "settle_s",
    "n_direction_changes",
)

#: Physical bounds per feature: facts about what the number MEANS, not properties
#: of any sample.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "distance_px": (0.0, math.inf),
    "duration_s": (0.0, math.inf),
    "path_length_px": (0.0, math.inf),
    "straightness": (0.0, 1.0),
    "max_dev_px": (-math.inf, math.inf),
    "max_dev_frac": (0.0, 1.0),
    "v_peak": (0.0, math.inf),
    "v_mean": (0.0, math.inf),
    "t_peak_frac": (0.0, 1.0),
    "accel_peak": (0.0, math.inf),
    "decel_peak": (-math.inf, 0.0),
    "jerk_rms": (0.0, math.inf),
    "n_submovements": (0.0, math.inf),
    "overshoot_px": (0.0, math.inf),
    "settle_s": (0.0, math.inf),
    "n_direction_changes": (0.0, math.inf),
    "pause_before_s": (0.0, math.inf),
    "dwell_s": (0.0, math.inf),
}
