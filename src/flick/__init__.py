"""FLICK: Feature-Level Imitation of Cursor Kinematics.

Record mouse movements, fit a swappable model of their features given distance, and
generate synthetic movement features for a start point and click target.
"""

from .capture import PointerSource, Recorder, ReplaySource, XQueryPointerSource, read_session
from .copula import GaussianCopula
from .demo import make_demo_features, write_demo_csv
from .features import (
    ExtractParams,
    FeatureRow,
    extract,
    extract_all,
    read_rows,
    select_rows,
    to_matrix,
    write_rows,
)
from .generate import ClickGenerator, SyntheticClick, SyntheticPath, measure_path
from .metrics import Standardizer, energy_distance, feature_gaps, noise_floor, wasserstein1
from .path import build_waypoints, generate_path, time_schedule, wind_walk
from .models import (
    CopulaModel,
    KNNModel,
    MovementModel,
    available_models,
    get_model,
    load_model,
    register_model,
)
from .schema import DEFAULT_COLUMNS, DISTANCE_COLUMN, FEATURE_BOUNDS
from .segment import Movement, SegmentParams, segment
from .validation import (
    BoundsReport,
    check_bounds,
    cross_validate,
    gaussian_baseline,
    sample_like,
)

from .view import build_view_data, render_page, write_view

__version__ = "0.2.0"

__all__ = [
    "BoundsReport", "ClickGenerator", "CopulaModel", "DEFAULT_COLUMNS",
    "DISTANCE_COLUMN", "ExtractParams", "FEATURE_BOUNDS", "FeatureRow",
    "GaussianCopula", "KNNModel", "Movement", "MovementModel", "PointerSource",
    "Recorder", "ReplaySource", "SegmentParams", "Standardizer", "SyntheticClick",
    "XQueryPointerSource", "available_models", "check_bounds", "cross_validate",
    "energy_distance", "extract", "extract_all", "feature_gaps", "gaussian_baseline",
    "get_model", "load_model", "make_demo_features", "noise_floor", "read_rows",
    "read_session", "register_model", "sample_like", "segment", "select_rows",
    "to_matrix", "wasserstein1", "write_demo_csv", "write_rows",
    "SyntheticPath", "build_waypoints", "generate_path", "measure_path",
    "time_schedule", "wind_walk", "build_view_data", "render_page", "write_view",
]
