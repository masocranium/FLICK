"""The mouse schema, feature-table IO, and the distance-conditioned sampler."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from norta_copula.copula import GaussianCopula, KDEMarginal
from norta_copula.demo import make_demo_features, write_demo_csv
from norta_copula.mouse import (
    DEFAULT_COLUMNS,
    FEATURE_BOUNDS,
    MAX_DURATION_S,
    MIN_DURATION_S,
    MovementSampler,
    fit_mouse_copula,
    read_rows,
    select_rows,
    to_matrix,
)
from norta_copula.validation import check_bounds


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

def test_default_columns_include_distance_for_conditioning():
    assert "distance_px" in DEFAULT_COLUMNS


def test_default_columns_exclude_exact_functions_of_other_columns():
    """path_length = distance/straightness and v_mean = path_length/duration, so
    including them makes the correlation matrix singular by construction."""
    assert "path_length_px" not in DEFAULT_COLUMNS
    assert "v_mean" not in DEFAULT_COLUMNS


def test_every_default_column_has_declared_physical_bounds():
    for c in DEFAULT_COLUMNS:
        assert c in FEATURE_BOUNDS, c


def test_fit_mouse_copula_applies_the_schema_bounds():
    cop = fit_mouse_copula(make_demo_features(200), rng=np.random.default_rng(0))
    duration = cop.marginals[DEFAULT_COLUMNS.index("duration_s")]
    assert isinstance(duration, KDEMarginal) and duration.transform == "log"


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def fitted():
    return fit_mouse_copula(make_demo_features(400), rng=np.random.default_rng(0))


def test_drawn_duration_tracks_the_distance_it_is_conditioned_on(fitted):
    s = MovementSampler(fitted, rng=np.random.default_rng(1))
    short = np.median([s.draw(30.0).duration_s for _ in range(60)])
    long = np.median([s.draw(800.0).duration_s for _ in range(60)])
    assert long > short * 1.5, f"short={short:.3f} long={long:.3f}"


def test_draw_returns_none_for_a_move_of_no_distance(fitted):
    s = MovementSampler(fitted, rng=np.random.default_rng(1))
    assert s.draw(0.0) is None
    assert s.draw(-5.0) is None
    assert s.draw(float("nan")) is None


def test_duration_is_clamped(fitted):
    s = MovementSampler(fitted, rng=np.random.default_rng(2))
    for d in (1.0, 50.0, 5000.0):
        for _ in range(40):
            assert MIN_DURATION_S <= s.draw(d).duration_s <= MAX_DURATION_S


def test_sampler_is_reproducible_from_a_seed(fitted):
    a = MovementSampler(fitted, rng=np.random.default_rng(42))
    b = MovementSampler(fitted, rng=np.random.default_rng(42))
    assert [a.draw(250.0).duration_s for _ in range(5)] == \
           [b.draw(250.0).duration_s for _ in range(5)]


def test_spec_carries_every_column_and_echoes_the_distance(fitted):
    spec = MovementSampler(fitted, rng=np.random.default_rng(3)).draw(300.0)
    assert set(spec.features) == set(DEFAULT_COLUMNS)
    assert spec.distance_px == 300.0
    assert spec.get("distance_px") == 300.0
    assert math.isnan(spec.get("not_a_column"))


def test_a_copula_without_distance_cannot_be_used_for_conditioning():
    gen = np.random.default_rng(0)
    data = np.column_stack([gen.lognormal(0, 0.5, 100), gen.normal(0, 1, 100)])
    cop = GaussianCopula.fit(data, ["duration_s", "max_dev_px"],
                             rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="distance_px"):
        MovementSampler(cop)


def test_sampler_loads_a_saved_model(fitted, tmp_path):
    path = fitted.save(tmp_path / "m.json")
    assert MovementSampler.from_file(path).draw(250.0) is not None
    with pytest.raises(FileNotFoundError):
        MovementSampler.from_file(tmp_path / "typo.json")


# --------------------------------------------------------------------------- #
# Feature tables
# --------------------------------------------------------------------------- #

def _jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_read_rows_jsonl_separates_the_header(tmp_path):
    p = _jsonl(tmp_path / "f.jsonl", [
        {"kind_record": "header", "version": 1, "extract_params": {"hz": 200.0}},
        {"kind_record": "row", "activity": "a", "kind": "click", "duration_s": 0.5},
    ])
    rows, header = read_rows(p)
    assert header["extract_params"] == {"hz": 200.0}
    assert rows == [{"activity": "a", "kind": "click", "duration_s": 0.5}]


def test_read_rows_rejects_an_unknown_version(tmp_path):
    p = _jsonl(tmp_path / "f.jsonl", [{"kind_record": "header", "version": 99}])
    with pytest.raises(ValueError, match="version"):
        read_rows(p)


def test_read_rows_csv_round_trips_the_demo_table(tmp_path):
    path = write_demo_csv(tmp_path / "demo.csv", n=20, seed=1)
    rows, header = read_rows(path)
    assert header == {}
    assert np.allclose(to_matrix(rows), make_demo_features(20, seed=1))


def test_select_rows_filters_only_what_it_is_given():
    rows = [{"activity": "a", "kind": "click"}, {"activity": "a", "kind": "drift"},
            {"activity": "b", "kind": "click"}]
    assert len(select_rows(rows)) == 3
    assert len(select_rows(rows, activity="a")) == 2
    assert len(select_rows(rows, activity="a", kind="click")) == 1


def test_to_matrix_drops_rows_with_missing_values():
    rows = [{"x": 1.0, "y": 2.0}, {"x": 3.0}, {"x": float("nan"), "y": 4.0},
            {"x": "5", "y": ""}, {"x": "6", "y": "7"}]
    assert to_matrix(rows, ["x", "y"]).tolist() == [[1.0, 2.0], [6.0, 7.0]]


def test_to_matrix_of_nothing_keeps_its_width():
    assert to_matrix([], ["x", "y"]).shape == (0, 2)


# --------------------------------------------------------------------------- #
# The demo data has the shapes it claims to
# --------------------------------------------------------------------------- #

def test_demo_data_is_itself_physically_possible():
    report = check_bounds(make_demo_features(2000), DEFAULT_COLUMNS, FEATURE_BOUNDS)
    # overshoot_px's dust straddles zero, which is the point of it.
    assert set(report.violations) <= {"overshoot_px"}
    assert report.nonfinite == 0


def test_demo_data_is_reproducible():
    assert np.array_equal(make_demo_features(50, seed=3), make_demo_features(50, seed=3))


def test_demo_data_exercises_every_marginal_family():
    cop = fit_mouse_copula(make_demo_features(400), rng=np.random.default_rng(0))
    kinds = {m.kind for m in cop.marginals}
    assert {"KDEMarginal", "DiscreteMarginal", "HurdleMarginal"} <= kinds
    transforms = {m.transform for m in cop.marginals if isinstance(m, KDEMarginal)}
    assert {"log", "logit", "neglog", "identity"} <= transforms
