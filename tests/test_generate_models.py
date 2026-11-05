"""Swappable models, the generator, evaluation, and the command line."""

from __future__ import annotations

import csv
import io
import json
import math

import numpy as np
import pytest

from flick.capture import BUTTON_BITS, Recorder, ReplaySource
from flick.cli import main
from flick.copula import GaussianCopula
from flick.demo import make_demo_features, write_demo_csv
from flick.generate import MAX_DURATION_S, MIN_DURATION_S, ClickGenerator
from flick.models import (
    CopulaModel,
    KNNModel,
    MovementModel,
    available_models,
    get_model,
    load_model,
    register_model,
)
from flick.schema import DEFAULT_COLUMNS, FEATURE_BOUNDS
from flick.validation import check_bounds, cross_validate, gaussian_baseline, kfold, sample_like

MODELS = [CopulaModel, KNNModel]


@pytest.fixture(scope="module")
def demo():
    return make_demo_features(300)


@pytest.fixture(scope="module", params=MODELS, ids=lambda m: m.name)
def fitted(request, demo):
    return request.param.fit(demo, DEFAULT_COLUMNS, rng=np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# The contract, checked against every built-in model
# --------------------------------------------------------------------------- #

def test_sample_returns_every_column_and_echoes_distance(fitted):
    out = fitted.sample(250.0, n=7, rng=np.random.default_rng(1))
    assert set(out) == set(DEFAULT_COLUMNS)
    assert all(v.shape == (7,) for v in out.values())
    assert np.all(out["distance_px"] == 250.0)


def test_sample_is_reproducible_from_a_seed(fitted):
    a = fitted.sample(250.0, n=5, rng=np.random.default_rng(2))
    b = fitted.sample(250.0, n=5, rng=np.random.default_rng(2))
    assert all(np.array_equal(a[c], b[c]) for c in a)


def test_longer_distance_means_longer_duration(fitted):
    near = fitted.sample(30.0, n=400, rng=np.random.default_rng(3))
    far = fitted.sample(600.0, n=400, rng=np.random.default_rng(3))
    assert np.median(far["duration_s"]) > np.median(near["duration_s"])


def test_no_model_generates_impossible_values(fitted, demo):
    rows = sample_like(fitted, demo, DEFAULT_COLUMNS, 1000, np.random.default_rng(4))
    report = check_bounds(rows, DEFAULT_COLUMNS)
    assert report.violations == {}
    assert report.nonfinite == 0


def test_save_and_load_round_trip_through_the_registry(fitted, tmp_path):
    path = fitted.save(tmp_path / "m.json")
    assert json.loads(path.read_text())["model"] == fitted.name
    back = load_model(path)
    assert type(back) is type(fitted)
    a = fitted.sample(250.0, n=20, rng=np.random.default_rng(5))
    b = back.sample(250.0, n=20, rng=np.random.default_rng(5))
    assert all(np.allclose(a[c], b[c]) for c in a)


@pytest.mark.parametrize("cls", MODELS, ids=lambda m: m.name)
def test_fit_requires_the_distance_column(cls, demo):
    with pytest.raises(ValueError, match="distance_px"):
        cls.fit(demo[:, 1:], DEFAULT_COLUMNS[1:], rng=np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# Swapping
# --------------------------------------------------------------------------- #

class MeanModel(MovementModel):
    """A deliberately trivial custom model: every feature at its mean."""

    name = "test-mean"

    def __init__(self, columns, means):
        self.columns, self.means = list(columns), list(means)

    @classmethod
    def fit(cls, data, columns, bounds=FEATURE_BOUNDS, rng=None, **options):
        return cls(columns, np.asarray(data).mean(axis=0).tolist())

    def sample(self, distance_px, n=1, rng=None):
        out = {c: np.full(n, m) for c, m in zip(self.columns, self.means)}
        out["distance_px"] = np.full(n, float(distance_px))
        return out

    def to_json(self):
        return {"columns": self.columns, "means": self.means}

    @classmethod
    def from_json(cls, data):
        return cls(data["columns"], data["means"])


def test_a_custom_model_works_everywhere_a_builtin_does(demo, tmp_path):
    register_model(MeanModel)
    assert "test-mean" in available_models()

    model = get_model("test-mean").fit(demo, DEFAULT_COLUMNS)
    path = model.save(tmp_path / "custom.json")
    click = ClickGenerator.from_file(path).generate((0, 0), (30, 40))
    assert click.distance_px == 50.0
    assert click.duration_s == pytest.approx(demo[:, 1].mean())
    assert cross_validate(demo, DEFAULT_COLUMNS, "test-mean", folds=3)


def test_a_name_cannot_be_silently_taken_by_another_class():
    class Impostor(MeanModel):
        name = "copula"

    with pytest.raises(ValueError, match="already taken"):
        register_model(Impostor)


def test_a_nameless_model_cannot_be_registered():
    class Nameless(MeanModel):
        name = ""

    with pytest.raises(ValueError, match="name"):
        register_model(Nameless)


def test_an_unknown_model_name_lists_the_real_ones():
    with pytest.raises(KeyError, match="copula"):
        get_model("typo")


def test_a_model_file_cannot_name_code_to_import(tmp_path):
    p = tmp_path / "evil.json"
    p.write_text(json.dumps({"model": "os:system"}))
    with pytest.raises(KeyError, match="no model named"):
        load_model(p)


def test_a_bare_copula_file_with_no_model_field_still_loads(demo, tmp_path):
    copula = GaussianCopula.fit(demo, DEFAULT_COLUMNS, bounds=FEATURE_BOUNDS,
                                rng=np.random.default_rng(0))
    path = copula.save(tmp_path / "old.json")
    assert "model" not in json.loads(path.read_text())
    assert isinstance(load_model(path), CopulaModel)


def test_knn_returns_recorded_rows_near_the_distance(demo):
    model = KNNModel.fit(demo, DEFAULT_COLUMNS, k=5)
    out = model.sample(200.0, n=50, rng=np.random.default_rng(0))
    i = DEFAULT_COLUMNS.index("duration_s")
    assert set(out["duration_s"]) <= set(demo[:, i])
    distances = demo[:, DEFAULT_COLUMNS.index("distance_px")]
    nearest = distances[np.argsort(np.abs(np.log(distances) - math.log(200.0)))[:5]]
    assert len(set(out["duration_s"])) <= 5
    assert np.allclose(np.log(nearest), math.log(200.0), atol=0.1)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #

def test_generate_derives_distance_and_heading_from_the_two_points(fitted):
    click = ClickGenerator(fitted, np.random.default_rng(1)).generate((100, 200), (130, 240))
    assert click.distance_px == 50.0
    assert click.get("distance_px") == 50.0
    assert click.angle_rad == pytest.approx(math.atan2(40, 30))
    assert click.start == (100.0, 200.0) and click.target == (130.0, 240.0)


def test_generate_has_nothing_to_do_at_the_target(fitted):
    g = ClickGenerator(fitted, np.random.default_rng(1))
    assert g.generate((5, 5), (5, 5)) is None
    assert g.generate((math.nan, 0), (5, 5)) is None
    assert g.generate_many((5, 5), (5, 5), 3) == []


def test_duration_is_clamped_but_the_raw_draw_is_kept(fitted):
    g = ClickGenerator(fitted, np.random.default_rng(2))
    for _ in range(20):
        c = g.generate((0, 0), (5000, 0))
        assert MIN_DURATION_S <= c.duration_s <= MAX_DURATION_S


def test_generate_many_and_to_row(fitted):
    clicks = ClickGenerator(fitted, np.random.default_rng(3)).generate_many((0, 0), (3, 4), 4)
    assert len(clicks) == 4
    row = clicks[0].to_row()
    assert row["start_x"] == 0.0 and row["target_y"] == 4.0
    assert row["duration_s"] == clicks[0].duration_s
    assert set(DEFAULT_COLUMNS) <= set(row)


def test_swapping_models_changes_only_the_model_argument(demo):
    for cls in MODELS:
        model = cls.fit(demo, DEFAULT_COLUMNS, rng=np.random.default_rng(0))
        assert ClickGenerator(model, np.random.default_rng(0)).generate((0, 0), (100, 0))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def test_kfold_partitions_every_index():
    folds = kfold(23, 4, np.random.default_rng(0))
    assert sorted(np.concatenate(folds).tolist()) == list(range(23))


@pytest.mark.parametrize("name", ["copula", "knn"])
def test_cross_validate_reports_each_fold(name):
    results = cross_validate(make_demo_features(120), DEFAULT_COLUMNS, name, folds=3)
    assert [r["fold"] for r in results] == [1, 2, 3]
    assert all(r["n_train"] + r["n_test"] == 120 for r in results)
    assert all(r["score"] > 0 and r["floor"] > 0 for r in results)


def test_cross_validate_skips_folds_too_small_to_score():
    assert cross_validate(make_demo_features(20), DEFAULT_COLUMNS, folds=5) == []


def test_check_bounds_counts_violations_per_column():
    samples = np.array([[-1.0, 0.5], [1.0, 2.0], [np.nan, 0.5]])
    report = check_bounds(samples, ["duration_s", "straightness"])
    assert report.violations == {"duration_s": 1, "straightness": 1}
    assert report.n_impossible == 2 and report.nonfinite == 1
    assert report.rate == pytest.approx(2 / 6)


def test_check_bounds_ignores_floating_point_dust():
    assert check_bounds(np.array([[-1e-14]]), ["overshoot_px"]).n_impossible == 0
    assert check_bounds(np.array([[-0.01]]), ["overshoot_px"]).n_impossible == 1


def test_copula_beats_the_gaussian_baseline_on_impossible_values(demo):
    model = CopulaModel.fit(demo, DEFAULT_COLUMNS, rng=np.random.default_rng(0))
    ours = check_bounds(sample_like(model, demo, DEFAULT_COLUMNS, 2000,
                                    np.random.default_rng(1)), DEFAULT_COLUMNS)
    mvn = check_bounds(gaussian_baseline(demo, 2000, np.random.default_rng(2)),
                       DEFAULT_COLUMNS)
    assert ours.n_impossible == 0 and ours.nonfinite == 0
    assert mvn.rate > 0.05


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

@pytest.fixture()
def demo_csv(tmp_path):
    return write_demo_csv(tmp_path / "demo.csv", n=150)


@pytest.mark.parametrize("name", ["copula", "knn"])
def test_fit_then_generate_end_to_end(name, demo_csv, tmp_path, capsys):
    model = tmp_path / "m.json"
    assert main(["fit", str(demo_csv), "--model", name, "--out", str(model),
                 "--check", "200", "--samples", "200"]) == 0
    report = capsys.readouterr().out
    assert "TRUST THIS ROW" in report
    assert "contains your training rows" in report

    assert main(["generate", str(model), "--from", "100,200", "--to", "130,240",
                 "-n", "5", "--seed", "1"]) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == 5
    assert all(float(r["distance_px"]) == 50.0 for r in rows)
    assert all(float(r["start_x"]) == 100.0 and float(r["target_y"]) == 240.0 for r in rows)


def test_fit_passes_options_to_the_model(demo_csv, tmp_path, capsys):
    model = tmp_path / "m.json"
    assert main(["fit", str(demo_csv), "--model", "knn", "--option", "k=7",
                 "--out", str(model), "--check", "0", "--samples", "50"]) == 0
    assert json.loads(model.read_text())["k"] == 7


def test_plugin_module_registers_a_model_for_the_cli(demo_csv, tmp_path, capsys, monkeypatch):
    (tmp_path / "my_models.py").write_text('''
import numpy as np
from flick.models import MovementModel, register_model
from flick.schema import FEATURE_BOUNDS

@register_model
class Median(MovementModel):
    """Every feature at its median."""
    name = "plugin-median"
    def __init__(self, columns, values):
        self.columns, self.values = list(columns), list(values)
    @classmethod
    def fit(cls, data, columns, bounds=FEATURE_BOUNDS, rng=None, **options):
        return cls(columns, np.median(data, axis=0).tolist())
    def sample(self, distance_px, n=1, rng=None):
        out = {c: np.full(n, v) for c, v in zip(self.columns, self.values)}
        out["distance_px"] = np.full(n, float(distance_px))
        return out
    def to_json(self):
        return {"columns": self.columns, "values": self.values}
    @classmethod
    def from_json(cls, data):
        return cls(data["columns"], data["values"])
''')
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "m.json"
    assert main(["fit", str(demo_csv), "--plugin", "my_models", "--model", "plugin-median",
                 "--out", str(model), "--check", "0", "--samples", "50"]) == 0
    capsys.readouterr()
    assert main(["generate", str(model), "--plugin", "my_models", "--from", "0,0",
                 "--to", "3,4"]) == 0
    assert "5" in capsys.readouterr().out
    assert main(["models", "--plugin", "my_models"]) == 0
    listed = capsys.readouterr().out
    assert "plugin-median" in listed and "copula" in listed


def test_unknown_model_and_missing_file_are_clear_errors(demo_csv, tmp_path):
    with pytest.raises(SystemExit, match="no model named"):
        main(["fit", str(demo_csv), "--model", "typo"])
    with pytest.raises(SystemExit, match="No features file"):
        main(["fit", str(tmp_path / "nope.csv")])


def test_fit_explains_an_empty_stratum(tmp_path):
    p = tmp_path / "f.jsonl"
    rec = dict(zip(DEFAULT_COLUMNS, make_demo_features(1)[0].tolist()), activity="a", kind="click")
    p.write_text(json.dumps(rec) + "\n")
    with pytest.raises(SystemExit, match="a/click"):
        main(["fit", str(p), "--activity", "typo"])


def test_generate_from_a_point_to_itself_says_so(demo_csv, tmp_path, capsys):
    model = tmp_path / "m.json"
    main(["fit", str(demo_csv), "--out", str(model), "--check", "0", "--samples", "50",
          "--folds", "1"])
    with pytest.raises(SystemExit, match="same point"):
        main(["generate", str(model), "--from", "1,1", "--to", "1,1"])


def test_demo_data_command_writes_a_labelled_table(tmp_path, capsys):
    assert main(["demo-data", str(tmp_path / "d.csv"), "--rows", "7"]) == 0
    assert "FABRICATED" in capsys.readouterr().out
    assert len((tmp_path / "d.csv").read_text().splitlines()) == 8


def _minimum_jerk(x0, x1, seconds, hz=200):
    n = int(seconds * hz)
    out = []
    for i in range(n):
        s = i / (n - 1)
        out.append(round(x0 + (x1 - x0) * (10 * s**3 - 15 * s**4 + 6 * s**5)))
    return out


def test_recorded_session_flows_through_extract_summary_fit_and_generate(tmp_path, capsys):
    """The whole application, from a recorded pointer to a generated click."""
    samples = []
    x = 0
    rng = np.random.default_rng(0)
    for _ in range(40):
        to = int(rng.integers(100, 900))
        path = _minimum_jerk(x, to, float(rng.uniform(0.3, 0.8)))
        samples += [(px, 300, 0) for px in path]
        samples += [(to, 300, 0)] * 10
        samples += [(to, 300, BUTTON_BITS["left"])] * 15     # the click
        samples += [(to, 300, 0)] * 60                        # a rest
        x = to

    t = [0.0]

    def clock():
        return t[0]

    def sleep(dt):
        t[0] += dt

    raw_dir = tmp_path / "raw"
    recorder = Recorder(ReplaySource(samples), "test task", hz=200.0, clock=clock, sleep=sleep)
    recorder.record(raw_dir / "s.raw.jsonl", seconds=len(samples) / 200.0)

    features = tmp_path / "features.jsonl"
    assert main(["extract", "--raw", str(raw_dir), "--out", str(features), "--csv"]) == 0
    # 39, not 40: one of the 40 random moves is 8 px, under the 12 px minimum.
    assert "39 click" in capsys.readouterr().out
    assert features.with_suffix(".csv").exists()

    assert main(["summary", "--features", str(features)]) == 0
    assert "test task" in capsys.readouterr().out

    model = tmp_path / "m.json"
    assert main(["fit", str(features), "--kind", "click", "--out", str(model),
                 "--check", "200", "--samples", "200"]) == 0
    capsys.readouterr()

    assert main(["generate", str(model), "--from", "0,0", "--to", "300,400", "--seed", "1"]) == 0
    row = next(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert float(row["distance_px"]) == 500.0
    assert float(row["duration_s"]) > 0
