"""Held-out scoring, bounds checks, and the command line end to end."""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest

from norta_copula.cli import main
from norta_copula.copula import GaussianCopula
from norta_copula.demo import make_demo_features, write_demo_csv
from norta_copula.mouse import DEFAULT_COLUMNS, FEATURE_BOUNDS
from norta_copula.validation import (
    check_bounds,
    cross_validate,
    gaussian_baseline,
    kfold,
)


def test_kfold_partitions_every_index():
    folds = kfold(23, 4, np.random.default_rng(0))
    assert sorted(np.concatenate(folds).tolist()) == list(range(23))


def test_cross_validate_reports_each_fold():
    data = make_demo_features(120)
    results = cross_validate(data, DEFAULT_COLUMNS, folds=3, bounds=FEATURE_BOUNDS)
    assert [r["fold"] for r in results] == [1, 2, 3]
    assert all(r["n_train"] + r["n_test"] == 120 for r in results)
    assert all(r["copula"] > 0 and r["floor"] > 0 for r in results)


def test_cross_validate_skips_folds_too_small_to_score():
    assert cross_validate(make_demo_features(20), DEFAULT_COLUMNS, folds=5,
                          bounds=FEATURE_BOUNDS) == []


def test_check_bounds_counts_violations_per_column():
    samples = np.array([[-1.0, 0.5], [1.0, 2.0], [np.nan, 0.5]])
    report = check_bounds(samples, ["duration_s", "straightness"], FEATURE_BOUNDS)
    assert report.violations == {"duration_s": 1, "straightness": 1}
    assert report.n_impossible == 2
    assert report.nonfinite == 1
    assert report.rate == pytest.approx(2 / 6)


def test_check_bounds_treats_an_unnamed_column_as_unbounded():
    assert check_bounds(np.array([[-1e9]]), ["unknown"], {}).n_impossible == 0


def test_copula_beats_the_gaussian_baseline_on_impossible_values():
    """The headline claim, measured rather than asserted in prose."""
    data = make_demo_features(400)
    cop = GaussianCopula.fit(data, DEFAULT_COLUMNS, bounds=FEATURE_BOUNDS,
                             rng=np.random.default_rng(0))
    ours = check_bounds(cop.sample(20000, np.random.default_rng(1)),
                        DEFAULT_COLUMNS, FEATURE_BOUNDS)
    mvn = check_bounds(gaussian_baseline(data, 20000, np.random.default_rng(2)),
                       DEFAULT_COLUMNS, FEATURE_BOUNDS)
    assert ours.n_impossible == 0 and ours.nonfinite == 0
    assert mvn.rate > 0.05


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

@pytest.fixture()
def demo_csv(tmp_path):
    return write_demo_csv(tmp_path / "demo.csv", n=200)


def test_fit_then_sample_end_to_end(demo_csv, tmp_path, capsys):
    model = tmp_path / "m.json"
    assert main(["fit", str(demo_csv), "--out", str(model), "--check", "2000",
                 "--samples", "500", "--condition", "distance_px=250"]) == 0
    report = capsys.readouterr().out
    assert "TRUST THIS ROW" in report
    assert "copula               : 0 impossible values, 0 non-finite" in report
    assert "contains your training rows" in report
    assert json.loads(model.read_text())["n_rows"] == 200

    assert main(["sample", str(model), "-n", "5", "--seed", "1",
                 "--given", "distance_px=250"]) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == 5
    assert all(float(r["distance_px"]) == 250.0 for r in rows)
    assert all(float(r["duration_s"]) > 0 for r in rows)


def test_sample_without_given_draws_from_the_joint(demo_csv, tmp_path, capsys):
    model = tmp_path / "m.json"
    main(["fit", str(demo_csv), "--out", str(model), "--check", "0", "--samples", "100"])
    capsys.readouterr()
    assert main(["sample", str(model), "-n", "3", "--seed", "1"]) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len({r["distance_px"] for r in rows}) == 3


def test_fit_filters_to_a_stratum_and_explains_an_empty_one(tmp_path, capsys):
    data = make_demo_features(60)
    p = tmp_path / "f.jsonl"
    with p.open("w") as fh:
        fh.write(json.dumps({"kind_record": "header", "version": 1,
                             "extract_params": {"hz": 200.0}}) + "\n")
        for i, row in enumerate(data):
            rec = dict(zip(DEFAULT_COLUMNS, row.tolist()))
            rec.update(activity="a" if i % 2 else "b", kind="click")
            fh.write(json.dumps(rec) + "\n")

    model = tmp_path / "m.json"
    assert main(["fit", str(p), "--activity", "a", "--kind", "click", "--out",
                 str(model), "--check", "0", "--samples", "100"]) == 0
    saved = json.loads(model.read_text())
    assert saved["n_rows"] == 30
    assert saved["activity"] == "a"
    assert saved["extract_params"] == {"hz": 200.0}

    with pytest.raises(SystemExit, match="a/click"):
        main(["fit", str(p), "--activity", "typo"])


def test_fit_says_when_there_is_too_little_for_cross_validation(tmp_path, capsys):
    path = write_demo_csv(tmp_path / "small.csv", n=10)
    assert main(["fit", str(path), "--check", "0", "--samples", "100"]) == 0
    assert "IN-SAMPLE" in capsys.readouterr().out


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="No features file"):
        main(["fit", str(tmp_path / "nope.csv")])


def test_demo_data_command_writes_a_labelled_table(tmp_path, capsys):
    assert main(["demo-data", str(tmp_path / "d.csv"), "--rows", "7"]) == 0
    assert "FABRICATED" in capsys.readouterr().out
    assert len((tmp_path / "d.csv").read_text().splitlines()) == 8
