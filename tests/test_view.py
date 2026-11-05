"""The page `flick view` writes. The data is tested here; the drawing runs in a browser."""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

from flick.capture import RawSession, SessionHeader
from flick.cli import main
from flick.demo import make_demo_features
from flick.generate import ClickGenerator
from flick.models import CopulaModel, KNNModel
from flick.schema import DEFAULT_COLUMNS
from flick.segment import Movement
from flick.view import (
    REAL_DISTANCE_BAND,
    SPEED_HZ,
    align_to,
    build_view_data,
    render_page,
    select_real_movements,
    write_view,
)

LEFT = 1 << 8


@pytest.fixture(scope="module")
def model():
    return CopulaModel.fit(make_demo_features(300), DEFAULT_COLUMNS, rng=np.random.default_rng(0))


def generator(model, seed=1):
    return ClickGenerator(model, np.random.default_rng(seed))


def embedded(html):
    return json.loads(re.search(
        r'<script id="flick-data" type="application/json">(.*?)</script>', html, re.S).group(1))


def session_with_moves(distances, activity="task"):
    """A session of left clicks, each a smooth move of the given length."""
    x, t, xs, ys, bs, ts = 100.0, 0.0, [], [], [], []

    def push(px, py, b):
        nonlocal t
        ts.append(t)
        xs.append(int(round(px)))
        ys.append(int(round(py)))
        bs.append(b)
        t += 1.0 / 200.0

    for d in distances:
        n = 100
        for i in range(n):
            s = i / (n - 1)
            push(x + d * (10 * s**3 - 15 * s**4 + 6 * s**5), 300.0, 0)
        x += d
        for _ in range(10):
            push(x, 300.0, 0)
        for _ in range(12):
            push(x, 300.0, LEFT)
        for _ in range(80):
            push(x, 300.0, 0)
    return RawSession(SessionHeader(activity, 200.0, ""), ts, xs, ys, bs)


# --------------------------------------------------------------------------- #
# Generated paths
# --------------------------------------------------------------------------- #

def test_view_data_has_every_path_from_start_to_target(model):
    data = build_view_data(generator(model), (100, 200), (640, 360), n=5, seed=7)
    assert len(data["movements"]) == 5
    assert data["seed"] == 7 and data["model"] == "copula"
    for m in data["movements"]:
        assert m["kind"] == "generated"
        assert (m["x"][0], m["y"][0]) == (100, 200)
        assert (m["x"][-1], m["y"][-1]) == (640, 360)
        assert m["t"][0] == 0.0
        assert len(m["t"]) == len(m["x"]) == len(m["y"])
        assert all(b >= a for a, b in zip(m["t"], m["t"][1:]))


def test_speed_curve_is_on_the_grid_the_page_assumes(model):
    data = build_view_data(generator(model), (0, 0), (300, 400), n=3)
    assert data["hz"] == SPEED_HZ
    for m in data["movements"]:
        duration = m["t"][-1]
        assert abs((len(m["speed"]) - 1) / SPEED_HZ - duration) <= 1.5 / SPEED_HZ
        assert max(m["speed"]) == pytest.approx(m["measured"]["v_peak"], rel=0.01)


def test_table_values_come_from_measuring_the_path_not_the_draw(model):
    data = build_view_data(generator(model), (0, 0), (300, 400), n=8)
    for m in data["movements"]:
        assert m["measured"]["duration_s"] == pytest.approx(m["drawn"]["duration_s"], rel=0.03)
        assert set(m["drawn"]) == {"duration_s", "n_submovements", "overshoot_px", "t_peak_frac"}


def test_data_is_strict_json_with_no_nan(model):
    data = build_view_data(generator(model), (0, 0), (300, 400), n=3)
    json.dumps(data, allow_nan=False)


def test_view_data_is_reproducible_from_a_seed(model):
    a = build_view_data(generator(model, 5), (0, 0), (300, 400), n=3)
    b = build_view_data(generator(model, 5), (0, 0), (300, 400), n=3)
    assert a == b


def test_any_model_can_be_displayed():
    knn = KNNModel.fit(make_demo_features(100), DEFAULT_COLUMNS)
    data = build_view_data(ClickGenerator(knn, np.random.default_rng(0)), (0, 0), (100, 0), n=2)
    assert data["model"] == "knn" and len(data["movements"]) == 2


def test_same_point_is_a_clear_error(model):
    with pytest.raises(ValueError, match="same point"):
        build_view_data(generator(model), (5, 5), (5, 5))


# --------------------------------------------------------------------------- #
# Recorded overlay
# --------------------------------------------------------------------------- #

def test_alignment_lands_on_the_start_and_target():
    m = Movement("click", "a", "s", t=np.arange(4.0), x=np.array([0.0, 10, 20, 30]),
                 y=np.array([0.0, 5, -5, 0]))
    x, y = align_to(m, (100, 100), (100, 160))
    assert (x[0], y[0]) == pytest.approx((100, 100))
    assert (x[-1], y[-1]) == pytest.approx((100, 160))


def test_alignment_keeps_the_shape():
    m = Movement("click", "a", "s", t=np.arange(3.0), x=np.array([0.0, 50, 100]),
                 y=np.array([0.0, 10, 0]))
    x, y = align_to(m, (0, 0), (0, 200))         # rotated 90 degrees, scaled by 2
    assert math.hypot(x[1] - 0, y[1] - 100) == pytest.approx(20.0)


def test_selection_prefers_the_nearest_distance_and_respects_the_band():
    session = session_with_moves([100, 150, 200, 400])
    picked = select_real_movements([session], 190.0, limit=2)
    assert [round(m.displacement_px, -1) for m in picked] == [200, 150]

    only_band = select_real_movements([session], 400.0, limit=10)
    assert all(abs(m.displacement_px / 400 - 1) <= REAL_DISTANCE_BAND for m in only_band)
    assert len(only_band) == 1


def test_selection_filters_by_activity():
    session = session_with_moves([200], activity="other")
    assert select_real_movements([session], 200.0, 5, activity="task") == []
    assert len(select_real_movements([session], 200.0, 5, activity="other")) == 1


def test_overlay_is_aligned_but_keeps_the_original_speed(model):
    session = session_with_moves([200])
    real = select_real_movements([session], 200.0, 5)
    data = build_view_data(generator(model), (0, 0), (0, 220), n=1, real=real)
    entry = [m for m in data["movements"] if m["kind"] == "real"][0]
    assert entry["id"] == "Recorded 1"
    assert (entry["x"][0], entry["y"][0]) == (0, 0)
    assert (entry["x"][-1], entry["y"][-1]) == pytest.approx((0, 220), abs=0.2)
    assert entry["measured"]["duration_s"] > 0
    # Peak speed is the recording's, not inflated by the 1.1x alignment scale.
    assert max(entry["speed"]) == pytest.approx(entry["measured"]["v_peak"], rel=0.01)
    assert all(v is None for v in entry["drawn"].values())


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #

def test_page_embeds_its_data_and_loads_nothing_external(model):
    data = build_view_data(generator(model), (0, 0), (300, 400), n=2)
    html = render_page(data)
    assert embedded(html) == data
    assert "/*__FLICK_DATA__*/" not in html
    assert not re.search(r'<script[^>]+src=|<link\b|@import|url\(', html)


def test_page_lets_the_hidden_attribute_win_over_flex_labels(model):
    """Regression: `label { display: inline-flex }` overrode `hidden`, so a page with no
    recordings still offered a "Show recorded movements" checkbox."""
    html = render_page(build_view_data(generator(model), (0, 0), (300, 400), n=1))
    assert "[hidden] { display: none !important; }" in html
    assert 'id="real-toggle" hidden' in html


def test_a_hostile_label_cannot_break_out_of_the_page(model):
    data = build_view_data(generator(model), (0, 0), (300, 400), n=1)
    data["model"] = '</script><img src=x onerror=alert(1)>'
    html = render_page(data)
    assert "<img src=x" not in html
    assert embedded(html)["model"] == data["model"]
    assert "innerHTML" not in html


def test_write_view_creates_parent_directories(model, tmp_path):
    path = tmp_path / "deep" / "er" / "p.view.html"
    write_view(path, generator(model), (0, 0), (300, 400), n=1)
    assert path.exists()


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

@pytest.fixture()
def saved_model(model, tmp_path):
    return model.save(tmp_path / "m.model.json")


def test_view_command_writes_a_page(saved_model, tmp_path, capsys):
    out = tmp_path / "p.view.html"
    assert main(["view", str(saved_model), "--from", "100,200", "--to", "640,360", "-n", "3",
                 "--seed", "1", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "3 generated paths" in printed
    assert "recorded" not in printed
    data = embedded(out.read_text())
    assert len(data["movements"]) == 3 and data["seed"] == 1


def _write_raw(session, path):
    h = session.header
    path.write_text(json.dumps(h.to_json()) + "\n" + json.dumps(
        {"kind": "samples", "t": session.t, "x": session.x, "y": session.y, "b": session.b}) + "\n")


def test_view_command_overlays_recordings_and_warns(saved_model, tmp_path, capsys):
    raw = tmp_path / "s.raw.jsonl"
    _write_raw(session_with_moves([300, 500]), raw)
    out = tmp_path / "p.view.html"
    assert main(["view", str(saved_model), "--from", "0,0", "--to", "300,400", "-n", "2",
                 "--real", str(raw), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "1 recorded movements" in printed
    assert "contains recorded movements" in printed
    assert sum(m["kind"] == "real" for m in embedded(out.read_text())["movements"]) == 1


def test_view_command_says_when_no_recording_is_comparable(saved_model, tmp_path, capsys):
    raw = tmp_path / "s.raw.jsonl"
    _write_raw(session_with_moves([50]), raw)
    out = tmp_path / "p.view.html"
    assert main(["view", str(saved_model), "--from", "0,0", "--to", "300,400", "-n", "1",
                 "--real", str(raw), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "no recorded clicks within 30%" in printed
    assert "contains recorded" not in printed


def test_view_command_from_a_point_to_itself_says_so(saved_model, tmp_path):
    with pytest.raises(SystemExit, match="same point"):
        main(["view", str(saved_model), "--from", "1,1", "--to", "1,1",
              "--out", str(tmp_path / "p.html")])
