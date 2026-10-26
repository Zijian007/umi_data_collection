"""Thresholds that turn a width curve into a marker, a grasp, and a release."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pilot = load("pilot")
split_tasks = load("split_tasks")
width = load("width_4k60")


def test_width_clamps_closed_and_extrapolates_past_open():
    anchors = [{"x": 277.35, "mm": 0.0}, {"x": 543.02, "mm": 112.0}]
    d = np.array([200.0, 277.35, 410.185, 600.0])
    w, top = width.width_mm(d, anchors)
    assert w[0] == 0.0 and w[1] == 0.0
    assert abs(w[2] - 56.0) < 0.05
    assert w[3] > 112.0 and bool(top[3]) and not bool(top[2])


def test_marker_is_the_first_open_close_after_a_closed_scan():
    w = np.zeros(80)
    w[40:52] = 110.0
    m = pilot.marker(w)
    assert m["found"] and m["scan_closed_ok"]
    assert m["lift_row"] == 40 and m["close_row"] == 52
    assert m["peak_mm"] == 110.0


def test_scan_that_opens_is_flagged():
    w = np.zeros(80)
    w[10] = 20.0
    w[40:52] = 110.0
    m = pilot.marker(w)
    assert m["found"] and not m["scan_closed_ok"]
    assert m["scan_max_open_mm"] == 20.0


def test_openings_closer_than_the_merge_gap_are_one_event():
    w = np.zeros(40)
    w[5:8] = 20.0
    w[10:12] = 15.0
    spans = pilot.open_spans(w, pilot.EVENT_OPEN_MM)
    assert spans == [(5, 12)]


def test_long_recording_splits_marker_grasp_and_release():
    n = 400
    w = np.zeros(n)
    p = np.zeros((n, 3))
    # Finger axis has to have a horizontal component; straight up has no heading and is not a marker.
    q = np.tile([0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)], (n, 1))
    w[30:45] = 112.0                       # marker: full, short, still, at the start
    w[100:130] = 40.0                      # grasp, closer to the marker
    p[100:130] = (0.35, 0.0, 0.0)
    w[200:240] = 60.0                      # release, farthest from the marker
    p[200:240] = (1.1, 0.0, 0.0)
    sp = split_tasks.split(w, p, q)
    assert sp["flags"] == []
    assert len(sp["tasks"]) == 1
    task = sp["tasks"][0]
    roles = [o["role"] for o in task["events"]["opens"]]
    assert roles == ["before_release", "release"]
    assert task["events"]["opens"][1]["peak_mm"] == 60.0
    assert task["training_part"]["rows"][0] == 45
    assert task["flags"] == []
