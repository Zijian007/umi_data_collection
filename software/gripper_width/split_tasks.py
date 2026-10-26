#!/usr/bin/env python3
"""split_tasks.py -- cut a long recording into one training part per task from gripper openings.  (from paper/cvpr candy_pilot/split_tasks.py)

Brought onto dev_Zijian with software/gripper_width/.  Needs WORK/pilot_analysis.json from `pilot.py analyse --long`.

Recording (负责人 2026-10-16): rest -> look around with the gripper closed -> at the start pose an empty open-close
(the marker) -> task (open, grasp the candy bag, turn, open over the paper bag, close, come back) -> hold at the start
pose -> marker -> next task ... about 6 tasks per video.  "Holding" is never still (hand tremor, 0.03-0.06 m/s), so no
rule here asks for zero motion.

Every opening of the gripper (>= EVENT_OPEN_MM, openings closer than MERGE_GAP_S merged) is described by
  peak opening, duration, how far the gripper moves during it (max displacement), and the angle between the finger
  direction during it and the finger direction at the first marker of the video ("heading").
Measured on the 24 single-task recordings of 2026-10-15 (paper/cvpr candy_pilot, review 20261015_candy_batch2):
  marker   : peak 110-116 mm, 0.5-0.93 s, moves 1.1-5.5 cm
  pre-grasp: heading 6-34 deg from the marker; moves 3.8-69 cm; never fully open AND short AND still at once
  release  : heading 54-114 deg from the marker (the gripper faces the table, not the shelf)
Rules:
  marker candidate  = peak >= MARKER_PEAK_MM and duration <= MARKER_MAX_S and moves <= MARKER_MAX_MOVE_M
                      and heading < FACING_SHELF_DEG
  release           = heading >= FACING_SHELF_DEG
Tasks: the openings facing the table split the video into cycles; in each cycle the task's marker is the LAST marker
candidate that is followed by another opening (the pre-grasp) before the release -- so an extra open-close at the end
of the previous task (it also looks like a marker) is skipped.  Training part = from the marker's close to 0.3 s before
the next marker candidate / extra opening after the release, with the final hold trimmed to KEEP_END_S (speed below
MOVE_MPS, 0.2 s smoothing).  The first opening of the video must be a marker (the gripper stays closed while looking
around); anything else is flagged.

Input: WORK/pilot_analysis.json from `pilot.py analyse --long` (chosen run per recording), the chosen run's
traj_tcp.txt, WORK/width/<tail>_rows.csv.  Output: WORK/tasks.json with one entry per task, key "<tail>_t<k>",
in the same shape pilot.py writes for single-task recordings (tail, use_run, training_part.rows, events, flags), so
candy_to_lerobot.py can read either.

    python split_tasks.py WORK --tails 030 031 ... [--out tasks.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

FPS = 60000 / 1001 / 2
EVENT_OPEN_MM = 10.0
MERGE_GAP_S = 0.25
MARKER_PEAK_MM = 90.0
MARKER_MAX_S = 1.5          # 040 had a 1.23 s marker (v1 used 1.2)
MARKER_NEAR_M = 0.6         # markers are made at the start pose: within this of the video's first marker. All 139 fully-open,
                            # short, still openings of 006-050: true markers <= 0.50 m (047 changed stance), releases >= 0.67 m
MARKER_MAX_MOVE_M = 0.08
FACING_SHELF_DEG = 45.0
MOVE_MPS = 0.08
KEEP_END_S = 1.0
CUT_BEFORE_S = 0.3


def read_tum(p: Path):
    a = np.array([[float(x) for x in l.split()] for l in open(p) if l.strip() and not l.startswith("#")])
    return a[:, 1:4], a[:, 4:8]


def width_rows(p: Path) -> np.ndarray:
    return np.array([float(r["width_mm"]) if r["width_mm"] != "" else np.nan for r in csv.DictReader(open(p))])


def open_spans(w: np.ndarray, thr: float) -> list[list[int]]:
    a = np.r_[False, np.nan_to_num(w, nan=0.0) > thr, False]
    e = np.flatnonzero(np.diff(a.astype(int)))
    out: list[list[int]] = []
    for s, t in zip(e[::2], e[1::2]):
        if out and s - out[-1][1] < MERGE_GAP_S * FPS:
            out[-1][1] = int(t)
        else:
            out.append([int(s), int(t)])
    return out


def describe(spans, w, p, z) -> list[dict]:
    ops = []
    for a, b in spans:
        seg = p[a:b + 1]
        ops.append({"rows": [a, b], "s": [round(a / FPS, 2), round(b / FPS, 2)], "dur_s": round((b - a) / FPS, 2),
                    "peak_mm": round(float(np.nanmax(w[a:b])), 1),
                    "move_m": round(float(np.linalg.norm(seg - seg[0], axis=1).max()), 3),
                    "z_mid": z[(a + b) // 2]})
    return ops


def angle_deg(u, v) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def split(w: np.ndarray, p: np.ndarray, q: np.ndarray) -> dict:
    """Tasks = the stretches between consecutive markers (v2, 2026-10-16 night).

    v1 told the release apart from the marker by heading alone (>= 45 deg from the first marker); in the long videos of
    10-16 the releases were often only 30-45 deg off and some markers took 1.23 s, so 7 of 15 videos lost tasks.  Now
    only the markers are found -- full open-close, short, the gripper barely moving, near the start area of the video
    -- and every task runs from one marker to the next.  The release (the opening farthest from the task's marker) is
    only used for the check sheets and flags.
    """
    from scipy.spatial.transform import Rotation as Rot

    n = min(len(w), len(p))
    w, p, q = w[:n], p[:n], q[:n]
    z = Rot.from_quat(q).apply([0.0, 0.0, 1.0])
    ops = describe(open_spans(w, EVENT_OPEN_MM), w, p, z)
    out = {"openings": len(ops), "tasks": [], "flags": []}
    if not ops:
        out["flags"].append("no_opening")
        return out
    shape_ok = lambda o: o["peak_mm"] >= MARKER_PEAK_MM and o["dur_s"] <= MARKER_MAX_S and o["move_m"] <= MARKER_MAX_MOVE_M
    first = ops[0]
    if not shape_ok(first):
        out["flags"].append("first_opening_is_not_marker_shaped")
    pos = lambda o: p[(o["rows"][0] + o["rows"][1]) // 2]
    p0, z0 = pos(first), first["z_mid"]
    z0h = np.r_[z0[:2], 0.0]
    for o in ops:
        zh = np.r_[o["z_mid"][:2], 0.0]
        o["heading_deg"] = round(angle_deg(zh / (np.linalg.norm(zh) + 1e-12), z0h / (np.linalg.norm(z0h) + 1e-12)), 1)
        o["from_first_m"] = round(float(np.linalg.norm(pos(o) - p0)), 3)
        o["marker_like"] = bool(shape_ok(o) and o["from_first_m"] <= MARKER_NEAR_M and o["heading_deg"] < FACING_SHELF_DEG)
    marks = [i for i, o in enumerate(ops) if o["marker_like"]]
    if not marks or marks[0] != 0:
        out["flags"].append("video_does_not_start_with_a_marker")
    # two markers with nothing between them: the earlier is an extra open-close (e.g. at the end of the previous task)
    keep, extras = [], []
    for a, b in zip(marks, marks[1:] + [None]):
        if b is not None and b == a + 1:
            extras.append(a)
        else:
            keep.append(a)
    v = np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1)] * FPS
    k = max(1, int(round(0.2 * FPS)))
    v = np.convolve(v, np.ones(k) / k, mode="same")
    for t, mi in enumerate(keep):
        mk = ops[mi]
        nxt_mark = keep[t + 1] if t + 1 < len(keep) else None
        inner = ops[mi + 1:(nxt_mark if nxt_mark is not None else len(ops))]
        inner_x = [o for j, o in enumerate(ops) if mi < j < (nxt_mark if nxt_mark is not None else len(ops)) and j in extras]
        inner = [o for o in inner if not o["marker_like"]]
        flags = []
        s0 = mk["rows"][1]
        # boundary: the first extra open-close or the next marker, whichever comes first
        cands = [o["rows"][0] for o in inner_x] + ([ops[nxt_mark]["rows"][0]] if nxt_mark is not None else [n])
        boundary = min(cands)
        if not inner:
            # a marker with nothing after it before the next marker / the end: an extra open-close (e.g. at the very end
            # of the video) -- not a task
            out.setdefault("trailing_or_empty_markers", []).append({k_: mk[k_] for k_ in ("s", "peak_mm", "move_m", "heading_deg", "from_first_m")})
            continue
        else:
            dist = [float(np.linalg.norm(pos(o) - pos(mk))) for o in inner]
            ri = int(np.argmax(dist))
            rel, pre = inner[ri], inner[:ri]
            after = inner[ri + 1:]
            if after:
                flags.append(f"{len(after)}_openings_after_release")
            if not pre and rel["dur_s"] < 2.0:
                flags.append("no_open_before_release")
            if len(pre) > 1:
                flags.append(f"regrasp_{len(pre)}_opens_before_release")
            if max(dist) < 0.4:
                flags.append(f"release_only_{max(dist):.2f}m_from_marker")
        e_cap = max(s0 + 1, boundary - int(round(CUT_BEFORE_S * FPS)))
        tail_from = rel["rows"][1] if rel is not None else s0
        moving = np.flatnonzero(v[tail_from:e_cap] > MOVE_MPS)
        last = tail_from + (int(moving[-1]) if len(moving) else 0)
        e_end = min(e_cap, last + int(round(KEEP_END_S * FPS)) + 1, n)
        if np.nanmax(w[max(s0, e_end - 5):e_end]) > 3.0:
            flags.append("not_closed_at_end")
        ev = []
        if rel is not None:
            ev = [{**{k_: o[k_] for k_ in ("rows", "s", "dur_s", "peak_mm", "move_m", "heading_deg")}, "role": "before_release"} for o in pre]
            ev.append({**{k_: rel[k_] for k_ in ("rows", "s", "dur_s", "peak_mm", "move_m", "heading_deg")}, "role": "release"})
            ev += [{**{k_: o[k_] for k_ in ("rows", "s", "dur_s", "peak_mm", "move_m", "heading_deg")}, "role": "after_release"} for o in inner[inner.index(rel) + 1:]]
        far_row = int(s0 + np.argmax(np.linalg.norm(p[s0:e_end] - pos(mk), axis=1)))
        out["tasks"].append({"farthest_row": far_row, "farthest_m": round(float(np.linalg.norm(p[far_row] - pos(mk))), 3),
                             "training_part": {"rows": [int(s0), int(e_end)], "s": [round(s0 / FPS, 2), round(e_end / FPS, 2)],
                                               "duration_s": round((e_end - s0) / FPS, 2)},
                             "marker": {k_: mk[k_] for k_ in ("rows", "s", "dur_s", "peak_mm", "move_m", "heading_deg", "from_first_m")},
                             "events": {"opens": ev},
                             "skipped_extra_open_close": [{k_: o[k_] for k_ in ("s", "peak_mm", "move_m", "heading_deg")} for o in inner_x],
                             "flags": flags})
    out["markers"] = len(keep)
    out["extras_skipped"] = len(extras)
    return out


def task_sheets(work: Path, tasks: dict, raw_dirs: list, per_image: int = 12, tile: int = 240) -> int:
    """One row per task: 0.3 s after the last pre-grasp opening closes (the bag should be in the gripper), the middle
    of the release, 0.4 s after the release closes (the bag should be in the paper bag).  WORK/task_check_N.jpg."""
    import av
    import cv2
    by_tail: dict = {}
    for key, t in tasks.items():
        by_tail.setdefault(t["tail"], []).append((key, t))
    rows = []
    for tail in sorted(by_tail):
        want = {}
        for key, t in by_tail[tail]:
            ops = t["events"]["opens"]
            pre = [o for o in ops if o["role"] == "before_release"]
            rel = [o for o in ops if o["role"] == "release"]
            if pre:
                want[2 * (pre[-1]["rows"][1] + int(0.3 * FPS))] = (key, 0, "grasp+0.3s")
            far = t.get("farthest_row")
            if far is not None:   # the gripper farthest from the task's marker = over the paper bag
                want[2 * far] = (key, 1, "farthest (over bag)")
                want[2 * (far + int(0.6 * FPS))] = (key, 2, "farthest+0.6s")
            elif rel:
                want[2 * ((rel[-1]["rows"][0] + rel[-1]["rows"][1]) // 2)] = (key, 1, f"release {rel[-1]['peak_mm']:.0f}mm")
                want[2 * (rel[-1]["rows"][1] + int(0.4 * FPS))] = (key, 2, "release+0.4s")
        hits = [f for d in raw_dirs for f in Path(d).glob(f"VID_*_{tail}.insv")]
        tiles: dict = {}
        with av.open(str(hits[0])) as c:
            st = c.streams.video[0]; st.thread_type = "AUTO"
            last = max(want) if want else -1
            for k, fr in enumerate(c.decode(st)):
                if k in want:
                    key, col, lab = want[k]
                    img = cv2.resize(fr.to_ndarray(format="bgr24")[:, 1920:][240:1680, 240:1680], (tile, tile))
                    cv2.putText(img, f"{key} {lab}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                    tiles[(key, col)] = img
                if k >= last:
                    break
        for key, _ in by_tail[tail]:
            r = [tiles.get((key, c_), np.zeros((tile, tile, 3), np.uint8)) for c_ in range(3)]
            rows.append(np.hstack(r))
    n = 0
    for i in range(0, len(rows), per_image):
        n += 1
        cv2.imwrite(str(work / f"task_check_{n}.jpg"), np.vstack(rows[i:i + per_image]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", type=Path)
    ap.add_argument("--tails", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sheets-raw", nargs="*", default=None, help="raw folders: also write WORK/task_check_N.jpg (one row per task)")
    a = ap.parse_args(argv)
    res = json.loads((a.work / "pilot_analysis.json").read_text())["recordings"]
    tasks, per_rec = {}, {}
    for tail in a.tails:
        rec = res[tail]
        run = next(p for p in (a.work / tail / "runs" / rec["use_run"]["run"]).iterdir() if p.is_dir())
        p, q = read_tum(run / "traj_tcp.txt")
        w = width_rows(a.work / "width" / f"{tail}_rows.csv")
        sp = split(w, p, q)
        per_rec[tail] = {k: v for k, v in sp.items() if k != "tasks"} | {"n_tasks": len(sp["tasks"])}
        for k, t in enumerate(sp["tasks"], 1):
            tasks[f"{tail}_t{k}"] = {"tail": tail, "use_run": rec["use_run"], **t}
        print(tail, json.dumps({"tasks": len(sp["tasks"]), "flags": sp["flags"], "extras": sp.get("extras_skipped"),
                                "parts": [(t["training_part"]["s"], t["flags"]) for t in sp["tasks"]]}), flush=True)
    out = a.out or (a.work / "tasks.json")
    out.write_text(json.dumps({"made": "split_tasks.py", "rules": {k: globals()[k] for k in (
        "EVENT_OPEN_MM", "MERGE_GAP_S", "MARKER_PEAK_MM", "MARKER_MAX_S", "MARKER_MAX_MOVE_M", "MARKER_NEAR_M", "FACING_SHELF_DEG",
        "MOVE_MPS", "KEEP_END_S", "CUT_BEFORE_S")}, "recordings": per_rec, "tasks": tasks}, indent=1, default=float))
    if a.sheets_raw:
        print("task check sheets", task_sheets(a.work, tasks, a.sheets_raw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
