#!/usr/bin/env python3
"""width_4k60.py -- gripper opening from the two fingertip ArUco tags in a 4K60 recording, through the 4K60 width
calibration of 2026-09-04 (recording 029, candidate_4k60_g029).  (2026-10-14, brought onto dev_Zijian from paper/cvpr gripper_width/)

Detection is the September tool (experiments/reviews/data/20260904_x3_4k60_calibration/tools/gripper_detect_4k60.py)
re-expressed for OpenCV >= 4.7, whose aruco API replaced Dictionary_get / DetectorParameters_create with ArucoDetector;
the parameters are the same: DICT_4X4_50, ID0 / ID1, right half of the packed 3840 x 1920 frame (working lens, native
1920 x 1920 pixels), adaptiveThreshWinSizeMax 53, minMarkerPerimeterRate 0.005, sub-pixel corner refinement, every frame.

Width: the tag-centre distance d (px) through the piecewise-linear mapping in the September plateaus.json
(7 anchors 277.35 px = 0 mm ... 543.02 px = 112 mm).  Below the 0 mm anchor the width is clamped to 0 (fingers touching
or pressing); above the 112 mm anchor it is extrapolated and flagged.  Frames without both tags get no width here.

Per trajectory row (every 2nd video frame, row k = frame 2k, as in the trajectory tool and the training-data builder)
the width is the median of the frames 2k-1, 2k, 2k+1 that have both tags; rows with none are filled by linear
interpolation if the gap is at most --max-gap-rows rows, else left empty (no width is made up for long gaps).

    python software/gripper_width/width_4k60.py INSV --calib plateaus.json --out OUT_DIR [--max-gap-rows 5]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def detector():
    import cv2
    aruco = cv2.aruco
    prm = aruco.DetectorParameters()
    prm.adaptiveThreshWinSizeMax = 53
    prm.minMarkerPerimeterRate = 0.005
    prm.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50), prm)


def width_mm(d: np.ndarray, anchors: list) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear px -> mm; clamp at 0 below the closed anchor; linear extrapolation above the top anchor (flagged)."""
    x = np.array([a["x"] for a in anchors]); y = np.array([a["mm"] for a in anchors])
    w = np.interp(d, x, y)
    top = d > x[-1]
    w[top] = y[-1] + (d[top] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return np.maximum(w, 0.0), top


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("insv", type=Path)
    ap.add_argument("--calib", type=Path, required=True, help="plateaus.json of the 4K60 width calibration")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-gap-rows", type=int, default=5)
    a = ap.parse_args()
    import av
    import cv2

    cal = json.loads(a.calib.read_text())["mapping"]
    det = detector()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with av.open(str(a.insv)) as c:
        s = c.streams.video[0]; s.thread_type = "AUTO"; tb = float(s.time_base)
        W, H = s.codec_context.width, s.codec_context.height
        assert W == 2 * H, f"not a packed dual-fisheye frame: {W}x{H}"
        half = W // 2
        for k, fr in enumerate(c.decode(s)):
            g = fr.to_ndarray(format="gray")[:, half:]
            corners, ids, _ = det.detectMarkers(g)
            r = {"frame": k, "t_s": round(float(fr.pts * tb), 5), "det0": 0, "det1": 0, "d_center_px": np.nan,
                 "side0_px": np.nan, "side1_px": np.nan, "u0": np.nan, "v0": np.nan, "u1": np.nan, "v1": np.nan}
            if ids is not None:
                cs = {int(i): corners[j][0] for j, i in enumerate(ids.flatten()) if int(i) in (0, 1)}
                for i in cs:
                    cc = cs[i].mean(0)
                    r[f"det{i}"] = 1; r[f"u{i}"], r[f"v{i}"] = float(cc[0]), float(cc[1])
                    r[f"side{i}_px"] = float(np.mean([np.linalg.norm(cs[i][m] - cs[i][(m + 1) % 4]) for m in range(4)]))
                if 0 in cs and 1 in cs:
                    r["d_center_px"] = float(np.linalg.norm(cs[0].mean(0) - cs[1].mean(0)))
            rows.append(r)
            if k % 600 == 0:
                print(f"frame {k} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    n = len(rows)
    d = np.array([r["d_center_px"] for r in rows])
    both = np.isfinite(d)
    w = np.full(n, np.nan); top = np.zeros(n, bool)
    w[both], top[both] = width_mm(d[both], cal["anchors"])
    with open(a.out / "frames.csv", "w", newline="") as f:
        cols = list(rows[0].keys()) + ["width_mm", "above_top_anchor"]
        wr = csv.writer(f); wr.writerow(cols)
        for r, wi, ti in zip(rows, w, top):
            wr.writerow([("" if isinstance(r[c], float) and np.isnan(r[c]) else (round(r[c], 3) if isinstance(r[c], float) else r[c])) for c in rows[0]]
                        + ["" if np.isnan(wi) else round(float(wi), 3), int(ti)])

    # per trajectory row (row k = frame 2k)
    nr = (n + 1) // 2
    rw = np.full(nr, np.nan); src = np.array(["none"] * nr, dtype=object)
    for k in range(nr):
        v = [w[f] for f in (2 * k - 1, 2 * k, 2 * k + 1) if 0 <= f < n and np.isfinite(w[f])]
        if v:
            rw[k] = float(np.median(v)); src[k] = "tags"
    have = np.flatnonzero(np.isfinite(rw))
    for i0, i1 in zip(have[:-1], have[1:]):
        if 1 < i1 - i0 <= a.max_gap_rows + 1:
            for k in range(i0 + 1, i1):
                rw[k] = rw[i0] + (rw[i1] - rw[i0]) * (k - i0) / (i1 - i0); src[k] = "interp"
    with open(a.out / "rows.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["row", "video_frame", "width_mm", "source"])
        for k in range(nr):
            wr.writerow([k, 2 * k, "" if np.isnan(rw[k]) else round(float(rw[k]), 3), src[k]])

    def runs(mask):
        out, i = [], 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j < len(mask) and mask[j]:
                    j += 1
                out.append(j - i); i = j
            else:
                i += 1
        return out

    gaps = runs(~np.isfinite(rw))
    side = np.array([r["side0_px"] for r in rows] + [r["side1_px"] for r in rows])
    summ = {"insv": a.insv.name, "insv_sha256": sha256(a.insv), "calib": str(a.calib), "calib_sha256": sha256(a.calib),
            "calib_status": cal["status"], "opencv": cv2.__version__, "frames": n,
            "det0_frac": round(float(np.mean([r["det0"] for r in rows])), 3), "det1_frac": round(float(np.mean([r["det1"] for r in rows])), 3),
            "both_frac": round(float(both.mean()), 3),
            "tag_side_px_p5_50_95": [round(float(v), 1) for v in np.nanpercentile(side, [5, 50, 95])],
            "d_center_px_p1_5_50_95_99": [round(float(v), 1) for v in np.nanpercentile(d, [1, 5, 50, 95, 99])] if both.any() else None,
            "below_closed_anchor_frac": round(float(np.mean(d[both] < cal["anchors"][0]["x"])), 3) if both.any() else None,
            "above_top_anchor_frac": round(float(top[both].mean()), 3) if both.any() else None,
            "width_mm_p1_5_50_95_99": [round(float(v), 1) for v in np.nanpercentile(w, [1, 5, 50, 95, 99])] if both.any() else None,
            "rows": nr, "rows_from_tags": int((src == "tags").sum()), "rows_interp": int((src == "interp").sum()),
            "rows_empty": int(np.isnan(rw).sum()), "rows_with_width_frac": round(float(np.isfinite(rw).mean()), 3),
            "empty_gaps_rows_count": len(gaps), "empty_gap_rows_max": max(gaps) if gaps else 0,
            "empty_gap_rows_ge_30": int(sum(g for g in gaps if g >= 30)),
            "wall_s": round(time.time() - t0, 1)}
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
