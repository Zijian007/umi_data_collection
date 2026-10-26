#!/usr/bin/env python3
"""apply_width.py -- gripper opening (mm) per frame and per trajectory row from the tags found by tags_4k60.py.
(2026-10-14, brought onto dev_Zijian from paper/cvpr gripper_width/)

1. Width calibration, re-derived with the same detector: the September calibration recording 029 was run through
   tags_4k60.py; for each of the 7 held openings (0/20/40/60/80/100/112 mm, windows exactly as chosen in September,
   plateaus.json labels_assigned) the anchor is the median tag-centre distance in that window.  Leave-one-anchor-out
   error and the shift against the September anchors (OpenCV 4.2, native scale only) are reported.
2. One tag only: the camera is fixed to the gripper body, so each tag moves along a fixed line in the image as the
   gripper opens.  For each tag, the distance two tags would have had is predicted from that tag's position along its
   line (quadratic), fitted on the frames of the SAME DAY's recordings that show both tags (the mount can differ between
   days: on 10-12 the tag pair sits ~30 px left of where it sat on 09-04).  Checked by leaving one recording out.
3. Width = the 7 anchors, piecewise linear; clamped to 0 below the closed anchor (fingers pressed together), extrapolated
   and flagged above the 112 mm anchor.
4. Per trajectory row k (= video frame 2k): median over frames 2k-1..2k+1, two-tag frames preferred over one-tag frames;
   gaps of at most --max-gap-rows rows are interpolated, longer gaps stay empty.

    python software/gripper_width/apply_width.py --calib-frames 029/frames.csv --sept plateaus.json --day 072=072/frames.csv ... --out DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def anchors_from(frames: pd.DataFrame, sept: dict) -> list[dict]:
    out = []
    for lab, pl in sorted(sept["labels_assigned"].items(), key=lambda kv: float(kv[0])):
        win = frames[(frames.frame >= pl["i0"]) & (frames.frame <= pl["i1"])].d_center_px.dropna()
        out.append({"mm": float(lab), "x": round(float(win.median()), 3), "n": int(len(win)),
                    "iqr_px": round(float(win.quantile(0.75) - win.quantile(0.25)), 3), "sept_x": round(float(pl["d_med"]), 3),
                    "window_frames": [pl["i0"], pl["i1"]]})
    return out


def to_mm(d: np.ndarray, anchors: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([a["x"] for a in anchors]); y = np.array([a["mm"] for a in anchors])
    w = np.interp(d, x, y)
    top = d > x[-1]
    w[top] = y[-1] + (d[top] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return np.maximum(w, 0.0), top


def loco(anchors: list[dict]) -> list[float]:
    x = np.array([a["x"] for a in anchors]); y = np.array([a["mm"] for a in anchors])
    return [round(float(abs(np.interp(x[i], np.delete(x, i), np.delete(y, i)) - y[i])), 3) for i in range(1, len(x) - 1)]


def fit_single(both: pd.DataFrame, i: int) -> dict:
    P = both[[f"u{i}", f"v{i}"]].values
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    s = (P - c) @ vt[0]
    co = np.polyfit(s, both.d_center_px.values, 2)
    return {"centre": c.tolist(), "axis": vt[0].tolist(), "poly2_d_of_s": co.tolist(),
            "s_range": [float(s.min()), float(s.max())], "across_axis_std_px": float(((P - c) @ vt[1]).std())}


def predict_single(m: dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    s = (np.c_[u, v] - np.array(m["centre"])) @ np.array(m["axis"])
    return np.polyval(m["poly2_d_of_s"], s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-frames", type=Path, required=True, help="tags_4k60.py output for the calibration recording 029")
    ap.add_argument("--sept", type=Path, required=True, help="plateaus.json of the September 4K60 width calibration")
    ap.add_argument("--day", nargs="+", required=True, help="NAME=frames.csv for every recording of one day (same mount)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-gap-rows", type=int, default=5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    sept = json.loads(a.sept.read_text())
    cal = pd.read_csv(a.calib_frames)
    anchors = anchors_from(cal, sept)
    x = [a_["x"] for a_ in anchors]
    calib = {"what": "4K60 gripper width calibration, recording 029 of 2026-09-04 re-measured with tags_4k60.py (3-scale "
                     "union detector); September windows and labels unchanged",
             "status": sept["mapping"]["status"], "monotonic": bool(np.all(np.diff(x) > 0)), "anchors": anchors,
             "loco_abs_err_mm_interior": loco(anchors), "shift_vs_september_px": [round(a_["x"] - a_["sept_x"], 3) for a_ in anchors],
             "calib_frames_both_frac": round(float(cal.d_center_px.notna().mean()), 3)}

    days = {}
    for item in a.day:
        name, p = item.split("=", 1)
        days[name] = pd.read_csv(p).assign(rec=name)
    allf = pd.concat(days.values(), ignore_index=True)
    both = allf[allf.d_center_px.notna()]
    single = {i: fit_single(both, i) for i in (0, 1)}
    # leave one recording out: fit on the others, predict the two-tag distance of the left-out recording from each tag alone
    lro = {}
    for i in (0, 1):
        pred, true = [], []
        for name in days:
            m = fit_single(both[both.rec != name], i)
            te = both[both.rec == name]
            pred.append(predict_single(m, te[f"u{i}"].values, te[f"v{i}"].values)); true.append(te.d_center_px.values)
        pred, true = np.concatenate(pred), np.concatenate(true)
        e = pred - true
        ew = np.abs(to_mm(pred, anchors)[0] - to_mm(true, anchors)[0])
        lro[f"tag{i}_alone"] = {"n": int(len(e)), "d_err_px_median_abs": round(float(np.median(np.abs(e))), 2),
                                "d_err_px_p90_abs": round(float(np.percentile(np.abs(e), 90)), 2),
                                "width_err_mm_median": round(float(np.median(ew)), 2), "width_err_mm_p90": round(float(np.percentile(ew, 90)), 2),
                                "width_err_mm_p99": round(float(np.percentile(ew, 99)), 2)}
    calib["single_tag"] = {"fitted_on": sorted(days), "models": single, "leave_one_recording_out": lro}
    (a.out / "calib_029_union.json").write_text(json.dumps(calib, indent=1))
    print(json.dumps({k: calib[k] for k in ("monotonic", "loco_abs_err_mm_interior", "shift_vs_september_px", "calib_frames_both_frac")}))
    print(json.dumps(lro))

    summary = {}
    for name, f in days.items():
        n = len(f)
        d = f.d_center_px.values.astype(float)
        src = np.where(np.isfinite(d), "both", np.where(f.det0 == 1, "only0", np.where(f.det1 == 1, "only1", "none")))
        dd = d.copy()
        for i in (0, 1):
            k = src == f"only{i}"
            if k.any():
                dd[k] = predict_single(single[i], f[f"u{i}"].values[k], f[f"v{i}"].values[k])
        w = np.full(n, np.nan); top = np.zeros(n, bool)
        ok = np.isfinite(dd)
        w[ok], top[ok] = to_mm(dd[ok], anchors)
        f = f.assign(source=src, d_used_px=np.round(dd, 3), width_mm=np.round(w, 3), above_top_anchor=top.astype(int))
        f.to_csv(a.out / f"{name}_frames_width.csv", index=False)

        nr = (n + 1) // 2
        rw = np.full(nr, np.nan); rsrc = np.array(["none"] * nr, dtype=object)
        for k in range(nr):
            fr = [j for j in (2 * k - 1, 2 * k, 2 * k + 1) if 0 <= j < n]
            b = [w[j] for j in fr if src[j] == "both"]
            s1 = [w[j] for j in fr if src[j] in ("only0", "only1")]
            if b:
                rw[k], rsrc[k] = float(np.median(b)), "two_tags"
            elif s1:
                rw[k], rsrc[k] = float(np.median(s1)), "one_tag"
        have = np.flatnonzero(np.isfinite(rw))
        for i0, i1 in zip(have[:-1], have[1:]):
            if 1 < i1 - i0 <= a.max_gap_rows + 1:
                for k in range(i0 + 1, i1):
                    rw[k] = rw[i0] + (rw[i1] - rw[i0]) * (k - i0) / (i1 - i0); rsrc[k] = "interp"
        with open(a.out / f"{name}_rows.csv", "w", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(["row", "video_frame", "width_mm", "source"])
            for k in range(nr):
                wr.writerow([k, 2 * k, "" if np.isnan(rw[k]) else round(float(rw[k]), 3), rsrc[k]])
        gaps, i = [], 0
        emp = np.isnan(rw)
        while i < nr:
            if emp[i]:
                j = i
                while j < nr and emp[j]:
                    j += 1
                gaps.append(j - i); i = j
            else:
                i += 1
        summary[name] = {"frames": n, **{f"frames_{s}": round(float((src == s).mean()), 3) for s in ("both", "only0", "only1", "none")},
                         "rows": nr, **{f"rows_{s}": int((rsrc == s).sum()) for s in ("two_tags", "one_tag", "interp", "none")},
                         "rows_with_width_frac": round(float(np.isfinite(rw).mean()), 3), "empty_gaps": len(gaps),
                         "empty_gap_rows_max": max(gaps) if gaps else 0, "empty_rows_in_gaps_ge_30": int(sum(g for g in gaps if g >= 30)),
                         "width_mm_p1_50_99": [round(float(v), 1) for v in np.nanpercentile(rw, [1, 50, 99])],
                         "frames_below_closed_anchor": round(float((dd[ok] < anchors[0]["x"]).mean()), 3),
                         "frames_above_top_anchor": round(float(top[ok].mean()), 3)}
        print(name, json.dumps(summary[name]))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
