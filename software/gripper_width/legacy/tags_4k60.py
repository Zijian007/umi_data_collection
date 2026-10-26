#!/usr/bin/env python3
"""tags_4k60.py -- fingertip ArUco tags (DICT_4X4_50, ID0 / ID1) in every frame of a packed 4K60 recording, with the
detector run at three scales of the tag band and the first hit per tag kept.  Detection only; widths come from
apply_width.py.  (2026-10-14, brought onto dev_Zijian from paper/cvpr gripper_width/)

Why three scales: in the 4K60 mode a tag is ~24 px wide (cells ~4 px).  On the 10-12 recordings the September setting
(native pixels; experiments/reviews/data/20260904_x3_4k60_calibration/tools/gripper_detect_4k60.py) found both tags in
22-35 % of frames although many missed tags are sharp and unoccluded; 2x / 3x cubic upscaling of the band finds
36-53 %, and the three settings miss different frames (detect_variants.py, 2026-10-14).  The tag-centre distance moves by
a median 0.03 px between settings (95 % within 0.8 px).

Per frame and tag: the first scale of (1, 2, 3) that finds the tag; centre in native right-half pixels
(native = (scaled + 0.5) / scale - 0.5 + band offset).  Same parameters at every scale as in September
(adaptiveThreshWinSizeMax 53, minMarkerPerimeterRate 0.005, sub-pixel corners).

    python software/gripper_width/legacy/tags_4k60.py INSV --out frames.csv [--band 520 1380 1200 1420]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

SCALES = (1, 2, 3)


def make_detector():
    import cv2
    aruco = cv2.aruco
    prm = aruco.DetectorParameters()
    prm.adaptiveThreshWinSizeMax = 53
    prm.minMarkerPerimeterRate = 0.005
    prm.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50), prm)


def tags_in(det, band_img, ou: int, ov: int) -> dict:
    """{tag id: (4, 2) corners in native right-half pixels, scale used}"""
    import cv2
    out = {}
    for sc in SCALES:
        g = band_img if sc == 1 else cv2.resize(band_img, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = det.detectMarkers(g)
        if ids is None:
            continue
        for j, i in enumerate(ids.flatten()):
            i = int(i)
            if i in (0, 1) and i not in out:
                out[i] = ((corners[j][0] + 0.5) / sc - 0.5 + np.array([ou, ov]), sc)
        if 0 in out and 1 in out:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("insv", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--band", type=int, nargs=4, default=[520, 1380, 1200, 1420], metavar=("U0", "U1", "V0", "V1"),
                    help="native right-half pixels where the tags move (10-12 recordings: u 674-1220, v 1290-1322)")
    a = ap.parse_args()
    import av

    det = make_detector()
    u0, u1, v0, v1 = a.band
    cols = ["frame", "t_s", "det0", "det1", "scale0", "scale1", "u0", "v0", "u1", "v1", "side0_px", "side1_px", "d_center_px"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); n = nb = 0
    with av.open(str(a.insv)) as c, open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        s = c.streams.video[0]; s.thread_type = "AUTO"; tb = float(s.time_base)
        W, H = s.codec_context.width, s.codec_context.height
        assert W == 2 * H, f"not a packed dual-fisheye frame: {W}x{H}"
        for k, fr in enumerate(c.decode(s)):
            g = fr.to_ndarray(format="gray")[v0:v1, W // 2 + u0:W // 2 + u1]
            found = tags_in(det, g, u0, v0)
            r = {"frame": k, "t_s": round(float(fr.pts * tb), 5)}
            for i in (0, 1):
                if i in found:
                    cs, sc = found[i]
                    cc = cs.mean(0)
                    r.update({f"det{i}": 1, f"scale{i}": sc, f"u{i}": round(float(cc[0]), 3), f"v{i}": round(float(cc[1]), 3),
                              f"side{i}_px": round(float(np.mean([np.linalg.norm(cs[m] - cs[(m + 1) % 4]) for m in range(4)])), 3)})
                else:
                    r.update({f"det{i}": 0, f"scale{i}": "", f"u{i}": "", f"v{i}": "", f"side{i}_px": ""})
            r["d_center_px"] = round(float(np.linalg.norm(found[0][0].mean(0) - found[1][0].mean(0))), 3) if (0 in found and 1 in found) else ""
            nb += r["d_center_px"] != ""
            w.writerow([r[c_] for c_ in cols]); n += 1
            if k % 600 == 0:
                print(f"frame {k} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    print(f"TAGS_DONE frames={n} both={nb} ({nb / max(n, 1):.3f}) wall_s={time.time() - t0:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
