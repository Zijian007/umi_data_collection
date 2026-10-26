#!/usr/bin/env python3
"""detect_variants.py -- why are the fingertip tags missed in clearly visible 4K60 frames, and which detector setting
finds them?  (2026-10-14, brought onto dev_Zijian from paper/cvpr gripper_width/)

On the 10-12 shelf recordings the September 4K60 detector (full right half, native pixels) finds both tags in only
22-35 % of frames, although in many missed frames both tags are sharp and unoccluded (tag edge ~24 px, cells ~4 px).
This script runs several settings on the same sampled frames and reports, per setting, how often each tag is found and,
on frames where both the reference setting and the variant find both tags, how far the tag-centre distance moves
(the width calibration is in native pixels, so a variant must not shift it).

Variants (all DICT_4X4_50, ID0 / ID1, sub-pixel corners; coordinates mapped back to native right-half pixels):
  ref        the September setting: full right half, native, adaptiveThreshWinSizeMax 53, minMarkerPerimeterRate 0.005
  band       only the band where the tags move (--band u0 u1 v0 v1, native px), native scale, same parameters
  band_x2    the band upscaled 2x (cubic)
  band_x3    the band upscaled 3x (cubic)
  band_x2_p  2x plus looser parameters (adaptiveThreshWinSizeMin 3, step 4, polygonalApproxAccuracyRate 0.05,
             minCornerDistanceRate 0.02, minDistanceToBorder 1)

    python software/gripper_width/legacy/detect_variants.py INSV --every 10 --band 450 1450 1150 1450 --out OUT.json [--first N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def make(params: dict):
    import cv2
    aruco = cv2.aruco
    prm = aruco.DetectorParameters()
    prm.adaptiveThreshWinSizeMax = 53
    prm.minMarkerPerimeterRate = 0.005
    prm.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    for k, v in params.items():
        setattr(prm, k, v)
    return aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50), prm)


LOOSE = {"adaptiveThreshWinSizeMin": 3, "adaptiveThreshWinSizeStep": 4, "polygonalApproxAccuracyRate": 0.05,
         "minCornerDistanceRate": 0.02, "minDistanceToBorder": 1}
VARIANTS = {"ref": (None, 1, {}), "band": ("band", 1, {}), "band_x2": ("band", 2, {}), "band_x3": ("band", 3, {}),
            "band_x2_p": ("band", 2, LOOSE)}


def detect(det, g, crop, scale, band):
    import cv2
    ou, ov = 0, 0
    if crop:
        u0, u1, v0, v1 = band
        g = g[v0:v1, u0:u1]; ou, ov = u0, v0
    if scale != 1:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    corners, ids, _ = det.detectMarkers(g)
    out = {}
    if ids is not None:
        for j, i in enumerate(ids.flatten()):
            if int(i) in (0, 1):
                # pixel centres: native = (scaled + 0.5) / scale - 0.5
                c = (corners[j][0] + 0.5) / scale - 0.5 + np.array([ou, ov])
                out[int(i)] = c
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("insv", type=Path)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--band", type=int, nargs=4, required=True, metavar=("U0", "U1", "V0", "V1"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first", type=int, default=0, help="stop after this many frames (0 = whole recording)")
    a = ap.parse_args()
    import av
    import cv2

    dets = {k: make(p) for k, (_, _, p) in VARIANTS.items()}
    res = {k: {"id0": 0, "id1": 0, "both": 0} for k in VARIANTS}
    dd = {k: [] for k in VARIANTS}
    n, t0 = 0, time.time()
    with av.open(str(a.insv)) as c:
        s = c.streams.video[0]; s.thread_type = "AUTO"
        half = s.codec_context.width // 2
        for k, fr in enumerate(c.decode(s)):
            if a.first and k >= a.first:
                break
            if k % a.every:
                continue
            g = fr.to_ndarray(format="gray")[:, half:]
            n += 1
            found = {}
            for name, (crop, scale, _) in VARIANTS.items():
                o = detect(dets[name], g, crop, scale, a.band)
                found[name] = o
                res[name]["id0"] += 0 in o; res[name]["id1"] += 1 in o; res[name]["both"] += (0 in o and 1 in o)
            r = found["ref"]
            if 0 in r and 1 in r:
                dref = float(np.linalg.norm(r[0].mean(0) - r[1].mean(0)))
                for name, o in found.items():
                    if 0 in o and 1 in o:
                        dd[name].append(float(np.linalg.norm(o[0].mean(0) - o[1].mean(0))) - dref)
    out = {"insv": a.insv.name, "opencv": cv2.__version__, "every": a.every, "band": a.band, "frames": n, "variants": {}}
    for name in VARIANTS:
        x = np.array(dd[name])
        out["variants"][name] = {k: round(v / n, 3) for k, v in res[name].items()}
        out["variants"][name]["d_minus_ref_px_on_common"] = (
            {"n": int(len(x)), "median": round(float(np.median(x)), 3), "p95_abs": round(float(np.percentile(np.abs(x), 95)), 3)} if len(x) else None)
    out["wall_s"] = round(time.time() - t0, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
