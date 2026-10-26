"""Read one umi-traj run the way the open/close cut expects it.

Brought from paper/cvpr shelf6.py (load_run, gaps_of, umeyama_rigid, FPS) so pilot.py can
label marker / grasp / release without the rest of the shelf analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

FPS = 60000 / 1001 / 2     # trajectory rows per second: every 2nd frame of 4K60 (59.94 fps)
SUPPORT = "full"


def read_tum(p: Path) -> np.ndarray:
    return np.array([list(map(float, l.split())) for l in Path(p).read_text().splitlines() if l.strip() and not l.startswith("#")], float)


def umeyama_rigid(src: np.ndarray, dst: np.ndarray):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d) / len(src)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mu_d - R @ mu_s


def gaps_of(support: np.ndarray) -> list[dict]:
    out, k, n = [], 0, len(support)
    while k < n:
        if not support[k]:
            j = k
            while j < n and not support[j]:
                j += 1
            out.append({"t0_s": round(k / FPS, 2), "t1_s": round(j / FPS, 2), "dur_s": round((j - k) / FPS, 2)})
            k = j
        else:
            k += 1
    return out


def load_run(d: Path) -> dict | None:
    try:
        summ = json.loads((d / "summary.json").read_text())
    except Exception:
        return None
    if summ.get("status") != "ok":
        return {"status": summ.get("status"), "reason": summ.get("reason"), "summary": summ}
    tcp, cam = read_tum(d / "traj_tcp.txt"), read_tum(d / "traj_camera.txt")
    fr = list(csv.DictReader(open(d / "frames.csv")))
    cons = np.array([r["constraint"] for r in fr])
    return {"status": "ok", "summary": summ, "tcp": tcp, "cam": cam, "constraint": cons, "support": cons == SUPPORT}
