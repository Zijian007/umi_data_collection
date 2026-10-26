"""公共 TUM 位姿轨迹读写；不选择设备、处理录像或定义世界坐标。"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [list(map(float, l.split())) for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    a = np.array(rows)
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def write_tum(path: Path, t, p, q, header: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for h in header.strip().splitlines():
            f.write("# " + h + "\n")
        for k in range(len(t)):
            f.write("%.6f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" % (t[k], *p[k], *q[k]))
