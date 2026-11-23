"""公共 TUM 位姿轨迹读写；不选择设备、处理录像或定义世界坐标。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [list(map(float, line.split())) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]
    trajectory = np.array(rows)
    return trajectory[:, 0], trajectory[:, 1:4], trajectory[:, 4:8]


def write_tum(path: Path, times_s: np.ndarray, positions: np.ndarray,
              quaternions_xyzw: np.ndarray, header: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for header_line in header.strip().splitlines():
            stream.write("# " + header_line + "\n")
        for frame_index in range(len(times_s)):
            stream.write("%.6f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" %
                         (times_s[frame_index], *positions[frame_index], *quaternions_xyzw[frame_index]))
