"""读取 main 脚本的像素距离—毫米标定，拒绝无效或非单调锚点。"""
from __future__ import annotations

import json
from pathlib import Path

from typing import TypedDict

import numpy as np


class Anchor(TypedDict):
    x: float
    mm: float


class WidthCalibration(TypedDict):
    anchors: list[Anchor]
    open_width_mm: float
    status: str


def read(path: Path) -> WidthCalibration:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('夹爪宽度标定必须为 JSON 对象')
    if {'closed_threshold_mm', 'max_gap_s'} & data.keys():
        raise ValueError('宽度处理参数请移到 scene.yaml 的 parameters.gripper_width_*，测量文件只保存实测关系')
    mapping = data.get('mapping', data)
    if not isinstance(mapping, dict) or not isinstance(mapping.get('anchors'), list) or len(mapping['anchors']) < 2:
        raise ValueError('宽度标定需要至少两个 anchors，每项包含 x（像素）和 mm')
    anchors = []
    for item in mapping['anchors']:
        if not isinstance(item, dict) or any(type(item.get(key)) not in (int, float) for key in ('x', 'mm')):
            raise ValueError('宽度标定的 x/mm 必须为有限数值')
        anchors.append(dict(x=float(item['x']), mm=float(item['mm'])))
    x = np.array([a['x'] for a in anchors])
    mm = np.array([a['mm'] for a in anchors])
    if (not np.isfinite(x).all() or not np.isfinite(mm).all() or x[0] <= 0 or mm[0] != 0
            or not np.all(np.diff(x) > 0) or not np.all(np.diff(mm) > 0)):
        raise ValueError('宽度标定锚点必须严格递增，首个宽度为 0 mm，像素距离为正')
    return dict(anchors=anchors, open_width_mm=float(mm[-1]),
                status=mapping.get('status', 'unspecified'))
