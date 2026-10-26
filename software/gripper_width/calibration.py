"""读取 main 脚本的像素距离—毫米标定，拒绝无效或非单调锚点。"""
from __future__ import annotations

import json
import math
from pathlib import Path

from typing import TypedDict

import numpy as np


class Anchor(TypedDict):
    x: float
    mm: float


class WidthCalibration(TypedDict):
    anchors: list[Anchor]
    open_width_mm: float
    closed_threshold_mm: float
    max_gap_s: float
    verified: bool
    status: str
    detector: str
    method: str


def read(path: Path) -> WidthCalibration:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('夹爪宽度标定必须为 JSON 对象')
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
    closed = data.get('closed_threshold_mm', 3.)
    max_gap = data.get('max_gap_s', .2)
    if (type(closed) not in (int, float) or not math.isfinite(closed) or not 0 <= closed < mm[-1]
            or type(max_gap) not in (int, float) or not math.isfinite(max_gap) or max_gap < 0):
        raise ValueError('closed_threshold_mm 或 max_gap_s 无效')
    verified = data.get('verified', False)
    if type(verified) is not bool:
        raise ValueError('宽度标定 verified 必须为布尔值')
    return dict(anchors=anchors, open_width_mm=float(mm[-1]), closed_threshold_mm=float(closed),
                max_gap_s=float(max_gap), verified=verified, status=mapping.get('status', 'unspecified'),
                detector='DICT_4X4_50:0,1', method='main_width_4k60_two_tags')
