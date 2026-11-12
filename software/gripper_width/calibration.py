"""读取 main 脚本的像素距离—毫米标定，拒绝无效或非单调锚点。"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict
import json

import numpy as np


class Anchor(TypedDict):
    x: float
    mm: float


class WidthCalibration(TypedDict):
    anchors: list[Anchor]
    open_width_mm: float
    status: str


def read(path: Path) -> WidthCalibration:
    width_calibration_document = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(width_calibration_document, dict):
        raise ValueError('夹爪宽度标定必须为 JSON 对象')
    if {'closed_threshold_mm', 'max_gap_s'} & width_calibration_document.keys():
        raise ValueError('宽度处理参数请移到 scene.yaml 的 parameters.gripper_width_*，测量文件只保存实测关系')
    mapping = width_calibration_document.get('mapping', width_calibration_document)
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


def pixel_distances_to_widths(distances_px: np.ndarray, anchors: list[Anchor]) -> tuple[np.ndarray, np.ndarray]:
    """分段线性换算；闭合以下截为零，最大锚点以上外推并标记。

    保留 main/f331ad9 width_4k60.py 的毫米映射，供正式场景流程使用。
    """
    anchor_pixels = np.array([anchor['x'] for anchor in anchors])
    anchor_widths_mm = np.array([anchor['mm'] for anchor in anchors])
    widths_mm = np.interp(distances_px, anchor_pixels, anchor_widths_mm)
    exceeds_calibrated_range = distances_px > anchor_pixels[-1]
    widths_mm[exceeds_calibrated_range] = anchor_widths_mm[-1] + (
        (distances_px[exceeds_calibrated_range] - anchor_pixels[-1])
        * (anchor_widths_mm[-1] - anchor_widths_mm[-2]) / (anchor_pixels[-1] - anchor_pixels[-2]))
    return np.maximum(widths_mm, 0.), exceeds_calibrated_range


def read_tag_sizes(path: Path) -> dict:
    tag_size_document = json.loads(path.read_text())
    if not isinstance(tag_size_document, dict) or tag_size_document.keys() != {'tag_sizes_mm'}:
        raise ValueError('标签间距测量文件需要且仅需 tag_sizes_mm，分别填写 ID 0/1 的黑色外框边长')
    sizes = tag_size_document['tag_sizes_mm']
    if not isinstance(sizes, dict) or sizes.keys() != {'0', '1'}:
        raise ValueError('tag_sizes_mm 必须包含字符串键 "0" 和 "1"')
    if any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in sizes.values()):
        raise ValueError('标签边长必须是正的有限毫米数')
    return dict(tag_sizes_mm={key: float(value) for key, value in sizes.items()})
