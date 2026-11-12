"""已知海报几何：ID → 米制角点；不将板在房间中的位姿保存为资产。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from software import provenance


DEFAULT_LAYOUT = provenance.ROOT / 'hardware/calibration_targets/environment_poster.json'


@dataclass(frozen=True)
class TagBoardLayout:
    board_id: str
    dictionary: str
    border_bits: int
    sheet_size_m: tuple[float, float]
    corners_m: dict[int, np.ndarray]
    geometry_status: str
    geometry_hash: str
    asset_hash: str
    source_pdf: dict

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.corners_m))

    def record(self) -> dict:
        return dict(board_id=self.board_id, dictionary=self.dictionary, border_bits=self.border_bits,
                    geometry_hash=self.geometry_hash, asset_hash=self.asset_hash,
                    geometry_status=self.geometry_status, source_pdf=self.source_pdf)


def _unique_keys(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f'海报布局含重复 JSON 字段：{name}')
        result[name] = value
    return result


def load(path: Path = DEFAULT_LAYOUT) -> TagBoardLayout:
    """读取显式角点，不按数字 ID 推断排列；验证平面正方形及检测器角点约定。"""
    path = path.expanduser().resolve()
    raw = path.read_bytes()
    doc = json.loads(raw, object_pairs_hook=_unique_keys)
    if not isinstance(doc, dict) or doc.get('kind') != 'umi_tag_board' or type(doc.get('schema_version')) is not int or doc['schema_version'] != 1:
        raise ValueError('不支持的海报布局格式')
    required = {'kind', 'schema_version', 'board_id', 'dictionary', 'border_bits', 'units',
                'sheet_size_m', 'coordinate_frame', 'corner_order', 'geometry_status', 'source_pdf', 'markers'}
    if set(doc) != required:
        raise ValueError('海报布局字段缺失或包含未知字段')
    if doc['dictionary'] != 'DICT_4X4_100' or doc['units'] != 'metres':
        raise ValueError('海报布局需要 DICT_4X4_100 和 metres 单位')
    if not isinstance(doc['board_id'], str) or not doc['board_id'].strip():
        raise ValueError('海报 board_id 必须非空')
    if type(doc['border_bits']) is not int or doc['border_bits'] < 1:
        raise ValueError('海报黑边格数必须为正整数')
    if doc['coordinate_frame'] != 'sheet_top_left_x_right_y_down_z_into_sheet' or doc['corner_order'] != 'opencv_marker_tl_tr_br_bl':
        raise ValueError('海报坐标轴或角点顺序不支持')
    if doc['geometry_status'] not in ('pdf_design', 'physically_measured'):
        raise ValueError('海报尺寸来源必须明确为 PDF 设计或实测')
    sheet = np.asarray(doc['sheet_size_m'], float)
    if sheet.shape != (2,) or not np.isfinite(sheet).all() or np.any(sheet <= 0):
        raise ValueError('海报纸张尺寸必须为两个有限正数')
    source = doc['source_pdf']
    if (not isinstance(source, dict) or set(source) != {'path', 'sha256'}
            or not isinstance(source['path'], str) or not source['path']
            or not isinstance(source['sha256'], str) or len(source['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in source['sha256'])):
        raise ValueError('海报需要 PDF 来源路径和 SHA256')
    markers = doc['markers']
    if not isinstance(markers, list) or not markers:
        raise ValueError('海报布局没有标签角点')
    corners = {}
    for marker in markers:
        if not isinstance(marker, dict) or set(marker) != {'id', 'corners_m'}:
            raise ValueError('海报标签需要 id 和 corners_m')
        tag_id = marker['id']
        if type(tag_id) is not int or not 0 <= tag_id < 100 or tag_id in corners:
            raise ValueError('海报 ID 必须是唯一的 0–99 整数')
        points = np.asarray(marker['corners_m'], float)
        if points.shape != (4, 3) or not np.isfinite(points).all():
            raise ValueError(f'海报 ID {tag_id} 角点必须为有限的 4×3 数组')
        if (not np.allclose(points[:, 2], 0, rtol=0, atol=1e-10)
                or np.any(points[:, :2] < 0) or np.any(points[:, :2] > sheet)):
            raise ValueError(f'海报 ID {tag_id} 必须位于纸张平面内')
        edges = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(edges, axis=1)
        if (np.min(lengths) <= 1e-6 or not np.allclose(lengths, lengths[0], rtol=1e-6, atol=1e-10)
                or abs(float(edges[0] @ edges[1])) > lengths[0]**2 * 1e-6
                or not np.allclose(edges[0], -edges[2], rtol=1e-6, atol=1e-10)
                or not np.allclose(edges[1], -edges[3], rtol=1e-6, atol=1e-10)
                or np.cross(edges[0], edges[1])[2] <= 0):
            raise ValueError(f'海报 ID {tag_id} 角点须按检测器顺序构成正方形')
        points.setflags(write=False)
        corners[tag_id] = points
    geometry = {key: doc[key] for key in ('dictionary', 'border_bits', 'units', 'sheet_size_m', 'coordinate_frame', 'corner_order')}
    geometry['markers'] = [{'id': tag_id, 'corners_m': corners[tag_id].tolist()} for tag_id in sorted(corners)]
    import hashlib
    return TagBoardLayout(doc['board_id'], doc['dictionary'], doc['border_bits'], tuple(sheet), corners,
                          doc['geometry_status'], provenance.canonical_hash(geometry), hashlib.sha256(raw).hexdigest(), source)
