"""共享场景：一次建图，多段任务，左右手在同一米制坐标和时间轴中处理。

架构分为三大模块：
- management: 采集任务生命周期与设备资产管理 (capture_config, ingest, episodes, inputs)
- spatial: 多相机空间几何、视觉定位与世界系 GTSAM 融合求解 (fusion, geometry, mapping, rear_calibration, slam)
- pipeline: 录像提取、质量预检与轨迹结果交付 (recordings, recording_checks, results)
"""
from __future__ import annotations
import sys

from .management import capture_config, episodes, ingest, inputs
from .spatial import fusion, geometry, mapping, rear_calibration, slam
from .pipeline import recording_checks, recordings, results

# 注册 sys.modules 别名，确保历史导入与 mock.patch 字符串 100% 向后兼容
_ALIASES = (
    ('capture_config', capture_config),
    ('episodes', episodes),
    ('ingest', ingest),
    ('inputs', inputs),
    ('fusion', fusion),
    ('geometry', geometry),
    ('mapping', mapping),
    ('rear_calibration', rear_calibration),
    ('slam', slam),
    ('recording_checks', recording_checks),
    ('recordings', recordings),
    ('results', results),
)

for _name, _mod in _ALIASES:
    sys.modules[f'software.scene.{_name}'] = _mod

__all__ = [
    'capture_config',
    'episodes',
    'ingest',
    'inputs',
    'fusion',
    'geometry',
    'mapping',
    'rear_calibration',
    'slam',
    'recording_checks',
    'recordings',
    'results',
]
