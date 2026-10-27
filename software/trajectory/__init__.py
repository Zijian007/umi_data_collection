"""X3 录像转夹爪轨迹（独立单目/双目 + IMU 融合）。

阅读入口：cli.main() -> workflow.run_batch() -> workflow.run_recording()。
架构模块：
- calibration: 设备规格与实测标定资产管理
- pipeline:    重建多轮重试、输入校验、运行设置与复用策略
- reporting:   结果交付、展示地图、轨迹预览图与质量分析
- algorithms:  图优化多传感器融合求解器内核
"""
from __future__ import annotations

import sys

# 优先注册底层资产模块，打破与 registry 的潜在循环导入
from .calibration import assets as calibration_assets
sys.modules["software.trajectory.calibration_assets"] = calibration_assets

from . import algorithms, calibration, models, pipeline, reporting

__all__ = [
    "algorithms",
    "calibration",
    "models",
    "pipeline",
    "reporting",
]

# 向后兼容映射与 sys.modules 别名注入（确保老测试与外部 import 无感知兼容）
_COMPAT_MAPPING = {
    "calibration_assets": calibration.assets,
    "selection": calibration.selection,
    "recording_data": pipeline.inputs,
    "reconstruction": pipeline.reconstruction,
    "reuse": pipeline.reuse,
    "runtime": pipeline.runtime,
    "map_data": reporting.map_data,
    "run_metadata": reporting.metadata,
    "plots": reporting.plots,
    "quality": reporting.quality,
    "reports": reporting.reports,
    "results": reporting.results,
}

for _alias, _module in _COMPAT_MAPPING.items():
    sys.modules[f"software.trajectory.{_alias}"] = _module
    globals()[_alias] = _module
