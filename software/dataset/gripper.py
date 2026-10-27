"""把场景结果中的宽度序列映射到训练时间轴，保留缺失值和来源。"""
from __future__ import annotations

from pathlib import Path

from typing import TypedDict

import numpy as np

from software.gripper_width import series


class GripperMeasurement(TypedDict):
    calibration: dict
    inputs: dict
    valid_fraction: float
    sources: dict[str, int]


class GripperReadiness(TypedDict):
    training_ready: bool
    missing_channels: list[str]
    not_ready_reasons: list[str]
    measurements: dict[str, GripperMeasurement]


def sample(directory: Path, manifest: dict, arms: dict, query: np.ndarray, max_gap_s: float) -> tuple[dict[str, np.ndarray], GripperReadiness]:
    arrays: dict[str, np.ndarray] = {}
    reasons: list[str] = []
    measurements: dict[str, GripperMeasurement] = {}
    missing = False
    for arm, trajectory in arms.items():
        report = manifest.get('gripper_width', {}).get(arm)
        if report is None:
            missing = True
            reasons.append(f"{arm}: missing gripper")
            continue
        measured = series.read(directory / arm / 'gripper_width.csv')
        if len(measured.times_s) != len(trajectory.times_s) or not np.allclose(measured.times_s, trajectory.times_s, atol=1e-8, rtol=0):
            raise ValueError(f'{arm} 夹爪宽度与 TCP 不在同一时间轴')
        calibration = report['calibration']
        opening = calibration['open_width_mm']
        if np.any((measured.width_mm[measured.valid] < 0) | (measured.width_mm[measured.valid] > opening)):
            raise ValueError(f'{arm} 有效宽度超出标定范围')
        sampled = series.resample(measured, query, max_gap_s)
        arrays[f"{arm}_gripper_width_m"] = sampled.width_mm / 1000.0
        arrays[f"{arm}_gripper_close_01"] = 1.0 - sampled.width_mm / opening
        arrays[f"{arm}_gripper_valid"] = sampled.valid
        arrays[f"{arm}_gripper_source"] = sampled.source
        measurements[arm] = dict(
            calibration=calibration,
            inputs=report['inputs'],
            valid_fraction=float(sampled.valid.mean()),
            sources={name: int((sampled.source == code).sum()) for code, name in series.SOURCE_NAMES.items()}
        )
        if not calibration['verified']:
            reasons.append(f"{arm}: gripper calibration unverified")
        if not sampled.valid.all():
            reasons.append(f"{arm}: gripper contains missing or out-of-range samples")
    return arrays, dict(training_ready=not reasons, missing_channels=['gripper'] if missing else [],
                        not_ready_reasons=reasons, measurements=measurements)
