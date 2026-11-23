"""请求、输出路径和阶段结果；不在数据结构中启动处理或创建目录。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from software.vision.models import RecordingPaths as VisualPaths, VisualSettings, VisualContext


if TYPE_CHECKING:
    import numpy as np
    from software.configuration import loader as config


@dataclass(frozen=True)
class FusedTrajectory:
    """IMU 位姿位于调用方声明的参考系；可为固定场景或单次局部坐标。"""
    T_reference_imu: np.ndarray
    tracked: np.ndarray
    front_tracked: np.ndarray
    evidence: dict



@dataclass(frozen=True)
class TrajectoryRequest:
    """One recording through the trajectory tool. The command line fills this in; other callers can too."""

    mode: str | None = None
    max_tries: int | None = None
    cores: str = ""
    keep_work: bool = False
    force: bool = False
    allow_irregular: bool | None = None
    device: str = ""
    tcp: Path | None = None
    mask: Path | None = None
    config: Path | None = None
    dry_run: bool = False
    configuration: config.Configuration | None = None
    intrinsics_source: str | None = None
    tag_layout: Path | None = None
    rear_calibration: str | None = None


@dataclass(frozen=True)
class RecordingPaths(VisualPaths):
    @property
    def reconstruction(self) -> Path:
        return self.debug / 'reconstruction'

    def attempt(self, number: int) -> Path:
        return self.reconstruction / f'try_{number}'


@dataclass(frozen=True)
class ReconstructionSettings(VisualSettings):
    """独立轨迹额外使用的离线融合设置。"""

    fusion_python: str
    fusion_script: Path
    time_offset_ns: str
    rear_q_ic: str
    rear_lever: str
    camera_json: Path
    parameters_file: Path
    poster_manifest: Path | None = None
    rear_manifest: Path | None = None


@dataclass(frozen=True)
class ReconstructionContext(VisualContext):
    """Resolved engine inputs for standalone trajectories and scene fallback."""

    paths: RecordingPaths
    settings: ReconstructionSettings


@dataclass(frozen=True)
class FusionArtifacts:
    trajectory: Path
    quality_report: Path


@dataclass(frozen=True)
class AttemptResult:
    attempts: list[dict]
    chosen: int | None
    directory: Path | None
    reason: str | None = None
    failed_stage: str | None = None
