"""请求、输出路径和阶段结果；不在数据结构中启动处理或创建目录。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from software import config
from software.vision.models import RecordingPaths as VisualPaths, VisualSettings, VisualContext


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
    gripper: str = ""
    config: Path | None = None
    dry_run: bool = False
    configuration: config.Configuration | None = None


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


@dataclass(frozen=True)
class ReconstructionContext:
    paths: RecordingPaths
    request: TrajectoryRequest
    settings: ReconstructionSettings
    frame_count: int

    def __post_init__(self):
        if self.request.configuration is None:
            raise ValueError("重建上下文需要 workflow 已解析的参数")

    @property
    def visual(self) -> VisualContext:
        return VisualContext(self.paths, self.settings, self.parameters, self.frame_count, self.request.cores)

    @property
    def parameters(self) -> config.TrajectoryParameters:
        return self.request.configuration.values


@dataclass(frozen=True)
class FusionResult:
    trajectory: Path
    info: Path


@dataclass(frozen=True)
class AttemptResult:
    attempts: list[dict]
    chosen: int | None
    directory: Path | None
    reason: str | None = None
    failed_stage: str | None = None


