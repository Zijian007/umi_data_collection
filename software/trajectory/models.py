"""请求、输出路径和阶段结果；不在数据结构中启动处理或创建目录。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from software import config


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
class RecordingPaths:
    output: Path

    @property
    def debug(self) -> Path:
        return self.output / "debug"

    @property
    def logs(self) -> Path:
        return self.debug / "logs"

    @property
    def dataset(self) -> Path:
        return self.debug / "dataset"

    @property
    def reconstruction(self) -> Path:
        return self.debug / "reconstruction"

    @property
    def lens_videos(self) -> Path:
        return self.debug / "lens_videos"

    def attempt(self, number: int) -> Path:
        return self.reconstruction / f"try_{number}"


@dataclass(frozen=True)
class RecordingData:
    manifest: dict
    imu_csv: Path
    times_csv: Path | None = None


@dataclass(frozen=True)
class ReconstructionSettings:
    environment: dict[str, str]
    tools_python: str
    fusion_python: str
    orb_directory: Path
    vocabulary: Path
    front_settings: Path
    front_mask: Path
    fusion_script: Path
    time_offset_ns: str
    rear_q_ic: str
    rear_lever: str
    camera_json: Path
    rear_settings: Path
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
    def parameters(self) -> config.TrajectoryParameters:
        return self.request.configuration.values


@dataclass(frozen=True)
class LensVideo:
    lens: str
    path: Path
    settings: Path
    mask: Path | None


@dataclass(frozen=True)
class MapAtlas:
    video: LensVideo
    archive: Path


@dataclass(frozen=True)
class AtlasMap:
    index: int
    keyframes: int


@dataclass(frozen=True)
class LensLocalization:
    lens: str
    frames_csv: Path


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


class ReconstructionError(RuntimeError):
    """阶段名和退出码供重试与报告使用；原因中保留具体日志位置。"""

    def __init__(self, stage: str, reason: str, *, return_code: int = 1):
        super().__init__(reason)
        self.stage = stage
        self.return_code = return_code


