"""视觉阶段的路径、设置与数据，不读取清单、不依赖业务请求。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from software import config

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
    def lens_videos(self) -> Path:
        return self.debug / "lens_videos"



@dataclass(frozen=True)
class RecordingData:
    manifest: dict
    imu_csv: Path
    times_csv: Path | None = None


@dataclass(frozen=True)
class VisualSettings:
    """镜头视频准备与 ORB 执行所需设置，供场景与独立轨迹共用。"""

    environment: dict[str, str]
    tools_python: str
    orb_directory: Path
    vocabulary: Path
    front_settings: Path
    front_mask: Path
    rear_settings: Path


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


class ProcessingError(RuntimeError):
    """阶段名和退出码供重试与报告使用；原因中保留具体日志位置。"""

    def __init__(self, stage: str, reason: str, *, return_code: int = 1):
        super().__init__(reason)
        self.stage = stage
        self.return_code = return_code


@dataclass(frozen=True)
class VisualContext:
    paths: RecordingPaths
    settings: VisualSettings
    parameters: config.TrajectoryParameters
    frame_count: int
    cores: str = ''
