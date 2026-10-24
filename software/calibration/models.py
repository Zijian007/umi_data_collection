"""Data passed between calibration stages; no solvers or filesystem mutations."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path

from software import config


@dataclass(frozen=True)
class SegmentPaths:
    a: Path
    b: Path
    c: Path
    d: Path

    def items(self) -> Iterator[tuple[str, Path]]:
        return zip('ABCD', (self.a, self.b, self.c, self.d))


@dataclass(frozen=True)
class CalibrationRequest:
    folder: Path | None
    recordings: SegmentPaths | None
    output: Path | None
    tag_size_mm: float
    cores: str | None = None
    keep_work: bool = False
    force: bool = False
    install: bool = False
    gripper: str | None = None
    compare_sample: bool = False
    config: Path | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class VerificationRequest:
    video: Path
    output: Path | None
    tag_size_mm: float
    cores: str | None = None
    keep_work: bool = False
    force: bool = False
    config: Path | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class VerificationInputs:
    video: Path
    camera_serial: str
    grid: dict
    calibration: dict
    calibration_name: str


@dataclass(frozen=True)
class RunPaths:
    output: Path

    @property
    def diagnostics(self) -> Path:
        return self.output / 'diagnostics'

    @property
    def work(self) -> Path:
        return self.diagnostics / 'work'

    @property
    def logs(self) -> Path:
        return self.diagnostics / 'logs'

    @property
    def board(self) -> Path:
        return self.diagnostics / 'aprilgrid.json'

    @property
    def bundle(self) -> Path:
        return self.output / 'bundle'

    @property
    def calibration(self) -> Path:
        return self.bundle / 'calibration_r1440.json'


@dataclass(frozen=True)
class CalibrationInputs:
    recordings: SegmentPaths
    camera_serial: str
    grid: dict


@dataclass(frozen=True)
class StageContext:
    """Shared paths, resolved parameters and execution options; stage results are explicit."""
    paths: RunPaths
    grid: dict
    cores: str | None
    workers: int
    parameters: config.CalibrationParameters
    trajectory_configuration: config.Configuration

    @property
    def cpu_prefix(self) -> list[str]:
        return ['taskset', '-c', self.cores] if self.cores else []


@dataclass(frozen=True)
class RecordingData:
    segments: SegmentPaths
    imu_metrics: dict[str, dict]


@dataclass(frozen=True)
class IntrinsicsResult:
    calibration_json: Path
    kb4: list[float]
    solver_metrics: dict
    board_metrics: dict


@dataclass(frozen=True)
class CameraImuResult:
    calibration_json: Path
    parameters: dict
    solver_metrics: dict
    board_metrics: dict


@dataclass(frozen=True)
class GravityResult:
    status: str | None
    windows: int | None
    directions: int | None
    max_residual: float | None


@dataclass(frozen=True)
class BundleResult:
    directory: Path
    calibration_json: Path
    calibration_id: str


@dataclass(frozen=True)
class TrajectoryValidation:
    candidate: dict
    reference: dict | None = None


class StageError(RuntimeError):
    """A failed stage and any measurements completed before it failed."""
    def __init__(self, stage: str, reason: str, *, imu_metrics: dict[str, dict] | None = None):
        super().__init__(reason)
        self.stage = stage
        self.imu_metrics = imu_metrics
