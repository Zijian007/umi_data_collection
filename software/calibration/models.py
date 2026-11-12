"""Data passed between calibration stages; no solvers or filesystem mutations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple


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
    compare_sample: bool = False
    config: Path | None = None
    dry_run: bool = False
    resume: bool = False
    workers: int | None = None
    lens: str = 'front'
    intrinsics_source: str = 'measured'

    @property
    def output_directory(self) -> Path:
        default_name = 'calibration_result' + ('_factory' if self.intrinsics_source == 'factory' else '')
        source_directory = self.folder or (self.recordings.a.parent if self.recordings else None)
        if self.output is None and source_directory is None:
            raise ValueError('需要标定输入目录或录像才能确定结果目录')
        return (self.output or source_directory/default_name).expanduser().resolve()


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
    resume: bool = False
    workers: int | None = None


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
    def run_directory(self) -> Path:
        return self.output / 'run'

    @property
    def record(self) -> Path:
        return self.run_directory / 'record.json'

    @property
    def work(self) -> Path:
        return self.run_directory / 'work'

    @property
    def logs(self) -> Path:
        return self.run_directory / 'logs'

    @property
    def board(self) -> Path:
        return self.work / 'aprilgrid.json'

    @property
    def calibration(self) -> Path:
        return self.output / 'calibration.json'

    @property
    def manifest(self) -> Path:
        return self.output / 'manifest.json'


@dataclass(frozen=True)
class CalibrationInputs:
    recordings: SegmentPaths
    camera_serial: str
    grid: dict


class Kb4(NamedTuple):
    """KB4 intrinsics (measured or factory-fitted), in 1440×1440 crop pixels."""
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    k4: float

    @property
    def principal_point(self) -> tuple[float, float]:
        return self.cx, self.cy


@dataclass(frozen=True)
class IntrinsicsResult:
    calibration_json: Path
    kb4: Kb4
    solver_metrics: dict
    board_metrics: dict

    def __post_init__(self):
        # Normalize solver values for named access throughout the pipeline.
        object.__setattr__(self, 'kb4', Kb4(*self.kb4))


@dataclass(frozen=True)
class CameraImuResult:
    calibration_json: Path
    parameters: dict
    solver_metrics: dict


@dataclass(frozen=True)
class GravityResult:
    status: str | None
    windows: int | None
    directions: int | None
    max_residual: float | None
    evidence: dict | None = None


@dataclass(frozen=True)
class TrajectoryValidation:
    candidate: dict
    reference: dict | None = None


class Decision(StrEnum):
    ACCEPTED = 'accepted'
    WARNINGS = 'accepted_with_warnings'
    REJECTED = 'rejected'
    INCOMPLETE = 'incomplete'

    @property
    def installable(self):
        return self in (self.ACCEPTED, self.WARNINGS)

    @property
    def label(self):
        return {self.ACCEPTED: 'PASS', self.WARNINGS: 'PASS（有提醒，见报告）',
                self.REJECTED: 'FAIL', self.INCOMPLETE: 'INCOMPLETE（证据不完整）'}[self]


class Step(StrEnum):
    PREPARE_INTRINSICS = 'prepare_intrinsics'
    PREPARE_CAMERA_IMU = 'prepare_camera_imu'
    PREPARE_VALIDATION = 'prepare_validation'
    PREPARE_GRAVITY = 'prepare_gravity'
    INTRINSICS = 'estimate_intrinsics'
    CAMERA_IMU = 'estimate_camera_imu'
    FACTORY = 'compare_insv_factory'
    GRAVITY = 'measure_gravity'
    TRAJECTORY = 'validate_trajectory'
    SAMPLE = 'sample_trajectory'

    @property
    def label(self):
        return {self.PREPARE_INTRINSICS: '准备 A 段图像与 IMU', self.PREPARE_CAMERA_IMU: '准备 B 段图像与 IMU',
                self.PREPARE_VALIDATION: '准备 C 段图像与 IMU', self.PREPARE_GRAVITY: '准备 D 段 IMU',
                self.INTRINSICS: '内参生成与 A 图像验证', self.CAMERA_IMU: 'B 相机–IMU 标定',
                self.GRAVITY: 'D 重力诊断', self.TRAJECTORY: 'C 轨迹验证',
                self.SAMPLE: '样例轨迹对照', self.FACTORY: 'INSV 工厂镜头参数对照'}[self]


ROLE_STEPS = dict(A=Step.PREPARE_INTRINSICS, B=Step.PREPARE_CAMERA_IMU,
                  C=Step.PREPARE_VALIDATION, D=Step.PREPARE_GRAVITY)


@dataclass(frozen=True)
class ExecutionOptions:
    paths: RunPaths
    cores: str | None
    workers: int
    timings: dict = field(default_factory=dict, compare=False)

    @property
    def cpu_prefix(self):
        return ['taskset', '-c', self.cores] if self.cores else []
