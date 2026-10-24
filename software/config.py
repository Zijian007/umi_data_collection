"""Typed algorithm settings: explicit overlays, validation and reproducible records.

Numerical defaults live in config/*.yaml. Load once at a workflow boundary and
pass the immutable stage settings down. This module has no numerical dependencies
and never creates output directories or changes environment variables.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import get_type_hints

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from software.camera.x3.frame import validate_video_size

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class InputParameters:
    mode: str
    allow_irregular: bool


@dataclass(frozen=True)
class VideoParameters:
    size_px: int
    crf: int
    preset: str


@dataclass(frozen=True)
class OrbParameters:
    features: int
    scale_factor: float
    levels: int
    initial_fast: int
    minimum_fast: int
    far_points_map_units: float
    aruco_dictionary: int
    initial_tag_id: int
    initial_tag_size_m: float


@dataclass(frozen=True)
class RetryParameters:
    max_tries: int
    minimum_full_fraction: float
    maximum_placements: int


@dataclass(frozen=True)
class ImuParameters:
    gyro_noise: float
    acc_noise: float
    gyro_bias_rw: float
    acc_bias_rw: float


@dataclass(frozen=True)
class FusionParameters:
    gravity_m_s2: float
    step: int
    node_stride: int
    position_sigma_m: float
    segment_s: float
    continuity_log_scale_sigma: float
    continuity_rotation_sigma_deg: float
    continuity_position_sigma_m: float
    maximum_relative_scale_std: float
    other_lens_rms_m: float
    other_lens_rotation_deg: float
    lever_sigma_m: float
    optimize_front_lever: bool
    split_rotation_deg: float
    refine_chain_pairs: int
    group_scale_ratio: float
    adaptive_sigma: bool
    map_residual: bool
    acc_noise: float
    gyro_noise: float
    acc_bias_rw: float
    gyro_bias_rw: float
    integration_covariance: float
    huber: float
    visual_rotation_sigma_deg: float
    scale_prior_sigma: float
    gauge_rotation_sigma_rad: float
    rotation_prior_sigma_rad: float
    gauge_position_sigma_m: float
    position_prior_sigma_m: float
    acc_bias_prior_sigma: float
    gyro_bias_prior_sigma: float
    velocity_prior_sigma_m_s: float
    initial_static_gyro_rad_s: float
    initial_bias_max_samples: int
    initial_bias_min_samples: int
    minimum_alignment_pairs: int
    minimum_chain_pairs: int
    gravity_tolerance_m_s2: float
    maximum_acc_bias_m_s2: float
    minimum_common_frames: int
    lm_iterations: int
    linear_iterations: int
    linear_tolerance: float


@dataclass(frozen=True)
class ZuptParameters:
    enabled: bool
    half_window_s: float
    minimum_samples: int
    maximum_gyro_rad_s: float
    maximum_acc_norm_std_m_s2: float
    velocity_sigma_m_s: float


@dataclass(frozen=True)
class NoteParameters:
    minimum_full_fraction: float
    maximum_leading_imu_s: float
    minimum_mapping_fraction: float


@dataclass(frozen=True)
class TrajectoryParameters:
    schema_version: int
    input: InputParameters
    video: VideoParameters
    orb: OrbParameters
    retry: RetryParameters
    fusion: FusionParameters
    zupt: ZuptParameters
    notes: NoteParameters


@dataclass(frozen=True)
class BoardParameters:
    intrinsics_step: int
    camera_imu_step: int
    validation_step: int
    detection_upscale: float
    minimum_pnp_tags: int
    minimum_reference_tags: int
    normalized_inlier_threshold: float
    minimum_bearing_z: float
    refinement_iterations: int
    minimum_comparison_frames: int
    coverage_radius_px: float
    coverage_reach: float
    coverage_percentile: float
    minimum_direction_frames: int


@dataclass(frozen=True)
class GravityParameters:
    window_s: float
    maximum_gyro_deg_s: float
    maximum_acc_norm_std_m_s2: float
    minimum_samples: int
    gravity_m_s2: float
    distinct_direction_cosine: float
    minimum_window_span_fraction: float


@dataclass(frozen=True)
class TrajectoryValidationParameters:
    config: str


@dataclass(frozen=True)
class CheckParameters:
    ate_fail_mm: float
    ate_warn_mm: float
    scale_warn: float
    scale_fail: float
    minimum_board_frames: int
    minimum_board_extent_m: float
    intrinsics_minimum_corners: int
    intrinsics_minimum_reach: float
    intrinsics_reprojection_px: float
    focal_warn_fraction: float
    focal_fail_fraction: float
    principal_point_warn_px: float
    camera_imu_minimum_corners: int
    minimum_gyro_p95_rad_s: float
    camera_imu_reprojection_px: float
    translation_warn_mm: float
    translation_fail_mm: float
    rotation_warn_deg: float
    rotation_fail_deg: float
    time_offset_warn_ms: float
    gravity_residual_m_s2: float
    minimum_gravity_directions: int


@dataclass(frozen=True)
class CalibrationParameters:
    schema_version: int
    imu: ImuParameters
    board: BoardParameters
    gravity: GravityParameters
    trajectory_validation: TrajectoryValidationParameters
    checks: CheckParameters


def canonical_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _merge(base: dict, overlay: dict, prefix: str = '') -> dict:
    result = dict(base)
    for name, value in overlay.items():
        if not isinstance(name, str):
            raise ValueError(f'{prefix or "顶层"} 配置项名称必须是字符串')
        key = prefix + name
        if name not in base:
            raise ValueError(f'未知配置项：{key}')
        if isinstance(base[name], dict):
            if not isinstance(value, dict):
                raise ValueError(f'{key} 必须是配置表')
            result[name] = _merge(base[name], value, key + '.')
        else:
            result[name] = value
    return result


def _typed(cls, data: dict, prefix: str = ''):
    if not isinstance(data, dict):
        raise ValueError(f"{prefix or cls.__name__} 必须是配置表")
    hints = get_type_hints(cls)
    result = {}
    if data.keys() != hints.keys():
        raise ValueError(f'{prefix or cls.__name__} 配置字段不完整或含未知项')
    for field in fields(cls):
        name, typ = field.name, hints[field.name]
        value, key = data[name], prefix + name
        if is_dataclass(typ):
            if not isinstance(value, dict):
                raise ValueError(f'{key} 必须是配置表')
            result[name] = _typed(typ, value, key + '.')
            continue
        if typ is float and type(value) in (float, int):
            value = float(value)
        elif type(value) is not typ:
            raise ValueError(f'{key} 类型错误：需要 {typ.__name__}')
        if typ in (float, int) and not math.isfinite(value):
            raise ValueError(f'{key} 必须是有限数值')
        result[name] = value
    return cls(**result)


def _validate(values) -> None:
    if values.schema_version != SCHEMA_VERSION:
        raise ValueError(f'不支持配置版本 {values.schema_version}；需要 {SCHEMA_VERSION}')
    # 除以下明确允许关闭/零起点的量，其余算法数值必须为正。
    zero_allowed = {'crf', 'initial_tag_id', 'aruco_dictionary', 'segment_s',
                    'split_rotation_deg', 'refine_chain_pairs', 'group_scale_ratio',
                    'minimum_full_fraction', 'minimum_mapping_fraction', 'maximum_leading_imu_s'}
    def walk(obj, prefix=''):
        for f in fields(obj):
            v, key = getattr(obj, f.name), prefix + f.name
            if is_dataclass(v):
                walk(v, key + '.')
            elif type(v) in (int, float) and (v < 0 if f.name in zero_allowed else v <= 0):
                raise ValueError(f'{key} 必须为{"非负" if f.name in zero_allowed else "正"}数')
    walk(values)
    if isinstance(values, TrajectoryParameters):
        p = values
        if p.input.mode not in ('handheld', 'dual'):
            raise ValueError('input.mode 必须为 handheld 或 dual')
        validate_video_size(p.video.size_px)
        if not 0 <= p.video.crf <= 51:
            raise ValueError('video.crf 必须在 0–51 之间')
        if p.video.preset not in ('ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'):
            raise ValueError('video.preset 不是有效的 x264 preset')
        if p.orb.scale_factor <= 1 or p.orb.minimum_fast > p.orb.initial_fast:
            raise ValueError('orb.scale_factor 必须 > 1；minimum_fast 不能大于 initial_fast')
        if p.orb.aruco_dictionary > 20:
            raise ValueError('orb.aruco_dictionary 必须在 0–20 之间（当前 OpenCV 引擎）')
        if 0 < p.fusion.group_scale_ratio < 1:
            raise ValueError('fusion.group_scale_ratio 必须为 0（关闭）或 >= 1')
        if p.fusion.initial_bias_min_samples > p.fusion.initial_bias_max_samples:
            raise ValueError('fusion.initial_bias_min_samples 不能大于 initial_bias_max_samples')
        for key, value in [('retry.minimum_full_fraction', p.retry.minimum_full_fraction),
                           ('notes.minimum_full_fraction', p.notes.minimum_full_fraction),
                           ('notes.minimum_mapping_fraction', p.notes.minimum_mapping_fraction)]:
            if not 0 <= value <= 1:
                raise ValueError(f'{key} 必须在 0–1 之间')
    else:
        p = values
        for warn, fail in [('ate_warn_mm', 'ate_fail_mm'), ('scale_warn', 'scale_fail'),
                           ('focal_warn_fraction', 'focal_fail_fraction'), ('translation_warn_mm', 'translation_fail_mm'),
                           ('rotation_warn_deg', 'rotation_fail_deg')]:
            if getattr(p.checks, warn) > getattr(p.checks, fail):
                raise ValueError(f'checks.{warn} 不能大于 {fail}')
        if not 0 < p.gravity.distinct_direction_cosine < 1:
            raise ValueError('gravity.distinct_direction_cosine 必须在 0–1 之间')
        if not 0 < p.gravity.minimum_window_span_fraction <= 1:
            raise ValueError('gravity.minimum_window_span_fraction 必须在 0–1 之间')
        if not 0 < p.board.minimum_bearing_z < 1 or not 0 < p.board.coverage_percentile <= 100:
            raise ValueError('board.minimum_bearing_z 或 coverage_percentile 超出范围')
        if p.board.minimum_reference_tags < p.board.minimum_pnp_tags or p.board.minimum_pnp_tags < 4:
            raise ValueError('board 需要 minimum_reference_tags >= minimum_pnp_tags >= 4')


def from_dict(kind: str, data: dict):
    """Decode a complete snapshot, including child-process boundaries; no defaults."""
    cls = {'calibration': CalibrationParameters, 'trajectory': TrajectoryParameters}[kind]
    values = _typed(cls, data)
    _validate(values)
    return values


@dataclass(frozen=True)
class Configuration:
    kind: str
    values: CalibrationParameters | TrajectoryParameters
    sources: tuple[str, ...]
    overrides_json: str
    cli_overrides_json: str = "{}"

    @property
    def computation_hash(self) -> str:
        data = asdict(self.values)
        data.pop('notes' if self.kind == 'trajectory' else 'checks')
        # Related trajectory values are fingerprinted separately, not their file location.
        data.pop('trajectory_validation', None)
        return canonical_hash(data)

    @property
    def evaluation_hash(self) -> str:
        return canonical_hash(asdict(self.values.notes if self.kind == 'trajectory' else self.values.checks))

    def snapshot(self) -> dict:
        return dict(kind=self.kind, schema_version=SCHEMA_VERSION, values=asdict(self.values),
                    sources=list(self.sources), overrides=json.loads(self.overrides_json),
                    cli_overrides=json.loads(self.cli_overrides_json),
                    computation_hash=self.computation_hash, evaluation_hash=self.evaluation_hash)

    def load_trajectory_validation(self) -> Configuration:
        path = self.values.trajectory_validation.config
        if not path:
            return load('trajectory')
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = Path(self.sources[-1]).parent / target
        return load('trajectory', target)


def _read_yaml(path: Path) -> dict:
    """Read one YAML 1.2 mapping; reject duplicate keys and executable tags."""
    parser = YAML(typ='safe', pure=True)
    parser.version = (1, 2)
    parser.allow_duplicate_keys = False
    with path.open(encoding='utf-8') as source:
        data = parser.load(source)
    if parser.version != (1, 2):
        raise ValueError(f'{path} 仅支持 YAML 1.2；请移除其他版本的 YAML 指令')
    if not isinstance(data, dict):
        raise ValueError(f'{path} 的配置顶层必须是映射（键: 值）；空覆盖请写 {{}}')
    return data


def load(kind: str, path: Path | None = None, overrides: dict | None = None) -> Configuration:
    default_path = ROOT / 'config' / f'{kind}.yaml'
    try:
        defaults = _read_yaml(default_path)
        sources = [str(default_path)]
        overlay = {}
        if path is not None:
            path = Path(path).expanduser().resolve()
            overlay = _read_yaml(path)
            sources.append(str(path))
        data = _merge(defaults, overlay)
        data = _merge(data, overrides or {})
        effective = from_dict(kind, data)
        changed = _differences(defaults, asdict(effective))
        return Configuration(kind, effective, tuple(sources), json.dumps(changed, sort_keys=True),
                             json.dumps(overrides or {}, sort_keys=True))
    except (OSError, YAMLError) as error:
        raise ValueError(f'读取 {kind} 配置失败：{error}') from error


def _differences(defaults: dict, effective: dict) -> dict:
    return {name: (_differences(defaults[name], value) if isinstance(value, dict) else value)
            for name, value in effective.items() if defaults[name] != value}


def resolve_trajectory_request(request):
    """Only explicit CLI/request fields override YAML; runtime options stay outside."""
    if request.configuration is not None:
        return request
    overrides = {}
    for section, name, value in [('input', 'mode', request.mode),
                                  ('input', 'allow_irregular', request.allow_irregular),
                                  ('retry', 'max_tries', request.max_tries)]:
        if value is not None:
            overrides.setdefault(section, {})[name] = value
    resolved = load('trajectory', request.config, overrides)
    p = resolved.values
    return replace(request, configuration=resolved, mode=p.input.mode, allow_irregular=p.input.allow_irregular,
                   max_tries=p.retry.max_tries)


def code_identity(domain: str) -> dict:
    """Content hashes cover uncommitted source too; never rely on Git HEAD alone."""
    paths = [ROOT / 'software/config.py', *sorted((ROOT / 'software' / domain).rglob('*.py')),
             *sorted((ROOT / 'software/camera').rglob('*.py'))]
    if domain == 'trajectory':
        paths += sorted((ROOT / 'software/trajectory/assets/config').glob('*.yaml'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def file_hash(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def engine_identity(domain: str) -> dict:
    """Hash installed engines and numerical package metadata without running them."""
    import os
    import shutil
    import sys
    paths = {}
    if domain == 'trajectory':
        orb = Path(os.environ.get('ORB3_DIR', ROOT / '.deps/orbslam3'))
        paths.update(orb_wrapper=orb / 'Examples/Monocular-Inertial/x3_mono_slam',
                     orb_library=orb / 'lib/libORB_SLAM3.so',
                     orb_vocabulary=Path(os.environ.get('ORB3_VOCAB') or orb / 'Vocabulary/ORBvoc.txt'))
    else:
        basalt = Path(os.environ.get('UMI_BASALT_DIR') or
                      Path(os.environ.get('UMI_DEPS', ROOT / '.deps')) / 'basalt/umi-basalt-0.1.7-fsoffset-linux-x86_64') / 'bin'
        paths.update({name: basalt / name for name in ('basalt_calibrate', 'basalt_calibrate_imu')})
    if domain == 'trajectory':
        for path in sorted(orb.glob('Thirdparty/*/lib/*.so*')):
            paths[str(path.relative_to(orb))] = path
    for name in ('ffmpeg', 'ffprobe'):
        executable = shutil.which(name)
        if executable:
            paths[name] = Path(executable)
    python = Path(os.environ.get('KIWI_PY') or sys.executable)
    paths['fusion_python'] = python
    # Package metadata identifies dependency versions without importing heavy modules.
    prefixes = {python.parent.parent, Path(sys.prefix)}
    for prefix in prefixes:
        for path in prefix.glob('lib/python*/site-packages/gtsam/*.so'):
            paths[f'{prefix.name}/gtsam/{path.name}'] = path
        for package in ('numpy', 'scipy', 'gtsam', 'opencv*', 'av'):
            for path in prefix.glob(f'lib/python*/site-packages/{package}-*.dist-info/METADATA'):
                paths[f'{prefix.name}/{path.parent.name}'] = path
    return {name: _engine_hash(path, path.stat().st_mtime_ns, path.stat().st_size) if path.is_file() else 'unavailable'
            for name, path in sorted(paths.items())}


def write_yaml(path: Path, values) -> None:
    """Emit the complete resolved parameters for a child process."""
    writer = YAML(typ='safe', pure=True)
    writer.default_flow_style = False
    writer.sort_base_mapping_type_on_output = False
    writer.allow_unicode = True
    with path.open('w', encoding='utf-8') as output:
        writer.dump(asdict(values), output)


# Engines are immutable during a run; cache only their hashes by filesystem
# identity. Camera/gripper assets are deliberately hashed on every selection.

@lru_cache(maxsize=128)
def _engine_hash(path: Path, mtime_ns: int, size: int) -> str:
    return file_hash(path)
