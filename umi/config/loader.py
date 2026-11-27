"""Typed algorithm settings: explicit overlays, validation and reproducible records.

Numerical defaults live in config/algorithms/*.yaml. Load once at a workflow boundary and
pass the immutable stage settings down. This module has no numerical dependencies
and never creates output directories or changes environment variables.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

from ruamel.yaml.error import YAMLError

from umi.config.parameter_models import CalibrationParameters, TrajectoryParameters
from umi.core.frame import validate_video_parameters
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import read_yaml
from umi.infra.file_store import write_yaml as write_mapping
from umi.infra.paths import CONFIG_ROOT

ALGORITHM_DIRECTORY = CONFIG_ROOT / 'algorithms'
# 内部 TrajectoryParameters 快照的结构版本；公开文件的版本见 reconstruction.SCHEMA_VERSION。
SCHEMA_VERSION = 1
# 标定 B 段的 IMU 噪声先验与融合共用 reconstruction.yaml 的 fusion.imu。
CALIBRATION_IMU_FIELDS = ('gyro_noise', 'acc_noise', 'gyro_bias_rw', 'acc_bias_rw')

_CALIBRATION_CHECK_FIELDS = {
    'intrinsics_mean_reprojection_px': 'intrinsics_reprojection_px',
    'camera_imu_mean_reprojection_px': 'camera_imu_reprojection_px',
}


def _read_calibration_yaml(path: Path) -> dict:
    document = read_yaml(path)
    checks = document.get('checks')
    if isinstance(checks, dict):
        unknown = checks.keys() & _CALIBRATION_CHECK_FIELDS.values()
        if unknown:
            raise ValueError(f'{path}: checks 含未知字段 {sorted(unknown)}')
        document['checks'] = {_CALIBRATION_CHECK_FIELDS.get(name, name): value for name, value in checks.items()}
    return document


def validate_environment_tags(parameters) -> None:
    """场景与独立轨迹共用的海报参数关系检查。标量类型在各入口解码。"""
    if parameters.tag_minimum_frames < 3:
        raise ValueError('tag_minimum_frames 至少为 3')
    if not 0 <= parameters.tag_maximum_cull_fraction < 1:
        raise ValueError('tag_maximum_cull_fraction 必须在 [0, 1) 内')
    if not 0 < parameters.tag_minimum_retained_time_fraction <= 1:
        raise ValueError('tag_minimum_retained_time_fraction 必须在 (0, 1] 内')
    if not 0 < parameters.tag_rear_rotation_maximum_correction_deg <= 180:
        raise ValueError('tag_rear_rotation_maximum_correction_deg 必须在 (0, 180] 内')


def merge_overrides(base: dict, overlay: dict, prefix: str = '') -> dict:
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
            result[name] = merge_overrides(base[name], value, key + '.')
        else:
            result[name] = value
    return result


def decode_dataclass(cls, configuration_values: dict, prefix: str = ''):
    if not isinstance(configuration_values, dict):
        raise ValueError(f"{prefix or cls.__name__} 必须是配置表")
    hints = get_type_hints(cls)
    result = {}
    if configuration_values.keys() != hints.keys():
        raise ValueError(f'{prefix or cls.__name__} 配置字段不完整或含未知项')
    for field in fields(cls):
        name, typ = field.name, hints[field.name]
        value, key = configuration_values[name], prefix + name
        if is_dataclass(typ):
            if not isinstance(value, dict):
                raise ValueError(f'{key} 必须是配置表')
            result[name] = decode_dataclass(typ, value, key + '.')
            continue
        if get_origin(typ) is tuple:
            item_types = get_args(typ)
            if (not isinstance(value, (list, tuple)) or len(value) != len(item_types)
                    or any(type(item) is not item_type for item, item_type in zip(value, item_types))):
                raise ValueError(f'{key} 必须包含 {len(item_types)} 个指定类型的值')
            value = tuple(value)
        elif typ is float and type(value) in (float, int):
            value = float(value)
        elif type(value) is not typ:
            raise ValueError(f'{key} 类型错误：需要 {typ.__name__}')
        if typ in (float, int) and not math.isfinite(value):
            raise ValueError(f'{key} 必须是有限数值')
        result[name] = value
    return cls(**result)


def _validate_parameters(values) -> None:
    if values.schema_version != SCHEMA_VERSION:
        raise ValueError(f'不支持配置版本 {values.schema_version}；需要 {SCHEMA_VERSION}')
    # 除以下明确允许关闭/零起点的量，其余算法数值必须为正。
    zero_allowed = {'crf', 'initial_tag_id', 'aruco_dictionary', 'segment_s',
                    'split_rotation_deg', 'refine_chain_pairs', 'group_scale_ratio',
                    'minimum_full_fraction', 'minimum_mapping_fraction', 'maximum_leading_imu_s', 'tag_sigma_per_radius_px',
                    'tag_maximum_cull_fraction'}
    def walk(parameter_group, prefix=''):
        for f in fields(parameter_group):
            v, key = getattr(parameter_group, f.name), prefix + f.name
            if is_dataclass(v):
                walk(v, key + '.')
            elif key in ('video.crop.offset_x_px', 'video.crop.offset_y_px'):
                continue  # 裁切中心可向任一方向偏移，边界统一由几何校验。
            elif type(v) in (int, float) and (v < 0 if f.name in zero_allowed else v <= 0):
                raise ValueError(f'{key} 必须为{"非负" if f.name in zero_allowed else "正"}数')
    walk(values)
    if isinstance(values, TrajectoryParameters):
        p = values
        validate_environment_tags(p.environment_tags)
        if p.input.mode not in ('handheld', 'dual'):
            raise ValueError('input.mode 必须为 handheld 或 dual')
        validate_video_parameters(p.video)
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
        if p.retry.maximum_scale_change_ratio <= 1:
            raise ValueError('retry.maximum_scale_change_ratio 必须 > 1')
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


def decode_parameters(kind: str, configuration_values: dict):
    """Decode a complete snapshot, including child-process boundaries; no defaults."""
    cls = {'calibration': CalibrationParameters, 'trajectory': TrajectoryParameters}[kind]
    values = decode_dataclass(cls, configuration_values)
    _validate_parameters(values)
    return values


def computation_values(kind: str, values: dict) -> dict:
    configuration_values = json.loads(json.dumps(values))
    configuration_values.pop('notes' if kind == 'trajectory' else 'checks', None)
    configuration_values.pop('trajectory_validation', None)
    if kind == 'calibration' and 'factory' in configuration_values:
        for field in ('minimum_holdout_frames', 'holdout_warn_rms_px'):
            configuration_values['factory'].pop(field, None)
    return configuration_values


@dataclass(frozen=True)
class Configuration:
    kind: str
    values: CalibrationParameters | TrajectoryParameters
    sources: tuple[str, ...]
    overrides_json: str
    cli_overrides_json: str = "{}"
    trajectory_validation_source: Path | None = None

    @property
    def computation_hash(self) -> str:
        return artifact_identity.canonical_hash(computation_values(self.kind, asdict(self.values)))

    @property
    def evaluation_hash(self) -> str:
        if self.kind == 'trajectory':
            return artifact_identity.canonical_hash(asdict(self.values.notes))
        return artifact_identity.canonical_hash(dict(checks=asdict(self.values.checks), factory=dict(
            minimum_holdout_frames=self.values.factory.minimum_holdout_frames,
            holdout_warn_rms_px=self.values.factory.holdout_warn_rms_px)))

    def snapshot(self) -> dict:
        return dict(kind=self.kind, schema_version=SCHEMA_VERSION, values=asdict(self.values),
                    sources=list(self.sources), overrides=json.loads(self.overrides_json),
                    cli_overrides=json.loads(self.cli_overrides_json),
                    computation_hash=self.computation_hash, evaluation_hash=self.evaluation_hash)

    def load_trajectory_validation(self, *, tag_size_m: float | None = None) -> Configuration:
        """Resolve C's config; its initialization tag shares the measured board size."""
        path = self.values.trajectory_validation.config
        target = Path(path).expanduser() if path else None
        if target is not None and not target.is_absolute():
            source_path = self.trajectory_validation_source or Path(self.sources[0])
            target = source_path.parent / target
        # C validates one candidate camera independently; task-only poster/dual
        # selections must never leak in from a shared trajectory configuration.
        overrides = {'environment_tags': {'tag_fusion_enabled': False}, 'input': {'mode': 'handheld'}}
        if tag_size_m is not None:
            overrides['orb'] = {'initial_tag_size_m': tag_size_m}
        return load('trajectory', target, overrides)


def overlay_trajectory(configuration: Configuration, overlay: dict, source: Path) -> Configuration:
    values = merge_overrides(asdict(configuration.values), overlay)
    effective = decode_parameters('trajectory', values)
    defaults, _ = _project_defaults('trajectory')
    return replace(configuration, values=effective, sources=(*configuration.sources, str(source)),
                   overrides_json=json.dumps(_changed_values(defaults, asdict(effective)), sort_keys=True))


def _project_defaults(kind: str) -> tuple[dict, list[str]]:
    from umi.config import reconstruction
    if kind == 'trajectory':
        return dict(reconstruction.defaults().trajectory, schema_version=SCHEMA_VERSION), [str(reconstruction.DEFAULT_PATH)]
    default_path = ALGORITHM_DIRECTORY / 'calibration.yaml'
    defaults = _read_calibration_yaml(default_path)
    if 'imu' in defaults:
        raise ValueError(f'{default_path}: IMU 默认值由 reconstruction.yaml 的 fusion.imu 提供')
    fusion = reconstruction.defaults().trajectory['fusion']
    defaults['imu'] = {name: fusion[name] for name in CALIBRATION_IMU_FIELDS}
    return defaults, [str(default_path), str(reconstruction.DEFAULT_PATH)]


def _overlay(kind: str, path: Path) -> dict:
    if kind == 'trajectory':
        from umi.config import reconstruction
        return reconstruction.read_overlay(path, trajectory_only=True).trajectory
    return _read_calibration_yaml(path)


def load(kind: str, path: Path | None = None, overrides: dict | None = None, *, inline: dict | None = None) -> Configuration:
    """path 是用户覆盖文件（轨迹为 reconstruction.yaml 同结构）；inline/overrides 是内部结构。"""
    try:
        defaults, sources = _project_defaults(kind)
        overlay = {}
        if path is not None:
            path = Path(path).expanduser().resolve()
            overlay = _overlay(kind, path)
            sources.append(str(path))
        configuration_values = merge_overrides(defaults, overlay)
        configuration_values = merge_overrides(configuration_values, inline or {})
        configuration_values = merge_overrides(configuration_values, overrides or {})
        effective = decode_parameters(kind, configuration_values)
        changed = _changed_values(defaults, asdict(effective))
        validation_source = None
        if kind == 'calibration':
            validation_source = path if 'config' in overlay.get('trajectory_validation', {}) else Path(sources[0])
        return Configuration(kind, effective, tuple(sources), json.dumps(changed, sort_keys=True),
                             json.dumps(overrides or {}, sort_keys=True), trajectory_validation_source=validation_source)
    except (OSError, YAMLError) as error:
        raise ValueError(f'读取 {kind} 配置失败：{error}') from error


def _changed_values(defaults: dict, effective: dict) -> dict:
    return {name: (_changed_values(defaults[name], value) if isinstance(value, dict) else value)
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


def write_yaml(path: Path, values) -> None:
    """Emit the complete resolved parameters for a child process."""
    document = asdict(values)
    if isinstance(values, TrajectoryParameters):
        # 子进程按用户覆盖文件的同一入口读取：输出完整的 reconstruction.yaml 同结构树。
        from umi.config import reconstruction
        document = reconstruction.public_tree(trajectory=document)
    else:
        for name, internal in _CALIBRATION_CHECK_FIELDS.items():
            document['checks'][name] = document['checks'].pop(internal)
    write_mapping(path, document)

