"""Intrinsic provenance, separate from registry/user/legacy-sample selection."""

from __future__ import annotations

from pathlib import Path

from software import provenance as artifact_identity


SOURCES = ('measured', 'factory')


def validate(value: str) -> str:
    if value not in SOURCES:
        raise ValueError(f'内参来源必须为 measured 或 factory：{value!r}')
    return value


def optical_hash(camera: dict) -> str:
    """B must use exactly the intrinsics selected before camera–IMU calibration."""
    return artifact_identity.canonical_hash(dict(camera_type=camera['camera_type'], resolution=camera['resolution'],
        fx=camera['fx'], fy=camera['fy'], cx=camera['cx'], cy=camera['cy'], k=camera['k']))


def provenance(metadata: dict) -> dict:
    calibration_metadata = metadata.get('provenance', metadata)
    return calibration_metadata if isinstance(calibration_metadata, dict) else {}


def source(metadata: dict) -> str:
    # Existing measured bundles precede this field. The old selector named
    # "factory" (the lab sample) is handled separately by assets.describe().
    return validate(provenance(metadata).get('intrinsics_source', 'measured'))


def require(camera: dict, expected: str | None, *, label: str = '标定') -> None:
    if expected is not None and camera.get('intrinsics_source', 'measured') != validate(expected):
        raise ValueError(f'{label}的内参来源与请求的 {expected} 不一致；请选择对应的完整标定版本')


def validate_factory(metadata: dict, camera: dict, *, serial: str, lens: str) -> None:
    if source(metadata) != 'factory':
        return
    origin = provenance(metadata)
    factory = origin.get('factory_intrinsics', {})
    digest = optical_hash(camera)
    if (origin.get('extrinsics_source') != 'measured_with_selected_intrinsics'
            or origin.get('b_intrinsics_sha256') != digest
            or factory.get('b_intrinsics_sha256') != digest
            or factory.get('camera_serial') != serial or factory.get('lens') != lens
            or factory.get('lens_index') != (1 if lens == 'front' else 0)
            or factory.get('geometry', {}).get('kind') != 'kb4_forward'
            or factory.get('geometry', {}).get('maximum_ray_angle_degrees') != 85):
        raise ValueError('工厂内参缺少匹配的实测 B、镜头身份或视野证据')


def check_recording(path, origin: dict) -> None:
    """A frozen factory calibration cannot be replaced by a new per-video fit."""
    import numpy as np
    from software.camera.x3.factory import read_factory, lens_model
    factory = origin['factory_intrinsics']
    native = read_factory(path)
    if native['serial'] != factory['camera_serial']:
        raise ValueError('原片序列号与工厂内参来源设备不一致')
    index = factory['lens_index']
    reference = lens_model(next(x for x in factory['native']['recordings']['B']['lenses'] if x['index'] == index), 1440)
    current = lens_model(next(x for x in native['lenses'] if x['index'] == index), 1440)
    axis = np.linspace(0, 1439, 17)
    xx, yy = np.meshgrid(axis, axis)
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    a = reference.unproject(pixels)
    valid = np.isfinite(a).all(axis=1) & (a[:, 2] >= np.cos(np.deg2rad(85.)))
    b = current.unproject(pixels[valid])
    angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(a[valid], b), axis=1), np.sum(a[valid]*b, axis=1)))
    limit = factory['limits']['maximum_recording_ray_difference_deg']
    if not len(angles) or not np.isfinite(angles).all() or np.max(angles) > limit:
        raise ValueError('当前原片工厂内参与已验收 B 的内参来源不一致；请重新标定和建图')


def read_rear(path: Path, *, serial: str, expected_source: str | None = None, video: Path | None = None) -> tuple[dict, dict]:
    """Shared rear identity/acceptance/source checks for registry, snapshots and queries.

    Registry hash validation belongs to registry.rear_path; a frozen file is
    checked against its run manifest. Video-specific checks require a recording.
    """
    import json
    from software.trajectory.calibration.assets import read_basalt

    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError('环境镜头测量必须为 JSON 对象')
    validation = metadata.get('validation')
    if metadata.get('camera_serial') != serial or metadata.get('lens') != 'rear':
        raise ValueError('环境镜头需要匹配的 camera_serial 和 lens: rear')
    if (metadata.get('kind') != 'umi_rear_calibration' or not isinstance(validation, dict)
            or not str(validation.get('verdict', '')).startswith('PASS')):
        raise ValueError('缺少通过验收的环境镜头测量（PASS）；请执行 umi-calib run <rear目录>')
    measured = read_basalt(path)
    rear_source = source(metadata)
    require(dict(intrinsics_source=rear_source), expected_source, label='环境镜头')
    validate_factory(metadata, measured, serial=serial, lens='rear')
    if video is not None and rear_source == 'factory':
        check_recording(video, provenance(metadata))
    return dict(measured, intrinsics_source=rear_source), metadata
