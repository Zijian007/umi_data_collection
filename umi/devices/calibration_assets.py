"""Read camera–IMU measurements and build or validate versioned assets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from umi.core.frame import CROP
from umi.infra.provenance import canonical_hash, file_hash

SAMPLE_CAMERA_JSON = Path(__file__).parent / 'assets/samples/calibration_r1440.json'
SAMPLE_REAR_GEOMETRY = SAMPLE_CAMERA_JSON.with_name('rear_sample.json')


class CalibrationError(ValueError):
    pass


def read_basalt(path: Path) -> dict:
    """The fields the pipeline needs from a Basalt calibration.json (single kb4 camera, 1440x1440 crop)."""
    raw_data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_basalt(raw_data, label=str(path))


def parse_basalt(raw_data: dict, *, label: str) -> dict:
    """Validate an in-memory measurement with the same rules as a frozen file."""
    import numpy as np

    path = label
    raw_data = raw_data.get("value0") or raw_data
    try:
        imu_cam_extrinsics = raw_data["T_imu_cam"][0]
        camera_intrinsics = raw_data["intrinsics"][0]
        resolution = raw_data["resolution"][0]
        cam_time_offset_ns = int(raw_data["cam_time_offset_ns"])
        intrinsics_params = camera_intrinsics["intrinsics"]
        parsed_calibration = {
            "q_xyzw": [imu_cam_extrinsics["qx"], imu_cam_extrinsics["qy"], imu_cam_extrinsics["qz"], imu_cam_extrinsics["qw"]],
            "p": [imu_cam_extrinsics["px"], imu_cam_extrinsics["py"], imu_cam_extrinsics["pz"]],
            "fx": intrinsics_params["fx"], "fy": intrinsics_params["fy"],
            "cx": intrinsics_params["cx"], "cy": intrinsics_params["cy"],
            "k": [intrinsics_params["k1"], intrinsics_params["k2"], intrinsics_params["k3"], intrinsics_params["k4"]],
            "cam_time_offset_ns": cam_time_offset_ns,
            "camera_type": camera_intrinsics["camera_type"],
            "resolution": list(resolution),
        }
    except (KeyError, IndexError, TypeError) as error:
        raise CalibrationError(f"{path} 不是有效的 Basalt 相机–IMU 标定：{error}") from error
    if parsed_calibration["camera_type"] != "kb4" or parsed_calibration["resolution"] != [CROP, CROP]:
        raise CalibrationError(f"{path} 需要单相机 kb4 标定，分辨率 {CROP}×{CROP}；实际为 {parsed_calibration['camera_type']} {parsed_calibration['resolution']}")
    numeric = [*parsed_calibration['q_xyzw'], *parsed_calibration['p'],
               parsed_calibration['fx'], parsed_calibration['fy'], parsed_calibration['cx'], parsed_calibration['cy'],
               *parsed_calibration['k']]
    if not np.all(np.isfinite(numeric)) or parsed_calibration['fx'] <= 0 or parsed_calibration['fy'] <= 0:
        raise CalibrationError(f'{path} 的标定数值必须有限，焦距必须为正')
    if len(raw_data["T_imu_cam"]) != 1 or abs(float(np.linalg.norm(parsed_calibration["q_xyzw"])) - 1.0) > 1e-6:
        raise CalibrationError(f"{path} 的 T_imu_cam 必须且只能包含一个单位四元数")
    return parsed_calibration


def make_result(calib_json: Path, out_dir: Path, *, provenance: dict) -> dict:
    """新标定结果只携带最终参数和身份清单，不复制数值摘要或算法。"""
    from umi.infra.file_store import write_json
    measured = read_basalt(calib_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / 'calibration.json'
    if calib_json.resolve() != target.resolve():
        shutil.copyfile(calib_json, target)
    write_json(out_dir / 'manifest.json', dict(kind='umi_camera_calibration_result', schema_version=3,
        camera_serial=provenance['camera_serial'], lens=provenance.get('lens', 'front'),
        camera_model=measured['camera_type'], resolution=measured['resolution'],
        provenance={key: value for key, value in provenance.items() if key not in ('camera_serial', 'lens')},
        files={'calibration.json': file_hash(target)}))
    return describe(out_dir, 'user')


def calibration_path(directory: Path) -> Path:
    return directory / ('calibration.json' if (directory / 'manifest.json').is_file() else 'calibration_r1440.json')


def describe(bundle_dir: Path, source: str) -> dict:
    json_path = SAMPLE_CAMERA_JSON if source == 'factory' else calibration_path(bundle_dir)
    calib_info = read_basalt(json_path)
    version = 2
    meta = {}
    if source != 'factory':
        try:
            modern = json_path.name == 'calibration.json'
            meta = json.loads((bundle_dir / ('manifest.json' if modern else 'bundle.json')).read_text())
            if not isinstance(meta, dict):
                raise ValueError('标定包元数据必须是对象')
            version = meta.get('schema_version', 1)
            if type(version) is not int or version not in ((3,) if modern else (2,)):
                raise ValueError(f'不支持标定包版本 {version}')
            if modern and (meta.get('kind') != 'umi_camera_calibration_result'
                           or not isinstance(meta.get('camera_serial'), str) or not meta['camera_serial']
                           or meta.get('lens') not in ('front', 'rear')):
                raise ValueError('标定清单缺少有效的相机身份或镜头')
            digest = meta['files']['calibration.json'] if modern else meta['calibration_sha256']
            if digest != file_hash(json_path):
                raise ValueError('原始标定 JSON 校验失败')
        except (OSError, ValueError, KeyError) as error:
            raise CalibrationError(f'无法读取标定包 {bundle_dir}：{error}') from error
    else:
        digest = file_hash(json_path)
    intrinsic_source = 'sample' if source == 'factory' else intrinsics_source(meta)
    if intrinsic_source == 'factory':
        try:
            validate_factory(meta, calib_info, serial=meta['camera_serial'], lens=meta['lens'])
        except ValueError as error:
            raise CalibrationError(str(error)) from error
    return {'source': source, 'intrinsics_source': intrinsic_source,
            'intrinsics_provenance': calibration_provenance(meta),
            'dir': str(bundle_dir), 'json': str(json_path),
            'id': digest[:12],
            'bundle_schema_version': version, 'measurement': calib_info,
            'cam_time_offset_ns': calib_info['cam_time_offset_ns'],
            'T_imu_cam': {'p': calib_info['p'], 'q_xyzw': calib_info['q_xyzw']}}


SOURCES = ('measured', 'factory')


def validate_intrinsics_source(value: str) -> str:
    if value not in SOURCES:
        raise ValueError(f'内参来源必须为 measured 或 factory：{value!r}')
    return value


def optical_hash(camera: dict) -> str:
    """B must use exactly the intrinsics selected before camera–IMU calibration."""
    return canonical_hash(dict(camera_type=camera['camera_type'], resolution=camera['resolution'],
        fx=camera['fx'], fy=camera['fy'], cx=camera['cx'], cy=camera['cy'], k=camera['k']))


def calibration_provenance(metadata: dict) -> dict:
    calibration_metadata = metadata.get('provenance', metadata)
    return calibration_metadata if isinstance(calibration_metadata, dict) else {}


def intrinsics_source(metadata: dict) -> str:
    # Existing measured bundles precede this field. The old selector named
    # "factory" (the lab sample) is handled separately by assets.describe().
    return validate_intrinsics_source(calibration_provenance(metadata).get('intrinsics_source', 'measured'))


def require_intrinsics_source(camera: dict, expected: str | None, *, label: str = '标定') -> None:
    if expected is not None and camera.get('intrinsics_source', 'measured') != validate_intrinsics_source(expected):
        raise ValueError(f'{label}的内参来源与请求的 {expected} 不一致；请选择对应的完整标定参数')


def validate_factory(metadata: dict, camera: dict, *, serial: str, lens: str) -> None:
    if intrinsics_source(metadata) != 'factory':
        return
    origin = calibration_provenance(metadata)
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

    from umi.devices.x3.factory_geometry import lens_model, read_factory
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
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError('环境镜头测量必须为 JSON 对象')
    validation = metadata.get('validation')
    if metadata.get('camera_serial') != serial or metadata.get('lens') != 'rear':
        raise ValueError('环境镜头需要匹配的 camera_serial 和 lens: rear')
    if (metadata.get('kind') != 'umi_rear_calibration' or not isinstance(validation, dict)
            or not str(validation.get('verdict', '')).startswith('PASS')):
        raise ValueError('缺少通过验收的环境镜头测量（PASS）；请执行 umi-calib run <rear目录>')
    measured = parse_basalt(metadata, label=str(path))
    rear_source = intrinsics_source(metadata)
    require_intrinsics_source(dict(intrinsics_source=rear_source), expected_source, label='环境镜头')
    validate_factory(metadata, measured, serial=serial, lens='rear')
    if video is not None and rear_source == 'factory':
        check_recording(video, calibration_provenance(metadata))
    return dict(measured, intrinsics_source=rear_source), metadata
