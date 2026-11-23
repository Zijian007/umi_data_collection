"""Read camera–IMU measurements and build or validate versioned assets."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil

import numpy as np

from umi.core.frame import CROP
from umi.devices import calibration_source


SAMPLE_CAMERA_JSON = Path(__file__).parent / 'assets/samples/calibration_r1440.json'
SAMPLE_REAR_GEOMETRY = SAMPLE_CAMERA_JSON.with_name('rear_sample.json')


class CalibrationError(ValueError):
    pass


def _sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def read_basalt(path: Path) -> dict:
    """The fields the pipeline needs from a Basalt calibration.json (single kb4 camera, 1440x1440 crop)."""
    raw_data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_basalt(raw_data, label=str(path))


def parse_basalt(raw_data: dict, *, label: str) -> dict:
    """Validate an in-memory measurement with the same rules as a frozen file."""
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


def make_bundle(calib_json: Path, out_dir: Path, note: dict | None = None) -> dict:
    """Create a v2 measurement bundle; never copy algorithm code into it."""
    calib_info = read_basalt(calib_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / 'calibration_r1440.json'
    if calib_json.resolve() != target.resolve():
        shutil.copyfile(calib_json, target)
    meta = {**(note or {}), 'kind': 'umi_traj_calibration_bundle', 'schema_version': 2,
            'calibration_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'T_imu_cam': {'p': calib_info['p'], 'q_xyzw': calib_info['q_xyzw']},
            'cam_time_offset_ns': calib_info['cam_time_offset_ns']}
    (out_dir / 'bundle.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding='utf-8')
    return describe(out_dir, 'user')


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
        files={'calibration.json': hashlib.sha256(target.read_bytes()).hexdigest()}))
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
            if digest != hashlib.sha256(json_path.read_bytes()).hexdigest():
                raise ValueError('原始标定 JSON 校验失败')
        except (OSError, ValueError, KeyError) as error:
            raise CalibrationError(f'无法读取标定包 {bundle_dir}：{error}') from error
    intrinsic_source = 'sample' if source == 'factory' else calibration_source.source(meta)
    if intrinsic_source == 'factory':
        try:
            calibration_source.validate_factory(meta, calib_info, serial=meta['camera_serial'], lens=meta['lens'])
        except ValueError as error:
            raise CalibrationError(str(error)) from error
    return {'source': source, 'intrinsics_source': intrinsic_source,
            'intrinsics_provenance': calibration_source.provenance(meta),
            'dir': str(bundle_dir), 'json': str(json_path),
            'id': _sha12(json_path),
            'bundle_schema_version': version, 'measurement': calib_info,
            'cam_time_offset_ns': calib_info['cam_time_offset_ns'],
            'T_imu_cam': {'p': calib_info['p'], 'q_xyzw': calib_info['q_xyzw']}}
