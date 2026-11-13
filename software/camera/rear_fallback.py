"""Derive an unvalidated rear camera from X3 factory geometry and measured front B.

This is a run asset, never an accepted rear calibration or a registry entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

from scipy.spatial.transform import Rotation

from software import config
from software.calibration.algorithms import factory_kb4
from software.camera import calibration_source
from software.camera.geometry import imu_from_camera, validate_transform
from software.camera.x3 import factory
from software.console import Console
from software.storage import atomic_file
from software.trajectory.calibration.assets import parse_basalt


KIND = 'umi_rear_factory_fallback'


@dataclass(frozen=True)
class FactoryRearCalibration:
    serialized: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.serialized.encode('utf-8')).hexdigest()

    @property
    def measurement(self) -> dict:
        document = json.loads(self.serialized)
        return dict(parse_basalt(document, label=KIND), intrinsics_source='factory',
                    calibration_kind=KIND, provenance=document['provenance'])

    def freeze(self, path: Path) -> None:
        with atomic_file(path) as temporary:
            temporary.write_text(self.serialized, encoding='utf-8')


def derive(video: Path, *, front_path: Path, serial: str, front_source: str) -> FactoryRearCalibration:
    """Read-only preflight. Invalid explicit rear bindings must never call here."""
    if not serial or front_source not in calibration_source.SOURCES:
        raise ValueError('rear 工厂兜底需要本设备 front 的有效 B 标定，不能使用样例')
    try:
        native = factory.read_factory(video)
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f'未提供 rear 标定且无法读取 INSV 工厂参数：{error}；请提供正式 rear 标定') from error
    if native['serial'] != serial:
        raise ValueError('rear 工厂兜底的原片序列号与 front 标定设备不一致')
    front_bytes = front_path.read_bytes()
    front_hash = hashlib.sha256(front_bytes).hexdigest()
    front = parse_basalt(json.loads(front_bytes), label=str(front_path))
    front_from_rear = factory.working_from_environment(native)
    imu_from_rear = validate_transform(imu_from_camera(front) @ front_from_rear)
    limits = config.load('calibration').values.factory
    environment_lens = next(lens for lens in native['lenses'] if lens['index'] == factory.ENVIRONMENT_LENS_INDEX)
    forward = factory_kb4.fit_forward(factory.lens_model(environment_lens, 1440), limits)
    translation = imu_from_rear[:3, 3].tolist()
    quaternion = Rotation.from_matrix(imu_from_rear[:3, :3]).as_quat().tolist()
    origin = dict(intrinsics_source='factory', extrinsics_source='front_b_composed_with_insv_offset_v3',
        front_calibration_sha256=front_hash, front_intrinsics_source=front_source,
        T_front_rear=front_from_rear.tolist(), native=native,
        time_offset_source='front_b_shared_video_frame_clock',
        timing_assumption='Both lens halves share the packed INSV frame timestamp; rear latency was not measured.',
        geometry=dict(kind='kb4_forward', maximum_ray_angle_degrees=85., crop_px=1440),
        optical_conversion=dict(fit_version=factory_kb4.FIT_VERSION,
                                limits=dict(maximum_fit_rms_px=limits.maximum_fit_rms_px,
                                            maximum_fit_p99_px=limits.maximum_fit_p99_px),
                                forward_pixels=forward.conversion),
        actual_error='not_measured')
    document = dict(kind=KIND, schema_version=1, camera_serial=serial, lens='rear',
        validation=dict(verdict='NOT_EVALUATED', rear_b_measured=False), provenance=origin,
        value0=dict(T_imu_cam=[dict(zip(('px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'), translation+quaternion))],
            intrinsics=[dict(camera_type='kb4', intrinsics=forward.fit['kb4'])],
            resolution=[[1440, 1440]], cam_time_offset_ns=front['cam_time_offset_ns']))
    derived = FactoryRearCalibration(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n')
    Console('环境镜头').write('未绑定 rear 标定：使用 INSV 工厂内参/双镜头外参 + front B；'
                             '沿用 front 时差，rear 精度未经独立验收', recording=video.name, state='兜底')
    return derived


def read_snapshot(path: Path, *, serial: str, expected_source: str) -> dict:
    """Run snapshots have hashes checked by their caller; registry remains strict."""
    document = json.loads(path.read_text())
    if document.get('kind') != KIND:
        return calibration_source.read_rear(path, serial=serial, expected_source=expected_source)[0]
    origin = document.get('provenance', {})
    if (document.get('schema_version') != 1 or document.get('camera_serial') != serial
            or document.get('lens') != 'rear' or expected_source != 'factory'
            or document.get('validation') != dict(verdict='NOT_EVALUATED', rear_b_measured=False)
            or origin.get('extrinsics_source') != 'front_b_composed_with_insv_offset_v3'
            or origin.get('time_offset_source') != 'front_b_shared_video_frame_clock'):
        raise ValueError('rear 工厂兜底快照身份或来源无效')
    return FactoryRearCalibration(path.read_text()).measurement
