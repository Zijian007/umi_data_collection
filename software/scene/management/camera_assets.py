"""按设备绑定选择完整镜头标定、冻结安装资产并生成相机身份。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import shutil

from software import provenance as artifact_identity
from software import registry
from software.camera import calibration_source, rear_fallback
from software.gripper_width import calibration as gripper_calibration
from software.scene.models import CalibratedCamera, Scene
from software.trajectory.calibration import selection


def rear_files(devices: dict[str, dict]) -> dict[str, Path]:
    return {serial: registry.rear_path(serial, device_binding['rear_calibration'])
            for serial, device_binding in devices.items() if 'rear_calibration' in device_binding}


def require_calibration_source(kit: dict, requested_source: str) -> None:
    """核对固定版本的来源，不把旧批次自动切换到另一套测量。"""
    calibration_source.validate(requested_source)
    for binding in kit['arms'].values():
        camera = registry.get(binding['serial'], calibration=binding['calibration'], allow_retired=True)
        calibration_source.require(camera, requested_source)
        if binding.get('rear_calibration'):
            calibration_source.read_rear(registry.rear_path(binding['serial'], binding['rear_calibration']),
                                         serial=binding['serial'], expected_source=requested_source)


def select_camera(recording_path: Path, scene: Scene) -> CalibratedCamera:
    # 场景路径不接受工程样例/跨设备覆盖；身份始终由原始录像决定。
    serial = registry.read_insv_serial(recording_path)
    if serial not in scene.devices:
        raise ValueError(f'相机 {serial} 未固定到场景/采集套件，请先登记设备配置')
    device_binding = scene.devices[serial]
    device_record = registry.get(serial, calibration=device_binding['calibration'], allow_retired=True)
    expected_source = device_binding.get('intrinsics_source') or device_record.get('intrinsics_source', 'measured')
    calibration_source.require(device_record, expected_source, label=f'相机 {serial} 工作镜头')
    if device_record.get('intrinsics_source') == 'factory':
        from software.vision.capabilities import require_forward_kb4
        require_forward_kb4()
        calibration_source.check_recording(recording_path, device_record['intrinsics_provenance'])
    if not re.fullmatch(r'[A-Za-z0-9_-]+', device_record['serial']):
        raise ValueError('相机序列号包含非法路径字符')
    if device_record['status'] not in ('active', 'retired') or device_record.get('calibration_owner', serial) != serial:
        raise ValueError('共享场景需要本设备已登记的完整相机标定，不能使用 sample')
    gripper = device_binding.get('gripper', {})
    tcp_path = Path(gripper['tcp']) if 'tcp' in gripper else None
    geometry = selection.read_tcp_geometry(tcp_path) if tcp_path is not None else None
    assets = selection.CalibrationAssets(Path(device_record['json']), tcp_path,
                                        Path(gripper['mask']) if 'mask' in gripper else selection.DEFAULT_FRONT_MASK)
    calibration_selection = selection.SelectedCalibration(
        assets=assets, directory=Path(device_record['dir']), calibration_id=device_binding['calibration'],
        camera_serial=device_record['serial'], source=device_record['source'], selection='scene_registry',
        status=device_record['status'], time_offset_ns=int(device_record['cam_time_offset_ns']), geometry=geometry,
        intrinsics_source=device_record.get('intrinsics_source', 'measured'))
    asset_paths = dict(front=assets.camera_json, mask=assets.front_mask)
    if tcp_path is not None:
        asset_paths['tcp'] = tcp_path
    width_measurement = None
    if 'width' in gripper:
        asset_paths['width'] = Path(gripper['width'])
        width_measurement = gripper_calibration.read(asset_paths['width'])
    tag_distance_measurement = None
    if 'tag_distance' in gripper:
        asset_paths['tag_distance'] = Path(gripper['tag_distance'])
        tag_distance_measurement = gripper_calibration.read_tag_sizes(asset_paths['tag_distance'])
    lens_calibrations = dict(front=dict(device_record['measurement'], intrinsics_source=device_record.get('intrinsics_source', 'measured')))
    manifest = Path(device_record['dir'])/'manifest.json'
    if manifest.is_file(): asset_paths['front_manifest'] = manifest
    factory_rear = None
    if scene.mode == 'dual' or scene.tag_layout is not None or serial in scene.rear_calibrations:
        rear_calibration_path = scene.rear_calibrations.get(device_record['serial'])
        if rear_calibration_path is None:
            factory_rear = rear_fallback.derive(recording_path, front_path=assets.camera_json,
                                               serial=serial, front_source=expected_source)
            measured = factory_rear.measurement
        else:
            measured, _ = calibration_source.read_rear(rear_calibration_path, serial=serial,
                expected_source=expected_source, video=recording_path)
            asset_paths['rear'] = rear_calibration_path
        if measured['intrinsics_source'] == 'factory' and scene.mode == 'dual':
            from software.vision.capabilities import require_forward_kb4
            require_forward_kb4()
        lens_calibrations['rear'] = measured
    if scene.parameters.tag_fusion_enabled and 'rear' not in lens_calibrations:
        raise ValueError('启用环境海报融合需要海报布局与本设备可用的环境镜头参数')
    return CalibratedCamera(device_record['serial'], calibration_selection, lens_calibrations, asset_paths,
                            width_measurement, tag_distance_measurement, factory_rear)


def snapshot_cameras(cameras: dict[str, CalibratedCamera], directory: Path) -> dict[str, CalibratedCamera]:
    """每次运行只复制一次小型资产，计算与结果均引用这份独立副本。"""
    snapshots = {}
    for arm, camera in cameras.items():
        expected_tag_hash = camera.asset_hashes.get('tag_distance')
        files = {}
        for name, source in camera.files.items():
            target = directory / 'devices' / camera.serial / (name + source.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            files[name] = target
        if camera.factory_rear is not None:
            files['rear'] = directory / 'devices' / camera.serial / 'rear.json'
            camera.factory_rear.freeze(files['rear'])
        if 'tag_distance' in files:
            if (artifact_identity.file_hash(files['tag_distance']) != expected_tag_hash
                    or gripper_calibration.read_tag_sizes(files['tag_distance']) != camera.tag_distance_measurement):
                raise ValueError('标签尺寸在读取与冻结之间发生变化，请重新运行')
        assets = selection.CalibrationAssets(files['front'], files.get('tcp'), files['mask'])
        snapshots[arm] = replace(camera, selected=replace(camera.selected, assets=assets), files=files, factory_rear=None)
    return snapshots


def camera_identity(camera: CalibratedCamera) -> dict:
    return dict(serial=camera.serial,
                assets={name: digest for name, digest in camera.asset_hashes.items() if name not in ('width', 'tag_distance')},
                intrinsics_source=camera.selected.intrinsics_source, calibration_version=camera.selected.calibration_id)


def select_cameras(recording_paths: dict, scene: Scene, expected_serials: dict | None = None) -> dict[str, CalibratedCamera]:
    cameras = {arm: select_camera(recording_path, scene) for arm, recording_path in recording_paths.items()}
    if expected_serials is not None and any(camera.serial != expected_serials[arm] for arm, camera in cameras.items()):
        raise ValueError('INSV 内部序列号与任务清单不一致')
    if len({camera.serial for camera in cameras.values()}) != len(cameras):
        raise ValueError('左右手必须来自两台不同序列号的相机')
    return cameras
