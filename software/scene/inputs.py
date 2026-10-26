"""读取场景清单、解析固定设备资产，并生成可复现的输入身份。"""
from __future__ import annotations

from dataclasses import asdict, fields
import json
import math
from pathlib import Path
import re

from ruamel.yaml import YAML

from software import config, registry
from software.scene.models import ARMS, CalibratedCamera, Episode, Scene, SceneParameters, Session
from software.trajectory import calibration_assets, selection

ROOT = Path(__file__).resolve().parents[2]
INTERNAL_DIRECTORIES = frozenset({'archive', 'cache', 'runs', 'versions', 'maps', 'imports', 'results'})


def read_yaml(path: Path) -> dict:
    parser = YAML(typ='safe', pure=True)
    parser.version = (1, 2)
    parser.allow_duplicate_keys = False
    with path.open(encoding='utf-8') as stream:
        data = parser.load(stream)
    if parser.version != (1, 2) or not isinstance(data, dict):
        raise ValueError(f'{path} 需要 YAML 1.2 映射')
    return data


def require_keys(data: dict, allowed: set, required: set, label: str) -> None:
    if not isinstance(data, dict) or data.keys() - allowed or required - data.keys():
        raise ValueError(f'{label} 字段缺失或含未知项；允许 {sorted(allowed)}，必需 {sorted(required)}')


def resolve_path(base: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('文件路径必须为非空字符串')
    return (base / Path(value).expanduser()).resolve()


def validate_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[\w-]+', value, re.UNICODE):
        raise ValueError('场景和任务名称只允许字母、数字、中文、下划线和连字符')
    return value


def validate_episode_name(value: str) -> str:
    value = validate_name(value)
    if value in INTERNAL_DIRECTORIES:
        raise ValueError(f'任务名不能使用内部目录名称：{value}')
    return value


def read_recordings(base: Path, data: dict, *, verify: bool = True) -> tuple[dict[str, Path], dict[str, str]]:
    require_keys(data, set(ARMS), set(), 'recordings')
    if not data:
        raise ValueError('至少需要一只手的录像')
    recording_paths, serials = {}, {}
    for arm, item in data.items():
        require_keys(item, {'serial', 'path', 'sha256', 'started_at_unix_s', 'duration_s'}, {'serial', 'path', 'sha256'}, '录像引用')
        serials[arm] = registry.validate_id(item['serial'])
        path = resolve_path(base, item['path'])
        if path.suffix.lower() != '.insv' or (verify and not path.is_file()):
            raise ValueError(f'需要原始 .insv 文件：{path}')
        if not isinstance(item['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise ValueError('录像引用需要完整 SHA256')
        if verify and registry.sha256_file(path) != item['sha256']:
            raise ValueError(f'录像已变化：SHA256 与导入记录不一致：{path}')
        recording_paths[arm] = path
    if len(set(recording_paths.values())) != len(recording_paths) or len(set(serials.values())) != len(serials):
        raise ValueError('左右手必须使用不同设备的不同录像')
    return recording_paths, serials


def load_scene_parameters(path: Path | None) -> SceneParameters:
    data = read_yaml(ROOT / 'config/scene_parameters.yaml')
    if path is not None:
        overlay = read_yaml(path)
        if overlay.keys() - data.keys():
            raise ValueError(f'未知场景参数：{sorted(overlay.keys()-data.keys())}')
        data.update(overlay)
    if data.keys() != {field.name for field in fields(SceneParameters)}:
        raise ValueError('场景参数字段不完整')
    for field in fields(SceneParameters):
        value = data[field.name]
        typ = int if field.type == 'int' else float
        if type(value) not in ((int,) if typ is int else (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{field.name} 必须为正的 {typ.__name__}')
        data[field.name] = typ(value)
    parameters = SceneParameters(**data)
    if parameters.schema_version != 1 or parameters.minimum_alignment_nodes < 5 or parameters.minimum_pose_pairs < 3:
        raise ValueError('场景参数版本无效，或最少对齐节点/位姿对不足')
    if not 0 < parameters.minimum_localization_fraction <= 1 or parameters.maximum_alignment_condition <= 1:
        raise ValueError('定位比例必须在 (0, 1] 内，最大条件数必须大于 1')
    return parameters


def read_devices(data: dict) -> dict[str, dict]:
    if not isinstance(data, dict) or not data:
        raise ValueError('devices 需要按完整序列号固定设备版本')
    for serial, device_binding in data.items():
        registry.validate_id(serial)
        require_keys(device_binding, {'calibration', 'gripper', 'gripper_version', 'rear_calibration'},
                     {'calibration', 'gripper', 'gripper_version'}, '设备版本')
        for value in device_binding.values():
            registry.validate_id(value)
    return data


def rear_files(devices: dict[str, dict]) -> dict[str, Path]:
    return {serial: registry.rear_path(serial, device_binding['rear_calibration'])
            for serial, device_binding in devices.items() if 'rear_calibration' in device_binding}


def read_scene(path: Path, *, verify_mapping: bool = False) -> Scene:
    path = path.expanduser().resolve()
    if path.is_dir():
        path /= 'scene.yaml'
    data = read_yaml(path)
    require_keys(data, {'schema_version', 'name', 'mode', 'reference_arm', 'mapping', 'devices',
                        'trajectory_config', 'parameters', 'description'},
                 {'schema_version', 'name', 'mode', 'reference_arm', 'mapping', 'devices'}, 'scene')
    if type(data['schema_version']) is not int or data['schema_version'] != 2:
        raise ValueError('场景清单需要 schema_version: 2')
    if data['mode'] not in ('shared_front', 'dual'):
        raise ValueError('场景 mode 必须为 shared_front 或 dual')
    recording_paths, serials = read_recordings(path.parent, data['mapping'], verify=verify_mapping) if data.get('mapping') else ({}, {})
    if recording_paths and data['mode'] == 'dual' and set(recording_paths) != set(ARMS):
        raise ValueError('dual 场景需要左右手两段建图录像；单相机验证可使用 shared_front')
    if data['reference_arm'] not in ARMS or (recording_paths and data['reference_arm'] not in recording_paths):
        raise ValueError('reference_arm 必须指向一段建图录像')
    device_bindings = read_devices(data['devices']) if data['devices'] else {}
    if set(serials.values()) - device_bindings.keys():
        raise ValueError('建图录像设备未登记到场景 devices')
    optional = lambda name: resolve_path(path.parent, data[name]) if data.get(name) else None
    trajectory_config = config.load('trajectory', optional('trajectory_config'))
    return Scene(path, validate_name(data['name']), data['mode'], data['reference_arm'], recording_paths,
                 rear_files(device_bindings), trajectory_config, load_scene_parameters(optional('parameters')), device_bindings, serials)


def read_arms(value: list | tuple, available=ARMS) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value or len(set(value)) != len(value) or set(value)-set(available):
        raise ValueError('预期使用的手必须为 left、right 或两者，且属于该套件')
    return tuple(arm for arm in ARMS if arm in value)


def validate_kit(data: dict) -> dict:
    """当前套件和批次快照使用相同结构；相机标定、夹爪测量仍固定实际版本。"""
    require_keys(data, {'id', 'arms', 'description'}, {'id', 'arms'}, '套件')
    validate_name(data['id'])
    if not isinstance(data.get('description', ''), str):
        raise ValueError('套件 description 必须为字符串')
    if not isinstance(data['arms'], dict) or not data['arms'] or data['arms'].keys()-set(ARMS):
        raise ValueError('套件至少配置一只手：left 或 right')
    serials = []
    for item in data['arms'].values():
        require_keys(item, {'serial', 'calibration', 'gripper', 'gripper_version', 'rear_calibration'},
                     {'serial', 'calibration', 'gripper', 'gripper_version'}, '套件设备')
        for value in item.values():
            registry.validate_id(value)
        serials.append(item['serial'])
    if len(set(serials)) != len(serials):
        raise ValueError('套件的左右手必须使用不同相机')
    return data


def read_kit(scene: Scene, name: str) -> dict:
    """新建批次读取当前配置；处理已有批次时直接使用 session.yaml 内的快照。"""
    kit = validate_kit(read_yaml(scene.root / 'kits' / validate_name(name) / 'kit.yaml'))
    if kit['id'] != name:
        raise ValueError('套件目录与配置 ID 不一致')
    return kit


def read_session(path: Path) -> Session:
    path = path.expanduser().resolve()
    if path.is_dir():
        path /= 'session.yaml'
    data = read_yaml(path)
    require_keys(data, {'schema_version', 'scene', 'map', 'kit', 'expected_arms', 'description'},
                 {'schema_version', 'scene', 'kit', 'expected_arms'}, 'session')
    if type(data['schema_version']) is not int or data['schema_version'] != 4:
        raise ValueError('采集批次需要 schema_version: 4')
    scene = read_scene(resolve_path(path.parent, data['scene']))
    kit = validate_kit(data['kit'])
    if path.parent.parent != scene.root / 'kits' / kit['id'] / 'sessions':
        raise ValueError('采集批次必须位于 scenes/<场景>/kits/<套件>/sessions/<批次>/，且与清单套件一致')
    expected = read_arms(data['expected_arms'], kit['arms'])
    version, digest = None, None
    if data.get('map') is not None:
        require_keys(data['map'], {'version', 'manifest_sha256'}, {'version', 'manifest_sha256'}, '批次地图')
        version, digest = data['map']['version'], data['map']['manifest_sha256']
        registry.validate_id(version)
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('批次地图需要完整 SHA256')
    return Session(path, scene, version, digest, kit, expected)


def read_synchronization(data: dict | None, arms: tuple[str, ...]) -> dict:
    """任务视频时钟：t_episode = t_video + offset；首只手为零偏移参考。"""
    if data is None:
        data = dict(offsets_s={arm: 0. for arm in arms}, confirmed=False, note='')
    require_keys(data, {'offsets_s', 'confirmed', 'note'}, {'offsets_s', 'confirmed'}, 'synchronization')
    offsets = data['offsets_s']
    if not isinstance(offsets, dict) or set(offsets) != set(arms):
        raise ValueError('synchronization.offsets_s 必须包含全部预期使用手')
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in offsets.values()):
        raise ValueError('同步偏移必须为有限秒数')
    if offsets[arms[0]] != 0:
        raise ValueError(f'参考手 {arms[0]} 的同步偏移必须为零')
    confirmed, note = data['confirmed'], data.get('note', '')
    if type(confirmed) is not bool or not isinstance(note, str):
        raise ValueError('同步 confirmed 必须为布尔值，note 必须为文本')
    if len(arms) == 1:
        return dict(method='single_camera', offsets_s={arms[0]: 0.}, confirmed=True, note=note)
    if confirmed and not note.strip():
        raise ValueError('确认双手同步时需要填写 note，说明核验依据')
    method = 'manual' if confirmed or any(offsets.values()) else 'assumed_simultaneous'
    return dict(method=method, offsets_s={arm: float(offsets[arm]) for arm in arms}, confirmed=confirmed, note=note)


def read_episode(path: Path, *, verify: bool = True) -> Episode:
    path = path.expanduser().resolve()
    if path.is_dir():
        path /= 'episode.yaml'
    data = read_yaml(path)
    require_keys(data, {'schema_version', 'session', 'recordings', 'expected_arms', 'instruction', 'synchronization'},
                 {'schema_version', 'session', 'recordings', 'expected_arms'}, 'episode')
    if type(data['schema_version']) is not int or data['schema_version'] not in (4, 5):
        raise ValueError('任务清单需要 schema_version: 4 或 5')
    if data['schema_version'] == 4 and 'synchronization' in data:
        raise ValueError('配置同步请将任务 schema_version 更新为 5')
    if data['schema_version'] == 5 and not {'instruction', 'synchronization'} <= data.keys():
        raise ValueError('schema 5 任务需要 instruction 和 synchronization')
    session = read_session(resolve_path(path.parent, data['session']))
    if path.parent.parent != session.path.parent:
        raise ValueError('任务必须直接位于采集批次的 <任务>/ 目录')
    validate_episode_name(path.parent.name)
    expected = read_arms(data['expected_arms'], session.expected_arms)
    recording_paths, serials = read_recordings(path.parent, data['recordings'], verify=verify)
    if serials.keys()-set(expected):
        raise ValueError('实际录像包含本次任务未声明使用的手')
    if any(serial != session.arm_serials[arm] for arm, serial in serials.items()):
        raise ValueError('录像清单的设备身份与套件的左右手配置不一致')
    if not isinstance(data.get('instruction', ''), str):
        raise ValueError('instruction 必须为操作描述字符串')
    synchronization = read_synchronization(data.get('synchronization'), expected)
    return Episode(path, session, recording_paths, serials, expected, data['recordings'],
                   data.get('instruction', ''), synchronization)


def select_camera(recording_path: Path, scene: Scene) -> CalibratedCamera:
    # 场景路径不接受工程样例/跨设备覆盖；身份始终由原始录像决定。
    serial = registry.read_insv_serial(recording_path)
    if serial not in scene.devices:
        raise ValueError(f'相机 {serial} 未固定到场景/采集套件，请先登记设备配置')
    device_binding = scene.devices[serial]
    device_record = registry.get(serial, allow_retired=True, **{key: device_binding[key] for key in ('calibration', 'gripper', 'gripper_version')})
    if not re.fullmatch(r'[A-Za-z0-9_-]+', device_record['serial']):
        raise ValueError('相机序列号包含非法路径字符')
    if device_record['status'] not in ('active', 'retired') or device_record.get('calibration_owner', serial) != serial:
        raise ValueError('共享场景需要已启用的实测相机标定，不能使用 sample')
    assets = selection.CalibrationAssets(Path(device_record['json']), Path(device_record['tcp_json']), Path(device_record['mask']))
    calibration_selection = selection.SelectedCalibration(assets, Path(device_record['dir']), device_record['id'], device_record['serial'],
        device_record['gripper'], device_record['source'], 'scene_registry', device_record['status'], int(device_record['cam_time_offset_ns']))
    asset_paths = dict(front=assets.camera_json, tcp=assets.tcp_geometry, mask=assets.front_mask)
    lens_calibrations = dict(front=calibration_assets.read_basalt(assets.camera_json))
    if scene.mode == 'dual':
        rear_calibration_path = scene.rear_calibrations.get(device_record['serial'])
        if rear_calibration_path is None:
            raise ValueError(f"相机 {device_record['serial']} 缺少后镜头实测标定；先安装后镜头标定，再在设备清单指定 rear_calibration 版本")
        rear_metadata = json.loads(rear_calibration_path.read_text())
        if not isinstance(rear_metadata, dict):
            raise ValueError(f'{rear_calibration_path} 的后镜头测量必须为 JSON 对象')
        if rear_metadata.get('camera_serial') != device_record['serial'] or rear_metadata.get('lens') != 'rear':
            raise ValueError(f'{rear_calibration_path} 必须明确包含匹配的 camera_serial 和 lens: rear')
        if rear_metadata.get('kind') != 'umi_rear_calibration' or not str(rear_metadata.get('validation', {}).get('verdict', '')).startswith('PASS'):
            raise ValueError(f'{rear_calibration_path} 缺少通过验收的后镜头测量；请运行 umi-scene calibrate-rear')
        lens_calibrations['rear'] = calibration_assets.read_basalt(rear_calibration_path)
        asset_paths['rear'] = rear_calibration_path
    return CalibratedCamera(device_record['serial'], calibration_selection, lens_calibrations, asset_paths)


def camera_identity(camera: CalibratedCamera) -> dict:
    return dict(serial=camera.serial, gripper=camera.selected.gripper_id,
                assets={name: config.file_hash(path) for name, path in sorted(camera.files.items())})


def recording_identity(path: Path) -> dict:
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def select_cameras(recording_paths: dict, scene: Scene, expected_serials: dict | None = None) -> dict[str, CalibratedCamera]:
    cameras = {arm: select_camera(recording_path, scene) for arm, recording_path in recording_paths.items()}
    if expected_serials is not None and any(camera.serial != expected_serials[arm] for arm, camera in cameras.items()):
        raise ValueError('INSV 内部序列号与任务清单不一致')
    if len({camera.serial for camera in cameras.values()}) != len(cameras):
        raise ValueError('左右手必须来自两台不同序列号的相机')
    return cameras


def artifact_hashes(directory: Path, roots: tuple[str, ...]) -> dict:
    return {str(path.relative_to(directory)): config.file_hash(path)
            for name in roots for path in sorted((directory / name).rglob('*')) if path.is_file()}


def validate_assets(directory: Path, hashes: dict) -> None:
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('场景/任务资产清单为空')
    for name, digest in hashes.items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory.resolve()) or not target.is_file() or config.file_hash(target) != digest:
            raise ValueError(f'场景/任务资产缺失、越界或内容已变：{name}')


def algorithm_identity() -> dict:
    identity = config.code_identity('scene')
    return {name: digest for name, digest in identity.items() if Path(name).name not in {
        'cli.py', 'capture_config.py', 'ingest.py', 'recording_checks.py', 'episodes.py', '__main__.py'}}


def episode_inputs(episode, cameras, map_manifest) -> dict:
    return dict(map_mode=map_manifest['mode'], scene_version=map_manifest['version'], frame_id=map_manifest['frame_id'], scene_assets=map_manifest['assets'],
        session=episode.session.name, session_manifest_sha256=config.file_hash(episode.session.path),
        episode_manifest_sha256=config.file_hash(episode.path), scene_manifest_sha256=config.file_hash(episode.scene.path),
        recordings={arm: recording_identity(video) for arm, video in episode.recordings.items()},
        cameras={arm: camera_identity(camera) for arm, camera in cameras.items()}, synchronization=episode.synchronization,
        parameters={key: value for key, value in asdict(episode.scene.parameters).items()
                    if not key.startswith(('recording_', 'pairing_'))}, trajectory_parameters=episode.scene.trajectory.snapshot()['values'],
        code=algorithm_identity(), trajectory_code=config.code_identity('trajectory'), engines=config.engine_identity('trajectory'))
