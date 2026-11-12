"""解析场景、套件和任务清单，校验参数与输入身份。"""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from pathlib import Path
import math
import re


from software import config, registry
from software import provenance as artifact_identity
from software.camera import calibration_source
from software.scene import layout
from software.scene.configuration import SceneOverrides, apply_overrides
from software.scene.management import camera_assets
from software.scene.models import ARMS, Episode, Scene, SceneParameters, Session
from software.scene.result_io import scene_map_identity
from software.storage import read_yaml  # noqa: F401  其他模块经 inputs.read_yaml 使用


ROOT = Path(__file__).resolve().parents[3]
KITS = ROOT / 'kits'
INTERNAL_DIRECTORIES = frozenset({'archive', 'cache', 'runs', 'versions', 'maps', 'mapping', 'videos', 'pending', 'imports', 'results', 'local_results'})


def require_keys(document: dict, allowed: set, required: set, label: str) -> None:
    if not isinstance(document, dict) or document.keys() - allowed or required - document.keys():
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


def validate_batch_name(value: str) -> str:
    value = validate_name(value)
    if value in INTERNAL_DIRECTORIES | {'kits', 'sessions'}:
        raise ValueError(f'批次名不能使用内部目录名称：{value}')
    return value


def task_video_sources(task: Path) -> list[Path]:
    """原片直接放在任务的左右手目录；旧 videos/ 仍可原地读取。"""
    roots = [task / arm for arm in ARMS] + [task / 'videos']
    return [path for path in roots if path.is_dir()]


def batch_task_directories(batch: Path) -> list[Path]:
    if not batch.is_dir():
        return []
    candidates = {task.name for task in batch.iterdir() if task.is_dir()}
    records = batch / '.umi/tasks'
    if records.is_dir():
        candidates.update(task.name for task in records.iterdir() if task.is_dir())
    tasks = []
    for name in sorted(candidates):
        try:
            validate_episode_name(name)
        except ValueError:
            continue
        task = batch / name
        if task_video_sources(task) or (task / 'task.yaml').is_file() or layout.manifest(task, 'episode.yaml').is_file():
            tasks.append(task)
    return tasks


def batch_video_sources(batch: Path) -> list[Path]:
    """只发现任务原片和待配对原片，跳过缓存、结果及其他内部目录。"""
    sources = [batch / 'pending'] if (batch / 'pending').is_dir() else []
    return sources + [source for task in batch_task_directories(batch) for source in task_video_sources(task)]


def read_recordings(base: Path, recording_references: dict, *, verify: bool = True) -> tuple[dict[str, Path], dict[str, str]]:
    require_keys(recording_references, set(ARMS), set(), 'recordings')
    if not recording_references:
        raise ValueError('至少需要一只手的录像')
    recording_paths, serials = {}, {}
    for arm, item in recording_references.items():
        require_keys(item, {'serial', 'path', 'sha256', 'started_at_unix_s', 'duration_s'}, {'serial', 'path', 'sha256'}, '录像引用')
        serials[arm] = registry.validate_id(item['serial'])
        path = resolve_path(base, item['path'])
        if path.suffix.lower() != '.insv' or (verify and not path.is_file()):
            raise ValueError(f'需要原始 .insv 文件：{path}')
        if not isinstance(item['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise ValueError('录像引用需要完整 SHA256')
        if verify and artifact_identity.file_hash(path) != item['sha256']:
            raise ValueError(f'录像已变化：SHA256 与导入记录不一致：{path}')
        recording_paths[arm] = path
    if len(set(recording_paths.values())) != len(recording_paths) or len(set(serials.values())) != len(serials):
        raise ValueError('左右手必须使用不同设备的不同录像')
    return recording_paths, serials


def load_scene_parameters(path: Path | None = None, *, overrides: dict | None = None) -> SceneParameters:
    parameter_values = read_yaml(ROOT / 'config/scene_parameters.yaml')
    if path is not None:
        overlay = read_yaml(path)
        parameter_values = config.merge_overrides(parameter_values, overlay, 'parameters.')
    if overrides is not None:
        parameter_values = config.merge_overrides(parameter_values, overrides, 'parameters.')
    if parameter_values.keys() != {field.name for field in fields(SceneParameters)}:
        raise ValueError('场景参数字段不完整')
    for field in fields(SceneParameters):
        value = parameter_values[field.name]
        if field.name == 'tag_distance_roi_px':
            from software.gripper_width.detection import validate_roi

            parameter_values[field.name] = list(validate_roi(value))
            continue
        if field.type == 'bool':
            if type(value) is not bool:
                raise ValueError(f'{field.name} 必须为布尔值')
            continue
        typ = int if field.type == 'int' else float
        allows_zero = field.name in {'gripper_width_max_gap_s', 'gripper_width_closed_threshold_mm',
                                    'tag_distance_max_missing_frames', 'tag_sigma_per_radius_px'}
        if (type(value) not in ((int,) if typ is int else (int, float)) or not math.isfinite(value)
                or value < 0 or value == 0 and not allows_zero):
            bound = '非负' if allows_zero else '正的'
            raise ValueError(f'{field.name} 必须为{bound} {typ.__name__}')
        parameter_values[field.name] = typ(value)
    parameters = SceneParameters(**parameter_values)
    if parameters.schema_version != 1 or parameters.minimum_alignment_nodes < 5 or parameters.minimum_pose_pairs < 3:
        raise ValueError('场景参数版本无效，或最少对齐节点/位姿对不足')
    if not 0 < parameters.minimum_localization_fraction <= 1 or parameters.maximum_alignment_condition <= 1:
        raise ValueError('定位比例必须在 (0, 1] 内，最大条件数必须大于 1')
    if not 0 < parameters.tag_distance_min_valid_fraction <= 1:
        raise ValueError('标签中心距最低有效比例必须在 (0, 1] 内')
    if parameters.localization_minimum_overlap_frames < 3 or parameters.localization_max_rotation_difference_deg > 180:
        raise ValueError('双向定位至少需要 3 个重叠帧，旋转差上限不能超过 180 度')
    if parameters.tag_minimum_frames < 3:
        raise ValueError('海报初始化至少需要 3 个有效时刻')
    if parameters.tag_mapping_minimum_frames < 3:
        raise ValueError('海报尺度估计至少需要 3 个有效时刻')
    if parameters.tag_mapping_scale_enabled and not parameters.tag_mapping_enabled:
        raise ValueError('海报尺度修正需要先开启 tag_mapping_enabled')
    return parameters


def gripper_paths(declaration: dict, directory: Path | None = None) -> dict[str, str]:
    """kit 中使用相对路径；持久化快照使用解析后的绝对路径。"""
    require_keys(declaration, {'tcp', 'width', 'mask', 'tag_distance'}, {'tcp'}, 'gripper')
    resolved = {}
    for asset, filename in declaration.items():
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f'gripper.{asset} 需要文件路径')
        path = Path(filename)
        if directory is not None:
            if path.is_absolute():
                raise ValueError(f'gripper.{asset} 请填写相对 kit.yaml 的文件路径')
            path = (directory / path).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError(f'gripper.{asset} 必须位于套件目录内')
        elif not path.is_absolute():
            raise ValueError('资产快照需要已解析的绝对路径')
        suffix = '.png' if asset == 'mask' else '.json'
        if path.suffix.lower() != suffix:
            raise ValueError(f'gripper.{asset} 需要 {suffix} 文件')
        resolved[asset] = str(path)
    return resolved


def read_devices(bindings: dict) -> dict[str, dict]:
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('devices 需要按完整序列号固定设备配置')
    devices = {}
    for serial, binding in bindings.items():
        registry.validate_id(serial)
        require_keys(binding, {'calibration', 'gripper', 'rear_calibration', 'intrinsics_source'}, {'calibration'}, '设备配置')
        devices[serial] = {field: registry.validate_id(value) for field, value in binding.items() if field != 'gripper'}
        if 'gripper' in binding:
            devices[serial]['gripper'] = gripper_paths(binding['gripper'])
    return devices


def scene_definition(scene_document: dict, name: str) -> dict:
    """在内存中补齐场景默认值，目录是场景身份的来源。"""
    version = scene_document.get('schema_version', 3)
    if type(version) is not int or version not in (2, 3):
        raise ValueError('场景配置需要 schema_version: 3；历史 schema 2 仍可读取')
    if version == 3:
        require_keys(scene_document, {'schema_version', 'mode', 'reference_arm', 'description', 'trajectory', 'parameters', 'tags'}, set(), 'scene')
        normalized = dict(schema_version=3, name=validate_name(name), mode='shared_front', reference_arm='left',
                          description='', trajectory={}, parameters={}, mapping={}, devices={})
        normalized.update(scene_document)
        for key in ('trajectory', 'parameters'):
            if not isinstance(normalized[key], dict):
                raise ValueError(f'scene.{key} 必须直接填写参数映射，不再引用覆盖文件')
        if not isinstance(normalized['description'], str):
            raise ValueError('场景 description 必须为字符串')
        return normalized
    # Read historical standalone-map definitions without creating new copies of them.
    require_keys(scene_document, {'schema_version', 'name', 'mode', 'reference_arm', 'mapping', 'devices',
                        'trajectory_config', 'parameters', 'description', 'tags'},
                 {'schema_version'}, 'scene')
    if type(scene_document['schema_version']) is not int or scene_document['schema_version'] != 2:
        raise ValueError('场景清单需要 schema_version: 2')
    if scene_document.get('name', name) != name:
        raise ValueError('场景目录与配置名称不一致')
    normalized = dict(schema_version=2, name=validate_name(name), mode='shared_front', mapping={}, devices={},
                      description='', trajectory_config=None, parameters=None)
    normalized.update(scene_document)
    if not isinstance(normalized['mapping'], dict) or not isinstance(normalized['devices'], dict):
        raise ValueError('场景 mapping 和 devices 必须为映射')
    reference = next((arm for arm in ARMS if arm in normalized['mapping']), 'left')
    normalized.setdefault('reference_arm', reference)
    if not isinstance(normalized['description'], str):
        raise ValueError('场景 description 必须为字符串')
    return normalized


def read_scene(path: Path, *, verify_mapping: bool = False, load_mapping: bool = True, overrides: SceneOverrides | None = None) -> Scene:
    path = path.expanduser().resolve()
    if path.is_dir():
        path /= 'scene.yaml'
    if not path.parent.is_dir():
        raise ValueError(f'场景目录不存在：{path.parent}')
    scene_document = scene_definition(read_yaml(path) if path.is_file() else {}, path.parent.name)
    if not load_mapping:
        scene_document.update(mapping={}, devices={})
    if scene_document['mode'] not in ('shared_front', 'dual'):
        raise ValueError('场景 mode 必须为 shared_front 或 dual')
    recording_paths, serials = read_recordings(path.parent, scene_document['mapping'], verify=verify_mapping) if scene_document.get('mapping') else ({}, {})
    if recording_paths and scene_document['mode'] == 'dual' and set(recording_paths) != set(ARMS):
        raise ValueError('dual 场景需要左右手两段建图录像；单相机验证可使用 shared_front')
    if scene_document['reference_arm'] not in ARMS or (recording_paths and scene_document['reference_arm'] not in recording_paths):
        raise ValueError('reference_arm 必须指向一段建图录像')
    device_bindings = read_devices(scene_document['devices']) if scene_document['devices'] else {}
    if set(serials.values()) - device_bindings.keys():
        raise ValueError('建图录像设备未登记到场景 devices')
    if scene_document['schema_version'] == 2:
        optional = lambda name: resolve_path(path.parent, scene_document[name]) if scene_document.get(name) else None
        trajectory_config = config.load('trajectory', optional('trajectory_config'))
        parameters = load_scene_parameters(optional('parameters'))
    else:
        trajectory_config = config.load('trajectory', inline=scene_document['trajectory'])
        parameters = load_scene_parameters(overrides=scene_document['parameters'])
    trajectory_config, parameters = apply_overrides(trajectory_config, parameters, overrides)
    tag_layout = None
    if 'tags' in scene_document:
        from software.vision.tag_board import load
        require_keys(scene_document['tags'], {'layout'}, {'layout'}, 'tags')
        tag_layout = resolve_path(path.parent, scene_document['tags']['layout'])
        load(tag_layout)  # 在创建运行目录或清理旧结果之前验证几何资产。
    if (parameters.tag_fusion_enabled or parameters.tag_mapping_enabled) and tag_layout is None:
        raise ValueError('启用海报融合或建图诊断需要 scene.yaml 的 tags.layout')
    return Scene(path, validate_name(scene_document['name']), scene_document['mode'], scene_document['reference_arm'], recording_paths,
                 camera_assets.rear_files(device_bindings), trajectory_config, parameters, device_bindings, serials, tag_layout, overrides)


def read_arms(value: list | tuple, available=ARMS) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value or len(set(value)) != len(value) or set(value) - set(available):
        raise ValueError('预期使用的手必须为 left、right 或两者，且属于该套件')
    return tuple(arm for arm in ARMS if arm in value)


def validate_kit(declaration: dict, *, name: str | None = None, intrinsics_source: str | None = None) -> dict:
    """解析套件/快照，按来源固定完整版本；安装资产在选择相机时读取。

    来源优先级：调用方的 CLI/批次选择、套件、每手配置；多来源未指定时选 measured。
    旧单版本配置保留原绑定，显式来源仍在 select_camera 中核对，不能自动替换版本。
    """
    require_keys(declaration, {'id', 'arms', 'description', 'intrinsics_source'}, {'arms'} if name is not None else {'id', 'arms'}, '套件')
    identity = name if name is not None else declaration['id']
    if declaration.get('id', identity) != identity:
        raise ValueError('套件目录与配置 ID 不一致')
    validate_name(identity)
    if not isinstance(declaration.get('description', ''), str):
        raise ValueError('套件 description 必须为字符串')
    arms = declaration['arms']
    if not isinstance(arms, dict) or not arms or arms.keys() - set(ARMS):
        raise ValueError('套件至少配置一只手：left 或 right')
    if 'intrinsics_source' in declaration:
        calibration_source.validate(declaration['intrinsics_source'])
    source = intrinsics_source or declaration.get('intrinsics_source')
    if source is not None:
        calibration_source.validate(source)
    resolved = {}
    for arm, binding in arms.items():
        require_keys(binding, {'serial', 'calibration', 'calibrations', 'gripper', 'rear_calibration', 'intrinsics_source'},
                     {'serial', 'gripper'}, '套件设备')
        selected = source or binding.get('intrinsics_source')
        profiles = binding.get('calibrations')
        if profiles is not None:
            if 'calibration' in binding or 'rear_calibration' in binding:
                raise ValueError('calibrations 与旧的 calibration/rear_calibration 不能混用')
            if not isinstance(profiles, dict) or not profiles or profiles.keys()-set(calibration_source.SOURCES):
                raise ValueError('calibrations 只允许 measured/factory 来源')
            for variant in profiles.values():
                require_keys(variant, {'front', 'rear'}, {'front'}, '镜头标定版本')
                for value in variant.values(): registry.validate_id(value)
            selected = selected or 'measured'
            calibration_source.validate(selected)
            if selected not in profiles:
                raise ValueError(f'套件 {identity} 的 {arm} 缺少 {selected} 完整标定版本')
            profile = profiles[selected]
            camera = dict(serial=registry.validate_id(binding['serial']), calibration=profile['front'],
                          intrinsics_source=selected)
            if 'rear' in profile: camera['rear_calibration'] = profile['rear']
        else:
            if 'calibration' not in binding:
                raise ValueError('套件设备需要 calibration 或 calibrations')
            camera = {field: registry.validate_id(value) for field, value in binding.items()
                      if field not in ('gripper', 'calibrations')}
            if selected is not None:
                camera['intrinsics_source'] = calibration_source.validate(selected)
        camera['gripper'] = gripper_paths(binding['gripper'], KITS / name if name is not None else None)
        resolved[arm] = camera
    if len({camera['serial'] for camera in resolved.values()}) != len(resolved):
        raise ValueError('套件的左右手必须使用不同相机')
    kit = dict(id=identity, arms=resolved)
    if declaration.get('description'):
        kit['description'] = declaration['description']
    return kit


def read_kit(name: str, *, intrinsics_source: str | None = None) -> dict:
    path = KITS / validate_name(name) / 'kit.yaml'
    return validate_kit(read_yaml(path), name=name, intrinsics_source=intrinsics_source)


def mapping_records(source: dict) -> dict:
    """单段建图不带手标签；展开为算法内部的参考槽位，兼容旧记录。"""
    if not isinstance(source, dict):
        raise ValueError('批次 mapping 必须为建图原片映射')
    return {'left': source} if 'path' in source else source


def write_batch_snapshot(path: Path, record: dict) -> None:
    # Keep a retrievable copy when the selected source or map changes.
    from software.storage import write_yaml
    archive = layout.metadata(layout.owner(path), 'snapshots') / (artifact_identity.canonical_hash(record)+'.yaml')
    if not archive.exists(): write_yaml(archive, record)
    write_yaml(path, record)


def read_batch(path: Path, *, verify_mapping=False, overrides: SceneOverrides | None = None) -> Session:
    path = layout.manifest(path, 'manifest.yaml')
    return batch_from_record(path, read_yaml(path), verify_mapping=verify_mapping, overrides=overrides)


def batch_from_record(path: Path, manifest_record: dict, *, verify_mapping=False, overrides: SceneOverrides | None = None) -> Session:
    """Parse the same resolved batch record in read-only previews and persisted runs."""
    root = layout.owner(path)
    require_keys(manifest_record, {'schema_version', 'scene', 'map', 'mapping', 'mapping_reference_arm', 'kit', 'expected_arms', 'description', 'intrinsics_source'},
                 {'schema_version', 'kit'}, 'session')
    if type(manifest_record['schema_version']) is not int or manifest_record['schema_version'] not in (4, 5):
        raise ValueError('采集批次需要 schema_version: 4 或 5')
    scene_path = root.parent / 'scene.yaml'
    if layout.manifest(root.parent, 'manifest.yaml').is_file() or (root.parent / 'batch.yaml').is_file():
        raise ValueError('采集批次不能嵌套在另一个批次内')
    if 'scene' in manifest_record and resolve_path(root, manifest_record['scene']) != scene_path:
        raise ValueError('采集批次必须位于所属场景目录下')
    scene = replace(read_scene(scene_path, load_mapping=False, overrides=overrides), batch_root=root)
    kit = validate_kit(manifest_record['kit'])
    source = mapping_records(manifest_record.get('mapping', {}))
    if source:
        read_arms(list(source))
        refs, devices = {}, {}
        for arm, item in source.items():
            require_keys(item, {'serial', 'path', 'sha256', 'calibration', 'gripper', 'rear_calibration', 'intrinsics_source'},
                         {'serial', 'path', 'sha256'}, '批次建图录像')
            refs[arm] = {key: item[key] for key in ('serial', 'path', 'sha256')}
            binding = {key: value for key, value in item.items() if key not in ('serial', 'path', 'sha256')}
            if not binding:
                binding = next(({key: value for key, value in camera.items() if key != 'serial'}
                                for camera in kit['arms'].values() if camera['serial'] == item['serial']), None)
                if binding is None:
                    raise ValueError('套件外的建图相机必须固定独立测量版本')
            devices[item['serial']] = binding
        paths, serials = read_recordings(root, refs, verify=verify_mapping)
        mode = 'shared_front' if len(paths) == 1 else 'dual'
        reference = manifest_record.get('mapping_reference_arm', next(iter(paths)))
        if reference not in paths:
            raise ValueError('批次建图参考手必须属于本轮建图录像')
        devices = read_devices(devices)
        scene = replace(scene, mode=mode, recordings=paths, recording_serials=serials, devices=devices,
                        reference_arm=reference, rear_calibrations=camera_assets.rear_files(devices))
    if path.name != 'manifest.yaml' or root.parent != scene.root:
        raise ValueError('采集批次必须位于 scenes/<场景>/<批次>/')
    validate_batch_name(root.name)
    expected = read_arms(manifest_record.get('expected_arms', list(kit['arms'])), kit['arms'])
    version, digest = None, None
    if manifest_record.get('map') is not None:
        require_keys(manifest_record['map'], {'version', 'manifest_sha256'}, {'version', 'manifest_sha256'}, '批次地图')
        version, digest = manifest_record['map']['version'], manifest_record['map']['manifest_sha256']
        registry.validate_id(version)
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('批次地图需要完整 SHA256')
    return Session(path, scene, version, digest, kit, expected, source)


def read_synchronization(synchronization_record: dict | None, arms: tuple[str, ...]) -> dict:
    """任务视频时钟：t_episode = t_video + offset；首只手为零偏移参考。"""
    if synchronization_record is None:
        synchronization_record = dict(offsets_s={arm: 0. for arm in arms}, confirmed=False, note='')
    require_keys(synchronization_record, {'offsets_s', 'confirmed', 'note'}, {'offsets_s', 'confirmed'}, 'synchronization')
    offsets = synchronization_record['offsets_s']
    if not isinstance(offsets, dict) or set(offsets) != set(arms):
        raise ValueError('synchronization.offsets_s 必须包含全部预期使用手')
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in offsets.values()):
        raise ValueError('同步偏移必须为有限秒数')
    if offsets[arms[0]] != 0:
        raise ValueError(f'参考手 {arms[0]} 的同步偏移必须为零')
    confirmed, note = synchronization_record['confirmed'], synchronization_record.get('note', '')
    if type(confirmed) is not bool or not isinstance(note, str):
        raise ValueError('同步 confirmed 必须为布尔值，note 必须为文本')
    if len(arms) == 1:
        return dict(method='single_camera', offsets_s={arms[0]: 0.}, confirmed=True, note=note)
    if confirmed and not note.strip():
        raise ValueError('确认双手同步时需要填写 note，说明核验依据')
    method = 'manual' if confirmed or any(offsets.values()) else 'assumed_simultaneous'
    return dict(method=method, offsets_s={arm: float(offsets[arm]) for arm in arms}, confirmed=confirmed, note=note)


def read_episode(path: Path, *, verify: bool = True, overrides: SceneOverrides | None = None) -> Episode:
    path = layout.manifest(path, 'episode.yaml')
    root = layout.owner(path)
    episode_document = read_yaml(path)
    require_keys(episode_document, {'schema_version', 'session', 'recordings', 'expected_arms', 'instruction', 'synchronization'},
                 {'schema_version', 'recordings'}, 'episode')
    if type(episode_document['schema_version']) is not int or episode_document['schema_version'] not in (4, 5):
        raise ValueError('任务清单需要 schema_version: 4 或 5')
    if episode_document['schema_version'] == 4 and 'synchronization' in episode_document:
        raise ValueError('配置同步请将任务 schema_version 更新为 5')
    episode_document = dict(episode_document, **layout.task_configuration(root))
    session_path = layout.manifest(root.parent, 'manifest.yaml')
    if 'session' in episode_document and layout.manifest(resolve_path(root, episode_document['session']), 'manifest.yaml') != session_path:
        raise ValueError('任务必须直接位于所属采集批次目录下')
    session = read_batch(session_path, overrides=overrides)
    if root.parent != session.root:
        raise ValueError('任务必须直接位于采集批次的 <任务>/ 目录')
    validate_episode_name(root.name)
    expected = read_arms(episode_document.get('expected_arms', session.expected_arms), session.expected_arms)
    recording_paths, serials = read_recordings(root, episode_document['recordings'], verify=verify)
    if set(serials.keys()) - set(expected):
        raise ValueError('实际录像包含本次任务未声明使用的手')
    if any(serial != session.arm_serials[arm] for arm, serial in serials.items()):
        raise ValueError('录像清单的设备身份与套件的左右手配置不一致')
    if not isinstance(episode_document.get('instruction', ''), str):
        raise ValueError('instruction 必须为操作描述字符串')
    synchronization = read_synchronization(episode_document.get('synchronization'), expected)
    return Episode(path, session, recording_paths, serials, expected, episode_document['recordings'],
                   episode_document.get('instruction', ''), synchronization)


def recording_identity(path: Path) -> dict:
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def artifact_hashes(directory: Path, roots: tuple[str, ...]) -> dict:
    return {str(path.relative_to(directory)): artifact_identity.file_hash(path)
            for name in roots for path in sorted((directory / name).rglob('*')) if path.is_file()}


def validate_assets(directory: Path, hashes: dict) -> None:
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('场景/任务资产清单为空')
    for name, digest in hashes.items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory.resolve()) or not target.is_file() or artifact_identity.file_hash(target) != digest:
            raise ValueError(f'场景/任务资产缺失、越界或内容已变：{name}')


def algorithm_identity() -> dict:
    identity = artifact_identity.code_identity('scene')
    return {name: digest for name, digest in identity.items() if Path(name).name not in {
        'cli.py', 'batches.py', 'ingest.py', 'recording_checks.py', 'episodes.py', '__main__.py'}}


def configuration_identity(batch_root: Path) -> str | None:
    declaration = batch_root / 'batch.yaml'
    if declaration.is_file():
        batch = read_yaml(declaration)
        return artifact_identity.canonical_hash(dict(batch=batch, kit=read_yaml(KITS / validate_name(batch['kit']) / 'kit.yaml')))
    return None


def episode_inputs(episode, cameras, map_manifest) -> dict:
    from software.gripper_width import pipeline as width_pipeline
    from software.gripper_width import tag_distance
    from software.vision.tag_board import load as load_tag_board

    widths = {arm: dict(sha256=camera.asset_hashes['width'])
              for arm, camera in cameras.items() if 'width' in camera.files}
    distances = {arm: dict(sha256=camera.asset_hashes['tag_distance'])
                 for arm, camera in cameras.items() if 'tag_distance' in camera.files}
    current_configuration = configuration_identity(episode.session.root)
    source_hashes = algorithm_identity()
    trajectory_hashes = artifact_identity.code_identity('trajectory')
    if any(source_hashes[name] != trajectory_hashes[name] for name in source_hashes.keys() & trajectory_hashes.keys()):
        raise ValueError('记录任务身份时源代码发生变化，请重新运行')
    source_hashes.update(trajectory_hashes)
    return dict(configuration=current_configuration, map_mode=map_manifest['mode'],
        scene_map_sha256=scene_map_identity(map_manifest), scene_assets=map_manifest['assets'],
        session_manifest_sha256=artifact_identity.file_hash(episode.session.path),
        episode_manifest_sha256=artifact_identity.file_hash(episode.path),
        task_configuration_sha256=layout.task_configuration_hash(episode.root),
        scene_manifest_sha256=artifact_identity.canonical_hash(scene_definition(
            read_yaml(episode.scene.path) if episode.scene.path.is_file() else dict(schema_version=2), episode.scene.name)),
        recordings={arm: recording_identity(video) for arm, video in episode.recordings.items()},
        cameras={arm: camera_assets.camera_identity(camera) for arm, camera in cameras.items()},
        synchronization_sha256=artifact_identity.canonical_hash(episode.synchronization),
        gripper_width=widths, gripper_width_code=width_pipeline.identity() if widths else {},
        tag_distance=distances, tag_distance_code=tag_distance.identity() if distances else {},
        environment_tags=load_tag_board(episode.scene.tag_layout).record() if episode.scene.tag_layout else None,
        parameters={key: value for key, value in asdict(episode.scene.parameters).items()
                    if not key.startswith(('recording_', 'pairing_'))}, trajectory_parameters=episode.scene.trajectory.snapshot()['values'],
        code=source_hashes, engines=artifact_identity.engine_identity('trajectory'))
