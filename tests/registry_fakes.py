"""在临时目录生成测试标定包，不依赖工作区里的设备登记或演示数据。"""
import shutil
from copy import deepcopy
from pathlib import Path

from umi.devices import calibration_assets
from umi.infra.file_store import read_yaml, write_json, write_yaml
from umi.infra.provenance import file_hash
from umi.scene.collection import kit as kit_inputs


def legacy_bundle(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    target = path / 'calibration_r1440.json'
    shutil.copyfile(calibration_assets.SAMPLE_CAMERA_JSON, target)
    measured = calibration_assets.read_basalt(target)
    write_json(path / 'bundle.json', dict(kind='umi_traj_calibration_bundle', schema_version=2,
        calibration_sha256=file_hash(target), T_imu_cam={'p': measured['p'], 'q_xyzw': measured['q_xyzw']},
        cam_time_offset_ns=measured['cam_time_offset_ns']))
    return path


def device_record() -> dict:
    record = calibration_assets.describe(calibration_assets.SAMPLE_CAMERA_JSON.parent, 'factory')
    return dict(record, serial='TEST_CAMERA', status='active')


def write_kit(name: str, arms: dict, *, description='') -> dict:
    """Write an explicit test kit exactly as an operator writes kit.yaml."""
    from umi.infra.file_store import write_yaml

    arms = deepcopy(arms)
    directory = kit_inputs.KITS / name
    directory.mkdir(parents=True, exist_ok=True)
    for binding in arms.values():
        # Older numerical fixtures name fixed assets; public kit files use
        # current lens selections. Historical snapshots are tested separately.
        if 'intrinsics' not in binding:
            binding['intrinsics'] = dict(front=binding.get('intrinsics_source', 'measured'),
                rear=binding.get('rear_intrinsics_source', 'measured' if 'rear_calibration' in binding else 'factory'))
            for field in ('calibration', 'rear_calibration', 'calibrations', 'intrinsics_source', 'rear_intrinsics_source'):
                binding.pop(field, None)
        for asset, filename in binding['gripper'].items():
            source = Path(filename)
            if source.is_absolute():
                binding['gripper'][asset] = source.name
                if source.exists() and source.resolve() != (directory / source.name).resolve():
                    shutil.copyfile(source, directory / source.name)
        tcp = directory / binding['gripper']['tcp']
        if not tcp.exists():
            tcp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(__file__).parent / 'fixtures/tcp_camera.json', tcp)
    data = dict(arms=arms)
    if description:
        data['description'] = description
    write_yaml(kit_inputs.KITS / name / 'kit.yaml', data)
    return kit_inputs.read_kit(name)


def _scene_text(source: dict) -> dict:
    """Scene fields of a public collection or of an internal scene document used as a fixture seed."""
    if 'arms' in source and 'kit' in source:
        return {key: source[key] for key in ('description',) if key in source}
    return dict(source.get('scene', {}))


def write_batch(scene: Path, name: str, *, kit: str, arms=None, map_version=None) -> Path:
    """Write a manual declaration and persist its resolved test snapshot."""
    from umi.scene.collection import batch as batches
    from umi.infra.file_store import write_yaml

    if scene.suffix in ('.yaml', '.yml'):
        root = scene.parent.parent if scene.name == 'collection.yaml' else scene.parent
    else:
        root = scene
    batch = root / name
    if scene.suffix in ('.yaml', '.yml'):
        source = read_yaml(scene)
        if 'arms' in source and 'map' in source and scene.name == 'collection.yaml':
            scene_document = dict(_scene_text(source),
                                  layout=source.get('map', {}).get('layout', 'auto'),
                                  map_reference_arm=source.get('map', {}).get('reference_arm', 'left'))
        else:
            scene_document = _scene_text(source) if scene.name == 'collection.yaml' else {}
    elif (scene / 'collection.yaml').is_file():
        source = read_yaml(scene / 'collection.yaml')
        if 'arms' in source and 'map' in source:
            scene_document = dict(_scene_text(source),
                                  layout=source.get('map', {}).get('layout', 'auto'),
                                  map_reference_arm=source.get('map', {}).get('reference_arm', 'left'))
        else:
            scene_document = _scene_text(source)
    else:
        candidates = sorted(scene.glob('*/collection.yaml')) if scene.is_dir() else []
        seed = [path for path in candidates if path.parent.name == 'seed']
        source = read_yaml((seed or candidates)[0]) if (seed or candidates) else {}
        if 'arms' in source and 'map' in source:
            scene_document = dict(_scene_text(source),
                                  layout=source.get('map', {}).get('layout', 'auto'),
                                  map_reference_arm=source.get('map', {}).get('reference_arm', 'left'))
        else:
            scene_document = _scene_text(source) if source else {}
    # Synthetic batches need a concrete topology when no mapping clips have
    # been staged yet; production auto mode still infers it from real clips.
    if not scene_document:
        scene_document = dict(layout='shared', map_reference_arm='left')
    elif scene_document.get('layout') == 'auto' and not (batch / 'mapping').exists():
        scene_document = dict(scene_document, layout='shared')
    selected_arms = list(arms) if arms is not None else list(kit_inputs.read_kit(kit)['arms'])
    old_tags = scene_document.get('tags', {})
    poster = None
    if old_tags:
        poster = dict(layout=old_tags.get('layout'), detection=old_tags.get('enabled', False))
        mapping = old_tags.get('mapping', {})
        poster_scale = ('correct' if mapping.get('scale', {}).get('enabled') else
                        'diagnose' if mapping.get('enabled') else 'off')
    data = dict(kit=kit, arms=selected_arms,
                description=scene_document.get('description', ''),
                map=dict(layout=scene_document.get('layout', 'auto'),
                         reference_arm=scene_document.get('map_reference_arm', 'left')),
                backend=dict(mode='scene_map'))
    if poster is not None:
        data['poster'] = poster
        data['map']['poster_scale'] = poster_scale
        fusion = old_tags.get('fusion', {})
        data['backend']['scene_map'] = dict(
            poster_constraint=fusion.get('method', 'staged') if fusion.get('enabled', False) else 'off')
    if map_version is not None:
        data['map']['version'] = map_version
    write_yaml(batch / 'collection.yaml', data)
    return batches.save_record(batches.read_inputs(batch, require_mapping=False))


def write_task(session, name: str, recording_paths: dict, *, expected_arms=None, instruction=None) -> Path:
    """Arrange a task record for tests of readers, maps and results; no CLI behavior."""
    from umi.store import layout
    from umi.scene.inputs import recordings
    from umi.infra.file_store import write_yaml
    import os

    directory = session.root / name
    task_record = dict(schema_version=5, recordings={})
    for arm, video in recording_paths.items():
        metadata = recordings.inspect(video)
        target = recordings.import_video(video, directory / arm, metadata=metadata, owned_root=directory)
        task_record['recordings'][arm] = dict(metadata, path=os.path.relpath(target, directory))
    configuration = {}
    if expected_arms is not None:
        configuration['expected_arms'] = list(expected_arms)
    if instruction is not None:
        configuration['instruction'] = instruction
    if configuration:
        collection_path = session.root / 'collection.yaml'
        collection = read_yaml(collection_path) if collection_path.is_file() else dict(
            kit=session.kit['id'], arms=list(session.expected_arms),
            description='', backend=dict(mode='scene_map'))
        collection.setdefault('tasks', {})[name] = dict(
            **({'arms': configuration.pop('expected_arms')} if 'expected_arms' in configuration else {}),
            **configuration)
        write_yaml(collection_path, collection)
    path = layout.metadata(directory, 'episode.yaml')
    write_yaml(path, task_record)
    return path


def write_task_configuration(task: Path, data: dict) -> Path:
    """Update the task entry in the owning batch collection document."""
    task = Path(task)
    if task.name == 'episode.yaml':
        task = task.parent
    if task.parent.name == 'tasks' and task.parent.parent.name == '.pipeline':
        batch = task.parent.parent.parent
    elif (task.parent / 'collection.yaml').is_file():
        batch = task.parent
    else:
        batch = task
    path = batch / 'collection.yaml'
    collection = read_yaml(path) if path.is_file() else dict(
        kit='rig', arms=['left', 'right'],
        description='', backend=dict(mode='scene_map'))
    task_data = deepcopy(data)
    if 'synchronization' in task_data:
        task_data['sync'] = task_data.pop('synchronization')
    if isinstance(task_data.get('sync'), dict) and 'mode' not in task_data['sync']:
        task_data['sync']['mode'] = 'manual'
    if 'expected_arms' in task_data:
        task_data['arms'] = task_data.pop('expected_arms')
    if 'reconstruction' in task_data and isinstance(task_data['reconstruction'], dict):
        task_data['backend'] = task_data.pop('reconstruction')
    task_map = {}
    if 'map_epoch' in task_data:
        task_map['epoch'] = task_data.pop('map_epoch')
    if 'map_version' in task_data:
        task_map['version'] = task_data.pop('map_version')
    if task_map:
        task_data['map'] = task_map
    collection.setdefault('tasks', {})[task.name] = task_data
    write_yaml(path, collection)
    return path


def read_task_configuration(task: Path) -> dict:
    task = Path(task)
    if task.name == 'episode.yaml':
        task = task.parent
    if task.parent.name == 'tasks' and task.parent.parent.name == '.pipeline':
        batch = task.parent.parent.parent
    elif (task.parent / 'collection.yaml').is_file():
        batch = task.parent
    else:
        batch = task
    from umi.config.collection_schema import normalize_collection
    collection = normalize_collection(read_yaml(batch / 'collection.yaml'))
    return deepcopy(collection.get('tasks', {}).get(task.name, {}))
