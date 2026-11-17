from software.scene.management import kit_inputs
"""在临时目录生成测试标定包，不依赖工作区里的设备登记或演示数据。"""
import json
import shutil
from copy import deepcopy
from pathlib import Path

from software.infrastructure.file_store import read_yaml, write_json, write_yaml
from software.trajectory.calibration import assets as calibration_assets


def legacy_bundle(path: Path) -> Path:
    calibration_assets.make_bundle(calibration_assets.SAMPLE_CAMERA_JSON, path)
    return path


def device_record() -> dict:
    record = calibration_assets.describe(calibration_assets.HERE, 'factory')
    return dict(record, serial='TEST_CAMERA', status='active')


def write_kit(name: str, arms: dict, *, description='') -> dict:
    """Write an explicit test kit exactly as an operator writes kit.yaml."""
    from software.infrastructure.file_store import read_yaml
    from software.scene.management import kit_inputs
    from software.infrastructure.file_store import write_yaml

    arms = deepcopy(arms)
    directory = kit_inputs.KITS / name
    directory.mkdir(parents=True, exist_ok=True)
    for binding in arms.values():
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


def write_batch(scene: Path, name: str, *, kit: str, arms=None, map_version=None) -> Path:
    """Write a manual declaration and persist its resolved test snapshot."""
    from software.scene.management import batches, kit_inputs
    from software.infrastructure.file_store import write_yaml

    if scene.suffix in ('.yaml', '.yml'):
        root = scene.parent.parent if scene.name == 'collection.yaml' else scene.parent
    else:
        root = scene
    batch = root / name
    if scene.suffix in ('.yaml', '.yml'):
        source = read_yaml(scene)
        scene_document = source.get('scene', {}) if scene.name == 'collection.yaml' else {}
    elif (scene / 'collection.yaml').is_file():
        source = read_yaml(scene / 'collection.yaml')
        scene_document = source.get('scene', {})
    else:
        candidates = sorted(scene.glob('*/collection.yaml')) if scene.is_dir() else []
        seed = [path for path in candidates if path.parent.name == 'seed']
        source = read_yaml((seed or candidates)[0]) if (seed or candidates) else {}
        scene_document = source.get('scene', {}) if source else {}
    # Synthetic batches need a concrete topology when no mapping clips have
    # been staged yet; production auto mode still infers it from real clips.
    if not scene_document:
        scene_document = dict(layout='shared', map_reference_arm='left')
    elif scene_document.get('layout') == 'auto' and not (batch / 'mapping').exists():
        scene_document = dict(scene_document, layout='shared')
    data = dict(schema_version=1, scene=scene_document, kit=kit)
    if arms is not None:
        data['arms'] = list(arms)
    if map_version is not None:
        data['map_version'] = map_version
    write_yaml(batch / 'collection.yaml', data)
    return batches.save_record(batches.read_inputs(batch, require_mapping=False))


def write_task(session, name: str, recording_paths: dict, *, expected_arms=None, instruction=None) -> Path:
    """Arrange a task record for tests of readers, maps and results; no CLI behavior."""
    from software.scene import layout
    from software.scene.pipeline import recordings
    from software.infrastructure.file_store import write_yaml
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
            schema_version=1, scene=dict(layout='auto', map_reference_arm='left'))
        collection.setdefault('tasks', {})[name] = configuration
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
        schema_version=1, scene=dict(layout='auto', map_reference_arm='left'))
    collection.setdefault('tasks', {})[task.name] = deepcopy(data)
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
    collection = read_yaml(batch / 'collection.yaml')
    return deepcopy(collection.get('tasks', {}).get(task.name, {}))
