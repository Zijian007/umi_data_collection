"""在临时目录生成测试标定包，不依赖工作区里的设备登记或演示数据。"""
import json
import shutil
from copy import deepcopy
from pathlib import Path

from software import registry
from software.storage import write_json
from software.trajectory.calibration import assets as calibration_assets


def legacy_bundle(path: Path) -> Path:
    calibration_assets.make_bundle(calibration_assets.SAMPLE_CAMERA_JSON, path)
    measured = calibration_assets.read_basalt(path / 'calibration_r1440.json')
    metadata = json.loads((path / 'bundle.json').read_text())
    metadata['schema_version'] = 1
    write_json(path / 'bundle.json', metadata)
    (path / 'front_x3_r1440_720.yaml').write_text(calibration_assets.yaml_text(measured))
    (path / 'kiwi_fuse3s.py').write_text(
        f'P_CfinI = np.array({measured["p"]!r})\n'
        f'R_ICf = Rot.from_quat({measured["q_xyzw"]!r}).as_matrix()\n')
    return path


def device_record() -> dict:
    record = calibration_assets.describe(calibration_assets.HERE, 'factory')
    return dict(record, serial='TEST_CAMERA', status='active')


def write_kit(name: str, arms: dict, *, description='') -> dict:
    """Write an explicit test kit exactly as an operator writes kit.yaml."""
    from software.scene.management import inputs
    from software.storage import write_yaml

    arms = deepcopy(arms)
    directory = inputs.KITS / name
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
    write_yaml(inputs.KITS / name / 'kit.yaml', data)
    return inputs.read_kit(name)


def write_batch(scene: Path, name: str, *, kit: str, arms=None, map_version=None) -> Path:
    """Write a manual declaration and persist its resolved test snapshot."""
    from software.scene.management import batches
    from software.storage import write_yaml

    root = scene.parent if scene.suffix in ('.yaml', '.yml') else scene
    batch = root / name
    data = dict(kit=kit)
    if arms is not None:
        data['arms'] = list(arms)
    if map_version is not None:
        data['map_version'] = map_version
    write_yaml(batch / 'batch.yaml', data)
    return batches.save_record(batches.read_inputs(batch, require_mapping=False))


def write_task(session, name: str, recording_paths: dict, *, expected_arms=None, instruction=None) -> Path:
    """Arrange a task record for tests of readers, maps and results; no CLI behavior."""
    from software.scene import layout
    from software.scene.pipeline import recordings
    from software.storage import write_yaml
    import os

    directory = session.root / name
    task_record = dict(schema_version=5, recordings={})
    for arm, video in recording_paths.items():
        metadata = recordings.inspect(video)
        target = recordings.import_video(video, directory / arm, metadata=metadata, owned_root=directory)
        task_record['recordings'][arm] = dict(metadata, path=os.path.relpath(target, directory))
    if expected_arms is not None:
        task_record['expected_arms'] = list(expected_arms)
    if instruction is not None:
        task_record['instruction'] = instruction
    path = layout.metadata(directory, 'episode.yaml')
    write_yaml(path, task_record)
    return path
