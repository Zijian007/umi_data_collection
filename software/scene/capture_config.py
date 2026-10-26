"""采集登记：kit.yaml 保存当前套件配置，创建批次时自动保存独立快照。"""
from __future__ import annotations

import os
from pathlib import Path

from software.scene import inputs
from software import registry
from software.scene.models import ARMS
from software.storage import locked, write_yaml


def register_kit(scene_path: Path, name: str, arms: dict[str, str], *, rear: dict[str, str] | None = None, description='') -> dict:
    """更新当前配置；已有批次使用自己的设备快照，不受此处修改影响。"""
    scene = inputs.read_scene(scene_path)
    name = inputs.validate_name(name)
    inputs.read_arms(list(arms))
    if len(set(arms.values())) != len(arms):
        raise ValueError('左右手不能使用同一台相机')
    device_bindings = {}
    for arm in ARMS:
        if arm not in arms:
            continue
        serial = registry.validate_id(arms[arm])
        device_binding = registry.pin(serial)
        if rear and arm in rear:
            registry.rear_path(serial, rear[arm])
            device_binding['rear_calibration'] = rear[arm]
        device_bindings[arm] = dict(serial=serial, **device_binding)
    kit_config = inputs.validate_kit(dict(id=name, description=description, arms=device_bindings))
    with locked(scene.root / 'kits'):
        write_yaml(scene.root / 'kits' / name / 'kit.yaml', kit_config)
    return kit_config


def list_kits(scene_path: Path) -> list[dict]:
    scene = inputs.read_scene(scene_path)
    root = scene.root / 'kits'
    return [inputs.read_kit(scene, path.name) for path in sorted(root.iterdir())
            if (path / 'kit.yaml').is_file()] if root.exists() else []


def create_session(scene_path: Path, name: str, *, kit: str,
                   arms: tuple[str, ...] | None = None, description='') -> Path:
    scene = inputs.read_scene(scene_path)
    kit_snapshot = inputs.read_kit(scene, kit)
    expected_arms = inputs.read_arms(arms or list(kit_snapshot['arms']), kit_snapshot['arms'])
    for item in kit_snapshot['arms'].values():
        registry.get(item['serial'], **{key: item[key] for key in ('calibration', 'gripper', 'gripper_version')})
        if 'rear_calibration' in item:
            registry.rear_path(item['serial'], item['rear_calibration'])
    session_dir = scene.root / 'kits' / kit_snapshot['id'] / 'sessions' / inputs.validate_name(name)
    session_path = session_dir / 'session.yaml'
    with locked(session_dir):
        if session_path.exists():
            raise ValueError(f'采集批次已存在：{session_path}')
        write_yaml(session_path, dict(schema_version=4, scene=os.path.relpath(scene.path, session_dir), description=description,
            kit=kit_snapshot, expected_arms=list(expected_arms), map=None))
    return session_path
