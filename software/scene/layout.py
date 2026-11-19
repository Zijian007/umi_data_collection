"""场景磁盘布局；用户输入留在外层，程序维护的地图、运行、缓存和状态统一位于 .pipeline。"""

from pathlib import Path
import json
import os

from software.infrastructure import provenance
from software.infrastructure.file_store import locked as storage_lock, read_yaml
from software.scene.management.collection_schema import normalize_collection


INTERNAL = '.pipeline'
TASK_FIELDS = {'instruction', 'expected_arms', 'synchronization', 'map_version', 'map_epoch', 'reconstruction'}
LEGACY_USER_FILES = {'scene.yaml', 'batch.yaml', 'task.yaml'}


def reject_legacy_user_file(path: Path) -> None:
    """Reject the retired split YAML entry points.

    Historical result directories remain readable through their frozen
    ``.pipeline`` manifests, but a new run must never silently migrate a
    split scene/batch/task declaration.
    """
    if path.name in LEGACY_USER_FILES:
        raise ValueError(
            f'{path.name} 已移除；请将场景、批次和任务配置合并到批次根目录的 collection.yaml')


def owner(path: Path) -> Path:
    """从当前布局的配置或集中存储清单还原原片目录。"""
    parent = path.parent
    if parent.parent.name == 'tasks' and parent.parent.parent.name == INTERNAL:
        return parent.parent.parent.parent / parent.name
    if parent.name == INTERNAL and path.name != 'episode.yaml':
        return parent.parent
    if path.name == 'collection.yaml':
        return parent
    reject_legacy_user_file(path)
    raise ValueError(f'不支持旧清单位置，请使用批次 {INTERNAL}/ 下的清单：{path}')


def is_batch(root: Path) -> bool:
    return any(path.is_file() for path in (root / 'collection.yaml', root / INTERNAL / 'manifest.yaml'))


def internal(root: Path) -> Path:
    if is_batch(root):
        return root / INTERNAL
    if is_batch(root.parent):
        return root.parent / INTERNAL / 'tasks' / root.name
    return root / INTERNAL


def metadata(root: Path, name: str) -> Path:
    return internal(root) / name


def manifest(path: Path, name: str) -> Path:
    """目录或当前清单入口，不创建文件，也不回退旧位置。"""
    path = path.expanduser().resolve()
    root = owner(path) if path.name == name else (path.parent if path.name == INTERNAL else path)
    return metadata(root, name)


def maps(root: Path) -> Path:
    return metadata(root, 'maps')


def locked(root: Path, *, blocking: bool = False):
    return storage_lock(internal(root), blocking=blocking)


def task_configuration(root: Path) -> dict:
    """读取任务选项；collection.yaml 是新的单文件用户入口。"""
    if (root / 'task.yaml').is_file():
        reject_legacy_user_file(root / 'task.yaml')
    batch = root
    if batch.parent.name == 'tasks' and batch.parent.parent.name == INTERNAL:
        batch = batch.parent.parent.parent
    elif (batch.parent / 'collection.yaml').is_file():
        # Raw task directories live directly under the batch root.
        batch = batch.parent
    collection = batch / 'collection.yaml'
    if not collection.is_file():
        return {}
    tasks = normalize_collection(read_yaml(collection)).get('tasks', {})
    if not isinstance(tasks, dict):
        raise ValueError('collection.yaml 的 tasks 必须是配置表')
    declaration = tasks.get(root.name, {})
    if not isinstance(declaration, dict):
        raise ValueError(f'collection.yaml 的 tasks.{root.name} 必须是配置表')
    if declaration.keys() - TASK_FIELDS:
        raise ValueError('collection.yaml 的任务只允许 instruction、arms、synchronization、map.epoch、map.version 和 reconstruction.backend')
    return declaration


def task_configuration_hash(root: Path) -> str:
    return provenance.canonical_hash(task_configuration(root))


def run_reference(root: Path, run: Path) -> str:
    return os.path.relpath(run, root)


def read_state(root: Path) -> dict:
    path = manifest(root, 'state.json')
    state = json.loads(path.read_text()) if path.is_file() else {}
    attempt = state.get('attempt', {})
    if attempt:
        attempt.setdefault('reason', '')
    if attempt.get('run'):
        run = (root / attempt['run']).resolve()
        if not run.is_relative_to(metadata(root, 'runs').resolve()):
            raise ValueError('任务运行记录超出所属目录')
        error = run / 'error.json'
        if error.is_file():
            attempt['reason'] = json.loads(error.read_text()).get('reason', '')
        report = run / 'preflight_report.json'
        if report.is_file():
            attempt['preflight_report'] = run_reference(root, report)
    return state
