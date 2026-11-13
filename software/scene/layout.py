"""场景磁盘布局；用户输入留在外层，程序维护的地图、运行、缓存和状态统一位于 .pipeline。"""

from pathlib import Path
import json
import os

from software import provenance
from software.storage import locked as storage_lock, read_yaml


INTERNAL = '.pipeline'
TASK_FIELDS = {'instruction', 'expected_arms', 'synchronization', 'map_version'}


def owner(path: Path) -> Path:
    """从当前布局的配置或集中存储清单还原原片目录。"""
    parent = path.parent
    if parent.parent.name == 'tasks' and parent.parent.parent.name == INTERNAL:
        return parent.parent.parent.parent / parent.name
    if parent.name == INTERNAL and path.name != 'episode.yaml':
        return parent.parent
    if path.name in {'batch.yaml', 'scene.yaml', 'task.yaml'}:
        return parent
    raise ValueError(f'不支持旧清单位置，请使用批次 {INTERNAL}/ 下的清单：{path}')


def is_batch(root: Path) -> bool:
    return any(path.is_file() for path in (
        root / 'batch.yaml', root / INTERNAL / 'manifest.yaml',
    ))


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
    """人工选项只从任务目录的 task.yaml 读取。"""
    path = root / 'task.yaml'
    if path.is_file():
        declaration = read_yaml(path)
        if declaration.keys() - TASK_FIELDS:
            raise ValueError('task.yaml 只允许 instruction、expected_arms、synchronization 和 map_version')
        return declaration
    return {}


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
