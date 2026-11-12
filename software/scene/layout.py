"""场景磁盘布局；用户输入留在外层，自动元数据统一位于 .umi。"""

from pathlib import Path
import json
import os

from ruamel.yaml import YAML

from software import provenance
from software.storage import locked as storage_lock


INTERNAL = '.umi'
TASK_FIELDS = {'instruction', 'expected_arms', 'synchronization'}


def owner(path: Path) -> Path:
    """从集中存储的清单还原原片目录；旧位置仅用于迁移和只读预览。"""
    for parent in path.parents:
        if parent.name == INTERNAL:
            relative = path.relative_to(parent).parts
            if len(relative) >= 3 and relative[0] == 'tasks':
                return parent.parent / relative[1]
            break
    return path.parent.parent if path.parent.name == INTERNAL else path.parent


def is_batch(root: Path) -> bool:
    return any(path.is_file() for path in (
        root / 'batch.yaml', root / 'manifest.yaml', root / INTERNAL / 'manifest.yaml',
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
    """目录或清单入口；已有旧清单可只读加载，不创建任何文件。"""
    path = path.expanduser().resolve()
    root = owner(path) if path.name == name else (path.parent if path.name == INTERNAL else path)
    current = metadata(root, name)
    return next((candidate for candidate in (current, root / INTERNAL / name, root / name)
                 if candidate.is_file()), current)


def maps(root: Path) -> Path:
    return metadata(root, 'maps')


def locked(root: Path, *, blocking: bool = False):
    return storage_lock(internal(root), blocking=blocking)


def task_configuration(root: Path) -> dict:
    """旧清单中的人工选项只读合并；迁移后仅保存在外层 task.yaml。"""
    reader = YAML(typ='safe', pure=True)
    capture_path = manifest(root, 'episode.yaml')
    capture = reader.load(capture_path.read_text()) if capture_path.is_file() else {}
    configuration = {key: value for key, value in capture.items() if key in TASK_FIELDS}
    path = root / 'task.yaml'
    if path.is_file():
        declaration = reader.load(path.read_text())
        if not isinstance(declaration, dict) or declaration.keys() - TASK_FIELDS:
            raise ValueError('task.yaml 只允许 instruction、expected_arms 和 synchronization')
        configuration.update(declaration)
    return configuration


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
        allowed = (metadata(root, 'runs'), root / INTERNAL / 'runs', root / 'runs')
        if not any(run.is_relative_to(directory.resolve()) for directory in allowed):
            raise ValueError('任务运行记录超出所属目录')
        error = run / 'error.json'
        if error.is_file():
            attempt['reason'] = json.loads(error.read_text()).get('reason', '')
        report = run / 'preflight_report.json'
        if report.is_file():
            attempt['preflight_report'] = run_reference(root, report)
    return state
