"""场景磁盘布局；用户输入留在外层，自动元数据统一位于 .umi。"""
from pathlib import Path
import json
from software.storage import locked as storage_lock


INTERNAL = '.umi'


def owner(path: Path) -> Path:
    """清单所属的批次或任务目录，兼容尚未迁移的旧清单。"""
    return path.parent.parent if path.parent.name == INTERNAL else path.parent


def metadata(root: Path, name: str) -> Path:
    return root / INTERNAL / name


def manifest(path: Path, name: str) -> Path:
    """目录或清单入口；已有旧清单可只读加载，不创建任何文件。"""
    path = path.expanduser().resolve()
    root = owner(path) if path.name == name else (path.parent if path.name == INTERNAL else path)
    current = metadata(root, name)
    return current if current.is_file() or not (root / name).is_file() else root / name


def maps(root: Path) -> Path:
    return metadata(root, 'maps')


def locked(root: Path, *, blocking: bool = False):
    return storage_lock(root / INTERNAL, blocking=blocking)


def read_state(root: Path) -> dict:
    path = manifest(root, 'state.json')
    state = json.loads(path.read_text()) if path.is_file() else {}
    attempt = state.get('attempt', {})
    if attempt:
        attempt.setdefault('reason', '')
    if attempt.get('run'):
        run = (root / attempt['run']).resolve()
        if not run.is_relative_to(root.resolve()):
            raise ValueError('任务运行记录超出所属目录')
        error = run / 'error.json'
        if error.is_file():
            attempt['reason'] = json.loads(error.read_text()).get('reason', '')
        report = run / 'preflight_report.json'
        if report.is_file():
            attempt['preflight_report'] = str(report.relative_to(root.resolve()))
    return state
