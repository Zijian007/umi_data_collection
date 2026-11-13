"""场景和设备档案共用的原子文件、写入锁与结果版本指针。"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import tempfile

from ruamel.yaml import YAML


def read_yaml(path: Path) -> dict:
    """读取一个 YAML 1.2 映射；拒绝重复键、可执行标签和其他版本指令。"""
    parser = YAML(typ='safe', pure=True)
    parser.version = (1, 2)
    parser.allow_duplicate_keys = False
    with path.open(encoding='utf-8') as stream:
        document = parser.load(stream)
    if parser.version != (1, 2) or not isinstance(document, dict):
        raise ValueError(f'{path} 需要 YAML 1.2 映射（键: 值）；空内容请写 {{}}')
    return document


def write_json(path: Path, document: dict) -> None:
    with atomic_file(path) as temporary:
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_yaml(path: Path, document: dict) -> None:
    writer = YAML()
    writer.default_flow_style = False
    writer.allow_unicode = True
    writer.width = 4096
    with atomic_file(path) as temporary, temporary.open('w', encoding='utf-8') as stream:
        writer.dump(document, stream)


@contextmanager
def atomic_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def locked(directory: Path, name: str = '.scene.lock', *, blocking: bool = False):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise ValueError(f'已有写入操作正在运行：{directory}') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def publish_link(source: Path, target: Path) -> None:
    """同目录原子替换指针；不删除上一次成功结果。"""
    temporary = target.with_name('.'+target.name+'.link')
    temporary.unlink(missing_ok=True)
    if target.exists() and not target.is_symlink():
        raise ValueError(f'发布目标不是本工具的版本指针，请先移走：{target}')
    temporary.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)
    temporary.replace(target)


def mark_used(directory: Path) -> None:
    """命中缓存条目时刷新目录时间；成功处理后据此删除本次未用到的条目。"""
    try:
        os.utime(directory)
    except FileNotFoundError:
        pass
