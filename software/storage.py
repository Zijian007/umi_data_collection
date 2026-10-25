"""场景和设备档案共用的原子文件、写入锁与结果版本指针。"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ruamel.yaml import YAML


def write_json(path: Path, data: dict) -> None:
    with atomic_file(path) as temporary:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_yaml(path: Path, data: dict) -> None:
    writer = YAML()
    writer.default_flow_style = False
    writer.allow_unicode = True
    writer.width = 4096
    with atomic_file(path) as temporary, temporary.open('w', encoding='utf-8') as stream:
        writer.dump(data, stream)


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
def locked(directory: Path, name: str = '.scene.lock'):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
