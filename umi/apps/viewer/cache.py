"""查看器可再生缓存：输入身份、跨进程锁与原子发布；不写录像目录。"""

from __future__ import annotations

import os
import threading
from functools import lru_cache, wraps
from pathlib import Path

from umi.config import operational_settings as settings
from umi.infra.file_store import atomic_file
from umi.infra.file_store import locked as directory_lock
from umi.infra.provenance import canonical_hash


def bounded(setting: str):
    """延迟创建内存缓存：导入模块与 --help 不读取配置、不分配工作线程。"""
    def decorate(function):
        cached, size = None, None
        lock = threading.Lock()

        @wraps(function)
        def call(*args, **kwargs):
            nonlocal cached, size
            configured = settings.values('viewer')[setting]
            with lock:
                if cached is None or size != configured:
                    cached, size = lru_cache(maxsize=configured)(function), configured
                invoke = cached
            return invoke(*args, **kwargs)

        def clear():
            with lock:
                if cached is not None:
                    cached.cache_clear()

        call.cache_clear = clear
        return call
    return decorate


def file_identity(path: Path) -> tuple:
    path = path.resolve()
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (str(path), None)
    return (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)


def root() -> Path:
    configured = os.environ.get('UMI_VIEW_CACHE')
    return (Path(configured).expanduser() if configured else
            Path(os.environ.get('XDG_CACHE_HOME') or Path.home()/'.cache')/'umi-view').resolve()


def path(kind: str, identity, suffix: str) -> Path:
    return root() / kind / (canonical_hash(identity) + suffix)


def locked(target: Path):
    return directory_lock(target.parent, target.name + '.lock', blocking=True)


def atomic_output(target: Path):
    # FFmpeg infers the output container from the temporary file's suffix.
    return atomic_file(target, prefix='.building-', suffix=target.suffix)
