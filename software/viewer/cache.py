"""查看器可再生缓存：输入身份、跨进程锁与原子发布；不写录像目录。"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
import fcntl
import hashlib
import json
import os
import tempfile
import threading

from software import settings


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


def key(identity) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def root() -> Path:
    configured = os.environ.get('UMI_VIEW_CACHE')
    return (Path(configured).expanduser() if configured else
            Path(os.environ.get('XDG_CACHE_HOME') or Path.home()/'.cache')/'umi-view').resolve()


def path(kind: str, identity, suffix: str) -> Path:
    return root() / kind / (key(identity) + suffix)


@contextmanager
def locked(target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(target.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def atomic_output(target: Path):
    """同一文件系统中先完成临时文件，成功后才替换缓存；失败不留下半成品。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.building-', suffix=target.suffix, dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
