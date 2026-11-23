"""资产内容哈希与代码版本记录；供缓存、设备档案和结果追溯共用。

缓存只依据数据输入（原片、标定、参数与配置）判定能否复用，不包含代码或引擎身份；
改了算法或重编引擎后用 ``--force`` 重算。结果中记录 git 提交号，仅用于追溯。
"""

from __future__ import annotations

from pathlib import Path
import functools
import hashlib
import json
import os
import threading

from umi.infra.paths import PROJECT_ROOT


ROOT = PROJECT_ROOT


def canonical_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


# 原片、地图和镜头视频在一次运行中会被多个阶段核对。大文件按文件系统身份
# 记忆摘要，避免重复整读；小文件始终重读，保留对配置/标定改动的即时检测。
LARGE_FILE_BYTES = 16 * 1024 * 1024
_large_file_hashes: dict[tuple, str] = {}
_large_file_lock = threading.Lock()


def file_hash(path: Path) -> str:
    with path.open('rb') as source:
        stat = os.fstat(source.fileno())
        if stat.st_size < LARGE_FILE_BYTES:
            return hashlib.file_digest(source, 'sha256').hexdigest()
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        with _large_file_lock:
            cached = _large_file_hashes.get(key)
        if cached is None:
            cached = hashlib.file_digest(source, 'sha256').hexdigest()
            with _large_file_lock:
                _large_file_hashes[key] = cached
        return cached


@functools.lru_cache(maxsize=1)
def git_revision() -> dict:
    """当前代码的 git 提交号与 umi/ 下是否有未提交改动；仅写入结果供追溯，不参与缓存判定。"""
    import subprocess
    try:
        with subprocess.Popen(['git', 'rev-parse', 'HEAD'], cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True) as process:
            commit = process.communicate(timeout=10)[0].strip()
        with subprocess.Popen(['git', 'status', '--porcelain', '--untracked-files=no', '--', 'umi'], cwd=ROOT,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) as process:
            status = process.communicate(timeout=10)[0]
    except (OSError, subprocess.SubprocessError):
        return dict(commit=None, dirty=None)
    return dict(commit=commit or None, dirty=bool(status.strip()) if commit else None)
