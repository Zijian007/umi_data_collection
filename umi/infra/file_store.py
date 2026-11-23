"""公共文件读写、缓存与原子发布。"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
import fcntl
import json
import os
import tempfile

from ruamel.yaml import YAML

if TYPE_CHECKING:
    import numpy as np


# --force：本次运行开始之前生成的缓存一律视为失效；本次运行中新生成的仍可复用（例如建图时
# 解码的录像随后被任务复用）。进程级开关，由命令入口在开始处理前设置。
_reuse_since: float | None = None


@contextmanager
def recompute(enabled: bool = True):
    """在此范围内只复用范围开始之后完成的缓存；嵌套时沿用最外层的起点。"""
    global _reuse_since
    previous = _reuse_since
    if enabled and previous is None:
        _reuse_since = _filesystem_now()
    try:
        yield
    finally:
        _reuse_since = previous


def recompute_when_forced(function):
    """命令入口装饰器：``force=True``（且非 ``dry_run``）时整个调用处于 recompute 范围。"""
    import functools

    @functools.wraps(function)
    def wrapper(*args, force: bool = False, dry_run: bool = False, **kwargs):
        with recompute(force and not dry_run):
            return function(*args, force=force, dry_run=dry_run, **kwargs)
    return wrapper


def _filesystem_now() -> float:
    """用文件系统时钟取当前时刻：内核写文件时间戳的粒度比 time.time() 粗，直接比较会把
    紧接着写出的缓存误判为旧缓存。"""
    with tempfile.NamedTemporaryFile() as stamp:
        return os.fstat(stamp.fileno()).st_mtime


def is_current(marker: Path) -> bool:
    """缓存完成标记是否可用：不存在为否；处于 recompute 范围时需晚于范围起点。"""
    try:
        modified = marker.stat().st_mtime
    except FileNotFoundError:
        return False
    return _reuse_since is None or modified >= _reuse_since


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


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 TUM 轨迹的秒时间戳、位置和 xyzw 四元数。"""
    import numpy as np

    rows = [list(map(float, line.split())) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]
    trajectory = np.array(rows)
    return trajectory[:, 0], trajectory[:, 1:4], trajectory[:, 4:8]


def write_tum(path: Path, times_s: np.ndarray, positions: np.ndarray,
              quaternions_xyzw: np.ndarray, header: str) -> None:
    """按项目既有精度写入 TUM 轨迹。"""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for header_line in header.strip().splitlines():
            stream.write("# " + header_line + "\n")
        for frame_index in range(len(times_s)):
            stream.write("%.6f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" %
                         (times_s[frame_index], *positions[frame_index], *quaternions_xyzw[frame_index]))


def read_npz_cache(path: Path, marker: Path, inputs: dict, *, mismatch: str | None = None) -> tuple[dict, str] | None:
    """读取已发布的数组缓存，并核对输入身份与文件内容。

    未命中（无标记、--force 之前生成）返回 None；身份或内容不符时，给了 ``mismatch`` 则报错，
    否则同样视为未命中。
    """
    if not is_current(marker):
        return None
    import numpy as np
    from umi.infra.provenance import file_hash

    try:
        record = json.loads(marker.read_text(encoding='utf-8'))
        # 按 JSON 规范形式比较：元组与列表、整数键等在落盘后不再区分。
        expected = json.loads(json.dumps(inputs, allow_nan=False))
        valid = record['inputs'] == expected and path.is_file() and file_hash(path) == record['sha256']
    except (OSError, ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        if mismatch is None:
            return None
        raise ValueError(mismatch)
    with np.load(path, allow_pickle=False) as arrays:
        return dict(arrays), record['sha256']


def write_npz_cache(path: Path, marker: Path, inputs: dict, **arrays) -> str:
    """先原子保存数组，再发布包含内容哈希的完成标记。"""
    import numpy as np
    from umi.infra.provenance import file_hash

    with atomic_file(path) as temporary, temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    digest = file_hash(path)
    write_json(marker, dict(inputs=inputs, sha256=digest))
    return digest


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


def publish_link(source: Path, target: Path, *, target_is_directory: bool = True) -> None:
    """在目标目录原子更新指向源路径的相对符号链接。"""
    temporary = target.with_name('.'+target.name+'.link')
    temporary.unlink(missing_ok=True)
    if target.exists() and not target.is_symlink():
        raise ValueError(f'发布目标不是本工具的版本指针，请先移走：{target}')
    temporary.symlink_to(os.path.relpath(source, target.parent), target_is_directory=target_is_directory)
    temporary.replace(target)


def mark_used(directory: Path) -> None:
    """命中缓存条目时刷新目录时间；成功处理后据此删除本次未用到的条目。"""
    try:
        os.utime(directory)
    except FileNotFoundError:
        pass
