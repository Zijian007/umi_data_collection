"""场景产物的文件身份：资产哈希清单、清单校验和原片轻量身份。

只依赖基础设施层；管理、流水线与缓存模块都从这里取用，不经过任务输入解析。
"""

from __future__ import annotations

from pathlib import Path

from software.infrastructure import provenance


def recording_identity(path: Path) -> dict:
    """不读内容的原片身份（路径、大小、修改时间），用于运行期间的变化检测。"""
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def artifact_hashes(directory: Path, roots: tuple[str, ...]) -> dict:
    """directory 下各 root 子树内全部文件的相对路径 → SHA256。"""
    return {str(path.relative_to(directory)): provenance.file_hash(path)
            for name in roots for path in sorted((directory / name).rglob('*')) if path.is_file()}


def validate_assets(directory: Path, hashes: dict) -> None:
    """核对资产清单：非空、不越界、文件存在且内容未变。"""
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('场景/任务资产清单为空')
    for name, digest in hashes.items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory.resolve()) or not target.is_file() or provenance.file_hash(target) != digest:
            raise ValueError(f'场景/任务资产缺失、越界或内容已变：{name}')
