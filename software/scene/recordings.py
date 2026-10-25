"""原片导入：以 INSV 序列号归档，保留原始文件名，拒绝同名覆盖。"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from software import registry

ROOT = Path(__file__).resolve().parents[2] / 'recordings' / 'scenes'


def inspect(video: Path) -> dict:
    video = video.expanduser().resolve()
    if not video.is_file() or video.suffix.lower() != '.insv':
        raise ValueError(f'需要原始 INSV 录像：{video}')
    serial = registry.validate_id(registry.read_insv_serial(video))
    return dict(serial=serial, sha256=registry.sha256_file(video))


def import_video(video: Path, directory: Path, *, metadata: dict) -> Path:
    """SD 卡原片默认复制；已在本场景归档的录像直接引用，避免建图/任务重复存储。"""
    video = video.expanduser().resolve()
    scene_root = directory.relative_to(ROOT).parts[0]
    managed = ROOT / scene_root
    if video.is_relative_to(managed) and video.parent.name == metadata['serial']:
        return video
    target = directory / metadata['serial'] / video.name
    if target.exists():
        if registry.sha256_file(target) != metadata['sha256']:
            raise ValueError(f'同名录像内容不同，拒绝覆盖：{target}')
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.import-', dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(video, temporary)
        if registry.sha256_file(temporary) != metadata['sha256']:
            raise ValueError(f'录像复制校验失败：{video}')
        # link 的排他创建语义保证并发导入也不能覆盖已有原片。
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
