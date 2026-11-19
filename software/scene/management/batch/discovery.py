"""批次目录发现；不读取算法配置，也不启动处理。"""

from __future__ import annotations

from pathlib import Path
import os

from software.scene import layout
from software.scene.management.inputs.scene import INTERNAL_DIRECTORIES


def batch_root(path: Path) -> Path:
    """把批次目录、collection.yaml 或 .pipeline 内的清单规范为批次根目录。"""
    path = path.expanduser().resolve()
    if path.is_file():
        path = layout.owner(path)
    return path.parent if path.name == layout.INTERNAL else path


def require_collection(batch: Path, command: str) -> None:
    if (batch / 'batch.yaml').is_file():
        layout.reject_legacy_user_file(batch / 'batch.yaml')
    if not (batch / 'collection.yaml').is_file():
        raise ValueError(f'{command} 需要包含 collection.yaml 的采集批次')


def configured_batches(paths: list[Path]) -> list[Path]:
    """发现真实 collection.yaml；示例、缓存和运行结果不参与发现。"""
    found = set()
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_file():
            layout.reject_legacy_user_file(path)
            if path.name not in {'collection.yaml', 'manifest.yaml'}:
                continue
            path = layout.owner(path)
        if not path.is_dir():
            raise ValueError(f'处理路径不存在：{path}')
        for current, children, files in os.walk(path):
            children[:] = sorted(name for name in children if not name.startswith('.')
                                 and name not in INTERNAL_DIRECTORIES)
            if 'batch.yaml' in files:
                layout.reject_legacy_user_file(Path(current) / 'batch.yaml')
            if 'scene.yaml' in files and 'collection.yaml' not in files:
                layout.reject_legacy_user_file(Path(current) / 'scene.yaml')
            if 'collection.yaml' in files:
                found.add(Path(current))
                children[:] = []
    return sorted(found)
