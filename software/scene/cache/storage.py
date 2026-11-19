"""共用阶段缓存：完整性校验、只读复用、原子发布、审计与计时。"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import json
import os
import shutil
import stat
import time
import uuid

from software.infrastructure import provenance as artifact_identity
from software.infrastructure.file_store import mark_used, write_json
from software.infrastructure.terminal import Console
from software.scene.asset_identity import artifact_hashes, validate_assets


console = Console('场景')


class StageCache:
    def __init__(self, root: Path, engines: dict, *, force: bool = False, audit=None):
        self.root = root
        self.engines = engines
        self.force = force
        self.audit = audit
        self.timings: list[dict] = []

    @contextmanager
    def measure(self, stage: str, label: str) -> Iterator[dict]:
        started = time.monotonic()
        record = dict(stage=stage, label=label, status='computed')
        console.write(label, state='开始')
        try:
            yield record
        except BaseException:
            record['status'] = 'failed'
            raise
        finally:
            record['elapsed_s'] = time.monotonic() - started
            self.timings.append(record)
            console.write(f'{label}；耗时 {record["elapsed_s"]:.1f} 秒',
                          state={'cached': '复用', 'computed': '完成', 'failed': '失败'}[record['status']])

    @contextmanager
    def stage(self, stage: str, label: str, identity: dict, directory: Path,
              *, roots: tuple[str, ...], refresh: bool = False, link: bool = False) -> Iterator[bool]:
        """只发布完整阶段；缓存命中复制到本次工作目录，避免写穿缓存。

        link=True 仅用于之后只读的产物（如解码视频）：同盘硬链接并设为只读，
        省去整份复制；误写会直接报错，跨盘时回退为复制。
        """
        place = _link_read_only if link else _copy
        key = artifact_identity.canonical_hash(identity)
        cache_dir = self.root / stage / key
        with self.measure(stage, label) as timing:
            timing['cache_key'] = key
            assets = None
            if not self.force and not refresh and (cache_dir / 'ready.json').is_file():
                try:
                    manifest = json.loads((cache_dir / 'ready.json').read_text())
                    if manifest['inputs'] != identity:
                        raise ValueError('缓存输入身份不匹配')
                    assets = manifest['assets']
                    validate_assets(cache_dir, assets)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    console.write(f'{label}缓存无效，将重新计算：{error}', state='提醒')
                    if self.audit is not None:
                        self.audit.record(stage=stage, key=key, path=str(cache_dir), status='invalidated',
                                          reason=f'缓存校验失败：{error}')
                    assets = None
            if assets is not None:
                mark_used(cache_dir)
                for name in assets:
                    place(cache_dir / name, directory / name)
                yield True
                timing['status'] = 'cached'
                if self.audit is not None:
                    self.audit.record(stage=stage, key=key, path=str(cache_dir), status='reused',
                                      reason='ready.json 输入身份与资产哈希校验通过')
                return

            try:
                yield False
                assets = artifact_hashes(directory, roots)
                if not assets:
                    raise ValueError(f'{label}未生成可缓存的产物')
                staging = cache_dir.with_name(f'.{key}-{uuid.uuid4().hex}')
                try:
                    for name in assets:
                        place(directory / name, staging / name)
                    write_json(staging / 'ready.json', dict(inputs=identity, assets=assets))
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir)
                    staging.rename(cache_dir)
                    if self.audit is not None:
                        self.audit.record(stage=stage, key=key, path=str(cache_dir), status='computed',
                                          reason='未命中可用 ready.json，完成计算并发布缓存')
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
            except BaseException as error:
                if self.audit is not None:
                    self.audit.record(stage=stage, key=key, path=str(cache_dir), status='failed',
                                      reason=f'缓存计算失败：{error}')
                raise


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _link_read_only(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    mode = target.stat().st_mode
    target.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
