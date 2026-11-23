"""场景阶段缓存、内容指纹与运行审计。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from umi.infra import provenance as artifact_identity
from umi.store.models import CalibratedCamera, PreparedRecording, Scene
from contextlib import contextmanager
from threading import Lock
from typing import Iterator
import json
import os
import shutil
import stat
import time
import uuid
from umi.infra.file_store import is_current, mark_used, write_json
from umi.infra.terminal import Console
from umi.store.result_io import artifact_hashes, validate_assets


def preparation_inputs(video: Path, camera: CalibratedCamera, scene: Scene) -> dict:
    """建图和任务共用；原片内容已确定设备，标定只决定生成哪些镜头。"""
    return dict(recording_sha256=artifact_identity.file_hash(video),
                lenses=sorted(camera.measurements), video=asdict(scene.trajectory.values.video))


def visual_inputs(recording: PreparedRecording, lens: str) -> dict:
    video = recording.lens_video(lens)
    return dict(video=artifact_identity.file_hash(video.path),
                settings=artifact_identity.file_hash(video.settings),
                frames=recording.context.frame_count, lens=lens,
                orb=asdict(recording.context.parameters.orb))


# 地图只依赖镜头标定和遮手图；TCP、夹爪宽度和标签测距不参与建图，也不冻结进地图版本。
MAP_CAMERA_ASSETS = ('front', 'front_manifest', 'rear', 'mask')


def map_camera_identity(camera: CalibratedCamera) -> dict:
    return dict(serial=camera.serial, intrinsics_source=camera.selected.intrinsics_source,
                calibration_version=camera.selected.calibration_id,
                assets={name: digest for name, digest in camera.asset_hashes.items()
                        if name in MAP_CAMERA_ASSETS})


def map_reuse_key(input_identity: dict) -> str:
    """路径留在输入记录中供追溯，复用比较使用内容和影响计算的配置。"""
    definition = {name: value for name, value in input_identity['definition'].items()
                  if name not in ('mapping', 'devices')}
    definition['devices'] = {
        serial: {name: value for name, value in binding.items() if name != 'gripper'}
        for serial, binding in input_identity['definition']['devices'].items()}
    return artifact_identity.canonical_hash(dict(
        schema_version=1, definition=definition,
        recordings={arm: recording['sha256'] for arm, recording in input_identity['recordings'].items()},
        cameras=input_identity['cameras']))


def matches_map_inputs(previous: dict, current: dict) -> bool:
    return previous.get('reuse_key') == current['reuse_key']


console = Console('场景')


class StageCache:
    def __init__(self, root: Path, *, force: bool = False, audit=None):
        self.root = root
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
            if not self.force and not refresh and is_current(cache_dir / 'ready.json'):
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


class CacheAudit:
    STATUSES = frozenset({'reused', 'computed', 'invalidated', 'failed', 'skipped', 'not_applicable'})

    def __init__(self):
        self._lock = Lock()
        self._events: list[dict] = []

    def record(self, *, stage: str, key: str, path: str, status: str, reason: str) -> None:
        if status not in self.STATUSES:
            raise ValueError(f'未知缓存状态：{status}')
        with self._lock:
            self._events.append(dict(stage=stage, cache_key=key, path=path,
                                     status=status, reason=reason))

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._events)
