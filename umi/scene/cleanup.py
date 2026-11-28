"""清理过期运行、阶段缓存与地图。"""

from __future__ import annotations

import json
import shutil
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from umi.infra.file_store import locked as storage_lock
from umi.infra.file_store import read_yaml
from umi.infra.terminal import Console
from umi.store import layout
from umi.store.result_io import result_paths

console = Console('清理')
# 目录时间精度与时钟抖动余量；只会多保留刚好在边界上的条目。
TIME_SLACK_S = 2.0


@dataclass
class Removal:
    paths: list[Path] = field(default_factory=list)
    freed_bytes: int = 0
    skipped: list[str] = field(default_factory=list)

    def remove(self, path: Path) -> None:
        self.paths.append(path)
        self.freed_bytes += _exclusive_bytes(path)
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
        else:
            shutil.rmtree(path)


def _exclusive_bytes(path: Path) -> int:
    """只计入没有其他硬链接的文件；与结果共享的视频删除后不释放空间。"""
    files = [path] if path.is_symlink() or not path.is_dir() else [
        item for item in path.rglob('*') if not item.is_symlink() and item.is_file()]
    total = 0
    for item in files:
        try:
            status = item.lstat()
        except FileNotFoundError:
            continue
        if status.st_nlink <= 1:
            total += status.st_size
    return total


def kept_runs(task: Path) -> set[Path]:
    keep = {result_paths(link).run for link in (task / 'results', task / 'local_results') if link.is_symlink()}
    state_path = layout.metadata(task, 'state.json')
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    # 已有人工记录引用的运行仍参与保留规则。
    for record in (state.get('attempt', {}), state.get('review', {}), *state.get('reviews', {}).values()):
        if record.get('run'):
            keep.add((task / record['run']).resolve())
    return keep


def prune_runs(task: Path, removal: Removal) -> None:
    """调用方持有任务锁。"""
    runs = layout.metadata(task, 'runs')
    if not runs.is_dir():
        return
    keep = kept_runs(task)
    for entry in sorted(runs.iterdir()):
        if entry.resolve() not in keep:
            audit = layout.run_audit(entry)
            if not audit.is_dir():
                removal.remove(entry)
                continue
            for child in entry.iterdir():
                if child != audit:
                    removal.remove(child)


def prune_cache(cache: Path, removal: Removal, *, used_since: float) -> None:
    """删除 used_since 之后没有命中或写入的条目；调用方持有对应写入锁。"""
    if not cache.is_dir():
        return
    for kind in sorted(cache.iterdir()):
        if not kind.is_dir():
            continue
        for entry in sorted(kind.iterdir()):
            # 以点开头的是中断留下的暂存目录。
            if entry.name.startswith('.') or entry.stat().st_mtime < used_since - TIME_SLACK_S:
                removal.remove(entry)


def prune_task(task: Path, removal: Removal, *, used_since: float | None = None) -> None:
    """任务处理结束后调用（持有任务锁）；只有成功计算时才传 used_since 清理缓存。"""
    prune_runs(task, removal)
    if used_since is not None:
        prune_cache(layout.metadata(task, 'cache'), removal, used_since=used_since)


def kept_map_versions(batch: Path) -> set[str]:
    versions = layout.maps(batch) / 'versions'
    keep = set()
    manifest = layout.metadata(batch, 'manifest.yaml')
    if manifest.is_file():
        version = (read_yaml(manifest).get('map') or {}).get('version')
        if version:
            keep.add(version)
    for task in task_roots(batch):
        for run in kept_runs(task):
            episode = result_paths(run).manifest
            if not episode.is_file():
                if (run / 'episode.json').is_file() and versions.is_dir():
                    # A retained flat publication may still reference any of these maps.
                    return {path.name for path in versions.iterdir()}
                continue
            reference = json.loads(episode.read_text()).get('map') or {}
            if reference.get('path'):
                target = (run / reference['path']).resolve()
                if target.is_relative_to(versions.resolve()):
                    keep.add(target.relative_to(versions.resolve()).parts[0])
    epoch_plan = layout.metadata(batch, 'map_epochs.yaml')
    if epoch_plan.is_file():
        from umi.scene.maps import epochs as map_epoch_selection
        for epoch in map_epoch_selection.read_plan(epoch_plan).epochs:
            if epoch.map_version:
                keep.add(epoch.map_version)
    # 最新一次建图保留其结果或失败现场。版本名以秒级本机时间开头，后缀随机，
    # 同一秒内的多个版本按目录修改时间区分先后。
    published = sorted((path for path in versions.iterdir() if not path.name.startswith('.')),
                       key=lambda path: (path.name.split('-', 1)[0], path.stat().st_mtime_ns)) if versions.is_dir() else []
    if published:
        keep.add(published[-1].name)
    return keep


def prune_maps(batch: Path, removal: Removal) -> None:
    """持有建图锁和全部任务锁时删除未引用的地图版本；有任务在运行则跳过。"""
    versions = layout.maps(batch) / 'versions'
    if not versions.is_dir():
        return
    with ExitStack() as stack:
        try:
            stack.enter_context(storage_lock(layout.maps(batch), '.build.lock'))
            for task in task_roots(batch):
                stack.enter_context(layout.locked(task))
        except ValueError:
            removal.skipped.append(f'{batch.name}：有任务或建图正在运行，暂不清理地图版本')
            return
        keep = kept_map_versions(batch)
        for entry in sorted(versions.iterdir()):
            if entry.name not in keep:
                removal.remove(entry)


def task_roots(batch: Path) -> list[Path]:
    tasks = layout.metadata(batch, 'tasks')
    return [batch / entry.name for entry in sorted(tasks.iterdir()) if entry.is_dir()] if tasks.is_dir() else []


def after_task(task: Path, *, used_since: float | None = None) -> None:
    """处理收尾的自动清理；失败只提醒，不影响已发布的结果。"""
    removal = Removal()
    try:
        prune_task(task, removal, used_since=used_since)
    except OSError as error:
        removal.skipped.append(f'{task.name}：自动清理未完成：{error}')
    report(removal)


def after_batch(batch: Path) -> None:
    removal = Removal()
    try:
        prune_maps(batch, removal)
    except OSError as error:
        removal.skipped.append(f'{batch.name}：地图版本清理未完成：{error}')
    report(removal)


def report(removal: Removal) -> None:
    if removal.paths:
        console.write(f'删除 {len(removal.paths)} 项已取代的运行、地图或缓存，释放约 {removal.freed_bytes / 2**20:.0f} MB')
    for message in removal.skipped:
        console.write(message, state='提醒')

