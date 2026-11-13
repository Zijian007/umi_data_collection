"""批次 .pipeline/ 的保留规则：只留当前在用和最近一次的产物，其余已被取代或可重建。

- 任务 runs/：results、local_results 指向的运行，最近一次尝试（可能是失败现场）和验收记录的运行。
- 地图 versions/：批次 manifest.yaml 绑定、保留运行引用的版本，以及最新一次建图（可能是失败现场）。
- 缓存条目：命中时刷新目录时间（storage.mark_used）；成功处理后删除本次未用到的条目。
自动清理在处理成功后由 workflow / map_lifecycle 调用；umi-scene clean 对已有数据执行相同规则。
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
import json
import shutil

from software.console import Console
from software.scene import layout
from software.storage import locked as storage_lock, read_yaml


console = Console('清理')
# 目录时间精度与时钟抖动余量；只会多保留刚好在边界上的条目。
TIME_SLACK_S = 2.0


@dataclass
class Removal:
    dry_run: bool = False
    paths: list[Path] = field(default_factory=list)
    freed_bytes: int = 0
    skipped: list[str] = field(default_factory=list)

    def remove(self, path: Path) -> None:
        self.paths.append(path)
        self.freed_bytes += _exclusive_bytes(path)
        if self.dry_run:
            return
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
        else:
            shutil.rmtree(path)

    def summary(self, root: Path | None = None) -> dict:
        def show(path: Path) -> str:
            return str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path)
        return dict(dry_run=self.dry_run, removed=[show(path) for path in self.paths],
                    freed_mb=round(self.freed_bytes / 2**20, 1), skipped=self.skipped)


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
    keep = {link.resolve() for link in (task / 'results', task / 'local_results') if link.is_symlink()}
    state_path = layout.metadata(task, 'state.json')
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    for record in (state.get('attempt', {}), state.get('review', {})):
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
            removal.remove(entry)


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
            episode = run / 'episode.json'
            if not episode.is_file():
                continue
            reference = json.loads(episode.read_text()).get('map') or {}
            if reference.get('path'):
                target = (run / reference['path']).resolve()
                if target.is_relative_to(versions.resolve()):
                    keep.add(target.relative_to(versions.resolve()).parts[0])
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
        legacy_pointer = layout.maps(batch) / 'current.json'  # 当前地图已只记录在 manifest.yaml
        if legacy_pointer.is_file():
            removal.remove(legacy_pointer)


def task_roots(batch: Path) -> list[Path]:
    tasks = layout.metadata(batch, 'tasks')
    return [batch / entry.name for entry in sorted(tasks.iterdir()) if entry.is_dir()] if tasks.is_dir() else []


def clean_batch(batch: Path, removal: Removal, *, cache: bool = False) -> None:
    """umi-scene clean：对已有批次执行保留规则；cache=True 时清空全部可重建缓存。"""
    for task in task_roots(batch):
        try:
            with layout.locked(task):
                prune_runs(task, removal)
                for name in ('work', *(('cache',) if cache else ())):
                    if layout.metadata(task, name).exists():
                        removal.remove(layout.metadata(task, name))
        except ValueError:
            removal.skipped.append(f'{batch.name}/{task.name}：任务正在运行，已跳过')
    prune_maps(batch, removal)
    if cache and (layout.maps(batch) / 'cache').exists():
        try:
            with storage_lock(layout.maps(batch), '.build.lock'):
                removal.remove(layout.maps(batch) / 'cache')
        except ValueError:
            removal.skipped.append(f'{batch.name}：建图正在运行，未清理建图缓存')


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
