"""场景批次处理与任务级编排。"""

from __future__ import annotations

from pathlib import Path
import os


from . import camera_assets, cleanup, episodes
from software.scene import layout
from software.scene.overrides import SceneOverrides
from software.scene import workflow
from software.scene.management.inputs.batch import read_batch
from software.scene.errors import ITEM_FAILURES
from software.scene.management.batch.discovery import configured_batches
from software.scene.management.batch.execution import build, prepare_batch
from software.scene.management.batch.epochs import build_epochs, plan_epoch_draft, run_epoch_pipeline

# 批次处理的对外入口；cli 与配置展示只从这里导入。
__all__ = ['build', 'build_epochs', 'clean', 'configured_batches', 'plan_epoch_draft',
           'prepare_batch', 'run', 'run_epoch_pipeline']


class _BatchInputError(ValueError):
    """已准备批次仍有原片问题；整批阻断，并保留结构化原因。"""

    def __init__(self, problems: list[dict]):
        self.problems = problems
        detail = '; '.join(f'{item["path"]}: {item.get("reason") or item.get("status", "待确认")}'
                           for item in problems)
        super().__init__('存在无法读取、设备不匹配或待补齐的原片：' + detail)


def _prepare_for_run(batch: Path, *, dry_run: bool, intrinsics_source: str | None,
                     overrides: SceneOverrides | None) -> dict:
    imported = prepare_batch(batch, dry_run=dry_run,
                             intrinsics_source=intrinsics_source, overrides=overrides)
    problems = imported['errors'] + imported['pending'] + imported['ignored']
    if problems:
        raise _BatchInputError(problems)
    return imported


def _failed_item(path: Path, error: Exception, *, name: str | None = None) -> dict:
    result = dict(path=str(path), name=path.name if name is None else name, status='failed', error=str(error))
    if isinstance(error, _BatchInputError):
        result['details'] = error.problems
    return result


def clean(paths: list[Path], *, dry_run: bool = False, cache: bool = False) -> dict:
    """对已有批次执行与自动清理相同的保留规则。"""
    batch_roots = configured_batches(paths)
    if not batch_roots:
        raise ValueError(f'没有找到批次 collection.yaml：{", ".join(map(str, paths))}')
    removal = cleanup.Removal(dry_run=dry_run)
    for batch in batch_roots:
        cleanup.clean_batch(batch, removal, cache=cache)
    common = Path(os.path.commonpath(batch_roots)) if len(batch_roots) > 1 else batch_roots[0]
    return removal.summary(common)


def run(paths: list[Path], *, jobs: int = 1, force: bool = False, cores: str = '', dry_run: bool = False,
        check_recordings: bool = True, map_version: str | None = None,
        map_epoch: str | None = None,
        intrinsics_source: str | None = None, overrides: SceneOverrides | None = None) -> list[dict]:
    if jobs < 1:
        raise ValueError('jobs 必须至少为 1')
    batch_roots = configured_batches(paths)
    summaries, ready = [], []
    for batch in batch_roots:
        try:
            imported = _prepare_for_run(batch, dry_run=dry_run,
                                        intrinsics_source=intrinsics_source, overrides=overrides)
            registered = episodes.episode_paths([batch])
            if dry_run:
                for item in imported['episodes']:
                    summaries.append(dict(item, status='planned', stages=['登记原片', '选择轨迹估计后端', '抽取轨迹']))
                # 已登记任务可能全部被入库复用，仍列出处理计划。
                planned = {item['path'] for item in imported['episodes']}
                for path in registered:
                    if str(path) not in planned:
                        summaries.append(dict(path=str(path), name=layout.owner(path).name, status='planned'))
            else:
                ready.extend(registered)
            if not imported['episodes'] and not registered:
                raise ValueError(f'没有任务原片，请放入 {batch}/take_001/left/ 或 right/')
        except ITEM_FAILURES as error:
            summaries.append(_failed_item(batch, error))
    # 已登记批次和显式任务路径继续使用相同的处理引擎。
    direct = episodes.episode_paths(paths, exclude=batch_roots)
    task_roots = {path: layout.owner(path) for path in direct}
    tasks_by_batch: dict[Path, list[Path]] = {}
    for path, task_root in task_roots.items():
        tasks_by_batch.setdefault(task_root.parent, []).append(path)
    handled = set()
    for batch, selected in sorted(tasks_by_batch.items()):
        if not (batch / 'collection.yaml').is_file():
            continue
        try:
            _prepare_for_run(batch, dry_run=dry_run,
                             intrinsics_source=intrinsics_source, overrides=overrides)
            if dry_run:
                summaries.extend(dict(path=str(path), name=task_roots[path].name, status='planned') for path in selected)
                handled.update(selected)
        except ITEM_FAILURES as error:
            summaries.extend(_failed_item(path, error, name=task_roots[path].name) for path in selected)
            handled.update(selected)
    direct = [path for path in direct if path not in handled]
    if intrinsics_source:
        checked = []
        source_errors: dict[Path, str | None] = {}
        for path in direct:
            batch = task_roots[path].parent
            if batch not in source_errors:
                try:
                    session = read_batch(batch, overrides=overrides)
                    camera_assets.require_calibration_source(session.kit, intrinsics_source)
                    source_errors[batch] = None
                except ITEM_FAILURES as error:
                    source_errors[batch] = str(error)
            if source_errors[batch] is None:
                checked.append(path)
            else:
                summaries.append(dict(path=str(path), name=task_roots[path].name, status='failed',
                    error=f'{source_errors[batch]}；来源切换需要 collection.yaml 和套件中对应的完整标定配置'))
        direct = checked
    ready.extend(direct)
    ready = list(dict.fromkeys(ready))
    if ready:
        summaries.extend(workflow.run_episodes(ready, jobs=jobs, force=force, cores=cores, dry_run=dry_run,
            check_recordings=check_recordings, map_version=map_version,
            map_epoch=map_epoch,
            overrides=overrides))
        if not dry_run:
            # 任务运行已各自清理；全部结束后再删除不再被引用的地图版本。
            for batch in sorted({layout.owner(path).parent for path in ready}):
                cleanup.after_batch(batch)
    if not summaries:
        raise ValueError('没有找到批次配置或任务清单；请准备 collection.yaml 和 take_001/left/、right/ 原片')
    return summaries
