"""从批次声明和原片启动处理；设备与地图快照沿用现有登记和处理实现。"""

from __future__ import annotations

from pathlib import Path
import os

from ruamel.yaml.error import YAMLError

from . import batches, camera_assets, episodes, ingest, inputs
from software.scene import layout
from software.scene.configuration import SceneOverrides
from software.scene import workflow
from software.scene.pipeline import map_lifecycle


def configured_batches(paths: list[Path]) -> list[Path]:
    """发现真实 batch.yaml；示例、缓存和运行结果不参与发现。"""
    found = set()
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_file():
            if path.name not in {'batch.yaml', 'manifest.yaml', 'scene.yaml'}:
                continue
            path = layout.owner(path)
        if not path.is_dir():
            raise ValueError(f'处理路径不存在：{path}')
        for current, children, files in os.walk(path):
            children[:] = sorted(name for name in children if not name.startswith('.')
                                 and name not in inputs.INTERNAL_DIRECTORIES)
            if 'batch.yaml' in files:
                found.add(Path(current))
                children[:] = []
    return sorted(found)


def prepare_batch(batch: Path, *, dry_run: bool = False, intrinsics_source: str | None = None, overrides: SceneOverrides | None = None) -> dict:
    """预览与实际运行共用原片规划；预览只构造内存中的批次快照。"""
    if not dry_run:
        from .migration import migrate_scene
        migrate_scene(batch.parent)
    batch_inputs = batches.read_inputs(batch, intrinsics_source=intrinsics_source, overrides=overrides)
    if dry_run:
        snapshot = inputs.batch_from_record(layout.metadata(batch, 'manifest.yaml'), batches.make_record(batch_inputs), overrides=overrides)
    else:
        snapshot = inputs.read_batch(batches.save_record(batch_inputs), overrides=overrides)
    return ingest.register_recordings(snapshot, dry_run=dry_run)


def build(batch: Path, *, force: bool = False, keep_work: bool = False,
          cores: str = '', dry_run: bool = False, intrinsics_source: str | None = None, overrides: SceneOverrides | None = None) -> dict:
    """Build only from a batch; run and build share configuration resolution."""
    batch = batch.expanduser().resolve()
    if batch.is_file():
        batch = layout.owner(batch)
    if batch.name == layout.INTERNAL:
        batch = batch.parent
    if not dry_run:
        from .migration import migrate_scene
        migrate_scene(batch.parent)
    if (batch / 'batch.yaml').is_file():
        batch_inputs = batches.read_inputs(batch, intrinsics_source=intrinsics_source, overrides=overrides)
        if dry_run:
            snapshot = inputs.batch_from_record(layout.metadata(batch, 'manifest.yaml'),
                                                batches.make_record(batch_inputs), verify_mapping=True, overrides=overrides)
            return map_lifecycle._build_scene(snapshot.scene, force=force, keep_work=keep_work, cores=cores, dry_run=True)
        path = batches.save_record(batch_inputs)
    else:
        path = layout.manifest(batch, 'manifest.yaml')
        if not path.is_file():
            raise ValueError('build 需要采集批次目录及 batch.yaml；将建图原片放入该批次的 mapping/')
        if intrinsics_source is not None:
            camera_assets.require_calibration_source(inputs.read_batch(path, overrides=overrides).kit, intrinsics_source)
    return map_lifecycle.build_scene(path, force=force, keep_work=keep_work, cores=cores, dry_run=dry_run,
                                     overrides=overrides)


def run(paths: list[Path], *, jobs: int = 1, force: bool = False, cores: str = '', dry_run: bool = False,
        check_recordings: bool = True, map_version: str | None = None,
        intrinsics_source: str | None = None, overrides: SceneOverrides | None = None) -> list[dict]:
    if jobs < 1:
        raise ValueError('jobs 必须至少为 1')
    batch_roots = configured_batches(paths)
    summaries, ready = [], []
    for batch in batch_roots:
        try:
            imported = prepare_batch(batch, dry_run=dry_run,
                                     intrinsics_source=intrinsics_source, overrides=overrides)
            problems = imported['errors'] + imported['pending'] + imported['ignored']
            if problems:
                detail = '; '.join(f'{item["path"]}: {item.get("reason") or item.get("status", "待确认")}'
                                   for item in problems)
                summaries.append(dict(path=str(batch), name=batch.name, status='failed',
                    error='存在无法读取、设备不匹配或待补齐的原片：' + detail, details=problems))
            registered = episodes.episode_paths([batch])
            if dry_run:
                for item in imported['episodes']:
                    summaries.append(dict(item, status='planned', stages=['登记原片', '固定或构建批次地图', '抽取轨迹']))
                # 已登记任务可能全部被入库复用，仍列出处理计划。
                planned = {item['path'] for item in imported['episodes']}
                for path in registered:
                    if str(path) not in planned:
                        summaries.append(dict(path=str(path), name=layout.owner(path).name, status='planned'))
            else:
                ready.extend(registered)
            if not imported['episodes'] and not registered and not problems:
                raise ValueError(f'没有任务原片，请放入 {batch}/take_001/left/ 或 right/')
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, YAMLError) as error:
            summaries.append(dict(path=str(batch), name=batch.name, status='failed', error=str(error)))
    # 已登记的旧批次和显式任务路径继续使用相同的处理引擎。
    direct = episodes.episode_paths(paths, exclude=batch_roots)
    if not dry_run:
        from .migration import migrate_scene
        legacy_scenes = {layout.owner(path).parent.parent for path in direct}
        for scene in sorted(legacy_scenes):
            migrate_scene(scene)
        direct = [layout.manifest(path, 'episode.yaml') for path in direct]
    for batch in sorted({layout.owner(path).parent for path in direct
                         if (layout.owner(path).parent / 'batch.yaml').is_file()}):
        selected = [path for path in direct if layout.owner(path).parent == batch]
        try:
            imported = prepare_batch(batch, dry_run=dry_run,
                                     intrinsics_source=intrinsics_source, overrides=overrides)
            if imported['errors'] or imported['pending'] or imported['ignored']:
                raise ValueError('当前批次存在无法读取、设备不匹配或待补齐的原片，请先核对')
            if dry_run:
                summaries.extend(dict(path=str(path), name=layout.owner(path).name, status='planned') for path in selected)
                direct = [path for path in direct if path not in selected]
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, YAMLError) as error:
            summaries.extend(dict(path=str(path), name=layout.owner(path).name, status='failed', error=str(error)) for path in selected)
            direct = [path for path in direct if path not in selected]
    if intrinsics_source:
        checked = []
        source_errors: dict[Path, str | None] = {}
        for path in direct:
            batch = layout.owner(path).parent
            if batch not in source_errors:
                try:
                    session = inputs.read_batch(batch, overrides=overrides)
                    camera_assets.require_calibration_source(session.kit, intrinsics_source)
                    source_errors[batch] = None
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    source_errors[batch] = str(error)
            if source_errors[batch] is None:
                checked.append(path)
            else:
                summaries.append(dict(path=str(path), name=layout.owner(path).name, status='failed',
                    error=f'{source_errors[batch]}；来源切换需要 batch.yaml 和套件中对应的完整标定配置'))
        direct = checked
    ready.extend(direct)
    ready = list(dict.fromkeys(ready))
    if ready:
        summaries.extend(workflow.run_episodes(ready, jobs=jobs, force=force, cores=cores, dry_run=dry_run,
            check_recordings=check_recordings, map_version=map_version,
            overrides=overrides))
    if not summaries:
        raise ValueError('没有找到批次配置或任务清单；请准备 batch.yaml 和 take_001/left/、right/ 原片')
    return summaries
