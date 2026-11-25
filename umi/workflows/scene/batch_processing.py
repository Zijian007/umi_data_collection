"""场景批次准备、建图与任务运行。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import os

from umi.infra import provenance as artifact_identity
from umi.infra.file_store import recompute_when_forced, read_yaml, write_yaml
from umi.store import layout
from umi.workflows.scene import workflow
from umi.workflows.scene.errors import ITEM_FAILURES
from umi.store.layout import batch_root, configured_batches, require_collection
from umi.workflows.scene import batch_snapshot as batches, cleanup, episodes, ingest
from umi.workflows.scene import camera_assets
from umi.workflows.scene.inputs.batch import batch_from_record, batch_task_directories, episode_from_recordings, read_batch
from umi.workflows.scene.inputs.scene import resolve_path
from umi.store.models import ARMS, Episode
from umi.workflows.scene.parameters import SceneOverrides
from umi.workflows.scene import recordings
from umi.workflows.scene.maps import build as map_build
from umi.workflows.scene.maps import epoch_selection as map_epoch_selection, repository as map_repository

# 批次处理的对外入口；cli 与配置展示只从这里导入。
__all__ = ['build', 'build_epochs', 'clean', 'configured_batches', 'prepare_batch', 'run']


class _BatchInputError(ValueError):
    """已准备批次仍有原片问题；整批阻断，并保留结构化原因。"""

    def __init__(self, problems: list[dict]):
        self.problems = problems
        detail = '; '.join(f'{item["path"]}: {item.get("reason") or item.get("status", "待确认")}'
                           for item in problems)
        super().__init__('存在无法读取、设备不匹配或待补齐的原片：' + detail)


def _prepare_for_run(batch: Path, *, dry_run: bool, intrinsics_source: str | None,
                     overrides: SceneOverrides | None, cores: str = '',
                     previews: dict[Path, Episode | Exception] | None = None,
                     map_epoch: str | None = None) -> dict:
    """准备一个批次：epoch 计划 → 登记原片 → 构建尚未构建的 epoch 地图。

    使用批次采集顺序生成地图计划，手工整理的批次可按录像时间起草。只读预览不写计划也不建图。
    """
    plan = _epoch_plan(batch, dry_run=dry_run, overrides=overrides, intrinsics_source=intrinsics_source)
    imported = _require_ready_recordings(prepare_batch(
        batch, dry_run=dry_run, intrinsics_source=intrinsics_source, overrides=overrides,
        epoch_plan=plan, previews=previews, map_epoch=map_epoch))
    if plan is not None:
        _build_pending_epochs(batch, plan, dry_run=dry_run, cores=cores, overrides=overrides)
    return imported


def _epoch_plan(batch: Path, *, dry_run: bool, overrides: SceneOverrides | None,
                intrinsics_source: str | None = None, build_only: bool = False) -> dict | None:
    """Return the multi-map plan when needed by the tasks or requested by build."""
    from umi.workflows.scene.collection_recordings import merge_epoch_plan, read_recordings, save_epoch_plan

    _, document, collection = batches.read_collection(batch)
    mapping_clips = recordings.insv_files(batch / 'mapping') if (batch / 'mapping').is_dir() else []
    plan_path = layout.metadata(batch, 'map_epochs.yaml')
    organizer = read_recordings(batch, document=document, verify=not dry_run)
    if organizer is not None and not organizer['maps']:
        if mapping_clips or plan_path.is_file():
            raise ValueError('批次采集记录声明没有建图，但批次已有建图原片或地图计划')
        return None
    has_mapping = bool(mapping_clips) or plan_path.is_file() or organizer is not None
    map_required, _ = batches.mapping_requirements(collection, has_mapping=has_mapping)
    if not map_required and not build_only:
        return None
    if organizer is not None:
        if dry_run:
            current = read_yaml(plan_path) if plan_path.is_file() else {}
            return merge_epoch_plan(current, organizer)
        return save_epoch_plan(batch, organizer)
    if plan_path.is_file():
        return read_yaml(plan_path)
    map_root = batch / 'mapping'
    single_pair = (len(mapping_clips) == 2
                   and all(sum(clip.is_relative_to(map_root / arm) for clip in mapping_clips) == 1
                           for arm in ARMS))
    if len(mapping_clips) < 2 or (single_pair and collection['scene']['layout'] != 'shared'):
        return None  # 单地图，包括每手一段录像组成的一对。
    draft = _draft_epoch_plan(batch, write=not dry_run, overrides=overrides,
                             intrinsics_source=intrinsics_source)
    if draft['status'] != 'ready':
        raise ValueError('mapping/ 有多段建图录像，但无法自动确定地图 epoch 顺序：' + '；'.join(draft['reasons']))
    return draft['plan']


def _build_pending_epochs(batch: Path, plan: dict, *, dry_run: bool, cores: str,
                          overrides: SceneOverrides | None) -> None:
    pending = [item['epoch'] for item in plan['maps'] if item.get('map_version') is None]
    if not pending:
        return
    if dry_run:
        workflow.console.write(f'将构建 {len(pending)} 张地图：{"、".join(pending)}', recording=batch.name, state='预览')
        return
    workflow.console.write(f'构建 {len(pending)} 张地图：{"、".join(pending)}', recording=batch.name, state='建图')
    build_epochs(batch, cores=cores, overrides=overrides)


def _require_ready_recordings(imported: dict) -> dict:
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


@recompute_when_forced
def run(paths: list[Path], *, jobs: int = 1, force: bool = False, cores: str = '', dry_run: bool = False,
        check_recordings: bool = True, map_version: str | None = None,
        map_epoch: str | None = None,
        intrinsics_source: str | None = None, overrides: SceneOverrides | None = None) -> list[dict]:
    if jobs < 1:
        raise ValueError('jobs 必须至少为 1')
    batch_roots = configured_batches(paths)
    summaries, ready = [], []
    previews = {} if dry_run and check_recordings else None
    for batch in batch_roots:
        try:
            imported = _prepare_for_run(batch, dry_run=dry_run, cores=cores,
                                        intrinsics_source=intrinsics_source, overrides=overrides,
                                        previews=previews, map_epoch=map_epoch)
            registered = episodes.episode_paths([batch])
            new_count = len(imported['episodes'])
            total = len({str(path) for path in registered} | {item['path'] for item in imported['episodes']})
            workflow.console.write(
                f'原片 {imported.get("total_scanned", 0)} 段 · '
                f'{"拟登记" if dry_run else "新登记"} {new_count} 个 · 共 {total} 个任务',
                recording=batch.name,
                state='预览' if dry_run else '登记',
            )
            if dry_run:
                for item in imported['episodes']:
                    summaries.append(_preview_item(Path(item['path']), previews, item=item))
                # 已登记任务可能全部被入库复用，仍列出处理计划。
                planned = {item['path'] for item in imported['episodes']}
                for path in registered:
                    if str(path) not in planned:
                        summaries.append(_preview_item(path, previews))
            else:
                ready.extend(registered)
            if not imported['episodes'] and not registered:
                raise ValueError(f'没有任务原片，请放入 {batch}/take_001/left/ 或 right/')
        except ITEM_FAILURES as error:
            if isinstance(error, _BatchInputError):
                workflow.console.write(f'原片存在 {len(error.problems)} 项问题，详见汇总',
                                       recording=batch.name, state='失败')
            summaries.append(_failed_item(batch, error))
    # 已登记批次和显式任务路径继续使用相同的处理引擎。
    direct = episodes.episode_paths(paths, exclude=batch_roots)
    # A selected take can have raw recordings but no saved task manifest yet.
    for target in paths:
        root = target.expanduser().resolve()
        if ((root.parent / 'collection.yaml').is_file() and root.parent not in batch_roots
                and root in batch_task_directories(root.parent)):
            manifest = layout.metadata(root, 'episode.yaml')
            if manifest not in direct:
                direct.append(manifest)
    task_roots = {path: layout.owner(path) for path in direct}
    tasks_by_batch: dict[Path, list[Path]] = {}
    for path, task_root in task_roots.items():
        tasks_by_batch.setdefault(task_root.parent, []).append(path)
    handled = set()
    for batch, selected in sorted(tasks_by_batch.items()):
        if not (batch / 'collection.yaml').is_file():
            continue
        try:
            imported = _prepare_for_run(batch, dry_run=dry_run, cores=cores,
                                        intrinsics_source=intrinsics_source, overrides=overrides,
                                        previews=previews, map_epoch=map_epoch)
            if dry_run:
                new = {Path(item['path']): item for item in imported['episodes']}
                for path in selected:
                    if path in new or path.is_file():
                        summaries.append(_preview_item(path, previews, item=new.get(path)))
                    else:
                        summaries.append(_failed_item(path, ValueError('未找到可预览的任务原片'),
                                                      name=task_roots[path].name))
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


def _preview_item(path: Path, previews: dict[Path, Episode | Exception] | None, *, item: dict | None = None) -> dict:
    root = layout.owner(path)
    planned = dict(item) if item is not None else dict(path=str(path), name=root.name)
    planned['status'] = 'planned'
    if item is not None:
        planned['stages'] = ['登记原片', '选择轨迹估计后端', '抽取轨迹']
    if previews is not None:
        try:
            episode = previews.get(path)
            if isinstance(episode, Exception):
                raise episode
            if episode is None:
                raise ValueError('没有可预检的任务原片')
            workflow.preflight_recordings(episode, workflow.console.child(recording=f'{root.parent.name}/{root.name}'),
                                          enabled=True, dry_run=True)
        except ITEM_FAILURES as error:
            return _failed_item(path, error, name=root.name)
    return planned


def prepare_batch(batch: Path, *, dry_run: bool = False, intrinsics_source: str | None = None,
                  overrides: SceneOverrides | None = None, epoch_plan: dict | None = None,
                  previews: dict[Path, Episode | Exception] | None = None,
                  map_epoch: str | None = None) -> dict:
    """Use the same registration plan with an in-memory batch snapshot for previews."""

    collection_source = batches.read_collection(batch)
    _, _, collection = collection_source
    mapping_clips = recordings.insv_files(batch / 'mapping')
    map_required, _ = batches.mapping_requirements(collection, has_mapping=bool(mapping_clips))
    epoch_mode = epoch_plan is not None or layout.metadata(batch, 'map_epochs.yaml').is_file()
    # 普通批次建图后，mapping/ 可能已经整理或清理；内部清单中的冻结地图
    # 指针足以支持后续 take。此时再次强制要求建图原片，会让有效地图的
    # 批量 CLI 运行无故失败。
    record_path = layout.metadata(batch, 'manifest.yaml')
    previous = read_yaml(record_path) if record_path.is_file() else {}
    has_bound_map = bool(previous.get('map') or collection.get('map_version'))
    batch_inputs = batches.read_inputs(
        batch,
        require_mapping=not epoch_mode and not has_bound_map and map_required,
        include_mapping=not epoch_mode and map_required,
        intrinsics_source=intrinsics_source,
        overrides=overrides,
        collection_source=collection_source,
        mapping_clips=mapping_clips,
        epoch_maps=epoch_plan['maps'] if epoch_plan is not None else None,
    )
    if dry_run:
        snapshot = batch_from_record(record_path, batches.make_record(batch_inputs), overrides=overrides,
                                     collection=collection)
    else:
        snapshot = read_batch(batches.save_record(batch_inputs), overrides=overrides, collection=collection)
    preview_recordings = {} if previews is not None else None
    plan = map_epoch_selection.parse_plan(epoch_plan) if dry_run and epoch_plan is not None else None
    report = ingest.register_recordings(snapshot, dry_run=dry_run, preview_recordings=preview_recordings,
                                        collection=collection, epoch_plan=plan)
    if preview_recordings and not (report['errors'] or report['pending'] or report['ignored']):
        for path, references in preview_recordings.items():
            try:
                previews[path] = episode_from_recordings(path, snapshot, references, collection=collection,
                    map_epoch_override=map_epoch, allow_pending_map=True, epoch_plan=plan)
            except ITEM_FAILURES as error:
                previews[path] = error
    return report


@recompute_when_forced
def build(batch: Path, *, force: bool = False, keep_work: bool = False,
          cores: str = '', dry_run: bool = False, intrinsics_source: str | None = None,
          overrides: SceneOverrides | None = None, epochs: bool = False) -> dict:
    """Build only from a batch; run and build share configuration resolution."""
    batch = batch_root(batch)
    if (batch / 'batch.yaml').is_file():
        layout.reject_legacy_user_file(batch / 'batch.yaml')
    epoch_plan = (_epoch_plan(batch, dry_run=dry_run, overrides=overrides, intrinsics_source=intrinsics_source,
                             build_only=True)
                  if (batch / 'collection.yaml').is_file() else None)
    if epochs or epoch_plan is not None:
        require_collection(batch, 'build')
        batch_inputs = batches.read_inputs(batch, require_mapping=False, include_mapping=False,
                                           intrinsics_source=intrinsics_source, overrides=overrides,
                                           epoch_maps=epoch_plan['maps'] if epoch_plan else None)
        if dry_run:
            session = batch_from_record(layout.metadata(batch, 'manifest.yaml'),
                                        batches.make_record(batch_inputs), overrides=overrides)
        else:
            batches.save_record(batch_inputs)
            session = None
        return build_epochs(batch, force=force, keep_work=keep_work, cores=cores,
                            dry_run=dry_run, overrides=overrides, session=session, preview_plan=epoch_plan if dry_run else None)
    if (batch / 'collection.yaml').is_file():
        batch_inputs = batches.read_inputs(batch, intrinsics_source=intrinsics_source, overrides=overrides)
        if dry_run:
            snapshot = batch_from_record(layout.metadata(batch, 'manifest.yaml'),
                                         batches.make_record(batch_inputs), verify_mapping=True,
                                         overrides=overrides)
            return map_build.build_version(snapshot.scene, force=force, keep_work=keep_work, cores=cores, dry_run=True)
        path = batches.save_record(batch_inputs)
    else:
        path = layout.manifest(batch, 'manifest.yaml')
        if not path.is_file():
            raise ValueError('build 需要采集批次目录及 collection.yaml；将建图原片放入该批次的 mapping/')
        if intrinsics_source is not None:
            camera_assets.require_calibration_source(read_batch(path, overrides=overrides).kit, intrinsics_source)
    result = map_repository.build_scene(path, force=force, keep_work=keep_work, cores=cores, dry_run=dry_run,
                                       overrides=overrides)
    if not dry_run:
        cleanup.after_batch(layout.owner(path))
    return result


def build_epochs(batch: Path, *, force: bool = False, keep_work: bool = False,
                 cores: str = '', dry_run: bool = False,
                 overrides: SceneOverrides | None = None,
                 session=None, preview_plan: dict | None = None) -> dict:
    """Build every unbuilt map in an epoch plan and atomically bind versions."""
    batch = batch_root(batch)
    require_collection(batch, 'build')
    plan_path = layout.metadata(batch, 'map_epochs.yaml')
    if not plan_path.is_file() and preview_plan is None:
        raise ValueError(f'缺少多地图清单：{plan_path}')

    with nullcontext() if dry_run else layout.locked(batch, blocking=True):
        before = None if dry_run else artifact_identity.file_hash(plan_path)
        # Epoch 清单自己声明建图原片；旧批次快照中的 mapping/ 可已被整理移动，
        # 不应阻塞按清单构建。每个 epoch 仍会在 _build_scene 内校验自身录像哈希。
        session = session or read_batch(layout.manifest(batch, 'manifest.yaml'), overrides=overrides)
        raw = preview_plan if preview_plan is not None else read_yaml(plan_path)
        plan = map_epoch_selection.parse_plan(raw)
        results = []
        for epoch in plan.epochs:
            if epoch.map_version is not None:
                directory, manifest = map_repository.load_published(session.scene,
                    version=epoch.map_version, allow_import=True)
                map_repository.validate_map_bindings(session.kit, manifest)
                if not dry_run:
                    directory = map_repository.import_map(session.scene, directory)
                results.append(dict(epoch=epoch.epoch, status='skipped', version=manifest['version'],
                                    frame_id=manifest['frame_id'], directory=str(directory)))
                continue
            if session.scene.mode == 'shared_front':
                if not isinstance(epoch.recording, str):
                    raise ValueError(f'地图 epoch {epoch.epoch} 的 shared_front recording 必须是单个路径')
                references = {'left': epoch.recording}
            elif session.scene.mode == 'dual':
                if not isinstance(epoch.recording, dict) or set(epoch.recording) != {'left', 'right'}:
                    raise ValueError(f'地图 epoch {epoch.epoch} 的 dual recording 必须包含 left 和 right')
                references = epoch.recording
            else:
                raise ValueError(f'地图 epoch {epoch.epoch} 的场景模式不支持：{session.scene.mode}')
            recording_paths, serials = {}, {}
            devices = dict(session.scene.devices)
            for arm, reference in references.items():
                recording = resolve_path(batch, reference)
                if not recording.is_relative_to(batch):
                    raise ValueError(f'地图 epoch {epoch.epoch} 的 recording 必须位于批次目录内：{recording}')
                metadata = recordings.inspect(recording)
                serial = metadata['serial']
                if serial not in session.arm_serials.values():
                    raise ValueError(f'地图 epoch {epoch.epoch} 的建图相机不属于当前套件：{serial}')
                recording_paths[arm] = recording
                serials[arm] = serial
                # epoch 原片的相机参数以当前 kit 为准；旧 mapping 快照可能来自上一轮标定。
                devices[serial] = dict(session.devices[serial])
            if len(set(serials.values())) != len(serials):
                raise ValueError(f'地图 epoch {epoch.epoch} 的左右建图录像不能来自同一台相机')
            if dry_run:
                results.append(dict(epoch=epoch.epoch, status='planned', recording=epoch.recording))
                continue
            scene = replace(session.scene, recording_serials=serials, devices=devices,
                            rear_calibrations=camera_assets.rear_files(devices),
                            recordings=recording_paths)
            result = map_build.build_version(scene, force=force, keep_work=keep_work,
                cores=cores, dry_run=False, map_epoch=epoch.epoch)
            directory = scene.map_root / 'versions' / result['version']
            results.append(dict(epoch=epoch.epoch, status=result['status'],
                                version=result['version'], frame_id=result['frame_id'],
                                directory=str(directory)))

        if dry_run:
            return dict(status='planned', plan=str(plan_path), epochs=results)
        after = artifact_identity.file_hash(plan_path)
        if after != before:
            raise ValueError('地图 epoch 清单在建图期间发生变化，请重新运行')
        by_epoch = {item['epoch']: item for item in raw['maps']}
        for result in results:
            if result['status'] in ('ok', 'skipped'):
                by_epoch[result['epoch']]['map_version'] = result['version']
        write_yaml(plan_path, raw)
        cleanup.after_batch(batch)
        return dict(status='ok', plan=str(plan_path), epochs=results)


def _draft_epoch_plan(batch: Path, *, write: bool = False,
                      intrinsics_source: str | None = None,
                      overrides: SceneOverrides | None = None) -> dict:
    """按录像内嵌时间戳为手工整理的多轮建图批次起草地图 epoch 计划。

    只在每个事件的时间和相机配对都无歧义时写入（write=True）。注意：时间戳取自各相机自己的
    时钟，两台相机的时钟可能不对齐；经 Organizer 整理的批次使用已对齐的采集顺序。
    """
    batch = batch_root(batch)
    require_collection(batch, 'run')
    batch_inputs = batches.read_inputs(batch, require_mapping=False, include_mapping=False,
                                      intrinsics_source=intrinsics_source, overrides=overrides)
    mapping_records, mapping_errors = ingest.scan_raw_videos([batch / 'mapping'])
    task_records = []
    scan_errors = list(mapping_errors)
    for task in batch_task_directories(batch):
        recordings_by_arm = {}
        for arm in ARMS:
            source = task / arm
            if not source.is_dir():
                continue
            records, errors = ingest.scan_raw_videos([source])
            scan_errors.extend(errors)
            recordings_by_arm[arm] = records
        task_records.append(dict(task=task.name, recordings=recordings_by_arm))
    result = map_epoch_selection.draft_plan(batch=batch, mode=batch_inputs.scene.mode,
                                   mapping_records=mapping_records,
                                   task_records=task_records,
                                   arm_serials={arm: item['serial'] for arm, item in batch_inputs.kit['arms'].items()})
    if scan_errors:
        result['status'] = 'needs_review'
        result['reasons'] = list(dict.fromkeys(result['reasons'] + [
            f"无法读取录像：{item.get('path', '')}；{item.get('reason', '未知错误')}"
            for item in scan_errors]))
    plan_path = layout.metadata(batch, 'map_epochs.yaml')
    result['batch'] = str(batch)
    result['plan_path'] = str(plan_path)
    result['scan_errors'] = scan_errors
    if write and result['status'] == 'ready':
        with layout.locked(batch, blocking=True):
            if plan_path.exists():
                raise ValueError(f'已有地图 epoch 清单：{plan_path}')
            write_yaml(plan_path, result['plan'])
        result['written'] = True
    else:
        result['written'] = False
    return result


