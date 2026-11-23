"""场景批次准备、建图与任务运行。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import os

from software.infrastructure import provenance as artifact_identity
from software.infrastructure.file_store import read_yaml, write_yaml
from software.scene import layout, workflow
from software.scene.errors import ITEM_FAILURES
from software.scene.layout import batch_root, configured_batches, require_collection
from software.scene.management import batch_snapshot as batches, camera_assets, cleanup, episodes, ingest
from software.scene.management.inputs.batch import batch_from_record, batch_task_directories, read_batch
from software.scene.management.inputs.scene import resolve_path
from software.scene.models import ARMS, Session
from software.scene.parameters import SceneOverrides
from software.scene.pipeline import recordings
from software.scene.pipeline.maps import build as map_build
from software.scene.pipeline.maps import epoch_selection as map_epoch_selection, repository as map_repository

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
    return _require_ready_recordings(prepare_batch(batch, dry_run=dry_run,
        intrinsics_source=intrinsics_source, overrides=overrides))


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


def run(paths: list[Path], *, jobs: int = 1, force: bool = False, cores: str = '', dry_run: bool = False,
        check_recordings: bool = True, map_version: str | None = None,
        map_epoch: str | None = None,
        intrinsics_source: str | None = None, overrides: SceneOverrides | None = None,
        _prepared: dict[Path, dict] | None = None) -> list[dict]:
    if jobs < 1:
        raise ValueError('jobs 必须至少为 1')
    batch_roots = configured_batches(paths)
    summaries, ready = [], []
    for batch in batch_roots:
        try:
            if _prepared is not None and batch in _prepared:
                imported = _require_ready_recordings(_prepared[batch])
            else:
                imported = _prepare_for_run(batch, dry_run=dry_run,
                                            intrinsics_source=intrinsics_source, overrides=overrides)
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
            if isinstance(error, _BatchInputError):
                workflow.console.write(f'原片存在 {len(error.problems)} 项问题，详见汇总',
                                       recording=batch.name, state='失败')
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


def prepare_batch(batch: Path, *, dry_run: bool = False, intrinsics_source: str | None = None, overrides: SceneOverrides | None = None) -> dict:
    _, report = _prepare_batch(batch, dry_run=dry_run, intrinsics_source=intrinsics_source,
                               overrides=overrides)
    return report


def _prepare_batch(batch: Path, *, dry_run: bool, overrides: SceneOverrides | None,
                   intrinsics_source: str | None = None) -> tuple[Session, dict]:
    """预览与实际运行共用原片规划；预览只构造内存中的批次快照。"""
    collection_source = batches.read_collection(batch)
    _, _, collection = collection_source
    mapping_clips = recordings.insv_files(batch / 'mapping')
    map_required, uses_scene_map = batches.mapping_requirements(collection, has_mapping=bool(mapping_clips))
    if not dry_run and uses_scene_map:
        from software.scene.management.import_manifest import ensure_epoch_plan
        ensure_epoch_plan(batch)
    epoch_mode = layout.metadata(batch, 'map_epochs.yaml').is_file()
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
    )
    if dry_run:
        snapshot = batch_from_record(record_path, batches.make_record(batch_inputs), overrides=overrides,
                                     collection=collection)
    else:
        snapshot = read_batch(batches.save_record(batch_inputs), overrides=overrides, collection=collection)
    return snapshot, ingest.register_recordings(snapshot, dry_run=dry_run)


def build(batch: Path, *, force: bool = False, keep_work: bool = False,
          cores: str = '', dry_run: bool = False, intrinsics_source: str | None = None,
          overrides: SceneOverrides | None = None, epochs: bool = False) -> dict:
    """Build only from a batch; run and build share configuration resolution."""
    batch = batch_root(batch)
    if (batch / 'batch.yaml').is_file():
        layout.reject_legacy_user_file(batch / 'batch.yaml')
    if epochs:
        require_collection(batch, 'build --epochs')
        batch_inputs = batches.read_inputs(batch, require_mapping=False, include_mapping=False,
                                           intrinsics_source=intrinsics_source, overrides=overrides)
        if dry_run:
            session = batch_from_record(layout.metadata(batch, 'manifest.yaml'),
                                        batches.make_record(batch_inputs), overrides=overrides)
        else:
            batches.save_record(batch_inputs)
            session = None
        return build_epochs(batch, force=force, keep_work=keep_work, cores=cores,
                            dry_run=dry_run, intrinsics_source=intrinsics_source,
                            overrides=overrides, session=session)
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
                 intrinsics_source: str | None = None,
                 overrides: SceneOverrides | None = None,
                 session=None) -> dict:
    """Build every unbuilt map in an epoch plan and atomically bind versions."""
    batch = batch_root(batch)
    require_collection(batch, 'build --epochs')
    plan_path = layout.metadata(batch, 'map_epochs.yaml')
    if not plan_path.is_file():
        raise ValueError(f'缺少多地图清单：{plan_path}')

    with nullcontext() if dry_run else layout.locked(batch, blocking=True):
        before = artifact_identity.file_hash(plan_path)
        # Epoch 清单自己声明建图原片；旧批次快照中的 mapping/ 可已被整理移动，
        # 不应阻塞按清单构建。每个 epoch 仍会在 _build_scene 内校验自身录像哈希。
        session = session or read_batch(layout.manifest(batch, 'manifest.yaml'), overrides=overrides)
        raw = read_yaml(plan_path)
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
            if dry_run:
                results.append(dict(epoch=epoch.epoch, status='planned', recording=epoch.recording))
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


def plan_epoch_draft(batch: Path, *, write: bool = False, force: bool = False,
                     intrinsics_source: str | None = None,
                     overrides: SceneOverrides | None = None) -> dict:
    """Infer a multi-map plan from embedded recording timestamps.

    Planning is intentionally separate from ``build --epochs``.  A draft is
    always safe to inspect; it is written only with ``write=True`` and only
    when every event has an unambiguous timestamp and camera pairing.
    """
    batch = batch_root(batch)
    require_collection(batch, 'plan-epochs')
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
    if write:
        if result['status'] != 'ready':
            raise ValueError('自动规划结果需要人工复核，未写入 map_epochs.yaml：' + '；'.join(result['reasons']))
        if plan_path.exists() and not force:
            raise ValueError(f'已有地图 epoch 清单：{plan_path}；需要 --force 才能覆盖')
        with layout.locked(batch, blocking=True):
            if plan_path.exists() and not force:
                raise ValueError(f'已有地图 epoch 清单：{plan_path}；需要 --force 才能覆盖')
            write_yaml(plan_path, result['plan'])
        result['written'] = True
    else:
        result['written'] = False
    return result


def run_epoch_pipeline(batch: Path, *, jobs: int = 1, force: bool = False,
                       keep_work: bool = False, cores: str = '', dry_run: bool = False,
                       check_recordings: bool = True,
                       overrides: SceneOverrides | None = None) -> dict:
    """Plan, build and process a multi-map batch in one deterministic command.

    An existing epoch plan is treated as the reviewed source of truth.  When
    it is absent, the timestamp-based draft is written only if it is
    unambiguous; ambiguous ordering still stops before any map or task result
    is created.
    """
    batch = batch_root(batch)
    # Materialize the batch snapshot before epoch planning/building.  This is
    # also the handoff point for Organizer's import_manifest.json.
    snapshot, imported = _prepare_batch(batch, dry_run=dry_run, overrides=overrides)
    plan_path = layout.metadata(batch, 'map_epochs.yaml')
    planning = None
    if not plan_path.is_file():
        planning = plan_epoch_draft(batch, write=not dry_run, overrides=overrides)
        if planning.get('status') != 'ready':
            return dict(status='needs_review', planning=planning, build=None, run=[])
        if dry_run:
            return dict(status='planned', planning=planning, build=None, run=[])
    build_result = build_epochs(batch, force=force, keep_work=keep_work, cores=cores,
                                dry_run=dry_run, overrides=overrides, session=snapshot if dry_run else None)
    run_result = run([batch], jobs=jobs, force=force, cores=cores, dry_run=dry_run,
                     check_recordings=check_recordings, overrides=overrides, _prepared={batch: imported})
    failed = any(item.get('status') not in ('ok', 'skipped', 'local_ok', 'local_skipped', 'planned')
                 for item in run_result)
    return dict(status='failed' if failed else 'ok', planning=planning,
                build=build_result, run=run_result)
