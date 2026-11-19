"""多地图 epoch 规划、建图和批量执行。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from software.infrastructure import provenance as artifact_identity
from software.scene.management.batch import snapshot as batches
from software.scene.management import camera_assets, cleanup, ingest
from software.scene import layout
from software.scene.overrides import SceneOverrides
from software.scene.pipeline.maps import lifecycle as map_lifecycle
from software.scene.pipeline.maps import epoch_planning as map_epoch_planning, epoch_selection as map_epoch_selection
from software.scene.pipeline import recordings
from software.infrastructure.file_store import read_yaml, write_yaml
from software.scene.management.inputs.scene import resolve_path
from software.scene.management.inputs.batch import batch_task_directories, read_batch
from software.scene.models import ARMS
from software.scene.management.batch.discovery import batch_root, require_collection


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

    with layout.locked(batch, blocking=True):
        before = artifact_identity.file_hash(plan_path)
        # Epoch 清单自己声明建图原片；旧批次快照中的 mapping/ 可已被整理移动，
        # 不应阻塞按清单构建。每个 epoch 仍会在 _build_scene 内校验自身录像哈希。
        session = session or read_batch(layout.manifest(batch, 'manifest.yaml'), overrides=overrides)
        plan = map_epoch_selection.read_plan(plan_path)
        raw = read_yaml(plan_path)
        results = []
        for epoch in plan.epochs:
            if epoch.map_version is not None:
                directory, manifest = map_lifecycle.load_published(session.scene,
                    version=epoch.map_version, allow_import=True)
                map_lifecycle.validate_map_bindings(session.kit, manifest)
                if not dry_run:
                    directory = map_lifecycle.import_map(session.scene, directory)
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
            result = map_lifecycle._build_scene(scene, force=force, keep_work=keep_work,
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
    result = map_epoch_planning.draft_plan(batch=batch, mode=batch_inputs.scene.mode,
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
    from software.scene.management.batch.execution import prepare_batch
    prepare_batch(batch, dry_run=dry_run, overrides=overrides)
    plan_path = layout.metadata(batch, 'map_epochs.yaml')
    planning = None
    if not plan_path.is_file():
        planning = plan_epoch_draft(batch, write=not dry_run, overrides=overrides)
        if planning.get('status') != 'ready':
            return dict(status='needs_review', planning=planning, build=None, run=[])
        if dry_run:
            return dict(status='planned', planning=planning, build=None, run=[])
    build_result = build_epochs(batch, force=force, keep_work=keep_work, cores=cores,
                                dry_run=dry_run, overrides=overrides)
    from software.scene.management.processing import run
    run_result = run([batch], jobs=jobs, force=force, cores=cores, dry_run=dry_run,
                     check_recordings=check_recordings, overrides=overrides)
    failed = any(item.get('status') not in ('ok', 'skipped', 'local_ok', 'local_skipped', 'planned')
                 for item in run_result)
    return dict(status='failed' if failed else 'ok', planning=planning,
                build=build_result, run=run_result)
