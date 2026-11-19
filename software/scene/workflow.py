"""任务编排：预检、固定地图绑定、每手计算、结果发布与失败收尾。

所有输入先预检；新版本全部成功后才发布指针。失败不替换已有成功地图或任务结果。
"""

from __future__ import annotations

from contextlib import nullcontext, suppress
from dataclasses import asdict, replace
from pathlib import Path
import concurrent.futures as cf
import json
import os
import shutil
import time
import traceback
import uuid


from software.infrastructure import provenance as artifact_identity
from software.infrastructure.terminal import Console
from software.scene import layout, cpu_budget
from software.scene.overrides import SceneOverrides
from software.scene.layout import locked
from software.scene.management import camera_assets
from software.scene.management import cleanup, episodes
from software.scene.asset_identity import validate_assets
from software.scene.management.inputs.episode import episode_inputs, read_episode
from software.infrastructure.file_store import read_yaml, write_json
from software.scene.models import Episode, EpisodeRunResult, TaskInputs
from software.scene.result_io import is_local_scene, is_poster_scene, scene_reference
from software.scene.pipeline.maps import lifecycle as map_lifecycle
from software.scene.pipeline.tasks import recording_preparation
from software.scene.pipeline.tasks.results import reporting
from software.scene.pipeline.tasks.results import publication as episode_publication, trajectory_writers
from software.scene.pipeline import recording_checks
from software.scene.pipeline.cache_audit import CacheAudit
from software.scene.pipeline import synchronization
from software.scene.pipeline.tasks import estimation
from software.scene.run_audit import RunAudit
from software.vision import runtime
from software.scene.errors import ITEM_FAILURES, MapCompatibilityError, fallback_record


console = Console('场景')


def record_failure(episode: Episode, directory: Path, reason: str, *, identity: dict,
                   status='failed', error: Exception | None = None) -> Path | None:
    """同一输入和原因只保留一份失败现场；成功运行不参与去重。"""
    key = artifact_identity.canonical_hash(dict(inputs=identity, reason=reason, status=status))[:24]
    target = directory.parent / ('failed-' + key)
    write_json(directory / 'error.json', dict(status=status, reason=reason,
        traceback=''.join(traceback.format_exception(error)) if error else '', input_hash=artifact_identity.canonical_hash(identity),
        inputs=identity, batch=read_yaml(episode.session.path)))
    retained = not target.exists()
    if not retained:
        shutil.rmtree(directory)
    else:
        directory.rename(target)
    episodes.record_attempt(episode, status, target)
    cleanup.after_task(episode.root)
    return target if retained else None


def _resolve_episode_synchronization(episode, *, force: bool, dry_run: bool):
    """Resolve audio-auto offsets and return the cache audit status."""
    if episode.synchronization.get('method') != 'audio_auto':
        if len(episode.expected_arms) > 1:
            console.write(
                f'双手同步：{episode.synchronization["method"]}；'
                f'offsets_s={synchronization.format_offsets(episode.synchronization["offsets_s"])}',
                recording=episode.root.name,
            )
        return episode, 'not_applicable', '任务未启用 audio_auto'

    sync_path = layout.metadata(episode.root, 'sync.json')
    console.write('双手音频同步：检查 sync.json', recording=episode.root.name)
    try:
        ready = synchronization.load_ready_audio_sync(sync_path, episode.recordings)
    except ValueError:
        if not force:
            raise
        ready = None
    if ready is not None:
        console.write(
            f'双手音频同步：复用；offsets_s={synchronization.format_offsets(ready["offsets_s"])}',
            recording=episode.root.name,
            state='复用',
        )
        return (replace(episode, synchronization=ready), 'reused',
                'sync.json 指纹匹配且 specialist session-clock 偏移已 ready')

    console.write('双手音频同步：开始估计', recording=episode.root.name)
    synced = synchronization.estimate_audio_sync(
        episode.recordings, output=None if dry_run else sync_path, force=force)
    resolved = dict(
        method='audio_auto', mode='audio_auto',
        offsets_s=synced['offsets_s'], confirmed=True,
        note=synced.get('note', 'specialist session-clock sync'),
        raw_audio_offset_s=synced.get('raw_audio_offset_s'),
        peak_ratio=synced.get('peak_ratio'),
        peak_value=synced.get('peak_value'),
        video_offset_status=synced.get('video_offset_status'),
        fine_method=synced.get('fine_method'),
        sync_record_sha256=(artifact_identity.file_hash(sync_path)
                            if sync_path.is_file() else None),
    )
    peak_ratio = synced.get('peak_ratio')
    peak_text = f'{float(peak_ratio):.2f}' if peak_ratio is not None else 'n/a'
    console.write(
        f'双手音频同步：完成；offsets_s={synchronization.format_offsets(resolved["offsets_s"])}；'
        f'峰值比={peak_text}',
        recording=episode.root.name,
        state='完成',
    )
    return replace(episode, synchronization=resolved), 'computed', '首次生成 specialist session-clock 偏移并写入 sync.json'


def _reuse_existing_result(episode: Episode, input_identity: dict, *, force: bool) -> dict | None:
    """Return a valid published result with the same input identity, if one exists."""
    if force:
        return None
    local_output = episode.root / 'local_results'
    candidates = [(episode.output, 'umi_scene_episode', 'skipped'),
                  (episode.output, 'umi_poster_episode', 'skipped')]
    if len(episode.expected_arms) == 1:
        candidates.append((local_output, 'umi_local_episode', 'local_skipped'))
        if layout.read_state(episode.root).get('attempt', {}).get('status') in ('local_ok', 'local_skipped'):
            candidates.reverse()
    for output, kind, status in candidates:
        previous = output / 'episode.json'
        if not previous.is_file():
            continue
        old = json.loads(previous.read_text())
        if old.get('kind') != kind or old.get('inputs') != input_identity:
            continue
        reference = scene_reference(old)
        validate_assets(output.resolve(), old['assets'])
        episodes.record_attempt(episode, status, output.resolve())
        cleanup.after_task(episode.root)
        quality_path = output / 'run_report.json'
        quality = None
        if quality_path.is_file():
            quality = json.loads(quality_path.read_text()).get('quality')
        return dict(status=status, output=str(output), frame_id=reference['frame_id'],
                    synchronization=old.get('synchronization'),
                    **({'quality': quality} if quality is not None else {}))
    return None


def _requested_map_version(episode: Episode, command_map_version: str | None) -> str | None:
    """Resolve one task's map declaration against the command-line override."""
    requested = command_map_version
    selection = episode.map_selection or {}
    explicit = selection.get('selection') in {
        'explicit_task', 'explicit_override', 'explicit_epoch'
    }
    planned = selection.get('selection') not in (None, 'batch_default')
    if episode.map_version is not None and (explicit or planned):
        if requested is not None and requested != episode.map_version:
            raise ValueError('命令行 map_version 与任务解析出的地图版本不一致')
        requested = episode.map_version
    return requested


def _reject_preflight(episode: Episode, check_report, *, dry_run: bool):
    """原片预检未通过：留下预检现场（非预览）后拒绝该任务。"""
    message = '原片检查未通过：' + '；'.join(check_report.reasons)
    if not dry_run:
        with locked(episode.root):
            run_dir = layout.metadata(episode.root, 'runs') / uuid.uuid4().hex
            write_json(run_dir / 'preflight_report.json', asdict(check_report))
            identity = dict(episode=artifact_identity.file_hash(episode.path),
                            session=artifact_identity.file_hash(episode.session.path),
                            preflight=asdict(check_report))
            audit = RunAudit(layout.metadata(episode.root, 'logs') / run_dir.name,
                             task=episode.root.name, session=episode.session.root.name,
                             input_identity=identity, run_directory=run_dir)
            audit.event('preflight', status='rejected', reasons=check_report.reasons)
            retained = record_failure(episode, run_dir, message, status='rejected_preflight',
                identity=identity)
            if retained is not None:
                audit.finish('rejected_preflight', error=recording_checks.RecordingCheckError(message),
                             directory=retained)
    raise recording_checks.RecordingCheckError(message)


def _require_map_views(scene, cameras: dict, map_manifest: dict) -> None:
    for arm in cameras:
        for lens in scene.localization_lenses(arm):
            if scene.role(arm, lens) not in map_manifest['maps']:
                raise MapCompatibilityError(f'场景地图缺少 {arm}/{lens} 可定位的视角')


def _create_cache_audit(episode: Episode, cameras: dict, sync_status: str, sync_reason: str) -> CacheAudit:
    """记录本次任务不经阶段缓存、但需要审计来源的输入：同步、标定和 TCP。"""
    audit = CacheAudit()
    audit.record(stage='synchronization', key=artifact_identity.canonical_hash(episode.synchronization),
                  path=str(layout.metadata(episode.root, 'sync.json')),
                  status=sync_status, reason=sync_reason)
    for arm, camera in cameras.items():
        audit.record(stage='calibration',
                      key=artifact_identity.canonical_hash(camera_assets.camera_identity(camera)),
                      path=str(camera.selected.assets.camera_json), status='reused',
                      reason='registry 不可变标定版本已固定到本次任务快照')
        if 'tcp' not in camera.files:
            audit.record(stage='tcp', key=f'{arm}:none', path='', status='not_applicable',
                          reason='套件未配置 TCP 几何资产')
    return audit


def _freeze_tag_layout(episode: Episode, run_dir: Path, input_identity: dict) -> Episode:
    """把海报布局复制进运行目录并核对身份；之后的计算只引用这份副本。"""
    if episode.scene.tag_layout is None:
        return episode
    from software.vision.poster.layout import load as load_tag_board
    frozen_layout = run_dir / 'devices/environment_poster.json'
    frozen_layout.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(episode.scene.tag_layout, frozen_layout)
    if load_tag_board(frozen_layout).asset_hash != input_identity['environment_tags']['asset_hash']:
        raise ValueError('海报布局在预检后发生变化，请重新运行')
    return replace(episode, session=replace(episode.session,
                   scene=replace(episode.scene, tag_layout=frozen_layout)))


def run_episode(path: Path, *, force=False, cores='', dry_run=False, check_recordings=True,
                map_version: str | None = None, overrides: SceneOverrides | None = None,
                parallel_arms: bool = True, batch_map: map_lifecycle.BatchMap | None = None,
                map_epoch: str | None = None, _preloaded_episode: Episode | None = None) -> EpisodeRunResult:
    """预检并固定地图 → 准备录像 → 各手独立求解 → 核对输入 → 发布结果。

    锁、缓存复用、失败记录与成功指针由本层管理；算法与结果写入分别
    交给 reconstruct_arm_in_scene() 和 results。失败保留已有成功入口。
    """
    episode = (_preloaded_episode if _preloaded_episode is not None else
               read_episode(path, overrides=overrides, map_epoch_override=map_epoch))
    episode, sync_cache_status, sync_cache_reason = _resolve_episode_synchronization(
        episode, force=force, dry_run=dry_run)
    video_offsets_s = episode.offsets()
    name = episode.root.name
    requested_map_version = _requested_map_version(episode, map_version)
    console.write(
        f'任务开始：手={"、".join(episode.expected_arms)}；'
        f'后端={episode.scene.reconstruction_mode}；'
        f'同步={episode.synchronization.get("method", "unknown")}',
        recording=name,
        state='开始',
    )

    # --- 阶段 0: 原片录像预检 (Preflight Check) ---
    check_report = recording_checks.check_episode(episode) if check_recordings else None
    if check_report is not None and not check_report.passed:
        _reject_preflight(episode, check_report, dry_run=dry_run)

    with nullcontext() if dry_run else locked(episode.root):
        # --- 阶段 1: 地图版本绑定与输入指纹比对 (Bind Map & Verify Cache) ---
        if batch_map is not None and batch_map.requested != requested_map_version:
            raise ValueError('批量运行中的任务需要不同地图版本，请按地图版本分组运行')
        mode = episode.scene.reconstruction_mode
        episode, map_selection = estimation.select_map(episode, requested_map_version,
            batch_map=batch_map, dry_run=dry_run, cores=cores)
        map_dir, map_manifest = map_selection.directory, map_selection.manifest
        if mode != 'scene_map':
            batch_map = None
        scene = episode.scene
        device_scene = replace(
            scene,
            devices=episode.session.devices,
            rear_calibrations=camera_assets.rear_files(episode.session.devices),
        )
        cameras = camera_assets.select_cameras(episode.recordings, device_scene, episode.serials)
        if map_manifest is not None:
            try:
                map_lifecycle.validate_map_calibrations(cameras, map_manifest)
                _require_map_views(scene, cameras, map_manifest)
            except MapCompatibilityError as error:
                if mode != 'auto':
                    raise
                map_selection = estimation.MapSelection.without_map(fallback_record(error))
                map_dir, map_manifest = None, None
        input_identity = episode_inputs(episode, cameras, map_manifest)

        map_text = (f'{map_manifest["version"]}（{map_manifest["frame_id"]}）'
                    if map_manifest is not None else '无预建场景地图')
        console.write(
            f'输入已固定：地图={map_text}；'
            f'镜头={", ".join(f"{arm}:{camera.serial}" for arm, camera in cameras.items())}；'
            f'结果={episode.output}',
            recording=name,
            state='就绪',
        )

        if dry_run:
            return dict(status='planned', inputs=input_identity, output=str(episode.output),
                        synchronization=episode.synchronization)

        if mode != 'poster_vio':
            runtime.validate_engines(dict(os.environ))

        reused = _reuse_existing_result(episode, input_identity, force=force)
        if reused is not None:
            return reused

        # --- 阶段 2: 独立计算隔离运行 (Isolated Run) ---
        run_dir = layout.metadata(episode.root, 'runs') / uuid.uuid4().hex
        run_dir.mkdir(parents=True)
        # 解码产物从缓存链接到运行期工作目录，结束即删；发布结果只保留自包含文件。
        work_dir = layout.metadata(episode.root, 'work') / run_dir.name
        started, started_at = time.monotonic(), time.time()
        cache_audit = _create_cache_audit(episode, cameras, sync_cache_status, sync_cache_reason)
        audit = RunAudit(layout.metadata(episode.root, 'logs') / run_dir.name,
                         task=name, session=episode.session.root.name,
                         input_identity=input_identity, run_directory=run_dir)
        run_console = console.child(line_sink=audit.write_line, event_sink=audit.event)
        audit.event('stage_started', stage='isolated_run')
        audit.event(
            'inputs_resolved',
            mode=mode,
            map_directory=map_dir,
            map_version=(map_manifest or {}).get('version') if map_manifest else None,
            synchronization_status=sync_cache_status,
            synchronization_reason=sync_cache_reason,
            preflight=(asdict(check_report) if check_report is not None else None),
        )

        try:
            if check_report is not None:
                write_json(run_dir / 'preflight_report.json', asdict(check_report))

            cameras = camera_assets.snapshot_cameras(cameras, run_dir)
            episode = _freeze_tag_layout(episode, run_dir, input_identity)
            # Step 1: 准备录像与公共时间区间
            audit.event('stage_started', stage='prepare_recordings')
            run_console.write('准备任务录像与时间轴', recording=name, step=(1, 3))
            arm_cpu_sets = cpu_budget.arm_cores(tuple(episode.recordings), cores, parallel=parallel_arms)
            prepared_recordings = recording_preparation.prepare_episode_recordings(episode, cameras, video_offsets_s,
                work_dir=work_dir, cores=cores, arm_cores=arm_cpu_sets, cache_audit=cache_audit)
            common_start = max(recording.video_times_s[0] + video_offsets_s[arm]
                               for arm, recording in prepared_recordings.items())
            common_end = min(recording.video_times_s[-1] + video_offsets_s[arm]
                             for arm, recording in prepared_recordings.items())
            run_console.write(
                f'录像准备完成：' + '；'.join(
                    f'{arm} {len(recording.video_times_s)} 帧/{recording.video_times_s[-1] - recording.video_times_s[0]:.2f}s'
                    for arm, recording in prepared_recordings.items())
                + f'；共同时间 {common_start:.3f}–{common_end:.3f}s',
                recording=name,
                state='完成',
            )
            # Step 2: 各手独立求解轨迹、夹爪测量与共享海报融合
            audit.event('stage_finished', stage='prepare_recordings')
            audit.event('stage_started', stage='estimate_tracks')
            task_inputs = TaskInputs(episode, prepared_recordings, video_offsets_s,
                                     run_dir, arm_cpu_sets, cache_audit, run_console)
            arm_execution = estimation.execute(task_inputs, map_selection)
            tracks = arm_execution.tracks
            coordinate_record = arm_execution.coordinate_record
            backend_names = sorted({
                track.metadata.get('fusion', {}).get('reconstruction_backend')
                for track in tracks.values()
                if track.metadata.get('fusion', {}).get('reconstruction_backend')
            })
            backend_text = ','.join(backend_names) if backend_names else '详见 reconstruction.json'
            fallback_text = '；发生自动回退' if coordinate_record.get('fallback') else ''
            run_console.write(
                f'轨迹求解完成：坐标={coordinate_record.get("frame_id", "unknown")}；'
                f'实际后端={backend_text}；手={", ".join(sorted(tracks))}{fallback_text}',
                recording=name,
                state='完成',
            )
            # 局部回退结果不引用共享地图。
            display_map_path = (None if is_local_scene(coordinate_record) or is_poster_scene(coordinate_record)
                                else map_dir / 'map.json')
            if mode == 'auto' and display_map_path is not None:
                from software.scene.asset_identity import validate_assets
                validate_assets(map_dir, map_manifest['assets'])

            audit.event('stage_finished', stage='estimate_tracks')
            # Step 3: 双手对齐、防篡改校验与原子发布
            audit.event('stage_started', stage='publish')
            run_console.write('写入轨迹、报告并发布结果', recording=name, step=(3, 3))
            common_interval_s = trajectory_writers.write_bimanual_trajectory(tracks, run_dir)

            publication = episode_publication.publish_episode(
                episode=episode,
                input_identity=input_identity,
                map_manifest=map_manifest,
                tracks=tracks,
                video_offsets_s=video_offsets_s,
                coordinate_record=coordinate_record,
                run_dir=run_dir,
                map_path=display_map_path,
                width_reports=arm_execution.width_reports,
                tag_distance_reports=arm_execution.tag_distance_reports,
                cache_audit=cache_audit,
                common_interval_s=common_interval_s,
                elapsed_s=time.monotonic() - started,
                started_at=started_at,
                batch_map=batch_map,
            )
            is_local = publication.is_local
            output = publication.output
            audit.event('stage_finished', stage='publish', output=output)
            completion_message = (
                f'任务完成：状态={"局部成功" if is_local else "成功"}；输出={output}；'
                f'共同时间={common_interval_s[0]:.3f}–{common_interval_s[1]:.3f}s')
            run_console.write(completion_message, recording=name, state='完成')
            audit.finish('ok', output=output)
        except Exception as error:
            retained = record_failure(episode, run_dir, str(error), identity=input_identity, error=error)
            if retained is not None:
                audit.finish('failed', error=error, directory=retained)
            raise
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            with suppress(OSError):
                work_dir.parent.rmdir()

        return dict(
            status='local_ok' if is_local else 'ok',
            output=str(output),
            frame_id=coordinate_record['frame_id'],
            common_interval_s=common_interval_s,
            synchronization=episode.synchronization,
            quality=reporting.quality_summary(tracks, coordinate_record, common_interval_s),
        )


def run_episodes(paths: list[Path], *, jobs: int = 1, force=False, cores='', dry_run=False, check_recordings=True,
                 map_version: str | None = None, map_epoch: str | None = None,
                 overrides: SceneOverrides | None = None) -> list[dict]:
    """批量处理任务；支持多任务并发调度与幂等断点续跑 (resume)。"""
    if jobs < 1:
        raise ValueError('--jobs 必须 >= 1')
    resolved = [p.resolve() for p in paths]
    if not resolved:
        return []

    console.write(f'共 {len(resolved)} 个任务；并发数 {jobs}')
    batch_maps = {}
    valid = []
    rejected = []
    for path in resolved:
        try:
            task_episode = read_episode(path, overrides=overrides, map_epoch_override=map_epoch)
            batch_root = layout.owner(path).parent
            requested = _requested_map_version(task_episode, map_version)
            batch_maps.setdefault((batch_root, requested), map_lifecycle.BatchMap(requested, dry_run, cores=cores))
            valid.append((path, requested, task_episode))
        except ITEM_FAILURES as error:
            console.write(f'{path.parent.name}：{error}', recording=path.parent.name, state='失败')
            rejected.append(dict(path=str(path), name=path.parent.name, status='failed', error=str(error)))

    def _execute(path: Path, requested: str | None, task_episode: Episode) -> dict:
        name = layout.owner(path).name
        try:
            episode_result = run_episode(path, force=force, cores=cores, dry_run=dry_run, check_recordings=check_recordings, map_version=map_version,
                                         map_epoch=map_epoch,
                                         overrides=overrides, parallel_arms=jobs == 1,
                                         batch_map=batch_maps[(layout.owner(path).parent, requested)],
                                         _preloaded_episode=task_episode)
            return dict(path=str(path), name=name, **episode_result)
        except ITEM_FAILURES as error:
            if not dry_run and path.is_file():
                try:
                    state = layout.read_state(layout.owner(path))
                    if state.get('attempt', {}).get('status') not in ('rejected_preflight', 'failed'):
                        episodes.record_attempt(task_episode, 'failed', None, reason=str(error))
                except ITEM_FAILURES:
                    pass
            console.write(f'{name}：{error}', recording=name, state='失败')
            return dict(path=str(path), name=name, status='failed', error=str(error))

    if jobs == 1 or len(valid) == 1:
        return rejected + [_execute(path, requested, task_episode)
                           for path, requested, task_episode in valid]
    with cf.ThreadPoolExecutor(max_workers=jobs) as executor:
        return rejected + list(executor.map(
            lambda item: _execute(*item), valid))


def format_batch_summary(summaries: list[dict]) -> str:
    """生成整洁易读的批量处理汇总表格。"""
    lines = [
        '',
        '=' * 80,
        f'场景任务批量处理结果 (共 {len(summaries)} 个任务)',
        '=' * 80,
        f'{"任务名":<15} {"状态":<10} {"有效时长":<15} {"质量摘要/输出目录"}',
        '-' * 80,
    ]
    ok_count = 0
    skipped_count = 0
    failed_count = 0
    local_count = 0

    def interval_text(summary_item: dict) -> str:
        interval = summary_item.get('common_interval_s')
        return f'{interval[1] - interval[0]:.2f}s' if interval else '-'

    def with_quality(summary_item: dict, tail: str) -> str:
        quality = quality_text(summary_item)
        return f'{quality}；{tail}' if quality else tail

    def quality_text(summary_item: dict) -> str:
        quality = summary_item.get('quality')
        if not quality:
            return ''
        parts = []
        for arm, arm_quality in quality.get('arms', {}).items():
            visual_fraction = arm_quality.get('visual_constraint_fraction')
            longest_gap = arm_quality.get('longest_visual_gap_s')
            fields = [arm]
            if visual_fraction is not None:
                fields.append(f'视觉 {visual_fraction:.1%}')
            if longest_gap is not None:
                fields.append(f'最长无视觉 {longest_gap:.2f}s')
            poster_status = arm_quality.get('environment_poster_status')
            if poster_status:
                fields.append(f'海报 {poster_status}')
            parts.append('，'.join(fields))
        return '；'.join(parts)

    for summary_item in summaries:
        name = summary_item.get('name', Path(summary_item['path']).parent.name)
        status = summary_item.get('status', 'unknown').upper()
        if status == 'OK':
            ok_count += 1
            duration_str = interval_text(summary_item)
            detail = with_quality(summary_item, summary_item.get('output', ''))
        elif status in ('LOCAL_OK', 'LOCAL_SKIPPED'):
            local_count += 1
            duration_str = interval_text(summary_item)
            detail = with_quality(summary_item, f'独立局部坐标（不与场景共享）: {summary_item.get("output", "")}')
        elif status == 'SKIPPED':
            skipped_count += 1
            duration_str = '-'
            detail = with_quality(summary_item, f'已完成复用: {summary_item.get("output", "")}')
        elif status == 'PLANNED':
            duration_str = '-'
            detail = '预检完成 (dry-run)'
        else:
            failed_count += 1
            duration_str = '-'
            detail = summary_item.get('error') or summary_item.get('reason') or '未知错误'
        lines.append(f'{name:<15} {status:<10} {duration_str:<15} {detail}')
    lines.append('=' * 80)
    local_summary = f' | 局部回退: {local_count}' if local_count else ''
    lines.append(f'成功: {ok_count}{local_summary} | 跳过: {skipped_count} | 失败: {failed_count} | 总计: {len(summaries)}')
    return '\n'.join(lines)
