"""执行场景任务与批量调度。"""

from __future__ import annotations

import os
import shutil
import time
import uuid
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import Lock

from umi.infra import provenance as artifact_identity
from umi.infra.concurrency import daemon_map
from umi.infra.file_store import recompute_when_forced, write_json
from umi.infra.orb_install import require_forward_kb4
from umi.infra.terminal import Console
from umi.perception import runtime
from umi.scene import cleanup, publication, routing, task_state
from umi.scene.collection import batch as batches
from umi.scene.collection.batch import batch_task_directories, read_batch, read_episode
from umi.scene.collection.configuration import SceneOverrides
from umi.scene.errors import ITEM_FAILURES
from umi.scene.inputs import recordings, synchronization
from umi.scene.inputs.identity import current_auto_identity, episode_inputs, map_input_fields, result_reuse_identity
from umi.scene.maps import binding, versions
from umi.scene.task_state import RunAudit
from umi.store import layout
from umi.store.cache import CacheAudit, camera_identity
from umi.store.layout import locked
from umi.store.models import CalibratedCamera, Episode, EpisodeRunResult, TaskInputs
from umi.store.result_io import is_local_scene, validate_assets

console = Console('场景')


@recompute_when_forced
def run_episode(path: Path, *, force=False, cores='', dry_run=False, check_recordings=True,
                map_version: str | None = None, overrides: SceneOverrides | None = None,
                parallel_arms: bool = True, batch_map: binding.BatchMap | None = None,
                map_epoch: str | None = None) -> EpisodeRunResult:
    """Resolve a task, reuse its published result or execute a new run."""
    episode = read_episode(path, overrides=overrides, map_epoch_override=map_epoch, allow_pending_map=True)
    requested = binding.requested_map_version(episode, map_version)
    return _run_episode(episode, requested, batch_map, force=force, cores=cores, dry_run=dry_run,
                        check_recordings=check_recordings, parallel_arms=parallel_arms)


def _run_episode(episode: Episode, requested_map: str | None, batch_map: binding.BatchMap | None, *,
                 force: bool, cores: str, dry_run: bool, check_recordings: bool,
                 parallel_arms: bool) -> EpisodeRunResult:
    task_started = time.monotonic()
    name = episode.root.name
    task_console = console.child(recording=f'{episode.session.root.name}/{name}')
    hands = '双手' if len(episode.expected_arms) > 1 else {'left': '左手', 'right': '右手'}[episode.expected_arms[0]]
    task_console.write(f'{hands} · {episode.scene.reconstruction_mode}', state='开始')
    episode, sync_cache_status, sync_cache_reason = resolve_synchronization(
        episode, task_console, force=force, dry_run=dry_run)
    video_offsets_s = episode.offsets()
    check_report = preflight_recordings(
        episode, task_console, enabled=check_recordings, dry_run=dry_run)

    with nullcontext() if dry_run else locked(episode.root):
        resolved = resolve_inputs(episode, task_console, map_version=requested_map,
                                            batch_map=batch_map, dry_run=dry_run, cores=cores, force=force,
                                            video_offsets_s=video_offsets_s, check_report=check_report,
                                            synchronization_audit=(sync_cache_status, sync_cache_reason))
        episode = resolved.episode
        if dry_run:
            return dict(status='planned', inputs=resolved.identity, output=str(episode.output),
                        synchronization=episode.synchronization, **backend_preview(episode, force=force))

        identify_previous = (lambda old: current_auto_identity(resolved, old)
                             ) if episode.scene.reconstruction_mode == 'auto' else None
        reused = publication.reuse_episode(episode, resolved.identity, force=force,
                                            identify_previous=identify_previous)
        if reused is not None:
            task_console.write('已复用匹配的轨迹结果', state='复用')
            return reused

        if episode.scene.reconstruction_mode not in ('auto', 'poster_vio'):
            runtime.validate_engines(dict(os.environ))
        return execute_episode(resolved, task_console, cores=cores,
                                         parallel_arms=parallel_arms, task_started=task_started)


@recompute_when_forced
def run_episodes(paths: list[Path], *, jobs: int = 1, force=False, cores='', dry_run=False, check_recordings=True,
                 map_version: str | None = None, map_epoch: str | None = None,
                 overrides: SceneOverrides | None = None) -> list[dict]:
    """批量处理任务；支持多任务并发调度与幂等断点续跑 (resume)。"""
    if jobs < 1:
        raise ValueError('--jobs 必须 >= 1')
    resolved = [p.resolve() for p in paths]
    if not resolved:
        return []

    total = len(resolved)
    console.write(f'{total} 个任务 · 并发上限 {jobs} · 原片预检{"开启" if check_recordings else "跳过"}',
                  state='计划')
    completion_lock = Lock()
    completed = 0

    def report_completion(name: str) -> None:
        nonlocal completed
        if total > 1:
            with completion_lock:
                completed += 1
                console.write(f'已结束 {completed}/{total} 个任务 · {name}', state='总进度')

    batch_maps = {}
    valid = []
    rejected = []
    for path in resolved:
        try:
            task_episode = read_episode(path, overrides=overrides, map_epoch_override=map_epoch, allow_pending_map=True)
            requested = binding.requested_map_version(task_episode, map_version)
            batch_binding = None
            if (task_episode.scene.reconstruction_mode in ('auto', 'scene_map')
                    and (force or not (task_episode.map_selection or {}).get('epoch'))):
                key = (task_episode.session.root, requested)
                if key not in batch_maps:
                    batch_maps[key] = binding.BatchMap(requested, dry_run, cores=cores, force=force)
                batch_binding = batch_maps[key]
            valid.append((path, task_episode, requested, batch_binding))
        except ITEM_FAILURES as error:
            name = path.parent.name
            console.write(str(error), recording=name, state='失败')
            rejected.append(dict(path=str(path), name=name, status='failed', error=str(error)))
            report_completion(name)

    def _execute(path: Path, task_episode: Episode, requested: str | None,
                 batch_map: binding.BatchMap | None) -> dict:
        root = layout.owner(path)
        name = root.name
        label = f'{root.parent.name}/{name}'
        try:
            episode_result = _run_episode(task_episode, requested, batch_map, force=force, cores=cores,
                                          dry_run=dry_run, check_recordings=check_recordings, parallel_arms=jobs == 1)
            result = dict(path=str(path), name=name, **episode_result)
        except ITEM_FAILURES as error:
            if not dry_run and path.is_file():
                try:
                    with locked(root):
                        state = layout.read_state(layout.owner(path))
                        if state.get('attempt', {}).get('status') not in ('rejected_preflight', 'failed'):
                            task_state.record_attempt(task_episode, 'failed', None, reason=str(error))
                except ITEM_FAILURES:
                    pass
            console.write(str(error), recording=label, state='失败')
            result = dict(path=str(path), name=name, status='failed', error=str(error))
        report_completion(label)
        return result

    if jobs == 1 or len(valid) == 1:
        return rejected + [_execute(*item) for item in valid]
    return rejected + daemon_map(lambda item: _execute(*item), valid, max_workers=jobs)


def format_batch_summary(summaries: list[dict]) -> str:
    """按任务分行展示状态、质量和路径，长路径可以自然换行。"""
    status_labels = {
        'ok': '成功', 'skipped': '复用', 'local_ok': '局部成功', 'local_skipped': '局部复用', 'planned': '预览',
    }
    counts = Counter(item.get('status') for item in summaries)
    failed = len(summaries) - sum(counts[status] for status in status_labels)
    totals = [f'共 {len(summaries)} 项', f'成功 {counts["ok"]}', f'复用 {counts["skipped"]}', f'失败 {failed}']
    if local := counts['local_ok'] + counts['local_skipped']:
        totals.append(f'局部回退 {local}')
    if counts['planned']:
        totals.append(f'预览 {counts["planned"]}')
    lines = ['', '场景处理结果', '  ' + ' · '.join(totals)]

    names = [item.get('name') or Path(item.get('path') or '.').parent.name or '任务' for item in summaries]
    name_counts = Counter(names)
    for name, item in zip(names, summaries):
        if name_counts[name] > 1 and item.get('path'):
            path = Path(item['path'])
            root = layout.owner(path) if path.match('*/.pipeline/tasks/*/episode.yaml') else path
            name = '/'.join(root.parts[-3:]).lstrip('/')
        status = item.get('status')
        quality = item.get('quality') or {}
        interval = item.get('common_interval_s') or quality.get('common_interval_s')
        title = [name, status_labels.get(status, '失败')]
        if interval:
            title.append(f'有效 {interval[1] - interval[0]:.2f}s')
        lines.extend(('', '  ' + ' · '.join(title)))

        if status == 'planned':
            if item.get('recompute'):
                lines.append('    重算本次执行的阶段')
            if order := item.get('auto_order'):
                lines.append('    后端顺序  ' + ' → '.join(order))
        elif status not in status_labels:
            if details := item.get('details'):
                source_states = {'missing_peer': '缺少配对原片', 'ambiguous': '配对不唯一',
                                 'ambiguous_task': '同一手有多段候选',
                                 'conflicting_tasks': '原片与已登记任务冲突'}
                for detail in details:
                    source_state = detail.get('status', '待确认')
                    reason = detail.get('reason') or source_states.get(source_state, source_state)
                    lines.append(f'    {detail["path"]} · {reason}')
            else:
                reason = item.get('error') or item.get('reason') or '未知错误'
                lines.extend(f'    {line}' for line in reason.splitlines())

        for arm, metrics in quality.get('arms', {}).items():
            fields = []
            if (visual := metrics.get('visual_constraint_fraction')) is not None:
                fields.append(f'视觉 {visual:.1%}')
            if (gap := metrics.get('longest_visual_gap_s')) is not None:
                fields.append(f'最长视觉缺口 {gap:.2f}s')
            if poster := metrics.get('environment_poster_status'):
                poster_label = {'fused': '已融合', 'diagnostics_only': '仅诊断',
                                'not_observed': '未观测', 'disabled': '关闭'}.get(poster, poster)
                fields.append(f'海报 {poster_label}')
            if distance := metrics.get('tag_distance'):
                fields.append(f'夹爪测距 {"通过" if distance["passed"] else "未通过"}（有效 {distance["valid_fraction"]:.1%}）')
            if fields:
                hand = {'left': '左手', 'right': '右手'}.get(arm, arm)
                lines.append(f'    {hand}  ' + ' · '.join(fields))
        if status in ('local_ok', 'local_skipped'):
            lines.append('    坐标  本任务的独立局部坐标')
        if status in ('ok', 'skipped', 'local_ok', 'local_skipped') and item.get('output'):
            lines.append(f'    结果  {item["output"]}')
    return '\n'.join(lines)


@dataclass(frozen=True)
class ResolvedInputs:
    episode: Episode
    cameras: dict[str, CalibratedCamera]
    map_selection: binding.MapSelection
    identity: dict
    video_offsets_s: dict[str, float]
    check_report: recordings.EpisodeCheckReport | None
    synchronization_audit: tuple[str, str]
    batch_map: binding.BatchMap | None
    requested_map: str | None = None
    force: bool = False
    map_attempt_episode: Episode | None = None


def backend_preview(episode: Episode, *, force=False) -> dict:
    plan = {'recompute': True} if force else {}
    if episode.scene.reconstruction_mode != 'auto':
        if force and episode.scene.reconstruction_mode == 'scene_map':
            plan['scene_map'] = dict(preparation='first_candidate', selection=episode.map_selection)
        return plan
    order = episode.scene.auto_order
    return dict(**plan, auto_order=list(order), scene_map=dict(
        preparation=('disabled' if 'scene_map' not in order else
                     'first_candidate' if order[0] == 'scene_map' else 'on_fallback'),
        selection=episode.map_selection))


def resolve_synchronization(episode, task_console: Console, *, force: bool, dry_run: bool):
    """Resolve audio-auto offsets and return the cache audit status."""
    if episode.synchronization.get('method') != 'audio_auto':
        if len(episode.expected_arms) > 1:
            method = episode.synchronization['method']
            strategy = {'manual': '人工偏移', 'assumed_simultaneous': '假设同时开录'}.get(method, method)
            offsets = synchronization.format_offsets(episode.synchronization['offsets_s'])
            confirmed = '已确认' if episode.synchronization['confirmed'] else '待确认'
            task_console.write(f'{strategy} · {offsets} · {confirmed}', state='同步')
        return episode, 'not_applicable', '任务未启用 audio_auto'

    sync_path = layout.metadata(episode.root, 'sync.json')
    algorithm = episode.synchronization['algorithm']
    with nullcontext() if dry_run else locked(episode.root):
        resolved, reused = synchronization.resolve_audio_sync(
            episode.recordings, sync_path, algorithm=algorithm, force=force, dry_run=dry_run,
            on_estimate=lambda: task_console.write(f'{algorithm} 音频偏移', state='同步中'))
    task_console.write(
        f'{algorithm}{"（复用）" if reused else ""} · {synchronization.format_offsets(resolved["offsets_s"])} · '
        f'{synchronization.format_audio_evidence(resolved)}', state='同步',
    )
    status = 'reused' if reused else 'computed'
    reason = f'sync.json 的原片和 {algorithm} 算法匹配' if reused else f'已估计 {algorithm} 音频偏移'
    return replace(episode, synchronization=resolved), status, reason


def preflight_recordings(episode: Episode, task_console: Console, *, enabled: bool,
                     dry_run: bool) -> recordings.EpisodeCheckReport | None:
    """Report recording warnings and reject hard failures before taking the run lock."""
    check_report = None
    if enabled:
        with task_console.heartbeat('检查原片与 IMU'):
            check_report = recordings.check_episode(episode)
        if not check_report.passed:
            _reject_preflight(episode, check_report, task_console, dry_run=dry_run)
        task_console.write('原片检查通过', state='预检')
        for arm, report in check_report.videos.items():
            for check in report.checks:
                if check.status in ('WARN', 'UNKNOWN'):
                    hand = '左手' if arm == 'left' else '右手'
                    message = ('开头静止不足' if check.name == 'initial_static' and check.status == 'WARN'
                               else check.message)
                    task_console.write(f'{hand} · {message}', state='提醒')
    else:
        task_console.write('已跳过原片预检', state='预检')
    return check_report


def _reject_preflight(episode: Episode, check_report, task_console: Console, *, dry_run: bool):
    """原片预检未通过：留下预检现场（非预览）后拒绝该任务。"""
    message = '原片检查未通过：' + '；'.join(check_report.reasons)
    if not dry_run:
        with locked(episode.root):
            run_dir = layout.metadata(episode.root, 'runs') / uuid.uuid4().hex
            write_json(run_dir / 'preflight_report.json', asdict(check_report))
            identity = dict(episode=artifact_identity.file_hash(episode.path),
                            session=artifact_identity.file_hash(episode.session.path),
                            preflight=asdict(check_report))
            audit = RunAudit(episode.root, session=episode.session.root.name,
                             input_identity=identity, run_directory=run_dir)
            task_console.child(line_sink=audit.write_line).write(message, state='预检', visible=False)
            audit.event('preflight', status='rejected', reasons=check_report.reasons)
            retained = publication.record_failure(episode, run_dir, message, status='rejected_preflight',
                identity=identity)
            audit.finish('rejected_preflight', error=recordings.RecordingCheckError(message))
            cleanup.after_task(episode.root)
            task_console.write(f'日志 {audit.log_path} · 预检报告 {retained / "preflight_report.json"}', state='诊断')
    raise recordings.RecordingCheckError(message)


def resolve_inputs(episode: Episode, task_console: Console, *, map_version: str | None,
                   batch_map: binding.BatchMap | None, dry_run: bool, cores: str,
                   video_offsets_s: dict[str, float], check_report: recordings.EpisodeCheckReport | None,
                   synchronization_audit: tuple[str, str], force=False) -> ResolvedInputs:
    """Select task inputs. Auto only resolves a map when its candidate is reached."""
    if batch_map is not None and batch_map.requested != map_version:
        raise ValueError('批量运行中的任务需要不同地图版本，请按地图版本分组运行')
    mode = episode.scene.reconstruction_mode
    map_selection = binding.MapSelection.without_map()
    if mode == 'scene_map' and not (dry_run and (force or (episode.map_selection or {}).get('status') == 'pending')):
        episode, map_selection = binding.select_map(episode, map_version,
            batch_map=batch_map, dry_run=dry_run, cores=cores, force=force)
    map_manifest = map_selection.manifest
    scene = episode.scene
    camera_mode = 'shared_front' if mode == 'auto' and scene.auto_order[0] != 'scene_map' else scene.mode
    cameras = recordings.select_task_cameras(episode, camera_mode)
    if map_manifest is not None:
        versions.validate_map_calibrations(cameras, map_manifest)
        binding.require_map_views(scene, cameras, map_manifest)
    input_identity = episode_inputs(episode, cameras, map_manifest)
    if mode != 'auto':
        input_identity['reuse_identity'] = result_reuse_identity(episode, input_identity, (mode,))

    map_text = (f'地图 {map_manifest["frame_id"]}' if map_manifest is not None else
                '到达 scene_map 时准备地图' if mode == 'auto' and 'scene_map' in scene.auto_order else '无预建场景地图')
    camera_text = ' · '.join(f'{arm} {camera.serial}' for arm, camera in cameras.items())
    task_console.write(f'{map_text} · 相机 {camera_text}', state='输入就绪')
    return ResolvedInputs(episode=episode, cameras=cameras, map_selection=map_selection,
                          identity=input_identity, video_offsets_s=video_offsets_s,
                          check_report=check_report, synchronization_audit=synchronization_audit,
                          batch_map=batch_map, requested_map=map_version, force=force)


def execute_episode(resolved: ResolvedInputs, task_console: Console, *, cores: str,
                    parallel_arms: bool, task_started: float) -> EpisodeRunResult:
    """Freeze assets and run reconstruction while the caller holds the task lock."""
    episode = resolved.episode
    video_offsets_s = resolved.video_offsets_s
    mode = episode.scene.reconstruction_mode
    # Auto validates a batch map at publication only after the map candidate was reached.
    active = replace(resolved, batch_map=None) if mode == 'auto' else resolved
    sync_cache_status, sync_cache_reason = resolved.synchronization_audit
    check_record = asdict(resolved.check_report) if resolved.check_report is not None else None

    run_dir = layout.metadata(episode.root, 'runs') / uuid.uuid4().hex
    run_dir.mkdir(parents=True)
    # 解码产物从缓存链接到运行期工作目录，结束即删；发布结果只保留自包含文件。
    work_dir = run_dir / 'work'
    started, started_at = time.monotonic(), time.time()
    published = None
    cache_audit = _create_cache_audit(episode, resolved.cameras, sync_cache_status, sync_cache_reason)
    audit = RunAudit(episode.root, session=episode.session.root.name,
                     input_identity=resolved.identity, run_directory=run_dir)
    run_console = task_console.child(line_sink=audit.write_line, event_sink=audit.event)
    audit.event('stage_started', stage='isolated_run')
    audit.event(
        'inputs_resolved',
        mode=mode,
        map_directory=resolved.map_selection.directory,
        map_version=(resolved.map_selection.manifest or {}).get('version'),
        synchronization_status=sync_cache_status,
        synchronization_reason=sync_cache_reason,
        preflight=check_record,
    )

    try:
        if check_record is not None:
            write_json(run_dir / 'preflight_report.json', check_record)

        cameras = recordings.snapshot_cameras(resolved.cameras, run_dir, share_identical=True)
        episode = _freeze_tag_layout(episode, run_dir, resolved.identity)
        active = replace(active, episode=episode, cameras=cameras)
        # Step 1: 准备录像与公共时间区间
        audit.event('stage_started', stage='prepare_recordings')
        run_console.write('准备录像和时间轴', step=(1, 3))
        arm_cpu_sets = recordings.arm_cores(tuple(episode.recordings), cores, parallel=parallel_arms)
        with run_console.heartbeat('准备录像和时间轴', step=(1, 3)):
            prepared_recordings = recordings.prepare_episode_recordings(episode, cameras, video_offsets_s,
                work_dir=work_dir, cores=cores, arm_cores=arm_cpu_sets, cache_audit=cache_audit)
        run_console.write(_recordings_summary(prepared_recordings, video_offsets_s), step=(1, 3), state='完成')
        # Step 2: 各手独立求解轨迹、夹爪测量与共享海报融合
        audit.event('stage_finished', stage='prepare_recordings')
        audit.event('stage_started', stage='estimate_tracks')
        task_inputs = TaskInputs(episode, prepared_recordings, video_offsets_s, run_dir, arm_cpu_sets, cache_audit,
                                 run_console, backend_services=routing.backend_services(), work_directory=work_dir)
        map_inputs = {}
        map_attempt_episode = None

        def resolve_candidate_map(inputs: TaskInputs):
            nonlocal map_inputs, map_attempt_episode
            bound, selection, map_cameras = binding.resolve_auto_map(
                resolved.episode, resolved.requested_map, resolved.cameras,
                batch_map=resolved.batch_map, allow_build=True, cores=cores, force=resolved.force)
            scene = replace(episode.scene, mode=bound.scene.mode)
            bound_episode = replace(episode, session=replace(bound.session, scene=scene),
                              map_version=bound.map_version, map_selection=bound.map_selection)
            reselect_cameras = map_cameras is not resolved.cameras
            map_inputs = map_input_fields(
                bound_episode, selection.manifest, auto_map_attempted=True,
                requested_map_version=None if resolved.force and selection.manifest else resolved.requested_map)
            map_inputs['auto_map_cameras'] = {arm: camera_identity(camera) for arm, camera in map_cameras.items()}
            map_attempt_episode = bound_episode
            audit.event('map_resolved', map_directory=selection.directory,
                        map_version=(selection.manifest or {}).get('version'),
                        input_hash=audit.update_input_identity({**resolved.identity, **map_inputs}), failure=selection.failure)
            if selection.manifest is not None:
                runtime.validate_engines(dict(os.environ))
                if scene.mode == 'dual' and any(camera.measurements.get('rear', {}).get('intrinsics_source') == 'factory'
                                                for camera in map_cameras.values()):
                    require_forward_kb4()
            if reselect_cameras:
                frozen_cameras = recordings.snapshot_cameras(map_cameras, inputs.directory, share_identical=True)
                inputs = replace(inputs, recordings=recordings.prepare_episode_recordings(
                    bound_episode, frozen_cameras, video_offsets_s, work_dir=inputs.work_directory / 'prepare',
                    cores=cores, arm_cores=arm_cpu_sets, cache_audit=cache_audit))
            elif selection.manifest is not None:
                inputs = freeze_candidate(inputs)
            return replace(inputs, episode=bound_episode), selection

        def freeze_candidate(inputs: TaskInputs):
            candidate_cameras = {arm: item.camera for arm, item in inputs.recordings.items()}
            frozen = recordings.snapshot_cameras(candidate_cameras, inputs.directory, share_identical=True)
            return replace(inputs, recordings={arm: replace(item, camera=frozen[arm])
                                               for arm, item in inputs.recordings.items()})

        def select_candidate(inputs: TaskInputs, selection: binding.MapSelection):
            nonlocal active
            identity = {**resolved.identity, **map_inputs}
            if selection.manifest is not None:
                identity['base_cameras'] = identity['cameras']
                identity['cameras'] = map_inputs['auto_map_cameras']
            active = replace(active, episode=inputs.episode, map_selection=selection, identity=identity,
                             batch_map=resolved.batch_map if map_attempt_episode else None,
                             map_attempt_episode=map_attempt_episode)

        with run_console.heartbeat('轨迹重建', step=(2, 3)):
            if mode == 'auto':
                arm_execution = routing.execute(task_inputs, active.map_selection, resolve_map=resolve_candidate_map,
                    prepare_independent=lambda: runtime.validate_engines(dict(os.environ)),
                    freeze_candidate=freeze_candidate, select_candidate=select_candidate)
            else:
                arm_execution = routing.execute(task_inputs, active.map_selection)
        coordinate_record = arm_execution.coordinate_record
        attempts = coordinate_record.get('attempts', ())
        selected = [item['backend'] for item in attempts] if mode == 'auto' else [mode]
        retry_preparation = (coordinate_record.get('front_map_selection', {}).get('retry_preparation', False)
                             or any(item.get('retry_preparation') for item in attempts))
        identity = dict(active.identity, reuse_identity=result_reuse_identity(
            episode, active.identity, selected, retry_preparation=retry_preparation))
        active = replace(active, identity=identity)
        audit.update_input_identity(identity)
        run_console.write(_backend_summary(arm_execution, mode), step=(2, 3), state='完成')
        # 局部回退结果不引用共享地图。
        display_map_path = None
        if arm_execution.coordinates.uses_map:
            selection = active.map_selection
            display_map_path = selection.directory / 'map.json'
            if mode == 'auto':
                validate_assets(selection.directory, selection.manifest['assets'])

        audit.event('stage_finished', stage='estimate_tracks')
        # Step 3: 双手对齐、防篡改校验与原子发布
        audit.event('stage_started', stage='publish')
        run_console.write('发布轨迹和报告', step=(3, 3))
        published = publication.publish_episode(
            active,
            result=arm_execution,
            run_dir=run_dir,
            map_path=display_map_path,
            cache_audit=cache_audit,
            preflight=check_record,
            started=started,
        )
        is_local = published.is_local
        output = published.output
        audit.event('stage_finished', stage='publish', output=output)
        run_console.write(
            f'{"局部轨迹已发布" if is_local else "轨迹已发布"} · 耗时 {time.monotonic() - task_started:.1f}s',
            step=(3, 3), state='完成')
        audit.finish('ok', output=output)
        task_console.write(str(audit.log_path), state='日志')
    except KeyboardInterrupt as error:
        run_console.write(f'处理被用户中断 · 日志 {audit.log_path}', state='诊断')
        publication.record_failure(active.episode, run_dir, '处理被用户中断',
                                   identity=active.identity, status='interrupted', error=error)
        audit.finish('interrupted', error=error)
        raise
    except Exception as error:
        run_console.write(f'{error} · 日志 {audit.log_path}', state='诊断')
        publication.record_failure(active.episode, run_dir, str(error), identity=active.identity, error=error)
        audit.finish('failed', error=error)
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        cleanup.after_task(episode.root, used_since=started_at if published is not None else None)

    common_interval_s = published.common_interval_s
    return dict(
        status='local_ok' if is_local else 'ok',
        output=str(output),
        frame_id=coordinate_record['frame_id'],
        common_interval_s=common_interval_s,
        synchronization=active.episode.synchronization,
        quality=published.quality,
    )


def _recordings_summary(prepared_recordings: dict, video_offsets_s: dict) -> str:
    common_start, common_end = recordings.shared_interval(prepared_recordings, video_offsets_s)
    return (' · '.join(
                f'{arm} {len(recording.video_times_s)} 帧 / {recording.video_times_s[-1] - recording.video_times_s[0]:.2f}s'
                for arm, recording in prepared_recordings.items())
            + f' · {"共同时间" if len(prepared_recordings) > 1 else "时间范围"} {common_start:.3f}–{common_end:.3f}s')


def _backend_summary(arm_execution, mode: str) -> str:
    """终端一行：实际使用的后端（auto 时为选中者）、输出坐标及是否回退。"""
    coordinate_record = arm_execution.coordinate_record
    backend_names = sorted({
        track.metadata.get('fusion', {}).get('reconstruction_backend')
        for track in arm_execution.tracks.values()
        if track.metadata.get('fusion', {}).get('reconstruction_backend')
    })
    selected = next((attempt['backend'] for attempt in coordinate_record.get('attempts', [])
                     if attempt['status'] == 'succeeded'), mode)
    backend_text = ('独立局部重建' if is_local_scene(coordinate_record)
                    else ', '.join(backend_names) or selected)
    fallback_text = ' · 已回退' if coordinate_record.get('fallback') else ''
    return f'{backend_text}{fallback_text}'


def _create_cache_audit(episode: Episode, cameras: dict, sync_status: str, sync_reason: str) -> CacheAudit:
    """记录本次任务不经阶段缓存、但需要审计来源的输入：同步、标定和 TCP。"""
    audit = CacheAudit()
    audit.record(stage='synchronization', key=artifact_identity.canonical_hash(episode.synchronization),
                  path=str(layout.metadata(episode.root, 'sync.json')),
                  status=sync_status, reason=sync_reason)
    for arm, camera in cameras.items():
        audit.record(stage='calibration',
                      key=artifact_identity.canonical_hash(camera_identity(camera)),
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
    from umi.perception.poster.layout import load as load_tag_board
    frozen_layout = run_dir / 'devices/environment_poster.json'
    frozen_layout.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(episode.scene.tag_layout, frozen_layout)
    if load_tag_board(frozen_layout).asset_hash != input_identity['environment_tags']['asset_hash']:
        raise ValueError('海报布局在预检后发生变化，请重新运行')
    return replace(episode, session=replace(episode.session,
                   scene=replace(episode.scene, tag_layout=frozen_layout)))


def _failed_item(path: Path, error: Exception, *, name: str | None = None) -> dict:
    result = dict(path=str(path), name=path.name if name is None else name, status='failed', error=str(error))
    if isinstance(error, batches.BatchInputError):
        result['details'] = error.problems
    return result


@recompute_when_forced
def run(paths: list[Path], *, jobs: int = 1, force: bool = False, cores: str = '', dry_run: bool = False,
        check_recordings: bool = True, map_version: str | None = None,
        map_epoch: str | None = None,
        intrinsics_source: str | None = None, overrides: SceneOverrides | None = None) -> list[dict]:
    if jobs < 1:
        raise ValueError('jobs 必须至少为 1')
    batch_roots = layout.configured_batches(paths)
    summaries, ready = [], []
    previews = {} if dry_run else None
    for batch in batch_roots:
        try:
            imported = batches.prepare_for_run(batch, dry_run=dry_run,
                                        intrinsics_source=intrinsics_source, overrides=overrides,
                                        previews=previews, map_epoch=map_epoch)
            registered = batches.episode_paths([batch])
            new_count = len(imported['episodes'])
            total = len({str(path) for path in registered} | {item['path'] for item in imported['episodes']})
            console.write(
                f'原片 {imported.get("total_scanned", 0)} 段 · '
                f'{"拟登记" if dry_run else "新登记"} {new_count} 个 · 共 {total} 个任务',
                recording=batch.name,
                state='预览' if dry_run else '登记',
            )
            if dry_run:
                for item in imported['episodes']:
                    summaries.append(_preview_item(Path(item['path']), previews, item=item,
                                                   check_recordings=check_recordings, force=force))
                # 已登记任务可能全部被入库复用，仍列出处理计划。
                planned = {item['path'] for item in imported['episodes']}
                for path in registered:
                    if str(path) not in planned:
                        summaries.append(_preview_item(path, previews, check_recordings=check_recordings, force=force))
            else:
                ready.extend(registered)
            if not imported['episodes'] and not registered:
                raise ValueError(f'没有任务原片，请放入 {batch}/take_001/left/ 或 right/')
        except ITEM_FAILURES as error:
            if isinstance(error, batches.BatchInputError):
                console.write(f'原片存在 {len(error.problems)} 项问题，详见汇总',
                                       recording=batch.name, state='失败')
            summaries.append(_failed_item(batch, error))
    # 已登记批次和显式任务路径继续使用相同的处理引擎。
    direct = batches.episode_paths(paths, exclude=batch_roots)
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
            imported = batches.prepare_for_run(batch, dry_run=dry_run,
                                        intrinsics_source=intrinsics_source, overrides=overrides,
                                        previews=previews, map_epoch=map_epoch)
            if dry_run:
                new = {Path(item['path']): item for item in imported['episodes']}
                for path in selected:
                    if path in new or path.is_file():
                        summaries.append(_preview_item(path, previews, item=new.get(path),
                                                       check_recordings=check_recordings, force=force))
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
                    recordings.require_calibration_source(session.kit, intrinsics_source)
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
        summaries.extend(run_episodes(ready, jobs=jobs, force=force, cores=cores, dry_run=dry_run,
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


def _preview_item(path: Path, previews: dict[Path, Episode | Exception] | None, *, item: dict | None = None,
                  check_recordings=True, force=False) -> dict:
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
            planned.update(backend_preview(episode, force=force))
            if check_recordings:
                preflight_recordings(episode, console.child(recording=f'{root.parent.name}/{root.name}'),
                                              enabled=True, dry_run=True)
        except ITEM_FAILURES as error:
            return _failed_item(path, error, name=root.name)
    return planned
