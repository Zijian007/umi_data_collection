"""Public scene task workflow and batch scheduling."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext, suppress
from pathlib import Path
from threading import Lock
import os
import time
from umi.infra.concurrency import daemon_map
from umi.infra.terminal import Console
from umi.store import layout
from umi.workflows.scene.errors import ITEM_FAILURES, MapCompatibilityError, fallback_record
from umi.store.layout import locked
from umi.workflows.scene import episodes
from umi.workflows.scene import camera_assets
from umi.workflows.scene.inputs.batch import read_episode, episode_inputs
from umi.store.models import Episode, EpisodeRunResult, CalibratedCamera, TaskInputs
from umi.workflows.scene.parameters import SceneOverrides
from umi.workflows.scene.maps import repository as map_repository
from umi.workflows.scene.tasks import publication, backend_routing as estimation
from umi.perception import runtime
from dataclasses import asdict, dataclass, replace
import shutil
import uuid
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import recompute_when_forced, write_json
from umi.store.result_io import validate_assets, is_local_scene, is_poster_scene
from umi.store.cache import CacheAudit
from umi.workflows.scene import recordings, synchronization
from umi.workflows.scene.tasks.publication import RunAudit
from umi.perception.audio.specialist_audio import DEFAULT_AUDIO_SYNC_ALGORITHM


console = Console('场景')


@recompute_when_forced
def run_episode(path: Path, *, force=False, cores='', dry_run=False, check_recordings=True,
                map_version: str | None = None, overrides: SceneOverrides | None = None,
                parallel_arms: bool = True, batch_map: map_repository.BatchMap | None = None,
                map_epoch: str | None = None) -> EpisodeRunResult:
    """Resolve a task, reuse its published result or execute a new run."""
    episode = read_episode(path, overrides=overrides, map_epoch_override=map_epoch)
    requested = requested_map_version(episode, map_version)
    return _run_episode(episode, requested, batch_map, force=force, cores=cores, dry_run=dry_run,
                        check_recordings=check_recordings, parallel_arms=parallel_arms)


def _run_episode(episode: Episode, requested_map: str | None, batch_map: map_repository.BatchMap | None, *,
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
                                            batch_map=batch_map, dry_run=dry_run, cores=cores,
                                            video_offsets_s=video_offsets_s, check_report=check_report,
                                            synchronization_audit=(sync_cache_status, sync_cache_reason))
        episode = resolved.episode
        if dry_run:
            return dict(status='planned', inputs=resolved.identity, output=str(episode.output),
                        synchronization=episode.synchronization)

        reused = publication.reuse_episode(episode, resolved.identity, force=force)
        if reused is not None:
            task_console.write('已复用匹配的轨迹结果', state='复用')
            return reused

        if episode.scene.reconstruction_mode != 'poster_vio':
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
            task_episode = read_episode(path, overrides=overrides, map_epoch_override=map_epoch)
            requested = requested_map_version(task_episode, map_version)
            binding = None
            if task_episode.scene.reconstruction_mode in ('auto', 'scene_map'):
                key = (task_episode.session.root, requested)
                if key not in batch_maps:
                    batch_maps[key] = map_repository.BatchMap(requested, dry_run, cores=cores)
                binding = batch_maps[key]
            valid.append((path, task_episode, requested, binding))
        except ITEM_FAILURES as error:
            name = path.parent.name
            console.write(str(error), recording=name, state='失败')
            rejected.append(dict(path=str(path), name=name, status='failed', error=str(error)))
            report_completion(name)

    def _execute(path: Path, task_episode: Episode, requested: str | None,
                 batch_map: map_repository.BatchMap | None) -> dict:
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
                    state = layout.read_state(layout.owner(path))
                    if state.get('attempt', {}).get('status') not in ('rejected_preflight', 'failed'):
                        episodes.record_attempt(task_episode, 'failed', None, reason=str(error))
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
            lines.append('    已列入处理计划')
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
    map_selection: estimation.MapSelection
    identity: dict
    video_offsets_s: dict[str, float]
    check_report: recordings.EpisodeCheckReport | None
    synchronization_audit: tuple[str, str]
    batch_map: map_repository.BatchMap | None


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
    algorithm = episode.synchronization.get('algorithm', DEFAULT_AUDIO_SYNC_ALGORITHM)
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


def requested_map_version(episode: Episode, command_map_version: str | None) -> str | None:
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
            audit = RunAudit(layout.metadata(episode.root, 'logs') / run_dir.name,
                             task=episode.root.name, session=episode.session.root.name,
                             input_identity=identity, run_directory=run_dir)
            audit.event('preflight', status='rejected', reasons=check_report.reasons)
            retained = publication.record_failure(episode, run_dir, message, status='rejected_preflight',
                identity=identity)
            audit.finish('rejected_preflight', error=recordings.RecordingCheckError(message),
                         directory=retained)
            task_console.write(f'预检报告 {retained / "preflight_report.json"}', state='诊断')
    raise recordings.RecordingCheckError(message)


def _require_map_views(scene, cameras: dict, map_manifest: dict) -> None:
    for arm in cameras:
        for lens in scene.localization_lenses(arm):
            if scene.role(arm, lens) not in map_manifest['maps']:
                raise MapCompatibilityError(f'场景地图缺少 {arm}/{lens} 可定位的视角')


def resolve_inputs(episode: Episode, task_console: Console, *, map_version: str | None,
                   batch_map: map_repository.BatchMap | None, dry_run: bool, cores: str,
                   video_offsets_s: dict[str, float], check_report: recordings.EpisodeCheckReport | None,
                   synchronization_audit: tuple[str, str]) -> ResolvedInputs:
    """Bind the map, select cameras and describe inputs while holding the task lock."""
    if batch_map is not None and batch_map.requested != map_version:
        raise ValueError('批量运行中的任务需要不同地图版本，请按地图版本分组运行')
    mode = episode.scene.reconstruction_mode
    episode, map_selection = estimation.select_map(episode, map_version,
        batch_map=batch_map, dry_run=dry_run, cores=cores)
    map_manifest = map_selection.manifest
    scene = episode.scene
    device_scene = replace(
        scene,
        devices=episode.session.devices,
        rear_calibrations=camera_assets.rear_files(episode.session.devices),
    )
    cameras = camera_assets.select_cameras(episode.recordings, device_scene, episode.serials)
    if map_manifest is not None:
        try:
            map_repository.validate_map_calibrations(cameras, map_manifest)
            _require_map_views(scene, cameras, map_manifest)
        except MapCompatibilityError as error:
            if mode != 'auto':
                raise
            map_selection = estimation.MapSelection.without_map(fallback_record(error))
            map_manifest = None
    input_identity = episode_inputs(episode, cameras, map_manifest)

    map_text = (f'地图 {map_manifest["frame_id"]}' if map_manifest is not None else '无预建场景地图')
    camera_text = ' · '.join(f'{arm} {camera.serial}' for arm, camera in cameras.items())
    task_console.write(f'{map_text} · 相机 {camera_text}', state='输入就绪')
    return ResolvedInputs(episode=episode, cameras=cameras, map_selection=map_selection,
                          identity=input_identity, video_offsets_s=video_offsets_s,
                          check_report=check_report, synchronization_audit=synchronization_audit,
                          batch_map=batch_map if mode == 'scene_map' else None)


def execute_episode(resolved: ResolvedInputs, task_console: Console, *, cores: str,
                    parallel_arms: bool, task_started: float) -> EpisodeRunResult:
    """Freeze assets and run reconstruction while the caller holds the task lock."""
    episode, cameras = resolved.episode, resolved.cameras
    input_identity, map_selection = resolved.identity, resolved.map_selection
    video_offsets_s = resolved.video_offsets_s
    map_dir, map_manifest = map_selection.directory, map_selection.manifest
    mode = episode.scene.reconstruction_mode
    sync_cache_status, sync_cache_reason = resolved.synchronization_audit
    check_record = asdict(resolved.check_report) if resolved.check_report is not None else None

    run_dir = layout.metadata(episode.root, 'runs') / uuid.uuid4().hex
    run_dir.mkdir(parents=True)
    # 解码产物从缓存链接到运行期工作目录，结束即删；发布结果只保留自包含文件。
    work_dir = layout.metadata(episode.root, 'work') / run_dir.name
    started, started_at = time.monotonic(), time.time()
    cache_audit = _create_cache_audit(episode, cameras, sync_cache_status, sync_cache_reason)
    audit = RunAudit(layout.metadata(episode.root, 'logs') / run_dir.name,
                     task=episode.root.name, session=episode.session.root.name,
                     input_identity=input_identity, run_directory=run_dir)
    run_console = task_console.child(line_sink=audit.write_line, event_sink=audit.event)
    audit.event('stage_started', stage='isolated_run')
    audit.event(
        'inputs_resolved',
        mode=mode,
        map_directory=map_dir,
        map_version=(map_manifest or {}).get('version') if map_manifest else None,
        synchronization_status=sync_cache_status,
        synchronization_reason=sync_cache_reason,
        preflight=check_record,
    )

    try:
        if check_record is not None:
            write_json(run_dir / 'preflight_report.json', check_record)

        cameras = camera_assets.snapshot_cameras(cameras, run_dir)
        episode = _freeze_tag_layout(episode, run_dir, input_identity)
        # Step 1: 准备录像与公共时间区间
        audit.event('stage_started', stage='prepare_recordings')
        run_console.write('准备录像和时间轴', step=(1, 3))
        arm_cpu_sets = recordings.arm_cores(tuple(episode.recordings), cores, parallel=parallel_arms)
        with run_console.heartbeat('准备录像和时间轴', step=(1, 3)):
            prepared_recordings = recordings.prepare_episode_recordings(episode, cameras, video_offsets_s,
                work_dir=work_dir, cores=cores, arm_cores=arm_cpu_sets, cache_audit=cache_audit)
        common_start, common_end = recordings.shared_interval(prepared_recordings, video_offsets_s)
        run_console.write(
            ' · '.join(
                f'{arm} {len(recording.video_times_s)} 帧 / {recording.video_times_s[-1] - recording.video_times_s[0]:.2f}s'
                for arm, recording in prepared_recordings.items())
            + f' · {"共同时间" if len(prepared_recordings) > 1 else "时间范围"} {common_start:.3f}–{common_end:.3f}s',
            step=(1, 3), state='完成',
        )
        # Step 2: 各手独立求解轨迹、夹爪测量与共享海报融合
        audit.event('stage_finished', stage='prepare_recordings')
        audit.event('stage_started', stage='estimate_tracks')
        task_inputs = TaskInputs(episode, prepared_recordings, video_offsets_s,
                                 run_dir, arm_cpu_sets, cache_audit, run_console)
        with run_console.heartbeat('轨迹重建', step=(2, 3)):
            arm_execution = estimation.execute(task_inputs, map_selection)
        tracks = arm_execution.tracks
        coordinate_record = arm_execution.coordinate_record
        backend_names = sorted({
            track.metadata.get('fusion', {}).get('reconstruction_backend')
            for track in tracks.values()
            if track.metadata.get('fusion', {}).get('reconstruction_backend')
        })
        selected = next((attempt['backend'] for attempt in coordinate_record.get('attempts', [])
                         if attempt['status'] == 'succeeded'), mode)
        backend_text = ('独立局部重建' if is_local_scene(coordinate_record)
                        else ', '.join(backend_names) or selected)
        fallback_text = ' · 已回退' if coordinate_record.get('fallback') else ''
        run_console.write(
            f'{backend_text} · 坐标 {coordinate_record.get("frame_id", "unknown")}{fallback_text}',
            step=(2, 3), state='完成',
        )
        # 局部回退结果不引用共享地图。
        display_map_path = (None if is_local_scene(coordinate_record) or is_poster_scene(coordinate_record)
                            else map_dir / 'map.json')
        if mode == 'auto' and display_map_path is not None:
            validate_assets(map_dir, map_manifest['assets'])

        audit.event('stage_finished', stage='estimate_tracks')
        # Step 3: 双手对齐、防篡改校验与原子发布
        audit.event('stage_started', stage='publish')
        run_console.write('发布轨迹和报告', step=(3, 3))
        published = publication.publish_episode(
            replace(resolved, episode=episode, cameras=cameras),
            result=arm_execution,
            run_dir=run_dir,
            map_path=display_map_path,
            cache_audit=cache_audit,
            preflight=check_record,
            started=started,
            started_at=started_at,
        )
        is_local = published.is_local
        output = published.output
        audit.event('stage_finished', stage='publish', output=output)
        run_console.write(
            f'{"局部轨迹已发布" if is_local else "轨迹已发布"} · 耗时 {time.monotonic() - task_started:.1f}s',
            step=(3, 3), state='完成')
        audit.finish('ok', output=output)
        task_console.write(str(audit.directory / 'run.log'), state='日志')
    except KeyboardInterrupt as error:
        retained = publication.record_failure(episode, run_dir, '处理被用户中断',
                                              identity=input_identity, status='interrupted', error=error)
        run_console.write(f'运行日志 {audit.directory / "run.log"}', state='诊断')
        audit.finish('interrupted', error=error, directory=retained)
        raise
    except Exception as error:
        retained = publication.record_failure(episode, run_dir, str(error), identity=input_identity, error=error)
        run_console.write(f'运行日志 {audit.directory / "run.log"}', state='诊断')
        audit.finish('failed', error=error, directory=retained)
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        with suppress(OSError):
            work_dir.parent.rmdir()

    common_interval_s = published.common_interval_s
    return dict(
        status='local_ok' if is_local else 'ok',
        output=str(output),
        frame_id=coordinate_record['frame_id'],
        common_interval_s=common_interval_s,
        synchronization=episode.synchronization,
        quality=published.quality,
    )


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
    from umi.perception.poster.layout import load as load_tag_board
    frozen_layout = run_dir / 'devices/environment_poster.json'
    frozen_layout.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(episode.scene.tag_layout, frozen_layout)
    if load_tag_board(frozen_layout).asset_hash != input_identity['environment_tags']['asset_hash']:
        raise ValueError('海报布局在预检后发生变化，请重新运行')
    return replace(episode, session=replace(episode.session,
                   scene=replace(episode.scene, tag_layout=frozen_layout)))
