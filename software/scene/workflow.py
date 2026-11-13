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

from ruamel.yaml.error import YAMLError

from software import provenance as artifact_identity
from software.console import Console
from software.scene import layout, scheduling
from software.scene.configuration import SceneOverrides
from software.scene.layout import locked
from software.scene.management import camera_assets
from software.scene.management import cleanup, episodes, inputs
from software.scene.models import Episode
from software.scene.models import EpisodeRunResult
from software.scene.result_io import scene_reference
from software.scene.pipeline import map_lifecycle, reconstruction, fallback, reporting
from software.scene.spatial.fusion import SceneFusionQualityError
from software.scene.pipeline import recording_checks, results, measurements, cache_audit
from software.scene.pipeline import synchronization
from software.storage import publish_link, write_json
from software.vision import runtime


console = Console('场景')


def record_failure(episode: Episode, directory: Path, reason: str, *, identity: dict,
                   status='failed', error: Exception | None = None) -> None:
    """同一输入和原因只保留一份失败现场；成功运行不参与去重。"""
    key = artifact_identity.canonical_hash(dict(inputs=identity, reason=reason, status=status))[:24]
    target = directory.parent / ('failed-' + key)
    write_json(directory / 'error.json', dict(status=status, reason=reason,
        traceback=''.join(traceback.format_exception(error)) if error else '', input_hash=artifact_identity.canonical_hash(identity),
        inputs=identity, batch=inputs.read_yaml(episode.session.path)))
    if target.exists():
        shutil.rmtree(directory)
    else:
        directory.rename(target)
    episodes.record_attempt(episode, status, target)
    cleanup.after_task(episode.root)


def run_episode(path: Path, *, force=False, cores='', dry_run=False, check_recordings=True,
                map_version: str | None = None, overrides: SceneOverrides | None = None,
                parallel_arms: bool = True, batch_map: map_lifecycle.BatchMap | None = None) -> EpisodeRunResult:
    """预检并固定地图 → 准备录像 → 各手独立求解 → 核对输入 → 发布结果。

    锁、缓存复用、失败记录与成功指针由本层管理；算法与结果写入分别
    交给 reconstruct_arm_in_scene() 和 results。失败保留已有成功入口。
    """
    episode = inputs.read_episode(path, overrides=overrides)
    if episode.synchronization.get('method') == 'audio_auto':
        sync_path = layout.metadata(episode.root, 'sync.json')
        raw = synchronization.estimate_audio_sync(episode.recordings,
            output=None if dry_run else sync_path)
        raise ValueError(
            f'audio_auto 已生成原始音频偏移 {raw["raw_audio_offset_s"]}，'
            f'记录在 {sync_path}；完成 recording-mode A/V 修正后才能运行轨迹抽取')
    video_offsets_s = episode.offsets()
    name = episode.root.name
    requested_map_version = map_version
    if episode.map_version is not None:
        if requested_map_version is not None and requested_map_version != episode.map_version:
            raise ValueError('命令行 map_version 与 task.yaml 的 map_version 不一致')
        requested_map_version = episode.map_version

    # --- 阶段 0: 原片录像预检 (Preflight Check) ---
    check_report = recording_checks.check_episode(episode) if check_recordings else None
    if check_report is not None and not check_report.passed:
        message = '原片检查未通过：' + '；'.join(check_report.reasons)
        if not dry_run:
            with locked(episode.root):
                run_dir = layout.metadata(episode.root, 'runs') / uuid.uuid4().hex
                write_json(run_dir / 'preflight_report.json', asdict(check_report))
                record_failure(episode, run_dir, message, status='rejected_preflight',
                    identity=dict(episode=artifact_identity.file_hash(episode.path), session=artifact_identity.file_hash(episode.session.path),
                                  preflight=asdict(check_report)))
        raise recording_checks.RecordingCheckError(message)

    with nullcontext() if dry_run else locked(episode.root):
        # --- 阶段 1: 地图版本绑定与输入指纹比对 (Bind Map & Verify Cache) ---
        if batch_map is not None and batch_map.requested != requested_map_version:
            raise ValueError('批量运行中的任务需要不同地图版本，请按地图版本分组运行')
        episode, map_dir, map_manifest = (batch_map.bind(episode) if batch_map is not None else
            map_lifecycle.bind_map(episode, requested_map_version, dry_run, cores=cores))
        scene = episode.scene
        device_scene = replace(
            scene,
            devices=episode.session.devices,
            rear_calibrations=camera_assets.rear_files(episode.session.devices),
        )
        cameras = camera_assets.select_cameras(episode.recordings, device_scene, episode.serials)
        map_lifecycle.validate_map_calibrations(cameras, map_manifest)
        input_identity = inputs.episode_inputs(episode, cameras, map_manifest)

        if dry_run:
            return dict(status='planned', inputs=input_identity, output=str(episode.output))

        runtime.validate_engines(dict(os.environ))
        for arm in cameras:
            for lens in scene.localization_lenses(arm):
                if scene.role(arm, lens) not in map_manifest['maps']:
                    raise ValueError(f'场景地图缺少 {arm}/{lens} 可定位的视角')

        # 检查是否可秒级复用已有结果
        local_output = episode.root / 'local_results'
        candidates = [(episode.output, 'umi_scene_episode', 'skipped')]
        if len(episode.expected_arms) == 1:
            candidates.append((local_output, 'umi_local_episode', 'local_skipped'))
            if layout.read_state(episode.root).get('attempt', {}).get('status') in ('local_ok', 'local_skipped'):
                candidates.reverse()
        for output, kind, status in candidates if not force else ():
            previous = output / 'episode.json'
            if not previous.is_file():
                continue
            old = json.loads(previous.read_text())
            if old.get('kind') == kind and old['inputs'] == input_identity:
                reference = scene_reference(old)
                inputs.validate_assets(output.resolve(), old['assets'])
                episodes.record_attempt(episode, status, output.resolve())
                cleanup.after_task(episode.root)
                return dict(status=status, output=str(output), frame_id=reference['frame_id'])

        # --- 阶段 2: 独立计算隔离运行 (Isolated Run) ---
        run_dir = layout.metadata(episode.root, 'runs') / uuid.uuid4().hex
        run_dir.mkdir(parents=True)
        # 解码产物从缓存链接到运行期工作目录，结束即删；发布结果只保留自包含文件。
        work_dir = layout.metadata(episode.root, 'work') / run_dir.name
        started, started_at = time.monotonic(), time.time()
        cache_events = cache_audit.CacheAudit()

        try:
            if check_report is not None:
                write_json(run_dir / 'preflight_report.json', asdict(check_report))

            cameras = camera_assets.snapshot_cameras(cameras, run_dir)
            if scene.tag_layout is not None:
                from software.vision.tag_board import load as load_tag_board
                frozen_layout = run_dir / 'devices/environment_poster.json'
                frozen_layout.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(scene.tag_layout, frozen_layout)
                if load_tag_board(frozen_layout).asset_hash != input_identity['environment_tags']['asset_hash']:
                    raise ValueError('海报布局在预检后发生变化，请重新运行')
                episode = replace(episode, session=replace(episode.session,
                    scene=replace(episode.scene, tag_layout=frozen_layout)))
            # Step 1: 准备录像与公共时间区间
            console.write('准备任务录像与时间轴', recording=name, step=(1, 3))
            allocation = scheduling.arm_cores(tuple(episode.recordings), cores, parallel=parallel_arms)
            prepared_recordings = reconstruction.prepare_episode_recordings(episode, cameras, video_offsets_s,
                work_dir=work_dir, cores=cores, arm_cores=allocation, cache_audit=cache_events)
            display_map_path = map_dir / 'map.json'
            scene_metadata = dict(
                name=scene.name,
                version=map_manifest['version'],
                frame_id=map_manifest['frame_id'],
                session=episode.session.name,
            )

            # Step 2: 各手独立求解轨迹与夹爪宽度
            tracks = {}
            widths = {}
            distances = {}
            def run_arm(arm):
                recording = prepared_recordings[arm]
                arm_scene = scene_metadata
                measurements.prepare_detections(episode, arm, recording)
                console.write(f'{arm}：已有场景地图定位 → 固定场景系融合', recording=name, step=(2, 3))
                try:
                    fusion_result = reconstruction.reconstruct_arm_in_scene(
                        episode, arm, recording, map_dir, map_manifest,
                        diagnostic_dir=run_dir / arm, cache_audit=cache_events)
                except SceneFusionQualityError as shared_error:
                    if len(episode.expected_arms) != 1 or len(prepared_recordings) != 1:
                        raise
                    console.write(f'{arm}：{shared_error}；自动回退到独立局部轨迹', recording=name, state='回退')
                    write_json(run_dir / 'fallback.json', dict(shared_failure=str(shared_error), shared_scene=scene_metadata))
                    try:
                        fusion_result = fallback.reconstruct(episode, arm, recording, run_dir / arm / 'independent')
                    except Exception as local_error:
                        raise ValueError(f'共享场景失败：{shared_error}；独立回退失败：{local_error}') from local_error
                    arm_scene = dict(name=scene.name, session=episode.session.name, version=None,
                        frame_id=f'local:{run_dir.name}:{arm}', scope='local',
                        fallback=dict(shared_failure=str(shared_error), shared_frame_id=map_manifest['frame_id']))
                measured = measurements.measure(episode, arm, recording, video_offsets_s[arm], run_dir / arm)
                track = results.write_arm(
                    arm, recording, fusion_result, video_offsets_s[arm], arm_scene, run_dir / arm,
                    width=measured.width, tag_distance=measured.tag_distance,
                    export_auxiliary_trajectories=scene.parameters.export_auxiliary_trajectories,
                    localization_lenses=('front',) if arm_scene.get('scope') == 'local' else scene.localization_lenses(arm),
                )
                return arm, track, measured, arm_scene

            if allocation:
                console.write('左右手并行重建', recording=name, step=(2, 3))
                with cf.ThreadPoolExecutor(max_workers=len(prepared_recordings), thread_name_prefix='scene-arm') as executor:
                    arm_results = list(executor.map(run_arm, prepared_recordings))
            else:
                arm_results = [run_arm(arm) for arm in prepared_recordings]
            for arm, track, measured, arm_scene in arm_results:
                tracks[arm] = track
                if measured.width is not None:
                    widths[arm] = measured.width.report
                if measured.tag_distance is not None:
                    distances[arm] = measured.tag_distance.report
                if arm_scene.get('scope') == 'local':
                    scene_metadata, display_map_path = arm_scene, None

            # Step 3: 双手对齐、防篡改校验与原子发布
            console.write('写入轨迹、报告并发布结果', recording=name, step=(3, 3))
            common_interval_s = results.write_bimanual(tracks, run_dir)

            # 安装资产已冻结；结束时只核对原片身份，不重新解析配置和校验标定。
            current_recordings = {arm: inputs.recording_identity(video) for arm, video in episode.recordings.items()}
            if input_identity['recordings'] != current_recordings:
                raise ValueError('任务期间录像发生变化，拒绝发布')
            epoch_plan = layout.metadata(episode.session.root, 'map_epochs.yaml')
            current_epoch_plan = artifact_identity.file_hash(epoch_plan) if epoch_plan.is_file() else None
            if input_identity.get('map_epoch_plan_sha256') != current_epoch_plan:
                raise ValueError('地图 epoch 清单在任务期间发生变化，拒绝发布')
            if batch_map is not None:
                batch_map.validate(check_configuration=False)

            reporting.write_run_report(
                run_dir,
                episode=episode,
                map_manifest=map_manifest,
                input_identity=input_identity,
                tracks=tracks,
                video_offsets_s=video_offsets_s,
                scene_record=scene_metadata,
                gripper_width=widths,
                tag_distance=distances,
                cache_audit=cache_events,
            )
            results.write_episode(
                episode,
                tracks,
                run_dir,
                common_interval_s=common_interval_s,
                scene_record=scene_metadata,
                input_identity=input_identity,
                video_offsets_s=video_offsets_s,
                elapsed_s=time.monotonic() - started,
                map_path=display_map_path,
                gripper_width=widths,
                tag_distance=distances,
            )
            # 原子更新结果指针：创建 results 软链接指向本次 run_dir
            is_local = scene_metadata.get('scope') == 'local'
            output = local_output if is_local else episode.output
            publish_link(run_dir, output)
            episodes.record_attempt(episode, 'local_ok' if is_local else 'ok', run_dir)
            cleanup.after_task(episode.root, used_since=started_at)
        except Exception as error:
            record_failure(episode, run_dir, str(error), identity=input_identity, error=error)
            raise
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            with suppress(OSError):
                work_dir.parent.rmdir()

        return dict(
            status='local_ok' if is_local else 'ok',
            output=str(output),
            frame_id=scene_metadata['frame_id'],
            common_interval_s=common_interval_s,
        )


def run_episodes(paths: list[Path], *, jobs: int = 1, force=False, cores='', dry_run=False, check_recordings=True, map_version: str | None = None, overrides: SceneOverrides | None = None) -> list[dict]:
    """批量处理任务；支持多任务并发调度与幂等断点续跑 (resume)。"""
    if jobs < 1:
        raise ValueError('--jobs 必须 >= 1')
    resolved = [p.resolve() for p in paths]
    if not resolved:
        return []

    console.write(f'共 {len(resolved)} 个任务；并发数 {jobs}')
    batch_maps = {}
    for path in resolved:
        batch_root = layout.owner(path).parent
        task_map_version = inputs.read_episode(path, verify=False).map_version
        requested = map_version
        if task_map_version is not None:
            if requested is not None and requested != task_map_version:
                raise ValueError('命令行 map_version 与 task.yaml 的 map_version 不一致')
            requested = task_map_version
        batch_maps.setdefault((batch_root, requested), map_lifecycle.BatchMap(requested, dry_run, cores=cores))

    def _execute(path: Path) -> dict:
        name = layout.owner(path).name
        try:
            episode_result = run_episode(path, force=force, cores=cores, dry_run=dry_run, check_recordings=check_recordings, map_version=map_version,
                                         overrides=overrides, parallel_arms=jobs == 1,
                                         batch_map=batch_maps[(layout.owner(path).parent,
                                                               inputs.read_episode(path, verify=False).map_version or map_version)])
            return dict(path=str(path), name=name, **episode_result)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, YAMLError) as error:
            if not dry_run and path.is_file():
                try:
                    episode = inputs.read_episode(path, verify=False)
                    state = layout.read_state(layout.owner(path))
                    if state.get('attempt', {}).get('status') not in ('rejected_preflight', 'failed'):
                        episodes.record_attempt(episode, 'failed', None, reason=str(error))
                except (OSError, ValueError, KeyError, RuntimeError, YAMLError):
                    pass
            console.write(f'{name}：{error}', recording=name, state='失败')
            return dict(path=str(path), name=name, status='failed', error=str(error))

    if jobs == 1 or len(resolved) == 1:
        return [_execute(p) for p in resolved]
    with cf.ThreadPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(_execute, resolved))


def format_batch_summary(summaries: list[dict]) -> str:
    """生成整洁易读的批量处理汇总表格。"""
    lines = [
        '',
        '=' * 80,
        f'场景任务批量处理结果 (共 {len(summaries)} 个任务)',
        '=' * 80,
        f'{"任务名":<15} {"状态":<10} {"有效时长":<15} {"详细信息/输出目录"}',
        '-' * 80,
    ]
    ok_count = 0
    skipped_count = 0
    failed_count = 0
    local_count = 0
    for summary_item in summaries:
        name = summary_item.get('name', Path(summary_item['path']).parent.name)
        status = summary_item.get('status', 'unknown').upper()
        if status == 'OK':
            ok_count += 1
            interval = summary_item.get('common_interval_s')
            duration_str = f'{interval[1] - interval[0]:.2f}s' if interval else '-'
            detail = summary_item.get('output', '')
        elif status in ('LOCAL_OK', 'LOCAL_SKIPPED'):
            local_count += 1
            interval = summary_item.get('common_interval_s')
            duration_str = f'{interval[1] - interval[0]:.2f}s' if interval else '-'
            detail = f'独立局部坐标（不与场景共享）: {summary_item.get("output", "")}'
        elif status == 'SKIPPED':
            skipped_count += 1
            duration_str = '-'
            detail = f'已完成复用: {summary_item.get("output", "")}'
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
