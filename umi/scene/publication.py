"""场景轨迹生成、结果发布与复用。"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from umi.core.frame import geometry_record
from umi.core.geometry import (
    camera_from_tcp,
    imu_from_camera,
    relative_pose_series,
    resample_pose_series,
    tcp_frame_definition,
)
from umi.infra import provenance as artifact_identity
from umi.infra.concurrency import daemon_map
from umi.infra.file_store import publish_link, read_yaml, write_json
from umi.perception.gripper.width_estimation import SOURCE_NAMES
from umi.perception.gripper.width_estimation import WidthResult
from umi.reconstruction.estimation.constraint_quality import position_support
from umi.reconstruction.estimation.models import BackendResult, FusedTrajectory
from umi.scene import cleanup, task_state
from umi.scene.collection.configuration import SceneOverrides
from umi.scene.inputs.identity import stored_reuse_identity
from umi.store import layout
from umi.store.models import ArmExecutionResult, ArmTrajectory, Episode, EpisodeRunResult, PreparedRecording, TaskInputs
from umi.store.result_io import (
    SCENE_RESULT_VERSIONS,
    artifact_hashes,
    is_local_scene,
    is_poster_scene,
    quality_summary,
    read_result,
    recording_identity,
    result_paths,
    scene_reference,
    uses_scene_map,
    validate_result,
)

if TYPE_CHECKING:
    from umi.scene.run import ResolvedInputs


@dataclass(frozen=True)
class PublicationResult:
    output: Path
    is_local: bool
    common_interval_s: list[float]
    quality: dict


def _verify_frozen_inputs(episode, input_identity: dict, *, batch_map, map_attempt_episode=None) -> None:
    """Reject publication if source inputs changed after reconstruction started."""
    current_recordings = {
        arm: recording_identity(video) for arm, video in episode.recordings.items()
    }
    if input_identity['recordings'] != current_recordings:
        raise ValueError('任务期间录像发生变化，拒绝发布')

    if episode.scene.reconstruction_mode == 'scene_map' or input_identity.get('auto_map_attempted'):
        from umi.scene.inputs.identity import map_epoch_identity

        current_epoch_plan = map_epoch_identity(map_attempt_episode or episode, selected_only='auto_order' in input_identity
                                               or input_identity.get('selected_epoch_identity', False))
        if input_identity.get('map_epoch_plan_sha256') != current_epoch_plan:
            raise ValueError('地图 epoch 清单在任务期间发生变化，拒绝发布')

    if batch_map is not None:
        batch_map.validate(check_configuration=False)


def publish_episode(resolved: ResolvedInputs, *, result: ArmExecutionResult,
                    run_dir: Path, map_path: Path | None, cache_audit, preflight: dict | None,
                    started: float) -> PublicationResult:
    """Write reports, atomically publish the result, and record task state."""
    episode, input_identity = resolved.episode, resolved.identity
    common_interval_s = write_bimanual_trajectory(result.tracks, run_dir)
    elapsed_s = time.monotonic() - started
    _verify_frozen_inputs(
        episode,
        input_identity,
        batch_map=resolved.batch_map,
        map_attempt_episode=resolved.map_attempt_episode,
    )
    quality = quality_summary({arm: track.metadata for arm, track in result.tracks.items()},
                              result.coordinate_record, common_interval_s)
    write_diagnostics(run_dir, result.tracks, cache_audit, resolved.map_selection.directory)
    write_episode_result(
        episode,
        result.tracks,
        run_dir,
        common_interval_s=common_interval_s,
        coordinate_record=result.coordinate_record,
        input_identity=input_identity,
        video_offsets_s=resolved.video_offsets_s,
        elapsed_s=elapsed_s,
        map_path=map_path,
        preflight=preflight,
        gripper_width=result.width_reports,
        tag_distance=result.tag_distance_reports,
    )

    is_local = is_local_scene(result.coordinate_record)
    output = episode.root / 'local_results' if is_local else episode.output
    publish_link(result_paths(run_dir).published, output)
    task_state.record_attempt(episode, 'local_ok' if is_local else 'ok', run_dir)
    return PublicationResult(output=output, is_local=is_local, common_interval_s=common_interval_s, quality=quality)


def reuse_episode(episode: Episode, input_identity: dict, *, force: bool, identify_previous=None) -> EpisodeRunResult | None:
    """Return a valid published result with the same input identity, if one exists."""
    if force:
        return None
    local_output = episode.root / 'local_results'
    candidates = [(episode.output, ('umi_scene_episode', 'umi_poster_episode'), 'skipped')]
    if len(episode.expected_arms) == 1:
        candidates.append((local_output, ('umi_local_episode',), 'local_skipped'))
        if layout.read_state(episode.root).get('attempt', {}).get('status') in ('local_ok', 'local_skipped'):
            candidates.reverse()
    for output, kinds, status in candidates:
        paths = result_paths(output)
        if not paths.manifest.is_file():
            continue
        old = read_result(paths)
        if old.get('kind') not in kinds or old.get('schema_version') not in SCENE_RESULT_VERSIONS:
            continue
        stored = stored_reuse_identity(old.get('inputs') or {})
        current = identify_previous(old) if identify_previous is not None else input_identity.get('reuse_identity')
        if not stored or stored.get('retry_preparation') or stored != current:
            continue
        if 'runtime_overrides' in old:
            try:
                SceneOverrides.from_snapshot(old['runtime_overrides'])
            except ValueError:
                continue
        if old['schema_version'] == 5 and (is_poster_scene(old['scene']) or
                (old['kind'] == 'umi_scene_episode' and episode.scene.scene_map_poster_constraint_enabled)):
            continue
        reference = scene_reference(old)
        validate_result(paths, old)
        task_state.record_attempt(episode, status, paths.run)
        task_state.restore_task_log_link(episode.root)
        cleanup.after_task(episode.root)
        quality = quality_summary(old['arm_results'], old['scene'], old['common_interval_s'])
        return dict(status=status, output=str(output), frame_id=reference['frame_id'],
                    synchronization=old.get('synchronization'), quality=quality)
    return None


def record_failure(episode: Episode, directory: Path, reason: str, *, identity: dict,
                   status='failed', error: BaseException | None = None) -> Path:
    """同一输入和原因只保留一份失败现场；成功运行不参与去重。"""
    # Publication is committed by replacing the result link. An audit or state
    # write can still fail afterward, so keep any directory the link points to.
    if any(output.is_symlink() and result_paths(output).run == directory.resolve()
           for output in (episode.output, episode.root / 'local_results')):
        return directory
    write_json(directory / 'error.json', dict(status=status, reason=reason,
        traceback=''.join(traceback.format_exception(error)) if error else '', input_hash=artifact_identity.canonical_hash(identity),
        inputs=identity, batch=read_yaml(episode.session.path)))
    task_state.record_attempt(episode, status, directory)
    return directory


def write_diagnostics(directory: Path, tracks: dict[str, ArmTrajectory],
                      cache_audit, map_directory: Path | None) -> None:
    """Write full solver evidence once, beside the published package."""
    poster_vio_quality = 'poster_vio/quality.json'
    has_poster_vio_quality = (directory / poster_vio_quality).is_file()
    duplicated = {'frame_provenance'}
    if has_poster_vio_quality:
        duplicated.update(('poster_vio', 'poster_vio_quality'))
    arms = {}
    for arm, track in tracks.items():
        metadata = track.metadata
        fusion = {key: value for key, value in metadata['fusion'].items() if key not in duplicated}
        if has_poster_vio_quality and 'environment_poster' in fusion:
            fusion['environment_poster'] = {
                key: value for key, value in fusion['environment_poster'].items() if key != 'final_reprojection_metrics'}
        arms[arm] = dict(fusion=fusion, localization=metadata.get('localization', {}),
                         frames=f'{arm}/details/frames.csv')
    write_json(directory / 'diagnostics.json', dict(
        schema_version=1, arms=arms,
        cache=dict(task=cache_audit.snapshot() if cache_audit is not None else [],
                   map_manifest=os.path.relpath(map_directory / 'manifest.json', directory)
                   if map_directory is not None else None),
        **({'poster_vio_quality': poster_vio_quality} if has_poster_vio_quality else {}),
    ))


def _published_arm_metadata(metadata: dict) -> dict:
    """Keep playback and quality evidence in the manifest as small summaries."""
    fusion = metadata['fusion']
    keys = ('reconstruction_backend', 'scene_map_priors_used', 'actual_error', 'maximum_visual_gap_s',
            'maximum_position_support_gap_s', 'visual_consistency_rms_m')
    summary = {key: fusion[key] for key in keys if key in fusion}
    if 'independent' in fusion:
        summary['independent'] = {key: fusion['independent'][key]
                                  for key in ('maximum_position_support_gap_s',) if key in fusion['independent']}
    poster = fusion.get('environment_poster')
    if isinstance(poster, dict):
        summary['environment_poster'] = {key: poster[key] for key in ('status', 'final_reprojection_rms_px')
                                         if key in poster}
        if 'position_support' in poster:
            summary['environment_poster']['position_support'] = {
                key: value for key, value in poster['position_support'].items() if key != 'frames'}
    return {**{key: value for key, value in metadata.items() if key not in ('fusion', 'localization')},
            'fusion': summary}


def _stage_publication(directory: Path, arms: dict) -> Path:
    published = directory / 'published'
    data = published / 'data'
    for arm in arms:
        destination = data / arm
        destination.mkdir(parents=True)
        for name in ('trajectory.csv', 'front.mp4'):
            (directory / arm / name).rename(destination / name)
        rear = directory / arm / 'rear.mp4'
        if rear.is_file():
            rear.rename(destination / rear.name)
    if len(arms) == 2:
        (directory / 'bimanual.csv').rename(data / 'bimanual.csv')
    return published


def _format_episode_report(episode: Episode, tracks: dict[str, ArmTrajectory], *,
                           common_interval_s: list[float], coordinate_record: dict,
                           video_offsets_s: dict[str, float], preflight: dict | None,
                           gripper_width: dict, tag_distance: dict) -> str:
    """Summarize the evidence needed for human review of a published task."""
    hand = {'left': '左手', 'right': '右手'}
    local = is_local_scene(coordinate_record)
    poster_frame = is_poster_scene(coordinate_record)
    mapless = not uses_scene_map(coordinate_record)
    backend = ('独立局部重建' if local else
               coordinate_record.get('reconstruction_backend', 'poster_independent' if mapless else 'scene_map'))
    scope = ('本录像局部米制坐标' if local else
             '任务海报米制坐标' if poster_frame else '共享场景米制坐标')
    start, end = common_interval_s
    lines = [
        f'任务：{episode.session.name}/{episode.root.name}',
        f'重建：{backend} · {scope}',
        f'指令：{episode.instruction or "未填写"}',
        f'共同有效时间：{start:.2f}–{end:.2f} s',
    ]
    sync = episode.synchronization
    method = sync['method']
    if method != 'single_camera':
        strategy = {'audio_auto': '自动音频', 'manual': '人工偏移',
                    'assumed_simultaneous': '假设同时开录'}[method]
        if method == 'audio_auto' and sync.get('algorithm'):
            algorithm = sync['algorithm']
            if fallback := sync.get('fallback'):
                algorithm += f' → {fallback["algorithm"]}'
            strategy += f'（{algorithm}）'
        status = '已确认' if sync['confirmed'] else '待确认'
        offsets = '，'.join(f'{hand[arm]} {value:+.3f} s' for arm, value in video_offsets_s.items())
        lines.append(f'同步：{strategy} · {status} · {offsets}')
    if method != 'audio_auto' and sync['note'].strip():
        lines.append(f'同步依据：{sync["note"]}')
    if local:
        lines.append(f'共享场景失败原因：{coordinate_record["fallback"]["shared_failure"]}')

    checks = '跳过' if preflight is None else 'PASS' if preflight['passed'] else 'FAIL'
    lines.append(f'原片检查：{checks}')
    if preflight is not None:
        for arm, video in preflight.get('videos', {}).items():
            lines.extend(f'{hand[arm]} [{check["status"]}] {check["message"]}'
                         for check in video['checks'] if check['status'] != 'PASS')
    trim = coordinate_record.get('recording_trim')
    if trim is not None:
        intervals = []
        for arm, item in trim['arms'].items():
            first, last = item['kept_video_interval_s']
            intervals.append(f'{hand[arm]} {first:.3f}–{last:.3f} s')
        lines.append('首尾裁剪后保留：' + '，'.join(intervals))

    for arm, track in tracks.items():
        metadata = track.metadata
        fusion = metadata['fusion']
        visual = '本录像视觉约束' if mapless else '地图位置约束'
        lines.extend([
            '',
            f'{hand[arm]}：{metadata["duration_s"]:.2f} s · TCP 路程 {metadata["path_length_m"]:.3f} m',
            f'{visual} {metadata["constraint_fraction"]["full"]:.1%} · '
            f'最长无视觉 {fusion["maximum_visual_gap_s"]:.3f} s',
        ])
        if not mapless:
            lines.append(f'视觉一致性 RMS {fusion["visual_consistency_rms_m"] * 1000:.2f} mm（内部指标）')
        poster = fusion.get('environment_poster')
        if poster is not None:
            status = '已融合' if poster['status'] == 'fused' else poster['status']
            rms = poster.get('final_reprojection_rms_px')
            detail = f' · 重投影 RMS {rms:.3f} px' if rms is not None else ''
            lines.append(f'环境海报：{status}{detail}')
            if 'reason' in poster:
                lines.append(f'海报诊断：{poster["reason"]}')
        for lens, recovery in metadata.get('localization', {}).items():
            if recovery['added_frames']:
                lines.append(f'{lens} 反向定位补入 {recovery["added_frames"]} 帧')
        measurements = []
        if arm in tag_distance:
            distance = tag_distance[arm]
            text = f'标签中心间距有效 {distance["valid_fraction"]:.1%}'
            if not distance.get('acceptance', {}).get('passed', True):
                longest = distance['acceptance']['longest_missing_frames']
                text += f'（未通过测距验收，最长缺失 {longest} 帧）'
            measurements.append(text)
        if arm in gripper_width:
            measurements.append(f'开口宽度有效 {gripper_width[arm]["valid_fraction"]:.1%}')
        if measurements:
            lines.append('夹爪测量：' + ' · '.join(measurements))
    accuracy = '轨迹实际误差：未测量。'
    if any(track.metadata.get('cad_status') == 'NOMINAL_CAD_CANDIDATE_UNVERIFIED' for track in tracks.values()):
        accuracy += ' TCP 使用 CAD 名义几何。'
    lines.extend(['', accuracy, '结果清单见 data/episode.json。'])
    return '\n'.join(lines) + '\n'


def write_episode_result(episode: Episode, tracks: dict[str, ArmTrajectory], directory: Path, *,
                  common_interval_s: list[float], coordinate_record: dict, input_identity: dict,
                  video_offsets_s: dict[str, float], elapsed_s: float, map_path: Path | None, preflight: dict | None,
                  gripper_width: dict | None = None, tag_distance: dict | None = None) -> None:
    """写任务报告与资产清单，供发布阶段更新结果指针。"""
    local = is_local_scene(coordinate_record)
    poster_frame = is_poster_scene(coordinate_record)
    mapless = not uses_scene_map(coordinate_record)
    if mapless and poster_frame and map_path is not None:
        raise ValueError('任务海报结果不能引用共享场景地图')
    if local and (len(tracks) != 1 or map_path is not None):
        raise ValueError('局部回退只允许单手且不能引用共享地图')
    if not mapless and map_path is None:
        raise ValueError('场景地图定位结果需要记录地图来源')
    if 'reconstruction_backend' not in coordinate_record:
        backend = 'independent' if local else 'poster_independent' if mapless else 'scene_map'
        coordinate_record = dict(coordinate_record, reconstruction_backend=backend)
    report = _format_episode_report(
        episode, tracks, common_interval_s=common_interval_s, coordinate_record=coordinate_record,
        video_offsets_s=video_offsets_s, preflight=preflight,
        gripper_width=gripper_width or {}, tag_distance=tag_distance or {},
    )
    published = _stage_publication(directory, tracks)
    (published / 'report.txt').write_text(report, encoding='utf-8')
    assets = artifact_hashes(published, ('report.txt', 'data'))
    evidence_assets = artifact_hashes(directory, (*tracks, 'devices', 'poster_vio', 'diagnostics.json',
        'reconstruction.json', 'tag_trim.json', 'fallback.json', 'reconstruction_attempts'))
    preflight_path = directory / 'preflight_report.json'
    primary_times = tracks.get('left', next(iter(tracks.values()))).episode_times_s
    metadata = dict(kind='umi_local_episode' if local else 'umi_poster_episode' if poster_frame and mapless else 'umi_scene_episode',
        schema_version=6 if poster_frame else 5,
        code_revision=artifact_identity.git_revision(),
        status='local_ok' if local else 'ok', name=episode.root.name, session=episode.session.name,
        kit=dict(id=episode.session.kit['id'], arms={arm: episode.session.kit['arms'][arm] for arm in tracks}),
        scene={key: value for key, value in coordinate_record.items() if key != 'session'},
        arms={arm: arm for arm in tracks}, common_interval_s=common_interval_s,
        rows=int(np.count_nonzero((primary_times >= common_interval_s[0]) & (primary_times <= common_interval_s[1]))),
        instruction=episode.instruction, synchronization=episode.synchronization,
        inputs=input_identity, assets=assets, elapsed_s=elapsed_s, actual_error='not_measured',
        tcp_frames={arm: track.tcp_frame for arm, track in tracks.items()},
        arm_results={arm: _published_arm_metadata(track.metadata) for arm, track in tracks.items()},
        evidence_assets=evidence_assets, diagnostics='diagnostics.json', preflight=preflight,
        map=None if mapless else dict(
            path=os.path.relpath(map_path, directory),
            sha256=artifact_identity.file_hash(map_path),
            map_epoch=(episode.map_selection or {}).get('epoch'),
            map_version=episode.map_version,
            selection=episode.map_selection,
        ),
        map_selection=None if mapless else episode.map_selection)
    if gripper_width:
        metadata['gripper_width'] = gripper_width
    if tag_distance:
        metadata['tag_distance'] = tag_distance
    if episode.scene.overrides is not None:
        metadata['runtime_overrides'] = episode.scene.overrides.snapshot()
    write_json(published / 'data/episode.json', metadata)
    preflight_path.unlink(missing_ok=True)


if TYPE_CHECKING:
    from umi.perception.gripper.distance import TagDistanceResult


def write_arm_result(inputs: TaskInputs, arm: str, fusion_result: FusedTrajectory, coordinate_record: dict):
    """Write one solved arm and return its track plus measurement reports."""
    from umi.scene import measurements

    episode = inputs.episode
    scene = episode.scene
    recording = inputs.recordings[arm]
    video_offset_s = inputs.offsets_s[arm]
    directory = inputs.directory / arm
    measured = measurements.measure(episode, arm, recording, video_offset_s)
    track = write_arm_trajectory(
        recording, fusion_result, video_offset_s, coordinate_record, directory,
        width=measured.width, tag_distance=measured.tag_distance,
        localization_lenses=scene.localization_lenses(arm) if uses_scene_map(coordinate_record) else ('front',),
    )
    return (
        track,
        measured.width.report if measured.width is not None else None,
        measured.tag_distance.report if measured.tag_distance is not None else None,
    )


def write_backend_result(inputs: TaskInputs, result: BackendResult, *, parallel=False) -> ArmExecutionResult:
    for arm, trajectory in result.trajectories.items():
        times = result.video_times_s[arm]
        if (len(trajectory.T_reference_imu) != len(times)
                or not np.array_equal(times, inputs.recordings[arm].video_times_s)):
            raise ValueError(f'{arm} 重建结果与候选录像时间轴不一致')
    written = write_task_results(inputs, result.trajectories, result.coordinate_record, parallel=parallel)
    return replace(written, coordinates=result.coordinates)


def write_task_results(inputs: TaskInputs, trajectories: dict[str, FusedTrajectory],
                       coordinate_record: dict, *, parallel=False) -> ArmExecutionResult:
    """Write arms after the backend has accepted their shared coordinate frame."""
    def write(arm):
        return arm, write_arm_result(inputs, arm, trajectories[arm], coordinate_record)

    arms = tuple(trajectories)
    written = daemon_map(write, arms) if parallel else map(write, arms)
    tracks, width_reports, tag_distance_reports = {}, {}, {}
    for arm, (track, width, tag_distance) in written:
        tracks[arm] = track
        if width is not None:
            width_reports[arm] = width
        if tag_distance is not None:
            tag_distance_reports[arm] = tag_distance
    return ArmExecutionResult(tracks, width_reports, tag_distance_reports, coordinate_record)


def write_arm_trajectory(recording: PreparedRecording, result: FusedTrajectory,
              offset_s: float, coordinate_record: dict, directory: Path,
              *, width: WidthResult | None = None, tag_distance: TagDistanceResult | None = None,
              localization_lenses: tuple[str, ...] = ('front',)) -> ArmTrajectory:
    directory.mkdir(parents=True, exist_ok=True)
    details = directory / 'details'
    details.mkdir(exist_ok=True)
    camera = recording.camera
    video_parameters = recording.context.parameters.video
    measured = camera.measurements['front']
    geometry = camera.selected.geometry
    tcp_frame = tcp_frame_definition(geometry)
    T_cam_tcp = camera_from_tcp(geometry)
    tcp_poses = result.T_reference_imu @ imu_from_camera(measured) @ T_cam_tcp
    # 文件时间为统一任务视频时钟。IMU 求解使用的 cam_time_offset 不应叠加两次。
    times = recording.video_times_s + offset_s
    if tag_distance is not None and (len(tag_distance.samples.times_s) != len(times)
            or not np.allclose(tag_distance.samples.times_s, times, rtol=0, atol=1e-8)):
        raise ValueError('标签间距与轨迹时间轴不一致')
    distance_valid = tag_distance.samples.valid if tag_distance is not None else None
    quaternions = Rotation.from_matrix(tcp_poses[:, :3, :3]).as_quat()
    if width is not None:
        width_valid = width.samples.valid
        close_01 = np.clip(1 - width.samples.width_mm / width.report['calibration']['open_width_mm'], 0, 1)
    with (directory / 'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row', 't_video_s', 't_episode_s', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
                        + (['width_mm', 'close_01', 'closed', 'valid', 'source'] if width is not None else [])
                        + (['tag_distance_mm', 'tag_distance_valid', 'tag_distance_source'] if tag_distance else []))
        for index, timestamp in enumerate(times):
            opening = []
            if width is not None:
                valid = bool(width_valid[index])
                measured_mm = width.samples.width_mm[index]
                opening = [f'{measured_mm:.6f}' if valid else '',
                    f'{close_01[index]:.9f}' if valid else '',
                    int(measured_mm <= width.closed_threshold_mm) if valid else '',
                    int(valid), SOURCE_NAMES[int(width.samples.source[index])]]
            distance_columns = []
            if tag_distance is not None:
                valid = bool(distance_valid[index])
                distance_columns = [f'{tag_distance.samples.distance_mm[index]:.6f}' if valid else '',
                                    int(valid), str(tag_distance.samples.source[index])]
            writer.writerow([index, f'{recording.video_times_s[index]:.9f}', f'{timestamp:.9f}',
                *[f'{value:.6f}' for value in tcp_poses[index, :3, 3]],
                *[f'{value:.9f}' for value in quaternions[index]], *opening, *distance_columns])
    support = position_support(recording.video_times_s, result.tracked,
                               result.evidence.get('environment_poster'))
    support_fields = tuple(support)
    frame_fields = ('row', 'video_frame', 'source_video_frame', 't_video_s', 't_episode_s',
                    'constraint', 'visual_lens', 'visual_lenses', 'map_role', 'map_index',
                    'map_epoch', 'map_version', 'map_frame_id', 'front_lens_tracked',
                    *support_fields, 'position_source')
    with (details / 'frames.csv').open('w') as stream:
        stream.write(','.join(frame_fields) + '\n')
        provenance = result.evidence.get('frame_provenance', [])
        for k, time in enumerate(times):
            source_frame = recording.source_frame_start + video_parameters.frame_step * k
            item = provenance[k] if k < len(provenance) else dict(
                constraint='full' if result.tracked[k] else 'imu', visual_lens=None,
                visual_lenses=[], map_role=None, map_index=None, map_epoch=None,
                map_version=None, map_frame_id=None)
            values = [item.get('constraint', 'full' if result.tracked[k] else 'imu'),
                      item.get('visual_lens') or '', '|'.join(item.get('visual_lenses') or []),
                      item.get('map_role') or '', item.get('map_index') if item.get('map_index') is not None else '',
                      item.get('map_epoch') or '', item.get('map_version') or '', item.get('map_frame_id') or '',
                      int(result.front_tracked[k])]
            flags = [int(support[field][k]) if support[field] is not None else '' for field in support_fields]
            visual = bool(result.tracked[k])
            tag = bool(support['poster_position_supported'][k]) if support['poster_position_supported'] is not None else None
            source = ('visual_and_poster' if visual and tag else 'visual' if visual else
                      'poster' if tag else 'imu' if tag is not None else 'unknown')
            columns = [str(k), str(k), str(source_frame), f'{recording.video_times_s[k]:.9f}',
                       f'{time:.9f}', *map(str, values), *map(str, flags), source]
            stream.write(','.join(columns) + '\n')
    front = recording.lens_video('front').path
    try:
        os.link(front, directory / 'front.mp4')
    except OSError:
        shutil.copy2(front, directory / 'front.mp4')
    length = float(np.linalg.norm(np.diff(tcp_poses[:, :3, 3], axis=0), axis=1).sum())
    metadata = dict(
        rows=len(times), duration_s=float(times[-1] - times[0]), path_length_m=length,
        constraint_fraction=dict(full=float(result.tracked.mean()), imu=float((~result.tracked).mean()), rot=0.),
        calibration_used=camera.selected.summary(),
        notes=[], fusion=result.evidence,
        measurements=camera.measurements, cad_status=geometry.get('status'),
        T_camera_tcp=geometry['T_camera_tcp'],
        frame_geometry=geometry_record(video_parameters.size_px,
            mode='dual' if 'rear' in localization_lenses else 'shared_front', video=video_parameters))
    if tag_distance is not None and (assessment := tag_distance.report.get('acceptance')):
        metadata['tag_distance'] = dict(valid_fraction=tag_distance.report['valid_fraction'],
            **{name: assessment[name] for name in ('passed', 'longest_missing_frames', 'maximum_missing_frames')})
    if recording.source_frame_start:
        metadata['video_time_offset_s'] = float(times[0])
    metadata['position_support'] = dict(
        association='nearest_output_time_cell_for_reporting_only',
        fractions={field: float(mask.mean()) if mask is not None else None for field, mask in support.items()},
        actual_error='not_measured')
    poster = result.evidence.get('environment_poster')
    if poster is not None:
        rear = metadata['frame_geometry']['rear_lens']
        rear['used_for_localization'] = 'rear' in localization_lenses
        rear['used_for_poster_fusion'] = poster['status'] == 'fused'
        rear['used'] = rear['used_for_localization'] or rear['used_for_poster_fusion']
        rear['poster_coordinates'] = 'calibrated_crop_1440'
    # Failed runs keep per-lens diagnostics; successful runs fold them into episode.json.
    localization = {}
    for lens in camera.measurements:
        diagnostic = directory / f'localization_{lens}.json'
        if diagnostic.is_file():
            localization[lens] = json.loads(diagnostic.read_text())
            diagnostic.unlink()
    if localization:
        metadata['localization'] = localization
    return ArmTrajectory(times, tcp_poses, coordinate_record['frame_id'], tcp_frame, metadata)


def write_bimanual_trajectory(tracks: dict[str, ArmTrajectory], directory: Path) -> list[float]:
    if len({track.frame_id for track in tracks.values()}) != 1:
        raise ValueError('双手轨迹必须使用同一个参考坐标系')
    start = max(track.episode_times_s[0] for track in tracks.values())
    end = min(track.episode_times_s[-1] for track in tracks.values())
    primary = tracks.get('left', next(iter(tracks.values()))).episode_times_s
    query = primary[(primary >= start) & (primary <= end)]
    if len(query) < 2:
        raise ValueError('任务录像没有足够共同有效时间，请检查原片完整性')
    if len(tracks) == 2:
        sampled = {arm: resample_pose_series(track.episode_times_s, track.T_reference_tcp, query)
                   for arm, track in tracks.items()}
        with (directory / 'bimanual.csv').open('w') as stream:
            writer = csv.writer(stream)
            axes = ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')
            writer.writerow(['t_episode_s'] + [f'{arm}_{axis}' for arm in ('left', 'right') for axis in axes]
                            + [f'left_tcp_to_right_tcp_{axis}' for axis in axes] + ['tcp_distance_m'])
            quaternions = {arm: Rotation.from_matrix(poses[:, :3, :3]).as_quat() for arm, poses in sampled.items()}
            relative = relative_pose_series(sampled['left'], sampled['right'])
            relative_q = Rotation.from_matrix(relative[:, :3, :3]).as_quat()
            for k, time in enumerate(query):
                writer.writerow([f'{time:.9f}'] + [f'{v:.9f}' for arm in ('left', 'right')
                    for v in np.r_[sampled[arm][k, :3, 3], quaternions[arm][k]]]
                    + [f'{v:.9f}' for v in np.r_[relative[k, :3, 3], relative_q[k], np.linalg.norm(relative[k, :3, 3])]])
    return [float(query[0]), float(query[-1])]
