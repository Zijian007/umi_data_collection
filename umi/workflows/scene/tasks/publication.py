"""Reuse, publish or retain the artifacts of one scene task attempt."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
import json
import shutil
import time
import traceback
from umi.infra import provenance as artifact_identity, provenance
from umi.infra.file_store import publish_link, read_yaml, write_json
from umi.store import layout
from umi.workflows.scene import cleanup, episodes
from umi.store.result_io import (artifact_hashes, is_local_scene, is_poster_scene, uses_scene_map, quality_summary,
                                 read_result, recording_identity, result_paths, scene_reference, validate_result)
from umi.store.models import ArmExecutionResult, Episode, EpisodeRunResult, ArmTrajectory
from umi.workflows.scene.tasks import trajectory_writers
import os
import numpy as np
from datetime import datetime, timezone
import threading


if TYPE_CHECKING:
    from umi.workflows.scene.workflow import ResolvedInputs


@dataclass(frozen=True)
class PublicationResult:
    output: Path
    is_local: bool
    common_interval_s: list[float]
    quality: dict


def _verify_frozen_inputs(episode, input_identity: dict, *, batch_map) -> None:
    """Reject publication if source inputs changed after reconstruction started."""
    current_recordings = {
        arm: recording_identity(video) for arm, video in episode.recordings.items()
    }
    if input_identity['recordings'] != current_recordings:
        raise ValueError('任务期间录像发生变化，拒绝发布')

    if episode.scene.reconstruction_mode not in ('poster_independent', 'poster_vio'):
        epoch_plan = layout.metadata(episode.session.root, 'map_epochs.yaml')
        current_epoch_plan = artifact_identity.file_hash(epoch_plan) if epoch_plan.is_file() else None
        if input_identity.get('map_epoch_plan_sha256') != current_epoch_plan:
            raise ValueError('地图 epoch 清单在任务期间发生变化，拒绝发布')

    if batch_map is not None:
        batch_map.validate(check_configuration=False)


def publish_episode(resolved: ResolvedInputs, *, result: ArmExecutionResult,
                    run_dir: Path, map_path: Path | None, cache_audit, preflight: dict | None,
                    started: float, started_at: float) -> PublicationResult:
    """Write reports, atomically publish the result, and record task state."""
    episode, input_identity = resolved.episode, resolved.identity
    common_interval_s = trajectory_writers.write_bimanual_trajectory(result.tracks, run_dir)
    elapsed_s = time.monotonic() - started
    _verify_frozen_inputs(
        episode,
        input_identity,
        batch_map=resolved.batch_map,
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
    episodes.record_attempt(episode, 'local_ok' if is_local else 'ok', run_dir)
    cleanup.after_task(episode.root, used_since=started_at)
    return PublicationResult(output=output, is_local=is_local, common_interval_s=common_interval_s, quality=quality)


def reuse_episode(episode: Episode, input_identity: dict, *, force: bool) -> EpisodeRunResult | None:
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
        if old.get('kind') not in kinds or old.get('inputs') != input_identity:
            continue
        if old.get('schema_version', 4) < 6 and (is_poster_scene(old['scene']) or
                (old['kind'] == 'umi_scene_episode' and episode.scene.scene_map_poster_constraint_enabled)):
            continue
        reference = scene_reference(old)
        validate_result(paths, old)
        episodes.record_attempt(episode, status, paths.run)
        restore_task_log_link(episode.root)
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
    key = artifact_identity.canonical_hash(dict(inputs=identity, reason=reason, status=status))[:24]
    target = directory.parent / ('failed-' + key)
    write_json(directory / 'error.json', dict(status=status, reason=reason,
        traceback=''.join(traceback.format_exception(error)) if error else '', input_hash=artifact_identity.canonical_hash(identity),
        inputs=identity, batch=read_yaml(episode.session.path)))
    if target.exists():
        shutil.rmtree(directory)
    else:
        directory.rename(target)
        _rewrite_failure_paths(target, directory, target)
    episodes.record_attempt(episode, status, target)
    cleanup.after_task(episode.root)
    return target


def _rewrite_failure_paths(root: Path, old_root: Path, new_root: Path) -> None:
    """Keep diagnostics valid after a failed run receives its stable name.

    External reconstruction stages receive the temporary run directory as an
    absolute path and persist it in logs and JSON.  Failed runs are renamed to
    ``failed-<content hash>`` during collection, so those references need to
    follow the rename for the retained evidence to remain directly usable.
    """
    old_text, new_text = str(old_root), str(new_root)
    if old_text == new_text:
        return
    text_suffixes = {'.json', '.log', '.txt', '.csv', '.yaml', '.yml'}
    for path in root.rglob('*'):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            content = path.read_text(encoding='utf-8')
        except (OSError, UnicodeError):
            continue
        if old_text in content:
            path.write_text(content.replace(old_text, new_text), encoding='utf-8')


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
            strategy += f'（{sync["algorithm"]}）'
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
            measurements.append(f'标签中心间距有效 {tag_distance[arm]["valid_fraction"]:.1%}')
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
    """写任务报告与资产清单；不更新 results 指针或人工验收状态。"""
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


def _publish_task_log(task: Path, source: Path) -> Path:
    link = task / 'run.log'
    try:
        publish_link(source, link, target_is_directory=False)
    except (OSError, ValueError):
        return source
    return link


def restore_task_log_link(task: Path) -> None:
    """Expose an existing audit log when reusing results from an earlier version."""
    with suppress(OSError):
        if (task / 'run.log').exists():
            return
        logs = (path for path in layout.metadata(task, 'logs').glob('*/run.log') if path.is_file())
        latest = max(logs, key=lambda path: (path.stat().st_mtime_ns, path), default=None)
        if latest is not None:
            _publish_task_log(task, latest)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


class RunAudit:
    """Append-only event log plus a final, human-readable run manifest."""

    schema_version = 1

    def __init__(self, task: Path, *, session: str, run_directory: Path,
                 input_identity: dict | None = None, command: str = 'umi-scene run'):
        self.directory = layout.metadata(task, 'logs') / run_directory.name
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._started_clock = time.monotonic()
        self._started_at = _now()
        self._command = command
        self._task = task.name
        self._session = session
        self._run_directory = run_directory
        self._input_hash = (provenance.canonical_hash(input_identity)
                            if input_identity is not None else None)
        self._manifest_path = self.directory / 'run_manifest.json'
        self._events_path = self.directory / 'events.jsonl'
        self._log_path = self.directory / 'run.log'
        self.log_path = self._log_path
        self._write_manifest(dict(
            schema_version=self.schema_version,
            status='running',
            command=command,
            task=self._task,
            session=session,
            input_hash=self._input_hash,
            started_at=self._started_at,
        ))
        self.event('run_started')
        with suppress(OSError):
            self._log_path.touch()
            self.log_path = _publish_task_log(task, self._log_path)

    def write_line(self, line: str) -> None:
        try:
            with self._lock, self._log_path.open('a', encoding='utf-8') as stream:
                stream.write(line.rstrip('\n') + '\n')
        except OSError:
            pass

    def event(self, kind: str | dict, **fields: Any) -> None:
        event = dict(kind=kind if isinstance(kind, str) else 'event',
                     timestamp=_now(), elapsed_s=round(time.monotonic() - self._started_clock, 3))
        if isinstance(kind, dict):
            event.update(kind)
        event.update(_json_safe(fields))
        try:
            with self._lock, self._events_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        except (OSError, TypeError, ValueError):
            pass

    def finish(self, status: str, *, output: Path | str | None = None,
               error: Exception | None = None, directory: Path | None = None) -> Path:
        """Write the final manifest and return the audit directory.

        ``directory`` is the retained scene run directory.  It is recorded as a
        reference but audit files stay in the task log store, so published result
        layouts remain backwards compatible.
        """
        if directory is not None:
            self._run_directory = directory
        self.event('run_finished', status=status)
        artifacts = []
        if self.directory.is_dir():
            for path in sorted(self.directory.rglob('*')):
                if not path.is_file() or path.name == 'run_manifest.json':
                    continue
                try:
                    artifacts.append(dict(path=str(path.relative_to(self.directory)),
                                         bytes=path.stat().st_size))
                except OSError:
                    continue
        document = dict(
            schema_version=self.schema_version,
            status=status,
            command=self._command,
            task=self._task,
            session=self._session,
            started_at=self._started_at,
            finished_at=_now(),
            duration_s=round(time.monotonic() - self._started_clock, 3),
            input_hash=self._input_hash,
            artifacts=artifacts,
        )
        if self._run_directory is not None:
            document['run_directory'] = str(self._run_directory)
        if output is not None:
            document['output'] = str(output)
        if error is not None:
            document['error'] = dict(type=type(error).__name__, message=str(error))
        self._write_manifest(document)
        return self.directory

    def _write_manifest(self, document: dict) -> None:
        try:
            write_json(self._manifest_path, _json_safe(document))
        except (OSError, TypeError, ValueError):
            pass
