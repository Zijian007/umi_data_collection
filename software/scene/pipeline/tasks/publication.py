"""Reuse, publish or retain the artifacts of one scene task attempt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
import json
import shutil
import time
import traceback
from software.infrastructure import provenance as artifact_identity, provenance
from software.infrastructure.file_store import publish_link, read_yaml, write_json
from software.scene import layout
from software.scene.management import cleanup, episodes
from software.scene.result_io import recording_identity, validate_assets, is_local_scene, scene_reference, is_poster_scene, artifact_hashes
from software.scene.models import ArmExecutionResult, Episode, EpisodeRunResult, ArmTrajectory
from software.scene.pipeline.tasks import trajectory_writers
from software.scene.management.collection_schema import apply_scene_map_poster_policy
import os
import numpy as np
from software.camera.geometry import describe_tcp_frame
from software.scene.pipeline import synchronization
from datetime import datetime, timezone
import threading


if TYPE_CHECKING:
    from software.scene.workflow import ResolvedInputs


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
    quality = quality_summary(result.tracks, result.coordinate_record, common_interval_s)
    write_run_report(
        run_dir,
        episode=episode,
        map_manifest=resolved.map_selection.manifest,
        input_identity=input_identity,
        tracks=result.tracks,
        video_offsets_s=resolved.video_offsets_s,
        coordinate_record=result.coordinate_record,
        gripper_width=result.width_reports,
        tag_distance=result.tag_distance_reports,
        quality=quality,
        cache_audit=cache_audit,
    )
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
    publish_link(run_dir, output)
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
        previous = output / 'episode.json'
        if not previous.is_file():
            continue
        old = json.loads(previous.read_text())
        if old.get('kind') not in kinds or old.get('inputs') != input_identity:
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


def record_failure(episode: Episode, directory: Path, reason: str, *, identity: dict,
                   status='failed', error: BaseException | None = None) -> Path:
    """同一输入和原因只保留一份失败现场；成功运行不参与去重。"""
    # Publication is committed by replacing the result link. An audit or state
    # write can still fail afterward, so keep any directory the link points to.
    if any(output.is_symlink() and output.resolve() == directory.resolve()
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


def _status_counts(events) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        status = event.get('status', 'unknown') if isinstance(event, dict) else 'unknown'
        counts[status] = counts.get(status, 0) + 1
    return counts


def _map_report(map_manifest: dict, manifest_sha256: str | None = None,
                scene_map_identity: str | None = None) -> dict:
    recording_inputs = map_manifest.get('inputs', {}).get('recordings', {})
    maps = []
    for role, item in sorted(map_manifest.get('maps', {}).items()):
        placement = item.get('placement', {})
        evidence = placement.get('evidence', {})
        maps.append({
            'role': role,
            'index': item.get('index'),
            'lens': item.get('lens'),
            'mapping_arm': item.get('mapping_arm'),
            'camera_serial': item.get('camera_serial'),
            'mapping_recording': recording_inputs.get(item.get('mapping_arm')),
            'archive': item.get('archive'),
            'scale': placement.get('scale'),
            'alignment_rms_m': evidence.get('rms_m'),
            'relative_scale_std': evidence.get('relative_scale_std'),
            'condition': evidence.get('condition'),
            'T_scene_metric_map': placement.get('T_scene_metric_map'),
            'environment_poster_used': evidence.get('environment_poster', {}).get('status')
                if isinstance(evidence.get('environment_poster'), dict) else None,
        })
    return dict(
        map_epoch=map_manifest.get('map_epoch'),
        map_version=map_manifest.get('version'),
        manifest_sha256=manifest_sha256 or map_manifest.get('manifest_sha256'),
        scene_map_identity=scene_map_identity,
        frame_id=map_manifest.get('frame_id'),
        cross_epoch_alignment=map_manifest.get('cross_epoch_alignment', dict(
            status='not_assumed',
            scene_frame_id=map_manifest.get('frame_id'),
            reason='环境视觉标签默认不是跨 epoch 的固定世界锚点')),
        mode=map_manifest.get('mode'),
        units=map_manifest.get('units'),
        gravity_axis=map_manifest.get('gravity_axis'),
        maps=maps,
        mapping_recordings=recording_inputs,
    )


def _environment_report(map_manifest: dict, tracks: dict, scene) -> dict:
    map_evidence = []
    for role, item in sorted(map_manifest.get('maps', {}).items()):
        evidence = item.get('placement', {}).get('evidence', {}).get('environment_poster')
        if evidence is not None:
            map_evidence.append(dict(role=role, **evidence))
    task_evidence = {}
    for arm, track in tracks.items():
        poster = track.metadata.get('fusion', {}).get('environment_poster')
        if poster is not None:
            task_evidence[arm] = poster
    mapping_status = _status_counts(map_evidence)
    fusion_status = _status_counts(task_evidence.values())
    mapping_observed = any(isinstance(item, dict) and isinstance(item.get('observations'), (int, float))
                           and item['observations'] > 0 for item in map_evidence)
    fusion_observed = any(isinstance(item, dict)
                          and isinstance(item.get('observations', item.get('frames')), (int, float))
                          and item.get('observations', item.get('frames')) > 0
                          for item in task_evidence.values())
    return dict(
        layout=str(scene.tag_layout) if scene.tag_layout else None,
        policy=scene.environment_tag_policy,
        detection_enabled=scene.environment_tag_detection,
        mapping_enabled=scene.parameters.tag_mapping_enabled,
        mapping_scale_enabled=scene.parameters.tag_mapping_scale_enabled,
        fusion_enabled=scene.parameters.tag_fusion_enabled,
        mapping_detected=mapping_observed,
        fusion_detected=fusion_observed,
        mapping_used=any(item.get('status') == 'corrected' for item in map_evidence),
        fusion_used=any(item.get('status') == 'fused' for item in task_evidence.values()),
        mapping_observation_count=len(map_evidence),
        mapping_status_counts=mapping_status,
        fusion_status_counts=fusion_status,
        cross_epoch_anchor=dict(status='not_assumed', used=False,
                                reason='标签可能移动；本次检测不建立跨 epoch 世界锚定'),
        scale_applied=bool(scene.parameters.tag_mapping_scale_enabled and any(
            isinstance(item, dict) and item.get('status') in ('accepted', 'corrected', 'fused', 'ok')
            for item in map_evidence)),
        mapping=map_evidence,
        task_fusion=task_evidence,
    )


def _frame_constraint_report(arm: str, rows: list[dict] | None) -> dict:
    """Summarize the exact visual/IMU rows while retaining frames.csv as source."""
    if rows is None:
        return dict(source=f'{arm}/details/frames.csv', total_rows=0, visual_rows=0, imu_rows=0,
                    constraint_counts={}, constraint_fractions={}, constraint_ranges={},
                    visual_ranges=[], imu_ranges=[], source_counts={}, source_ranges=[])
    groups = {'full': [], 'imu': []}
    source_counts = {}
    source_ranges = []
    previous = None
    start = None
    for row in rows:
        groups.setdefault(row['constraint'], []).append(int(row['row']))
        source = '|'.join(filter(None, (row.get('map_role'), row.get('visual_lens'), row.get('constraint')))) or row.get('constraint', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
        if source != previous:
            if previous is not None:
                source_ranges.append(dict(start=start, end=int(row['row']) - 1, source=previous))
            previous, start = source, int(row['row'])
    if previous is not None:
        source_ranges.append(dict(start=start, end=int(rows[-1]['row']), source=previous))

    def ranges(values: list[int]) -> list[list[int]]:
        if not values:
            return []
        values = sorted(values)
        result = [[values[0], values[0]]]
        for value in values[1:]:
            if value == result[-1][1] + 1:
                result[-1][1] = value
            else:
                result.append([value, value])
        return result

    total = len(rows)
    counts = {key: len(values) for key, values in sorted(groups.items())}
    fractions = {key: (value / total if total else 0.) for key, value in counts.items()}
    constraint_ranges = {key: ranges(values) for key, values in sorted(groups.items())}
    return dict(source=f'{arm}/details/frames.csv', total_rows=total,
                visual_rows=len(groups['full']), imu_rows=len(groups['imu']),
                constraint_counts=counts, constraint_fractions=fractions,
                constraint_ranges=constraint_ranges,
                visual_ranges=constraint_ranges['full'], imu_ranges=constraint_ranges['imu'],
                source_counts=source_counts, source_ranges=source_ranges)


def quality_summary(tracks: dict, coordinate_record: dict, common_interval_s: list[float]) -> dict:
    """Return the compact, raw quality evidence used by terminal and JSON output.

    These values describe what the pipeline verified; they are not an independent
    accuracy score or a substitute for reviewing the trajectory against video.
    """
    arms = {}
    for arm, track in tracks.items():
        metadata = track.metadata
        constraints = metadata.get('constraint_fraction', {})
        fusion = metadata.get('fusion', {})
        poster = fusion.get('environment_poster')
        arms[arm] = dict(
            rows=metadata.get('rows'),
            duration_s=metadata.get('duration_s'),
            path_length_m=metadata.get('path_length_m'),
            visual_constraint_fraction=constraints.get('full'),
            imu_constraint_fraction=constraints.get('imu'),
            longest_visual_gap_s=fusion.get('maximum_visual_gap_s'),
            longest_position_support_gap_s=fusion.get('independent', {}).get('maximum_position_support_gap_s'),
            position_support=metadata.get('position_support'),
            poster_geometry_support=({key: value for key, value in poster.get('position_support', {}).items()
                                      if key != 'frames'} if isinstance(poster, dict) else None),
            visual_consistency_rms_m=fusion.get('visual_consistency_rms_m'),
            environment_poster_status=(poster.get('status')
                                        if isinstance(poster, dict) else None),
        )
    return dict(
        schema_version=1,
        scope='task_poster' if is_poster_scene(coordinate_record) else 'local' if is_local_scene(coordinate_record) else 'shared_scene',
        frame_id=coordinate_record.get('frame_id'),
        common_interval_s=list(common_interval_s),
        arms=arms,
        accuracy='not_measured',
    )


def write_run_report(directory: Path, *, episode, map_manifest: dict, input_identity: dict,
                     tracks: dict, video_offsets_s: dict, coordinate_record: dict,
                     gripper_width: dict | None, tag_distance: dict | None,
                     quality: dict,
                     cache_audit=None) -> Path:
    """Write a compact, machine-readable explanation of a completed run."""

    poster_frame = is_poster_scene(coordinate_record)
    local = is_local_scene(coordinate_record) or poster_frame
    scene = episode.scene if poster_frame else apply_scene_map_poster_policy(episode.scene)
    map_manifest = map_manifest or {}
    scope = quality['scope']
    arms = {}
    for arm, track in tracks.items():
        metadata = track.metadata
        arm_quality = quality['arms'][arm]
        frames = track.frame_rows
        arms[arm] = dict(
            camera_serial=input_identity.get('cameras', {}).get(arm, {}).get('serial'),
            visual_lenses=([lens for lens, evidence in metadata.get('fusion', {}).get('poster_vio', {}).get(
                               'natural_by_camera', {}).items() if evidence['tracks'] > 0]
                           if coordinate_record.get('reconstruction_backend') == 'poster_vio' else
                           list(episode.scene.localization_lenses(arm)) if not local else ['front']),
            rows=arm_quality['rows'],
            constraints=metadata.get('constraint_fraction', {}),
            longest_visual_gap_s=arm_quality['longest_visual_gap_s'],
            visual_consistency_rms_m=arm_quality['visual_consistency_rms_m'],
            gripper_tags=(tag_distance or {}).get(arm),
            gripper_width=(gripper_width or {}).get(arm),
            gripper_tag_detected=arm in (tag_distance or {}),
            gripper_width_detected=arm in (gripper_width or {}),
            frame_constraints=_frame_constraint_report(arm, frames),
            frame_sources=dict(source=f'{arm}/details/frames.csv', rows=frames or []),
        )

    map_report = _map_report(map_manifest, episode.session.map_manifest_sha256,
                             input_identity.get('scene_map_sha256'))
    report = dict(
        schema_version=1,
        kind='umi_scene_run_report',
        status='ok',
        task=episode.root.name,
        session=episode.session.name,
        map_selection=episode.map_selection,
        map=None if local else map_report,
        mapping_scope=scope,
        reconstruction=dict(requested_mode=episode.scene.reconstruction_mode,
                            selected_backend=(coordinate_record.get('reconstruction_backend', 'poster_independent')
                                              if poster_frame else 'independent' if local else 'scene_map'),
                            attempts=coordinate_record.get('attempts'),
                            fallback=coordinate_record.get('fallback')),
        synchronization=dict(episode.synchronization, offsets_s=video_offsets_s),
        environment_tags=_environment_report(map_manifest, tracks, scene),
        quality=quality,
        coordinate=dict(
            frame_id=coordinate_record['frame_id'],
            scope=scope,
            units='metres',
            gravity_axis=map_manifest.get('gravity_axis') if not local else None,
            pose_frame='tcp',
            transform_chain=(['joint_poster_visual_inertial_ba', 'world_to_board',
                              'camera_imu_extrinsic', 'camera_tcp_geometry']
                             if coordinate_record.get('reconstruction_backend') == 'poster_vio' else
                             ['independent_visual_imu_poster_fusion', 'local_world_to_board',
                              'camera_imu_extrinsic', 'camera_tcp_geometry'] if poster_frame else
                             ['metric_map_placement', 'visual_localization',
                              'camera_imu_extrinsic', 'camera_tcp_geometry']),
            cross_epoch_comparable=False,
            cross_epoch_alignment='not_assumed_without_verified_static_anchor',
            map_placement_transforms={item['role']: item.get('T_scene_metric_map')
                                      for item in map_report['maps']},
        ),
        arms=arms,
        inputs=dict(
            recordings=input_identity.get('recordings'),
            cameras=input_identity.get('cameras'),
            scene_map_sha256=input_identity.get('scene_map_sha256'),
            map_epoch_plan_sha256=input_identity.get('map_epoch_plan_sha256'),
            synchronization_sha256=input_identity.get('synchronization_sha256'),
            code=input_identity.get('code'),
            engines=input_identity.get('engines'),
        ),
        cache=dict(
            task=cache_audit.snapshot() if cache_audit is not None else [],
            map=dict(status='manifest_audit', version=map_manifest.get('version'),
                     events=map_manifest.get('cache_audit', []),
                     timings=map_manifest.get('timings', []),
                     reuse_scope='published_map_binding',
                     selection=episode.map_selection,
                     reason='地图版本不可变；本次任务绑定已发布地图，阶段状态来自 map manifest',
                     status_counts=_status_counts(map_manifest.get('cache_audit', []))),
        ),
    )
    path = directory / 'run_report.json'
    write_json(path, report)
    return path


def write_episode_result(episode: Episode, tracks: dict[str, ArmTrajectory], directory: Path, *,
                  common_interval_s: list[float], coordinate_record: dict, input_identity: dict,
                  video_offsets_s: dict[str, float], elapsed_s: float, map_path: Path | None, preflight: dict | None,
                  gripper_width: dict | None = None, tag_distance: dict | None = None) -> None:
    """写任务报告与资产清单；不更新 results 指针或人工验收状态。"""
    local = is_local_scene(coordinate_record)
    poster_frame = is_poster_scene(coordinate_record)
    joint_vio = coordinate_record.get('reconstruction_backend') == 'poster_vio'
    mapless = local or poster_frame
    if poster_frame and map_path is not None:
        raise ValueError('任务海报结果不能引用共享场景地图')
    if local and (len(tracks) != 1 or map_path is not None):
        raise ValueError('局部回退只允许单手且不能引用共享地图')
    sync_record = episode.synchronization
    sync_lines = [
        f'同步：{sync_record["method"]}；offsets_s={synchronization.format_offsets(video_offsets_s)}',
        f'同步确认：{sync_record["confirmed"]}；依据：{sync_record["note"]}',
    ]
    if sync_record.get('raw_audio_offset_s') is not None:
        sync_lines.append(
            '同步原始音频偏移：'
            f'{synchronization.format_offsets(sync_record["raw_audio_offset_s"])}')
    if 'algorithm' in sync_record:
        sync_lines.append(f'同步算法：{sync_record["algorithm"]}')
    if sync_record.get('diagnostics'):
        sync_lines.append(synchronization.format_audio_evidence(sync_record))
    if sync_record.get('peak_ratio') is not None:
        sync_lines.append(f'同步音频峰值比：{float(sync_record["peak_ratio"]):.2f}')
    if sync_record.get('video_offset_status') is not None:
        sync_lines.append(f'同步视频时间轴：{sync_record["video_offset_status"]}')
    text = (f'场景任务：{episode.root.name}\n采集批次：{episode.session.name}\n坐标：{coordinate_record["frame_id"]}\n'
        f'操作指令：{episode.instruction or "未填写"}\n'
        f'手：{", ".join(tracks)}\n共同有效时间：{common_interval_s[0]:.6f}–{common_interval_s[1]:.6f} s\n'
        + '\n'.join(sync_lines) + '\n'
        + (f'夹爪宽度：{", ".join(gripper_width)}\n' if gripper_width else '')
        + (f'夹爪标签中心三维间距：{", ".join(tag_distance)}\n' if tag_distance else '')
        + '每手 trajectory.csv 保存 TCP 位姿及已配置的测量；缺失测量不代表闭合。\n'
        + ('独立回退：仅本次录像的局部米制坐标，不与其他任务/场景共享。\n'
           if local else '任务海报米制坐标；共享海报视觉惯性联合图，不跨任务共享。\n'
           if joint_vio else '任务海报米制坐标；左右手各自持续融合海报，不跨任务共享。\n'
           if poster_frame else '轨迹共享米制坐标；左右手分别融合。\n')
        + '实际误差：未测量；TCP 使用 CAD 名义几何。\n')
    trim = coordinate_record.get('recording_trim')
    if trim is not None:
        text += '\n环境标签首尾裁剪（原片保留）：\n'
        for arm, item in trim['arms'].items():
            text += (f'{arm}：去头 {item["removed_head_s"]:.3f}s；去尾 {item["removed_tail_s"]:.3f}s；'
                     f'原片保留区间 {item["kept_video_interval_s"][0]:.6f}–{item["kept_video_interval_s"][1]:.6f}s\n')
    if local:
        text += f'共享场景失败原因：{coordinate_record["fallback"]["shared_failure"]}\n'
    # 预检先独立落盘，保证算法失败时也有证据；成功后收入唯一运行清单。
    preflight_path = directory / 'preflight_report.json'
    text += '\n原片检查：' + ('跳过' if preflight is None else 'PASS' if preflight['passed'] else 'FAIL') + '\n'
    if preflight is not None:
        for arm, video in preflight.get('videos', {}).items():
            for check in video['checks']:
                if check['status'] != 'PASS':
                    text += f'{arm} [{check["status"]}] {check["name"]}：{check["message"]}\n'
    for arm, track in tracks.items():
        arm_metadata = track.metadata
        text += '\n' + '\n'.join([
            f'{arm} {"独立局部" if local else "场景"}轨迹', f'相机：{input_identity["cameras"][arm]["serial"]}',
            'TCP 自身坐标：' + describe_tcp_frame(track.tcp_frame),
            f'视频时间偏移：{video_offsets_s[arm]:+.9f} s（t_episode = t_video + offset）',
            f'时长：{arm_metadata["duration_s"]:.2f} s；TCP 路程：{arm_metadata["path_length_m"]:.3f} m',
            f'{"本录像视觉位置约束" if poster_frame else "地图位置约束"}：{arm_metadata["constraint_fraction"]["full"]:.1%}',
            f'最长无视觉：{arm_metadata["fusion"]["maximum_visual_gap_s"]:.3f} s',
            ('联合图诊断和冻结数值输入保存在 poster_vio/。' if joint_vio else
             '独立融合诊断保存在 independent/debug。' if mapless else
             f'视觉一致性 RMS：{arm_metadata["fusion"]["visual_consistency_rms_m"] * 1000:.2f} mm（内部指标）'),
        ]) + '\n'
        poster = arm_metadata['fusion'].get('environment_poster')
        if poster is not None:
            text += f'环境海报：{poster["status"]}；真实轨迹误差未测量。\n'
            if 'final_reprojection_rms_px' in poster:
                text += f'海报重投影 RMS：{poster["final_reprojection_rms_px"]:.3f} px（内部指标）\n'
            metrics = (poster.get('final_reprojection_metrics') or poster.get('reprojection_metrics')
                       or poster.get('initial_reprojection_metrics'))
            if metrics:
                text += (f'海报角点残差：median {metrics.get("median_px", float("nan")):.3f} px；'
                         f'P95 {metrics.get("p95_px", float("nan")):.3f} px；'
                         f'帧数 {len(metrics.get("frame_rms_px", []))}；'
                         f'ID 数 {len(metrics.get("id_rms_px", {}))}\n')
            if 'reason' in poster:
                text += f'海报诊断原因：{poster["reason"]}\n'
        for lens, recovery in arm_metadata.get('localization', {}).items():
            text += (f'{lens} 双向定位：正向 {recovery["forward_frames"]} 帧，'
                     f'反向补入 {recovery["added_frames"]} 帧；{recovery["status"]}\n')
        if arm in (tag_distance or {}):
            text += f'夹爪标签中心间距有效比例：{tag_distance[arm]["valid_fraction"]:.1%}；单位 mm\n'
    (directory / 'report.txt').write_text(text, encoding='utf-8')
    assets = artifact_hashes(directory, (*tracks, 'devices', 'poster_vio'))
    for name in ('report.txt', 'bimanual.csv', 'run_report.json', 'reconstruction.json', 'tag_trim.json'):
        if (directory / name).is_file():
            assets[name] = artifact_identity.file_hash(directory / name)
    if (directory/'fallback.json').exists():
        assets['fallback.json'] = artifact_identity.file_hash(directory/'fallback.json')
    primary_times = tracks.get('left', next(iter(tracks.values()))).episode_times_s
    metadata = dict(kind='umi_local_episode' if local else 'umi_poster_episode' if poster_frame else 'umi_scene_episode', schema_version=4,
        status='local_ok' if local else 'ok', name=episode.root.name, session=episode.session.name,
        kit=dict(id=episode.session.kit['id'], arms={arm: episode.session.kit['arms'][arm] for arm in tracks}),
        scene={key: value for key, value in coordinate_record.items() if key != 'session'},
        arms={arm: arm for arm in tracks}, common_interval_s=common_interval_s,
        rows=int(np.count_nonzero((primary_times >= common_interval_s[0]) & (primary_times <= common_interval_s[1]))),
        instruction=episode.instruction, synchronization=episode.synchronization,
        inputs=input_identity, assets=assets, elapsed_s=elapsed_s, actual_error='not_measured',
        tcp_frames={arm: track.tcp_frame for arm, track in tracks.items()},
        arm_results={arm: track.metadata for arm, track in tracks.items()}, preflight=preflight,
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
    write_json(directory / 'episode.json', metadata)
    preflight_path.unlink(missing_ok=True)


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

    def __init__(self, directory: Path, *, task: str, session: str,
                 input_identity: dict | None = None, command: str = 'umi-scene run',
                 run_directory: Path | None = None):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._started_clock = time.monotonic()
        self._started_at = _now()
        self._command = command
        self._task = task
        self._session = session
        self._run_directory = run_directory
        self._input_hash = (provenance.canonical_hash(input_identity)
                            if input_identity is not None else None)
        self._manifest_path = self.directory / 'run_manifest.json'
        self._events_path = self.directory / 'events.jsonl'
        self._log_path = self.directory / 'run.log'
        self._write_manifest(dict(
            schema_version=self.schema_version,
            status='running',
            command=command,
            task=task,
            session=session,
            input_hash=self._input_hash,
            started_at=self._started_at,
        ))
        self.event('run_started')

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
