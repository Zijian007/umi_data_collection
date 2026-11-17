"""Structured evidence emitted for every successful scene run."""

from __future__ import annotations

from pathlib import Path
import csv

from software.infrastructure.file_store import write_json
from software.scene.result_io import is_local_scene


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


def _read_frames(directory: Path, arm: str) -> list[dict] | None:
    path = directory / arm / 'details' / 'frames.csv'
    if not path.is_file():
        return None
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


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
    return dict(source=f'{arm}/details/frames.csv', total_rows=total,
                visual_rows=len(groups['full']), imu_rows=len(groups['imu']),
                constraint_counts=counts, constraint_fractions=fractions,
                constraint_ranges={key: ranges(values) for key, values in sorted(groups.items())},
                visual_ranges=ranges(groups['full']), imu_ranges=ranges(groups['imu']),
                source_counts=source_counts, source_ranges=source_ranges)


def _frame_source_report(arm: str, rows: list[dict] | None) -> dict:
    """Expose exact per-frame map/lens/visual-IMU provenance in the formal report."""
    if rows is None:
        return dict(source=f'{arm}/details/frames.csv', rows=[])
    fields = ('row', 'video_frame', 'source_video_frame', 't_video_s', 't_episode_s',
              'constraint', 'visual_lens', 'visual_lenses', 'map_role', 'map_index',
              'map_epoch', 'map_version', 'map_frame_id', 'front_lens_tracked')
    return dict(source=f'{arm}/details/frames.csv', rows=[{field: row.get(field, '') for field in fields} for row in rows])


def quality_summary(tracks: dict, scene_record: dict, common_interval_s: list[float]) -> dict:
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
            visual_consistency_rms_m=fusion.get('visual_consistency_rms_m'),
            environment_poster_status=(poster.get('status')
                                        if isinstance(poster, dict) else None),
        )
    return dict(
        schema_version=1,
        scope='local' if is_local_scene(scene_record) else 'shared_scene',
        frame_id=scene_record.get('frame_id'),
        common_interval_s=list(common_interval_s),
        arms=arms,
        accuracy='not_measured',
    )


def write_run_report(directory: Path, *, episode, map_manifest: dict, input_identity: dict,
                     tracks: dict, video_offsets_s: dict, scene_record: dict,
                     gripper_width: dict | None, tag_distance: dict | None,
                     common_interval_s: list[float],
                     cache_audit=None) -> Path:
    """Write a compact, machine-readable explanation of a completed run."""

    local = is_local_scene(scene_record)
    arms = {}
    for arm, track in tracks.items():
        metadata = track.metadata
        fusion = metadata.get('fusion', {})
        frames = _read_frames(directory, arm)
        arms[arm] = dict(
            camera_serial=input_identity.get('cameras', {}).get(arm, {}).get('serial'),
            visual_lenses=list(episode.scene.localization_lenses(arm)) if not local else ['front'],
            rows=metadata.get('rows'),
            constraints=metadata.get('constraint_fraction', {}),
            longest_visual_gap_s=fusion.get('maximum_visual_gap_s'),
            visual_consistency_rms_m=fusion.get('visual_consistency_rms_m'),
            gripper_tags=(tag_distance or {}).get(arm),
            gripper_width=(gripper_width or {}).get(arm),
            gripper_tag_detected=arm in (tag_distance or {}),
            gripper_width_detected=arm in (gripper_width or {}),
            frame_constraints=_frame_constraint_report(arm, frames),
            frame_sources=_frame_source_report(arm, frames),
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
        mapping_scope='local' if local else 'shared_scene',
        synchronization=dict(episode.synchronization, offsets_s=video_offsets_s),
        environment_tags=_environment_report(map_manifest, tracks, episode.scene),
        quality=quality_summary(tracks, scene_record, common_interval_s),
        coordinate=dict(
            frame_id=scene_record['frame_id'],
            scope='local' if local else 'shared_scene',
            units='metres',
            gravity_axis=map_manifest.get('gravity_axis') if not local else None,
            pose_frame='tcp',
            transform_chain=['metric_map_placement', 'visual_localization',
                             'camera_imu_extrinsic', 'camera_tcp_geometry'],
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
