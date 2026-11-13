"""Structured evidence emitted for every successful scene run."""

from __future__ import annotations

from pathlib import Path
import csv
import json

from software.storage import write_json


def _map_report(map_manifest: dict, manifest_sha256: str | None = None,
                scene_map_identity: str | None = None) -> dict:
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
            'archive': item.get('archive'),
            'scale': placement.get('scale'),
            'alignment_rms_m': evidence.get('rms_m'),
            'relative_scale_std': evidence.get('relative_scale_std'),
            'condition': evidence.get('condition'),
        })
    return dict(
        map_epoch=map_manifest.get('map_epoch'),
        map_version=map_manifest.get('version'),
        manifest_sha256=manifest_sha256 or map_manifest.get('manifest_sha256'),
        scene_map_identity=scene_map_identity,
        frame_id=map_manifest.get('frame_id'),
        mode=map_manifest.get('mode'),
        units=map_manifest.get('units'),
        gravity_axis=map_manifest.get('gravity_axis'),
        maps=maps,
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
    return dict(
        layout=str(scene.tag_layout) if scene.tag_layout else None,
        mapping_enabled=scene.parameters.tag_mapping_enabled,
        mapping_scale_enabled=scene.parameters.tag_mapping_scale_enabled,
        fusion_enabled=scene.parameters.tag_fusion_enabled,
        mapping_detected=bool(map_evidence),
        fusion_detected=bool(task_evidence),
        scale_applied=bool(scene.parameters.tag_mapping_scale_enabled and map_evidence),
        mapping=map_evidence,
        task_fusion=task_evidence,
    )


def _frame_constraint_report(directory: Path, arm: str) -> dict:
    """Summarize the exact visual/IMU rows while retaining frames.csv as source."""
    path = directory / arm / 'details' / 'frames.csv'
    if not path.is_file():
        return dict(source=f'{arm}/details/frames.csv', visual_rows=0, imu_rows=0,
                    visual_ranges=[], imu_ranges=[])
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    groups = {'full': [], 'imu': []}
    for row in rows:
        groups.setdefault(row['constraint'], []).append(int(row['row']))

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

    return dict(source=f'{arm}/details/frames.csv', visual_rows=len(groups['full']),
                imu_rows=len(groups['imu']), visual_ranges=ranges(groups['full']),
                imu_ranges=ranges(groups['imu']))


def write_run_report(directory: Path, *, episode, map_manifest: dict, input_identity: dict,
                     tracks: dict, video_offsets_s: dict, scene_record: dict,
                     gripper_width: dict | None, tag_distance: dict | None,
                     cache_audit=None) -> Path:
    """Write a compact, machine-readable explanation of a completed run."""

    local = scene_record.get('scope') == 'local'
    arms = {}
    for arm, track in tracks.items():
        metadata = track.metadata
        fusion = metadata.get('fusion', {})
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
            frame_constraints=_frame_constraint_report(directory, arm),
        )

    report = dict(
        schema_version=1,
        kind='umi_scene_run_report',
        status='ok',
        task=episode.root.name,
        session=episode.session.name,
        map_selection=episode.map_selection,
        map=_map_report(map_manifest, episode.session.map_manifest_sha256,
                        input_identity.get('scene_map_sha256')) if not local else None,
        mapping_scope='local' if local else 'shared_scene',
        synchronization=dict(episode.synchronization, offsets_s=video_offsets_s),
        environment_tags=_environment_report(map_manifest, tracks, episode.scene),
        coordinate=dict(
            frame_id=scene_record['frame_id'],
            scope='local' if local else 'shared_scene',
            units='metres',
            gravity_axis=map_manifest.get('gravity_axis') if not local else None,
            pose_frame='tcp',
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
            map=dict(status='manifest_timings', version=map_manifest.get('version'),
                     timings=map_manifest.get('timings', []),
                     reason='地图版本不可变；阶段状态来自 map manifest'),
        ),
    )
    path = directory / 'run_report.json'
    write_json(path, report)
    return path
