"""Execute the per-arm part of one scene episode.

This module owns reconstruction, single-arm fallback, shared poster refinement,
and arm-level result writing.  Episode locking and final publication remain in
``scene.workflow``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import concurrent.futures as cf

from software.infrastructure.file_store import write_json
from software.scene.pipeline import fallback, measurements, reconstruction, trajectory_writers
from software.scene.spatial.fusion import SceneFusionQualityError
from software.scene.result_io import is_local_scene


@dataclass(frozen=True)
class ArmExecutionResult:
    tracks: dict[str, object]
    widths: dict[str, dict]
    distances: dict[str, dict]
    scene_metadata: dict  # 单手独立回退时替换为局部坐标记录（scope='local'）


def write_arm_result(episode, arm, recording, fusion_result, video_offset_s, arm_scene,
                     run_dir: Path, scene):
    """Write one solved arm and return its track plus measurement reports."""
    measured = measurements.measure(episode, arm, recording, video_offset_s, run_dir / arm)
    track = trajectory_writers.write_arm_trajectory(
        arm, recording, fusion_result, video_offset_s, arm_scene, run_dir / arm,
        width=measured.width, tag_distance=measured.tag_distance,
        export_auxiliary_trajectories=scene.parameters.export_auxiliary_trajectories,
        localization_lenses=('front',) if is_local_scene(arm_scene)
        else scene.localization_lenses(arm),
    )
    return (
        track,
        measured.width.report if measured.width is not None else None,
        measured.tag_distance.report if measured.tag_distance is not None else None,
    )


def execute_arm_tracks(*, episode, prepared_recordings: dict, video_offsets_s: dict[str, float],
                       map_dir: Path, map_manifest: dict, run_dir: Path, scene, name: str,
                       allocation: bool, cache_events, scene_metadata: dict,
                       console) -> ArmExecutionResult:
    """Run all arms, optionally solve one shared environment-poster pose, and write tracks."""
    tracks = {}
    widths = {}
    distances = {}
    # 双手共享同一张海报：各手先只做 map+IMU，最后在一张图里联合求解海报位姿。
    shared_poster = (
        len(prepared_recordings) > 1
        and scene.environment_tag_detection
        and scene.parameters.tag_fusion_enabled
    )
    def write(arm, recording, fusion_result, arm_scene):
        track, width_report, tag_distance_report = write_arm_result(
            episode, arm, recording, fusion_result, video_offsets_s[arm], arm_scene, run_dir, scene)
        tracks[arm] = track
        if width_report is not None:
            widths[arm] = width_report
        if tag_distance_report is not None:
            distances[arm] = tag_distance_report

    def run_arm(arm):
        recording = prepared_recordings[arm]
        arm_scene = scene_metadata
        measurements.prepare_detections(episode, arm, recording, cache_audit=cache_events)
        console.write(f'{arm}：已有场景地图定位 → 固定场景系融合', recording=name, step=(2, 3))
        poster_inputs = None
        try:
            solved = reconstruction.reconstruct_arm_in_scene(
                episode, arm, recording, map_dir, map_manifest, diagnostic_dir=run_dir / arm,
                cache_audit=cache_events, share_poster=shared_poster)
            fusion_result, poster_inputs = solved.trajectory, solved.poster_inputs
        except SceneFusionQualityError as shared_error:
            if len(episode.expected_arms) != 1 or len(prepared_recordings) != 1:
                raise
            console.write(f'{arm}：{shared_error}；自动回退到独立局部轨迹', recording=name, state='回退')
            write_json(run_dir / 'fallback.json',
                       dict(shared_failure=str(shared_error), shared_scene=scene_metadata))
            try:
                fusion_result = fallback.reconstruct(
                    episode, arm, recording, run_dir / arm / 'independent')
            except Exception as local_error:
                raise ValueError(
                    f'共享场景失败：{shared_error}；独立回退失败：{local_error}'
                ) from local_error
            arm_scene = dict(
                name=scene.name,
                session=episode.session.name,
                version=None,
                frame_id=f'local:{run_dir.name}:{arm}',
                scope='local',
                fallback=dict(
                    shared_failure=str(shared_error),
                    shared_frame_id=map_manifest['frame_id'],
                ),
            )

        # Single-arm runs publish each successful arm immediately so a failed
        # peer still leaves useful diagnostics. Shared poster refinement waits
        # until every arm has supplied its observation context.
        if not shared_poster:
            write(arm, recording, fusion_result, arm_scene)
        return arm, recording, fusion_result, arm_scene, poster_inputs

    if allocation:
        console.write('左右手并行重建', recording=name, step=(2, 3))
        with cf.ThreadPoolExecutor(
            max_workers=len(prepared_recordings), thread_name_prefix='scene-arm'
        ) as executor:
            arm_results = list(executor.map(run_arm, prepared_recordings))
    else:
        arm_results = [run_arm(arm) for arm in prepared_recordings]

    if shared_poster:
        from software.scene.spatial import poster

        refined = poster.refine_shared_poster_pose(
            [inputs for *_, inputs in arm_results if inputs is not None])
        for arm, recording, fusion_result, arm_scene, _ in arm_results:
            write(arm, recording, refined.get(arm, fusion_result), arm_scene)

    # 只有单手任务会独立回退，此时整个任务改用该手的局部坐标记录。
    for *_, arm_scene, _ in arm_results:
        if is_local_scene(arm_scene):
            scene_metadata = arm_scene
    return ArmExecutionResult(tracks, widths, distances, scene_metadata)
