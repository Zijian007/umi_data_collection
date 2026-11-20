"""场景地图后端：手级重建、单手局部回退与双手海报联合求解。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from software.infrastructure.concurrency import daemon_map
from software.infrastructure.file_store import write_json
from software.scene.pipeline.tasks.backends import independent_reconstruction, scene_reconstruction
from software.scene.pipeline.tasks import measurements
from software.scene.pipeline.tasks.results import arm_writer
from software.scene.spatial.fusion import SceneFusionQualityError
from software.scene.result_io import is_local_scene
from software.scene.models import TaskInputs, ArmExecutionResult, FusedTrajectory
from software.scene.spatial.poster import PosterInputs


@dataclass(frozen=True)
class _ArmSolveResult:
    arm: str
    trajectory: FusedTrajectory
    coordinate_record: dict
    poster_inputs: PosterInputs | None


def execute_task(inputs: TaskInputs, *, map_dir: Path, map_manifest: dict,
                       coordinate_record: dict, allow_local_fallback=True) -> ArmExecutionResult:
    """Run all arms, optionally solve one shared environment-poster pose, and write tracks."""
    episode = inputs.episode
    scene, name = episode.scene, episode.root.name
    prepared_recordings, run_dir = inputs.recordings, inputs.directory
    cache_audit, console = inputs.cache_audit, inputs.console
    tracks = {}
    width_reports = {}
    tag_distance_reports = {}
    # 双手共享同一张海报：各手先只做 map+IMU，最后在一张图里联合求解海报位姿。
    shared_poster = (
        len(prepared_recordings) > 1
        and scene.environment_tag_detection
        and scene.parameters.tag_fusion_enabled
    )
    def write(arm, fusion_result, coordinate_record):
        track, width_report, tag_distance_report = arm_writer.write_arm_result(inputs, arm, fusion_result, coordinate_record)
        tracks[arm] = track
        if width_report is not None:
            width_reports[arm] = width_report
        if tag_distance_report is not None:
            tag_distance_reports[arm] = tag_distance_report

    def run_arm(arm):
        recording = prepared_recordings[arm]
        arm_coordinate_record = coordinate_record
        measurements.prepare_detections(episode, arm, recording, cache_audit=cache_audit)
        console.write(f'{arm}：已有场景地图定位 → 固定场景系融合', recording=name, step=(2, 3))
        poster_inputs = None
        try:
            solved = scene_reconstruction.reconstruct_arm_in_scene(
                episode, arm, recording, map_dir, map_manifest, diagnostic_dir=run_dir / arm,
                cache_audit=cache_audit, defer_poster_fusion=shared_poster)
            fusion_result, poster_inputs = solved.trajectory, solved.poster_inputs
        except SceneFusionQualityError as shared_error:
            if not allow_local_fallback or len(episode.expected_arms) != 1 or len(prepared_recordings) != 1:
                raise
            console.write(f'{arm}：{shared_error}；自动回退到独立局部轨迹', recording=name, state='回退')
            write_json(run_dir / 'fallback.json',
                       dict(shared_failure=str(shared_error), shared_scene=coordinate_record))
            try:
                fusion_result = independent_reconstruction.reconstruct(
                    episode, arm, recording, run_dir / arm / 'independent')
            except Exception as local_error:
                raise ValueError(
                    f'共享场景失败：{shared_error}；独立回退失败：{local_error}'
                ) from local_error
            arm_coordinate_record = dict(
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
            write(arm, fusion_result, arm_coordinate_record)
        return _ArmSolveResult(arm, fusion_result, arm_coordinate_record, poster_inputs)

    if inputs.arm_cpu_sets:
        console.write('左右手并行重建', recording=name, step=(2, 3))
        arm_results = daemon_map(run_arm, tuple(prepared_recordings))
    else:
        arm_results = [run_arm(arm) for arm in prepared_recordings]

    if shared_poster:
        from software.scene.spatial import poster

        refined = poster.refine_shared_poster_pose(
            [result.poster_inputs for result in arm_results if result.poster_inputs is not None])
        for result in arm_results:
            write(result.arm, refined.get(result.arm, result.trajectory), result.coordinate_record)

    # 只有单手任务会独立回退，此时整个任务改用该手的局部坐标记录。
    for result in arm_results:
        if is_local_scene(result.coordinate_record):
            coordinate_record = result.coordinate_record
    return ArmExecutionResult(tracks, width_reports, tag_distance_reports, coordinate_record)
