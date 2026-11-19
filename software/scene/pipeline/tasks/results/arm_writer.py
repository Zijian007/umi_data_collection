"""所有轨迹后端共用的手级测量与结果写入。"""

from __future__ import annotations

from software.scene.models import TaskInputs, FusedTrajectory
from software.scene.pipeline.tasks import measurements
from software.scene.pipeline.tasks.results import trajectory_writers
from software.scene.result_io import is_local_scene, is_poster_scene


def write_arm_result(inputs: TaskInputs, arm: str, fusion_result: FusedTrajectory, coordinate_record: dict):
    """Write one solved arm and return its track plus measurement reports."""
    episode = inputs.episode
    scene = episode.scene
    recording = inputs.recordings[arm]
    video_offset_s = inputs.offsets_s[arm]
    directory = inputs.directory / arm
    measured = measurements.measure(episode, arm, recording, video_offset_s, directory)
    track = trajectory_writers.write_arm_trajectory(
        arm, recording, fusion_result, video_offset_s, coordinate_record, directory,
        width=measured.width, tag_distance=measured.tag_distance,
        export_auxiliary_trajectories=scene.parameters.export_auxiliary_trajectories,
        localization_lenses=('front',) if is_local_scene(coordinate_record) or is_poster_scene(coordinate_record)
        else scene.localization_lenses(arm),
    )
    return (
        track,
        measured.width.report if measured.width is not None else None,
        measured.tag_distance.report if measured.tag_distance is not None else None,
    )
