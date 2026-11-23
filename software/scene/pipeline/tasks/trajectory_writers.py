"""后端共用的单手测量、轨迹表和双手对齐表写入。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import csv
import json
import os
import shutil

from scipy.spatial.transform import Rotation
import numpy as np

from software.camera.geometry import (imu_from_camera, camera_from_tcp, tcp_frame_definition,
                                      resample_pose_series, relative_pose_series)
from software.camera.x3.frame import geometry_record
from software.gripper_width.width_estimation import WidthResult
from software.gripper_width.series import SOURCE_NAMES
from software.trajectory.tum import write_tum
from software.trajectory.reporting.constraint_quality import position_support
from software.infrastructure.concurrency import daemon_map
from software.scene.models import ArmExecutionResult, ArmTrajectory, PreparedRecording, TaskInputs
from software.trajectory.models import FusedTrajectory
from software.scene.result_io import is_local_scene, is_poster_scene

if TYPE_CHECKING:
    from software.gripper_width.distance.measurement import TagDistanceResult


def write_arm_result(inputs: TaskInputs, arm: str, fusion_result: FusedTrajectory, coordinate_record: dict):
    """Write one solved arm and return its track plus measurement reports."""
    from software.scene.pipeline.tasks import measurements

    episode = inputs.episode
    scene = episode.scene
    recording = inputs.recordings[arm]
    video_offset_s = inputs.offsets_s[arm]
    directory = inputs.directory / arm
    measured = measurements.measure(episode, arm, recording, video_offset_s, directory)
    track = write_arm_trajectory(
        recording, fusion_result, video_offset_s, coordinate_record, directory,
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


def collect_arm_results(written, coordinate_record: dict) -> ArmExecutionResult:
    tracks, width_reports, tag_distance_reports = {}, {}, {}
    for arm, (track, width, tag_distance) in written:
        tracks[arm] = track
        if width is not None:
            width_reports[arm] = width
        if tag_distance is not None:
            tag_distance_reports[arm] = tag_distance
    return ArmExecutionResult(tracks, width_reports, tag_distance_reports, coordinate_record)


def write_task_results(inputs: TaskInputs, trajectories: dict[str, FusedTrajectory],
                       coordinate_record: dict, *, parallel=False) -> ArmExecutionResult:
    """Write arms after the backend has accepted their shared coordinate frame."""
    def write(arm):
        return arm, write_arm_result(inputs, arm, trajectories[arm], coordinate_record)

    arms = tuple(trajectories)
    written = daemon_map(write, arms) if parallel else map(write, arms)
    return collect_arm_results(written, coordinate_record)


def write_arm_trajectory(recording: PreparedRecording, result: FusedTrajectory,
              offset_s: float, coordinate_record: dict, directory: Path,
              *, width: WidthResult | None = None, tag_distance: TagDistanceResult | None = None,
              export_auxiliary_trajectories: bool = False,
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
    poses = dict(imu=result.T_reference_imu, camera=result.T_reference_imu @ imu_from_camera(measured))
    poses['tcp'] = poses['camera'] @ T_cam_tcp
    # 文件时间为统一任务视频时钟。IMU 求解使用的 cam_time_offset 不应叠加两次。
    times = recording.video_times_s + offset_s
    if tag_distance is not None and (len(tag_distance.samples.times_s) != len(times)
            or not np.allclose(tag_distance.samples.times_s, times, rtol=0, atol=1e-8)):
        raise ValueError('标签间距与轨迹时间轴不一致')
    distance_valid = tag_distance.samples.valid if tag_distance is not None else None
    for name in ('camera', 'imu') if export_auxiliary_trajectories else ():
        matrices = poses[name]
        write_tum(details / f'traj_{name}.txt', times, matrices[:, :3, 3], Rotation.from_matrix(matrices[:, :3, :3]).as_quat(),
            f'{name} pose: t x y z qx qy qz qw\nframe_id={coordinate_record["frame_id"]}; metres; t_episode=t_video+{offset_s:.9f}\n'
            + ('Pose maps body coordinates to this recording local frame; not a shared scene.'
             if is_local_scene(coordinate_record) else 'Pose maps body coordinates to the task poster layout frame.'
             if is_poster_scene(coordinate_record) else
             'Pose maps body coordinates to the fixed scene; no per-recording recentering.'))
    quaternions = Rotation.from_matrix(poses['tcp'][:, :3, :3]).as_quat()
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
                *[f'{value:.6f}' for value in poses['tcp'][index, :3, 3]],
                *[f'{value:.9f}' for value in quaternions[index]], *opening, *distance_columns])
    support = position_support(recording.video_times_s, result.tracked,
                               result.evidence.get('environment_poster'))
    support_fields = tuple(support)
    frame_fields = ('row', 'video_frame', 'source_video_frame', 't_video_s', 't_episode_s',
                    'constraint', 'visual_lens', 'visual_lenses', 'map_role', 'map_index',
                    'map_epoch', 'map_version', 'map_frame_id', 'front_lens_tracked',
                    *support_fields, 'position_source')
    frame_rows = []
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
            frame_rows.append(dict(zip(frame_fields, columns, strict=True)))
    front = recording.lens_video('front').path
    try:
        os.link(front, directory / 'front.mp4')
    except OSError:
        shutil.copy2(front, directory / 'front.mp4')
    length = float(np.linalg.norm(np.diff(poses['tcp'][:, :3, 3], axis=0), axis=1).sum())
    metadata = dict(
        rows=len(times), duration_s=float(times[-1] - times[0]), path_length_m=length,
        constraint_fraction=dict(full=float(result.tracked.mean()), imu=float((~result.tracked).mean()), rot=0.),
        calibration_used=camera.selected.summary(),
        notes=[], fusion=result.evidence,
        measurements=camera.measurements, cad_status=geometry.get('status'),
        T_camera_tcp=geometry['T_camera_tcp'],
        frame_geometry=geometry_record(video_parameters.size_px,
            mode='dual' if 'rear' in localization_lenses else 'shared_front', video=video_parameters))
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
    return ArmTrajectory(times, poses['tcp'], coordinate_record['frame_id'], tcp_frame, metadata, frame_rows)


def write_bimanual_trajectory(tracks: dict[str, ArmTrajectory], directory: Path) -> list[float]:
    if len({track.frame_id for track in tracks.values()}) != 1:
        raise ValueError('双手轨迹必须使用同一个固定场景坐标系')
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

