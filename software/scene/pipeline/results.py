"""场景任务结果：同场景 TUM、显式视频时钟、双手对齐表与易读报告。"""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from software import config
from software.camera.x3.frame import geometry_record
from software.camera.geometry import (
    imu_from_camera, camera_from_tcp, tcp_frame_definition, validate_transform,
    describe_tcp_frame, resample_pose_series,
)
from software.scene.models import Episode, PreparedRecording, ArmTrajectory
from software.scene.spatial import fusion
from software.scene.management import inputs
from software.storage import write_json
from software.pose_io import write_tum


def write_arm(arm: str, recording: PreparedRecording, result: fusion.FusionResult,
              offset_s: float, scene_record: dict, directory: Path) -> ArmTrajectory:
    directory.mkdir(parents=True)
    details = directory / 'details'
    details.mkdir()
    camera = recording.camera
    video_parameters = recording.context.parameters.video
    measured = camera.measurements['front']
    geometry = json.loads(camera.files['tcp'].read_text())
    tcp_frame = tcp_frame_definition(geometry)
    T_cam_tcp = camera_from_tcp(geometry)
    validate_transform(T_cam_tcp)
    poses = dict(imu=result.T_scene_imu, camera=result.T_scene_imu @ imu_from_camera(measured))
    poses['tcp'] = poses['camera'] @ T_cam_tcp
    # 文件时间为统一任务视频时钟。IMU 求解使用的 cam_time_offset 不应叠加两次。
    times = recording.video_times_s + offset_s
    for name, matrices in poses.items():
        target = directory / 'traj_tcp.txt' if name == 'tcp' else details / f'traj_{name}.txt'
        write_tum(target, times, matrices[:, :3, 3], Rotation.from_matrix(matrices[:, :3, :3]).as_quat(),
            f'{name} pose: t x y z qx qy qz qw\nframe_id={scene_record["frame_id"]}; metres, Z up; t_episode=t_video+{offset_s:.9f}\n'
            'Pose maps body coordinates to the fixed scene; no per-recording recentering.' +
            ('\nTCP frame: ' + json.dumps(tcp_frame, ensure_ascii=False) if name == 'tcp' else ''))
    with (details / 'frames.csv').open('w') as stream:
        stream.write('row,video_frame,source_video_frame,t_video_s,t_scene_s,constraint,front_lens_tracked\n')
        for k, time in enumerate(times):
            source_frame = video_parameters.frame_step * k
            stream.write(f'{k},{k},{source_frame},{recording.video_times_s[k]:.9f},{time:.9f},{"full" if result.tracked[k] else "imu"},{int(result.front_tracked[k])}\n')
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
            mode='dual' if 'rear' in camera.measurements else 'shared_front', video=video_parameters))
    return ArmTrajectory(times, poses['tcp'], scene_record['frame_id'], tcp_frame, metadata)


def write_bimanual(tracks: dict[str, ArmTrajectory], directory: Path) -> list[float]:
    if len({track.frame_id for track in tracks.values()}) != 1:
        raise ValueError('双手轨迹必须使用同一个固定场景坐标系')
    start = max(track.episode_times_s[0] for track in tracks.values())
    end = min(track.episode_times_s[-1] for track in tracks.values())
    primary = tracks.get('left', next(iter(tracks.values()))).episode_times_s
    query = primary[(primary >= start) & (primary <= end)]
    if len(query) < 2:
        raise ValueError('任务录像没有足够共同有效时间，请检查原片完整性')
    if len(tracks) == 2:
        sampled = {arm: resample_pose_series(track.episode_times_s, track.T_scene_tcp, query)
                   for arm, track in tracks.items()}
        with (directory / 'bimanual.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['t_scene_s'] + [f'{arm}_{axis}' for arm in ('left', 'right') for axis in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')])
            quaternions = {arm: Rotation.from_matrix(poses[:, :3, :3]).as_quat() for arm, poses in sampled.items()}
            for k, time in enumerate(query):
                writer.writerow([f'{time:.9f}'] + [f'{v:.9f}' for arm in ('left', 'right')
                    for v in np.r_[sampled[arm][k, :3, 3], quaternions[arm][k]]])
    return [float(query[0]), float(query[-1])]


def write_episode(episode: Episode, tracks: dict[str, ArmTrajectory], directory: Path, *,
                  common_interval_s: list[float], scene_record: dict, input_identity: dict,
                  video_offsets_s: dict[str, float], elapsed_s: float, map_path: Path,
                  gripper_width: dict | None = None) -> None:
    """写任务报告与资产清单；不更新 results 指针或人工验收状态。"""
    text = (f'场景任务：{episode.root.name}\n采集批次：{episode.session.name}\n坐标：{scene_record["frame_id"]}\n'
        f'操作指令：{episode.instruction or "未填写"}\n'
        f'手：{", ".join(tracks)}\n共同有效时间：{common_interval_s[0]:.6f}–{common_interval_s[1]:.6f} s\n'
        f'同步：{episode.synchronization["method"]}；offsets_s={video_offsets_s}\n'
        f'同步确认：{episode.synchronization["confirmed"]}；依据：{episode.synchronization["note"]}\n'
        f'夹爪宽度：{", ".join(gripper_width or {}) or "未配置标定"}\n'
        '轨迹共享米制坐标；左右手分别融合。\n实际误差：未测量；TCP 使用 CAD 名义几何。\n')
    # 预检先独立落盘，保证算法失败时也有证据；成功后收入唯一运行清单。
    preflight_path = directory / 'preflight_report.json'
    preflight = json.loads(preflight_path.read_text()) if preflight_path.is_file() else None
    text += '\n原片检查：' + ('跳过' if preflight is None else 'PASS' if preflight['passed'] else 'FAIL') + '\n'
    if preflight is not None:
        for arm, video in preflight.get('videos', {}).items():
            for check in video['checks']:
                if check['status'] != 'PASS':
                    text += f'{arm} [{check["status"]}] {check["name"]}：{check["message"]}\n'
    for arm, track in tracks.items():
        data = track.metadata
        text += '\n' + '\n'.join([
            f'{arm} 场景轨迹', f'相机：{input_identity["cameras"][arm]["serial"]}',
            'TCP 自身坐标：' + describe_tcp_frame(track.tcp_frame),
            f'视频时间偏移：{video_offsets_s[arm]:+.9f} s（t_episode = t_video + offset）',
            f'时长：{data["duration_s"]:.2f} s；TCP 路程：{data["path_length_m"]:.3f} m',
            f'地图位置约束：{data["constraint_fraction"]["full"]:.1%}',
            f'最长无视觉：{data["fusion"]["maximum_visual_gap_s"]:.3f} s',
            f'视觉一致性 RMS：{data["fusion"]["visual_consistency_rms_m"] * 1000:.2f} mm（内部指标）',
        ]) + '\n'
    (directory / 'report.txt').write_text(text, encoding='utf-8')
    assets = inputs.artifact_hashes(directory, tuple(tracks))
    for name in ('report.txt', 'bimanual.csv'):
        if (directory / name).is_file():
            assets[name] = config.file_hash(directory / name)
    primary_times = tracks.get('left', next(iter(tracks.values()))).episode_times_s
    metadata = dict(kind='umi_scene_episode', schema_version=3, status='ok', name=episode.root.name, session=episode.session.name, kit=episode.session.kit, scene=scene_record,
        frame_id=scene_record['frame_id'], arms={arm: arm for arm in tracks}, common_interval_s=common_interval_s,
        rows=int(np.count_nonzero((primary_times >= common_interval_s[0]) & (primary_times <= common_interval_s[1]))),
        instruction=episode.instruction, synchronization=episode.synchronization,
        gripper_width=gripper_width or {},
        inputs=input_identity, assets=assets, elapsed_s=elapsed_s, actual_error='not_measured',
        tcp_frames={arm: track.tcp_frame for arm, track in tracks.items()},
        arm_results={arm: track.metadata for arm, track in tracks.items()}, preflight=preflight,
        map=dict(path=os.path.relpath(map_path, directory), sha256=config.file_hash(map_path)))
    write_json(directory / 'episode.json', metadata)
    preflight_path.unlink(missing_ok=True)
