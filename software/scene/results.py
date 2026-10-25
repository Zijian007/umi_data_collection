"""场景任务结果：同场景 TUM、显式视频时钟、双手对齐表与易读报告。"""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from software.scene import engine, fusion
from software.scene.geometry import camera_to_imu, validate_transform
from software.storage import write_json
from software.trajectory.algorithms import postprocess


def resample(times: np.ndarray, poses: np.ndarray, target: np.ndarray) -> np.ndarray:
    if target[0] < times[0]-1e-9 or target[-1] > times[-1]+1e-9:
        raise ValueError('双手对齐不能外推轨迹')
    result = np.tile(np.eye(4), (len(target), 1, 1))
    result[:, :3, :3] = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))(target).as_matrix()
    result[:, :3, 3] = np.column_stack([np.interp(target, times, poses[:, k, 3]) for k in range(3)])
    return result


def write_arm(arm: str, recording: engine.PreparedRecording, result: fusion.FusionResult,
              offset_s: float, scene_record: dict, display: dict, directory: Path) -> tuple[np.ndarray, np.ndarray]:
    directory.mkdir(parents=True)
    details = directory / 'details'
    details.mkdir()
    camera = recording.camera
    measured = camera.measurements['front']
    geometry = json.loads(camera.files['tcp'].read_text())
    tcp = geometry['T_camera_tcp']
    T_cam_tcp = np.eye(4)
    T_cam_tcp[:3, :3], T_cam_tcp[:3, 3] = tcp['R_rows'], tcp['t_m']
    validate_transform(T_cam_tcp)
    poses = dict(imu=result.poses, camera=result.poses @ camera_to_imu(measured))
    poses['tcp'] = poses['camera'] @ T_cam_tcp
    # 文件时间为统一任务视频时钟。IMU 求解使用的 cam_time_offset 不应叠加两次。
    times = recording.times+offset_s
    for name, matrices in poses.items():
        target = directory / 'traj_tcp.txt' if name == 'tcp' else details / f'traj_{name}.txt'
        postprocess.write_tum(target, times, matrices[:, :3, 3], Rotation.from_matrix(matrices[:, :3, :3]).as_quat(),
            f'{name} pose: t x y z qx qy qz qw\nframe_id={scene_record["frame_id"]}; metres, Z up; t_episode=t_video+{offset_s:.9f}\n'
            'Pose maps body coordinates to the fixed scene; no per-recording recentering.')
    with (details / 'frames.csv').open('w') as stream:
        stream.write('row,video_frame,t_video_s,t_scene_s,constraint,front_lens_tracked\n')
        for k, time in enumerate(times):
            stream.write(f'{k},{2*k},{recording.times[k]:.9f},{time:.9f},{"full" if result.tracked[k] else "imu"},{int(result.front_tracked[k])}\n')
    front = recording.video('front').path
    try:
        os.link(front, directory / 'front.mp4')
    except OSError:
        shutil.copy2(front, directory / 'front.mp4')
    length = float(np.linalg.norm(np.diff(poses['tcp'][:, :3, 3], axis=0), axis=1).sum())
    summary = dict(status='ok', mode='scene', arm=arm, scene=scene_record, video=camera.serial,
        rows=len(times), duration_s=float(times[-1]-times[0]), path_length_m=length,
        constraint_fraction=dict(full=float(result.tracked.mean()), imu=float((~result.tracked).mean()), rot=0.),
        calibration_used=camera.selected.summary(), video_time_offset_s=offset_s,
        notes=[], fusion=result.evidence)
    write_json(details / 'summary.json', summary)
    write_json(details / 'map.json', display)
    write_json(details / 'run_settings.json', dict(scene=scene_record, camera=camera.identity(),
        measurements=camera.measurements, video_time_offset_s=offset_s, cad_status=geometry.get('status')))
    text = [f'{arm} 场景轨迹', f'场景坐标：{scene_record["frame_id"]}', f'相机：{camera.serial}',
            f'视频时间偏移：{offset_s:+.9f} s（t_episode = t_video + offset）',
            f'时长：{times[-1]-times[0]:.2f} s；TCP 路程：{length:.3f} m',
            f'地图位置约束：{result.tracked.mean():.1%}',
            f'最长无视觉：{result.evidence["maximum_visual_gap_s"]:.3f} s',
            f'视觉一致性 RMS：{result.evidence["visual_consistency_rms_m"]*1000:.2f} mm（内部指标）',
            '实际误差：未测量；TCP 几何：CAD 名义值。']
    (directory / 'report.txt').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return times, poses['tcp']


def write_bimanual(tracks: dict, directory: Path) -> list[float]:
    start = max(item[0][0] for item in tracks.values())
    end = min(item[0][-1] for item in tracks.values())
    primary = tracks.get('left', next(iter(tracks.values())))[0]
    query = primary[(primary >= start) & (primary <= end)]
    if len(query) < 2:
        raise ValueError('双手录像在统一时钟上没有足够重叠，检查同步偏移')
    if len(tracks) == 2:
        sampled = {arm: resample(t, poses, query) for arm, (t, poses) in tracks.items()}
        with (directory / 'bimanual.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['t_scene_s']+[f'{arm}_{axis}' for arm in ('left', 'right') for axis in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')])
            quaternions = {arm: Rotation.from_matrix(poses[:, :3, :3]).as_quat() for arm, poses in sampled.items()}
            for k, time in enumerate(query):
                writer.writerow([f'{time:.9f}']+[f'{v:.9f}' for arm in ('left', 'right')
                    for v in np.r_[sampled[arm][k, :3, 3], quaternions[arm][k]]])
    return [float(query[0]), float(query[-1])]
