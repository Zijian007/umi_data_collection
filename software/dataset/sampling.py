"""以真实捕获时间对齐 TCP 与视频帧；保留视频 PTS 供独立读取。"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from software.camera.geometry import resample_pose_series
from software.pose_io import read_tum


@dataclass(frozen=True)
class ArmSamples:
    times_s: np.ndarray
    poses: np.ndarray
    video_indices: np.ndarray
    video_pts_s: np.ndarray
    full_constraint: np.ndarray
    front_tracked: np.ndarray


def video_timestamps(path: Path) -> np.ndarray:
    import av

    values = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f'视频帧缺少 PTS：{path}')
            values.append(float(frame.pts * frame.time_base))
    times = np.asarray(values, dtype=np.float64)
    validate_times(times, '视频 PTS')
    return times


def validate_times(times: np.ndarray, label: str) -> None:
    if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError(f'{label} 必须包含至少两个有限且严格递增的时间戳')


def read_arm(directory: Path, offset_s: float) -> ArmSamples:
    times, positions, quaternions = read_tum(directory / 'traj_tcp.txt')
    validate_times(times, 'TCP 时间')
    if (positions.shape != (len(times), 3) or quaternions.shape != (len(times), 4)
            or not np.isfinite(positions).all() or not np.isfinite(quaternions).all()
            or not np.allclose(np.linalg.norm(quaternions, axis=1), 1., atol=1e-5)):
        raise ValueError('TCP 位置或四元数无效')
    with (directory / 'details/frames.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(times):
        raise ValueError('frames.csv 与 TCP 行数不一致')
    frame_times = np.array([float(row['t_scene_s']) for row in rows])
    capture_times = np.array([float(row['t_video_s']) for row in rows])
    validate_times(frame_times, '帧映射时间')
    if not np.allclose(frame_times, times, atol=1e-6, rtol=0):
        raise ValueError('帧映射与 TCP 时间不一致')
    if not np.allclose(capture_times + offset_s, frame_times, atol=1e-8, rtol=0):
        raise ValueError('帧映射不符合 t_episode = t_video + offset')
    indices = np.array([int(row['video_frame']) for row in rows], dtype=np.int64)
    pts = video_timestamps(directory / 'front.mp4')
    if indices.min() < 0 or indices.max() >= len(pts) or not np.all(np.diff(indices) > 0):
        raise ValueError('frames.csv 视频帧索引越界或非递增，请重新生成结果')
    if any(row['constraint'] not in ('full', 'imu') or row['front_lens_tracked'] not in ('0', '1') for row in rows):
        raise ValueError('帧映射包含未知跟踪状态')
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    poses[:, :3, 3] = positions
    poses[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
    return ArmSamples(frame_times, poses, indices, pts,
                      np.array([row['constraint'] == 'full' for row in rows]),
                      np.array([row['front_lens_tracked'] == '1' for row in rows]))


def brackets(times: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.clip(np.searchsorted(times, query), 0, len(times)-1)
    left = np.maximum(right-1, 0)
    left = np.where(np.abs(times[right]-query) <= 1e-9, right, left)
    return left, right


def sample(arms: dict[str, ArmSamples], fps: float, max_gap_s: float) -> tuple[dict[str, np.ndarray], dict]:
    start = max(arm.times_s[0] for arm in arms.values())
    end = min(arm.times_s[-1] for arm in arms.values())
    if end <= start:
        raise ValueError('轨迹没有共同有效时间区间')
    first = int(np.ceil(start*fps))
    last = int(np.floor(end*fps))
    query = np.arange(first, last+1, dtype=np.float64)/fps
    keep = np.ones(len(query), dtype=bool)
    for arm in arms.values():
        left, right = brackets(arm.times_s, query)
        keep &= arm.times_s[right]-arm.times_s[left] <= max_gap_s
    dropped = int(np.count_nonzero(~keep))
    query = query[keep]
    if len(query) < 2:
        raise ValueError('目标时间轴有效样本不足')
    segments = np.r_[0, np.cumsum(np.diff(query) > 1.5/fps)].astype(np.int64)
    arrays = dict(t_episode_s=query, segment_id=segments)
    quality = dict(dropped_gap_samples=dropped, segments=int(segments[-1])+1, arms={})
    for name, arm in arms.items():
        left, right = brackets(arm.times_s, query)
        nearest = np.where(query-arm.times_s[left] <= arm.times_s[right]-query, left, right)
        time_error = arm.times_s[nearest]-query
        tolerance = .5*float(np.median(np.diff(arm.times_s)))+1e-6
        if np.any(np.abs(time_error) > tolerance):
            raise ValueError(f'{name} 图像到样本的时间误差过大')
        indices = arm.video_indices[nearest]
        arrays[name+'_T_scene_tcp'] = resample_pose_series(arm.times_s, arm.poses, query)
        arrays[name+'_video_frame_index'] = indices
        arrays[name+'_video_pts_s'] = arm.video_pts_s[indices]
        arrays[name+'_image_time_s'] = arm.times_s[nearest]
        arrays[name+'_image_time_error_s'] = time_error
        arrays[name+'_full_constraint'] = arm.full_constraint[left] & arm.full_constraint[right]
        arrays[name+'_front_tracked'] = arm.front_tracked[left] & arm.front_tracked[right]
        quality['arms'][name] = dict(maximum_image_time_error_s=float(np.abs(time_error).max()),
            full_constraint_fraction=float(arrays[name+'_full_constraint'].mean()),
            front_tracked_fraction=float(arrays[name+'_front_tracked'].mean()))
    return arrays, quality
