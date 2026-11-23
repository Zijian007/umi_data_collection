"""按真实捕获时间对齐 TCP、夹爪宽度和视频帧，保留视频 PTS 与测量来源。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
import csv

from scipy.spatial.transform import Rotation
import numpy as np

from umi.core.geometry import resample_pose_series, relative_pose_series
from umi.dataset.schema import validate_quality_policy
from umi.perception.gripper import series
from umi.infra.file_store import read_tum
from umi.store.result_io import TCP_COLUMNS, arm_evidence, result_paths, trajectory_path

@dataclass(frozen=True)
class ArmSamples:
    times_s: np.ndarray
    poses: np.ndarray
    video_indices: np.ndarray
    video_pts_s: np.ndarray
    full_constraint: np.ndarray
    front_tracked: np.ndarray
    width_samples: series.WidthSeries | None = None
    position_support: dict[str, np.ndarray] | None = None


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
    path = trajectory_path(directory)
    width_samples = None
    if path.suffix == '.csv':
        with path.open(encoding='utf-8', newline='') as stream:
            trajectory_rows = list(csv.DictReader(stream))
        if trajectory_rows and 'width_mm' in trajectory_rows[0]:
            width_samples = series.from_rows(trajectory_rows)
        values = np.array([[float(row[column]) for column in TCP_COLUMNS[1:]] for row in trajectory_rows])
        times = np.array([float(row['t_episode_s']) for row in trajectory_rows])
        validate_times(times, 'TCP 时间')
        positions, quaternions = values[:, :3], values[:, 3:]
    else:
        times, positions, quaternions = read_tum(path)
        validate_times(times, 'TCP 时间')
    if (positions.shape != (len(times), 3) or quaternions.shape != (len(times), 4)
            or not np.isfinite(positions).all() or not np.isfinite(quaternions).all()
            or not np.allclose(np.linalg.norm(quaternions, axis=1), 1., atol=1e-5)):
        raise ValueError('TCP 位置或四元数无效')
    with arm_evidence(directory, 'frames.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(times):
        raise ValueError('frames.csv 与 TCP 行数不一致')
    # 历史结果的同一任务时钟列名为 t_scene_s。
    time_column = 't_episode_s' if rows and 't_episode_s' in rows[0] else 't_scene_s'
    frame_times = np.array([float(row[time_column]) for row in rows])
    capture_times = np.array([float(row['t_video_s']) for row in rows])
    validate_times(frame_times, '帧映射时间')
    time_tolerance = 1e-8 if path.suffix == '.csv' else 1e-6  # 历史 TUM 时间只保存六位小数。
    if not np.allclose(frame_times, times, atol=time_tolerance, rtol=0):
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
    support = _read_position_support(rows)
    return ArmSamples(frame_times, poses, indices, pts,
                      np.array([row['constraint'] == 'full' for row in rows]),
                      np.array([row['front_lens_tracked'] == '1' for row in rows]), width_samples, support)


def _read_position_support(rows: list[dict]) -> dict[str, np.ndarray] | None:
    fields = ('visual_position_supported', 'poster_observed', 'poster_position_supported', 'position_supported')
    if not rows or not any(field in rows[0] for field in fields):
        return None  # 历史结果的未知证据不能冒充海报未使用。
    if any(field not in row or row[field] not in ('', '0', '1') for row in rows for field in fields):
        raise ValueError('逐帧位置支持状态不完整或无效')
    support = {field: np.array([row[field] == '1' for row in rows])
               for field in fields if all(row[field] != '' for row in rows)}
    visual = support.get('visual_position_supported')
    if visual is None or not np.array_equal(visual, [row['constraint'] == 'full' for row in rows]):
        raise ValueError('自然视觉位置支持与原约束状态不一致')
    poster, observed = support.get('poster_position_supported'), support.get('poster_observed')
    combined = support.get('position_supported')
    if poster is not None:
        if observed is None or np.any(poster & ~observed):
            raise ValueError('海报位置支持缺少观测证据')
        if combined is None or not np.array_equal(combined, visual | poster):
            raise ValueError('共同位置支持必须是自然视觉与海报的并集')
    elif combined is not None:
        raise ValueError('共同位置支持缺少海报几何证据')
    return support


def brackets(times: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.clip(np.searchsorted(times, query), 0, len(times)-1)
    left = np.maximum(right-1, 0)
    left = np.where(np.abs(times[right]-query) <= 1e-9, right, left)
    return left, right


def _supported_samples(arm, left, right):
    """三态判定：任一端已知无支持则拒绝，两端有支持才通过，其余未知。"""
    combined = (arm.position_support or {}).get('position_supported')
    # 缺少海报证据时，自然视觉支持仍是已知的正证据；其余帧保持未知。
    supported = arm.full_constraint if combined is None else combined
    known = arm.full_constraint if combined is None else np.ones(len(combined), bool)
    accepted = supported[left] & supported[right]
    rejected = (known[left] & ~supported[left]) | (known[right] & ~supported[right])
    return accepted, ~accepted & ~rejected


def _segment_ids(query, arms, fps, max_gap_s, quality_policy):
    """低频采样也保留源时间轴中的缺口，防止窗口跨过未被采到的坏段。"""
    breaks = np.diff(query) > 1.5 / fps
    for arm in arms.values():
        gap_prefix = np.r_[0, np.cumsum(np.diff(arm.times_s) > max_gap_s)]
        first = np.searchsorted(arm.times_s[1:], query[:-1], side='right')
        last = np.searchsorted(arm.times_s[:-1], query[1:], side='left')
        breaks |= gap_prefix[last] > gap_prefix[first]
        if quality_policy == 'position_supported':
            indices = np.arange(len(arm.times_s))
            supported, _ = _supported_samples(arm, indices, indices)
            bad_prefix = np.r_[0, np.cumsum(~supported)]
            first = np.searchsorted(arm.times_s, query[:-1], side='right')
            last = np.searchsorted(arm.times_s, query[1:], side='left')
            breaks |= bad_prefix[last] > bad_prefix[first]
    return np.r_[0, np.cumsum(breaks)].astype(np.int64)


def sample(arms: dict[str, ArmSamples], fps: float, max_gap_s: float, *,
           quality_policy: str = 'all', unknown_support: str = 'drop',
           time_window_s: tuple[float, float] | None = None) -> tuple[dict[str, np.ndarray], dict]:
    validate_quality_policy(quality_policy, unknown_support)
    start = max(arm.times_s[0] for arm in arms.values())
    end = min(arm.times_s[-1] for arm in arms.values())
    if time_window_s is not None:
        if (len(time_window_s) != 2 or not np.isfinite(time_window_s).all()
                or time_window_s[1] <= time_window_s[0]):
            raise ValueError('time_window_s 必须是有限的 [start, end] 秒区间，且 end 大于 start')
        start = max(start, time_window_s[0])
        end = min(end, time_window_s[1])
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
    gap_keep = keep.copy()
    selection = dict(quality_policy=quality_policy, unknown_support=unknown_support, arms={})
    if quality_policy == 'position_supported':
        for name, arm in arms.items():
            left, right = brackets(arm.times_s, query)
            supported, unknown = _supported_samples(arm, left, right)
            selection['arms'][name] = dict(
                unsupported_samples=int(np.count_nonzero(gap_keep & ~supported & ~unknown)),
                unknown_samples=int(np.count_nonzero(gap_keep & unknown)))
            if unknown_support == 'error' and np.any(gap_keep & unknown):
                raise ValueError(f'{name} 存在未知位置支持证据，无法按严格质量策略导出')
            keep &= supported
    selection['dropped_quality_samples'] = int(np.count_nonzero(gap_keep & ~keep))
    query = query[keep]
    if len(query) < 2:
        if selection['dropped_quality_samples']:
            raise ValueError('质量筛选后有效样本不足两个：删除了 '
                             f"{selection['dropped_quality_samples']} 个位置支持不足或未知样本")
        raise ValueError('目标时间轴有效样本不足')
    segments = _segment_ids(query, arms, fps, max_gap_s, quality_policy)
    arrays = dict(t_episode_s=query, segment_id=segments)
    quality = dict(dropped_gap_samples=dropped, segments=int(segments[-1])+1,
                   selection=selection, window_s=[float(start), float(end)], arms={})
    for name, arm in arms.items():
        left, right = brackets(arm.times_s, query)
        nearest = np.where(query-arm.times_s[left] <= arm.times_s[right]-query, left, right)
        time_error = arm.times_s[nearest]-query
        tolerance = .5*float(np.median(np.diff(arm.times_s)))+1e-6
        if np.any(np.abs(time_error) > tolerance):
            raise ValueError(f'{name} 图像到样本的时间误差过大')
        indices = arm.video_indices[nearest]
        arrays[f'{name}_T_scene_tcp'] = resample_pose_series(arm.times_s, arm.poses, query)
        arrays[f'{name}_video_frame_index'] = indices
        arrays[f'{name}_video_pts_s'] = arm.video_pts_s[indices]
        arrays[f'{name}_image_time_s'] = arm.times_s[nearest]
        arrays[f'{name}_image_time_error_s'] = time_error
        arrays[f'{name}_full_constraint'] = arm.full_constraint[left] & arm.full_constraint[right]
        arrays[f'{name}_front_tracked'] = arm.front_tracked[left] & arm.front_tracked[right]
        quality['arms'][name] = dict(maximum_image_time_error_s=float(np.abs(time_error).max()),
            full_constraint_fraction=float(arrays[f'{name}_full_constraint'].mean()),
            front_tracked_fraction=float(arrays[f'{name}_front_tracked'].mean()))
        support = arm.position_support or {}
        for field, mask in support.items():
            arrays[f'{name}_{field}'] = mask[left] & mask[right]
        quality['arms'][name]['position_support_available'] = 'position_supported' in support
        quality['arms'][name]['position_support_fractions'] = {
            field: float(arrays[f'{name}_{field}'].mean()) for field in support}
    if 'left' in arms and 'right' in arms:
        arrays['T_left_tcp_right_tcp'] = relative_pose_series(arrays['left_T_scene_tcp'], arrays['right_T_scene_tcp'])
    return arrays, quality


class GripperMeasurement(TypedDict):
    calibration: dict
    inputs: dict
    valid_fraction: float
    sources: dict[str, int]


class GripperReadiness(TypedDict):
    training_ready: bool
    missing_channels: list[str]
    not_ready_reasons: list[str]
    measurements: dict[str, GripperMeasurement]


def _sample_gripper(directory: Path, manifest: dict, arms: dict[str, ArmSamples],
                    arrays: dict[str, np.ndarray], max_gap_s: float) -> GripperReadiness:
    query = arrays['t_episode_s']
    reasons: list[str] = []
    measurements: dict[str, GripperMeasurement] = {}
    missing = False
    for arm, trajectory in arms.items():
        report = manifest.get('gripper_width', {}).get(arm)
        if report is None:
            missing = True
            reasons.append(f"{arm}: missing gripper")
            continue
        measured = trajectory.width_samples
        if measured is None:  # 历史结果的宽度独立存放。
            measured = series.read(directory / arm / 'gripper_width.csv')
            if len(measured.times_s) != len(trajectory.times_s) or not np.allclose(measured.times_s, trajectory.times_s, atol=1e-8, rtol=0):
                raise ValueError(f'{arm} 夹爪宽度与 TCP 不在同一时间轴')
        calibration = report['calibration']
        opening = calibration['open_width_mm']
        if np.any((measured.width_mm[measured.valid] < 0) | (measured.width_mm[measured.valid] > opening)):
            raise ValueError(f'{arm} 有效宽度超出标定范围')
        sampled = series.resample(measured, query, max_gap_s)
        arrays[f"{arm}_gripper_width_m"] = sampled.width_mm / 1000.0
        arrays[f"{arm}_gripper_close_01"] = 1.0 - sampled.width_mm / opening
        arrays[f"{arm}_gripper_valid"] = sampled.valid
        arrays[f"{arm}_gripper_source"] = sampled.source
        measurements[arm] = dict(
            calibration=calibration,
            inputs=report['inputs'],
            valid_fraction=float(sampled.valid.mean()),
            sources={name: int((sampled.source == code).sum()) for code, name in series.SOURCE_NAMES.items()}
        )
        if not sampled.valid.all():
            reasons.append(f"{arm}: gripper contains missing or out-of-range samples")
    return dict(training_ready=not reasons, missing_channels=['gripper'] if missing else [],
                not_ready_reasons=reasons, measurements=measurements)


def sample_episode(directory: Path, manifest: dict, fps: float, max_gap_s: float, *,
                   quality_policy: str, unknown_support: str,
                   time_window_s: tuple[float, float] | None) -> tuple[dict[str, np.ndarray], dict, GripperReadiness]:
    """Sample the accepted episode's poses, frames, and gripper values on one timeline."""
    directory = result_paths(directory).data
    offsets = manifest['synchronization']['offsets_s']
    arms = {arm: read_arm(directory / arm, offsets[arm]) for arm in manifest['arms']}
    arrays, quality = sample(arms, fps, max_gap_s, quality_policy=quality_policy,
                             unknown_support=unknown_support, time_window_s=time_window_s)
    readiness = _sample_gripper(directory, manifest, arms, arrays, max_gap_s)
    return arrays, quality, readiness
