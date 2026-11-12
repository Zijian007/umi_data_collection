"""Fill fixed-map localization gaps with independently verified reverse tracks."""

from __future__ import annotations

from dataclasses import asdict

from scipy.spatial.transform import Rotation
import numpy as np

from .geometry import VisualObservations
from software.scene.models import SceneParameters


def policy(parameters: SceneParameters) -> dict:
    return {key: value for key, value in asdict(parameters).items() if key.startswith('localization_')}


def reverse_window(forward: VisualObservations, video_times_s: np.ndarray,
                   parameters: SceneParameters) -> tuple[int, int] | None:
    """覆盖所有缺失帧，并保留原始时间轴上的定位上下文；长窗口直接整段运行。"""
    present = np.zeros(len(video_times_s), dtype=bool)
    present[forward.indices] = True
    missing = np.flatnonzero(~present)
    if not len(missing):
        return None
    context = parameters.localization_reverse_context_s
    start = int(np.searchsorted(video_times_s, video_times_s[missing[0]] - context))
    end = int(np.searchsorted(video_times_s, video_times_s[missing[-1]] + context, side='right'))
    # 范围过大时，重复启动引擎的代价可能高于省下的帧处理。
    return (start, end) if end - start < .75 * len(video_times_s) else None


def merge(forward: VisualObservations, reverse: VisualObservations, *, metric_scale: float,
          parameters: SceneParameters) -> tuple[VisualObservations, dict]:
    """Keep forward poses; admit reverse-only frames from consistent continuous runs.

    Both inputs use original ascending frame indices/timestamps in the same frozen
    visual map. Translation differences are converted to metres using its scale.
    No alignment, averaging, interpolation or propagation across lost frames.
    """
    if forward.reference_frame != 'visual_map' or reverse.reference_frame != 'visual_map':
        raise ValueError('双向定位必须使用同一视觉地图坐标')
    if not np.isfinite(metric_scale) or metric_scale <= 0:
        raise ValueError('双向定位需要有效的已固定地图尺度')
    for obs in (forward, reverse):
        if (len(obs.indices) != len(obs.imu_times_s) or len(obs.indices) != len(obs.T_reference_camera)
                or np.any(np.diff(obs.indices) <= 0) or np.any(np.diff(obs.imu_times_s) <= 0)
                or not np.isfinite(obs.imu_times_s).all() or not np.isfinite(obs.T_reference_camera).all()):
            raise ValueError('双向定位帧索引、时间或位姿无效')
    lookup = {int(index): row for row, index in enumerate(forward.indices)}
    accepted = []
    segments = []
    boundaries = np.flatnonzero(np.diff(reverse.indices) != 1) + 1
    for rows in np.split(np.arange(len(reverse.indices)), boundaries):
        if not len(rows):
            continue
        overlap = [int(row) for row in rows if int(reverse.indices[row]) in lookup]
        missing = [int(row) for row in rows if int(reverse.indices[row]) not in lookup]
        if not missing:
            continue
        segment = dict(start_frame=int(reverse.indices[rows[0]]), end_frame=int(reverse.indices[rows[-1]]),
                       overlap_frames=len(overlap), candidate_frames=len(missing), accepted_frames=0)
        if len(overlap) < parameters.localization_minimum_overlap_frames:
            segment['status'] = 'insufficient_overlap'
        else:
            original = [lookup[int(reverse.indices[row])] for row in overlap]
            if not np.allclose(forward.imu_times_s[original], reverse.imu_times_s[overlap], rtol=0, atol=1e-7):
                raise ValueError('双向定位同帧时间戳不一致')
            a, b = forward.T_reference_camera[original], reverse.T_reference_camera[overlap]
            position = np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1) * metric_scale
            rotation = np.degrees((Rotation.from_matrix(a[:, :3, :3]).inv()
                                   * Rotation.from_matrix(b[:, :3, :3])).magnitude())
            segment.update(max_position_difference_m=float(position.max()),
                           max_rotation_difference_deg=float(rotation.max()))
            consistent = (position.max() <= parameters.localization_max_position_difference_m
                          and rotation.max() <= parameters.localization_max_rotation_difference_deg)
            segment['status'] = 'accepted' if consistent else 'inconsistent'
            if consistent:
                accepted.extend(missing)
                segment['accepted_frames'] = len(missing)
        segments.append(segment)
    indices = np.r_[forward.indices, reverse.indices[accepted]]
    order = np.argsort(indices)
    merged = VisualObservations(
        np.r_[forward.imu_times_s, reverse.imu_times_s[accepted]][order],
        np.concatenate([forward.T_reference_camera, reverse.T_reference_camera[accepted]])[order],
        indices[order], 'visual_map')
    return merged, dict(status='completed', policy=policy(parameters), metric_scale=metric_scale,
                        forward_frames=len(forward.indices), reverse_frames=len(reverse.indices),
                        added_frames=len(accepted), merged_frames=len(indices), segments=segments)
