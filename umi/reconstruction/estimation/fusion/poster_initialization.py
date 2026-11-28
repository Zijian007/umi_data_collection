"""Metric initialization from rear poster observations and the recording IMU."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from umi.core.geometry import imu_from_camera, resample_pose_series
from umi.perception.poster.geometry import camera_model
from umi.perception.poster.pnp import solve_frame
from umi.perception.poster.quality import position_support_metrics
from umi.reconstruction.estimation.fixed_map.geometry import gravity_rotation
from umi.reconstruction.estimation.fusion.data import TrajectorySeed
from umi.reconstruction.estimation.fusion.poster import eligible_sightings
from umi.reconstruction.estimation.metric_initialization import INERTIAL_KEYFRAME_S, fit_training_seed


@dataclass(frozen=True)
class PosterSeed:
    trajectory: TrajectorySeed
    supported_times_s: np.ndarray
    gyro_bias: np.ndarray
    acc_bias: np.ndarray
    diagnostics: dict
    covers_recording: bool = True

    def supported_frames(self) -> np.ndarray:
        """Exclude extrapolation and interpolation across long gaps in rear observations."""
        times, supported = self.trajectory.video_times_s, self.supported_times_s
        right = np.searchsorted(supported, times)
        before = np.maximum(right - 1, 0)
        after = np.minimum(right, len(supported) - 1)
        bracketed = ((right > 0) & (right < len(supported))
                     & (supported[after] - supported[before] <= INERTIAL_KEYFRAME_S))
        return bracketed | (np.abs(supported[after] - times) <= 1e-9)


def _poster_poses(inputs, video, parameters, audit):
    tag_parameters = parameters.environment_tags
    tags = inputs.poster
    observations = tags.observations
    observations.validate()
    rows, times = eligible_sightings(inputs, video, tag_parameters)
    order = np.argsort(times, kind='stable')
    rows, times = rows[order], times[order]
    unique_times, starts, counts = np.unique(times, return_index=True, return_counts=True)
    model = camera_model(tags.measured)
    camera_from_imu = np.linalg.inv(imu_from_camera(tags.measured))
    pixels = observations.corners_px
    accepted_times, accepted_poses, accepted_rms = [], [], []
    for time, start, count in zip(unique_times, starts, counts):
        indices = rows[start:start + count]
        ids = observations.tag_ids[indices]
        # Metric seeding requires multiple distinct markers and a well-conditioned pose.
        if len(np.unique(ids)) < 3 or any(int(tag) not in tags.board.corners_m for tag in ids):
            continue
        points = np.array([tags.board.corners_m[int(tag)] for tag in ids])
        corners = pixels[indices]
        try:
            camera_from_board, rms = solve_frame(points.reshape(-1, 3), corners.reshape(-1, 2), model,
                                                 tag_parameters.tag_maximum_reprojection_rms_px)
        except ValueError:
            continue
        body = np.linalg.inv(camera_from_board) @ camera_from_imu
        geometry = position_support_metrics(np.broadcast_to(body, (count, 4, 4)), np.eye(4), camera_from_imu,
                                            points, corners, model, times[start:start + count], tag_parameters)
        if geometry['supported_frames']:
            accepted_times.append(time)
            accepted_poses.append(body)
            accepted_rms.append(rms)
    times = np.asarray(accepted_times, dtype=float)
    maximum_gap = float(np.diff(np.r_[video[0], times, video[-1]]).max())
    audit.update(eligible_exposures=len(unique_times), supported_exposures=len(times), maximum_pnp_gap_s=maximum_gap,
                 median_pnp_rms_px=float(np.median(accepted_rms)) if accepted_rms else None)
    if len(times) < max(3, tag_parameters.tag_minimum_frames):
        raise ValueError('多标签海报位置观测不足')
    return times, np.asarray(accepted_poses)


def estimate_poster_seed(inputs, parameters, audit, *, video_times_s=None, require_full_coverage=True) -> PosterSeed:
    """Initialize gravity and metric states using exact rear timestamps and the measured lever arm."""
    video = inputs.video_times_s if video_times_s is None else video_times_s
    imu = inputs.imu_video_clock
    if video is None or len(video) < 2 or not np.isfinite(video).all() or np.any(np.diff(video) <= 0):
        raise ValueError('海报初始化需要完整且严格递增的原片时间轴')
    if len(imu) < 2 or video[0] < imu[0, 0] - 1e-9 or video[-1] > imu[-1, 0] + 1e-9:
        raise ValueError('IMU 未覆盖海报初始化的输出时间轴')
    gap = (imu[1:, 0] > video[0]) & (imu[:-1, 0] < video[-1])
    if np.any(np.diff(imu[:, 0])[gap] >= INERTIAL_KEYFRAME_S):
        raise ValueError('海报初始化区间出现整关键帧长度的 IMU 缺口')
    times, body = _poster_poses(inputs, video, parameters, audit)
    covers_recording = audit['maximum_pnp_gap_s'] <= parameters.retry.maximum_position_support_gap_s
    if require_full_coverage and not covers_recording:
        raise ValueError('多标签海报位置观测缺口过长')
    body, gravity, acc_bias, velocities, inertial = fit_training_seed(times, body, imu, parameters.fusion)
    world_from_board = np.eye(4)
    world_from_board[:3, :3] = gravity_rotation(gravity)
    query = np.clip(video, times[0], times[-1])
    poses = world_from_board @ resample_pose_series(times, body, query)
    speeds = np.column_stack([np.interp(query, times, axis) for axis in velocities.T]) @ world_from_board[:3, :3].T
    audit.update(status='accepted', covers_recording=covers_recording, inertial=inertial)
    return PosterSeed(TrajectorySeed(video, poses, speeds), times, np.asarray(inertial['initial_gyro_bias_rad_s']),
                      acc_bias, dict(audit), covers_recording)


def use_poster_seed(aligned, seed):
    return replace(aligned, initial_lens='poster', initial_gyro_bias=seed.gyro_bias,
                   estimates=dict(aligned.estimates, poster={'ba': seed.acc_bias}),
                   diagnostics=dict(aligned.diagnostics, poster_initialization=seed.diagnostics))


def fit_front_placement(centers, rotations, reference, camera_rotation, lever, parameters):
    relative_rotations = reference[:, :3, :3] @ (rotations @ camera_rotation).transpose(0, 2, 1)
    rotation = Rotation.from_matrix(relative_rotations).mean().as_matrix()
    angle = np.degrees(Rotation.from_matrix(rotation.T @ relative_rotations).magnitude())
    source = centers @ rotation.T
    target = reference[:, :3, 3] - (rotation @ rotations) @ lever
    centered = source - source.mean(axis=0)
    spread = float(np.sum(centered**2))
    scale = float(np.sum(centered * (target - target.mean(axis=0))) / spread) if spread > 1e-12 else 0.
    translation = np.mean(target - scale * source, axis=0)
    error = np.linalg.norm(scale * source + translation - target, axis=1)
    relative_std = None
    if scale > 0. and np.linalg.norm(np.ptp(target, axis=0)) >= 2 * parameters.position_sigma_m:
        relative_std = float(np.sqrt(np.sum(error**2) / ((3 * len(centers) - 4) * spread)) / scale)
    placement = {'s': scale, 'R': rotation, 't': translation, 'kind': 'full'}
    return placement, relative_std, angle, error


def front_map_placements(aligned, seed, parameters):
    """Fit one metric placement per ORB map on the locally bracketed common support."""
    front = aligned.lenses['front']
    times = seed.trajectory.video_times_s
    usable = seed.supported_frames()
    candidates, placements, groups, gates = {}, {}, {}, []
    for i, (start, end) in enumerate(front['chains']):
        frames = np.flatnonzero(usable[start:end + 1]) + start
        map_id = front['mapid'].get(start)
        first_node = start + (-start % parameters.node_stride)
        has_graph_node = first_node <= end or end == len(times) - 1
        if len(frames) >= 3 and has_graph_node:
            identity = ('map', map_id) if map_id is not None and parameters.group_scale_ratio > 1 else ('chain', i)
            candidates.setdefault(identity, {})[i] = frames
        else:
            gates.append(dict(lens='front', chain=[start, end], kind='dropped', n_common=len(frames),
                              why='insufficient_poster_overlap' if has_graph_node else 'no_graph_node'))
    for by_chain in candidates.values():
        frames = np.concatenate(list(by_chain.values()))
        if len(frames) < parameters.minimum_common_frames:
            gates.extend(dict(lens='front', chain=list(front['chains'][chain]), kind='dropped',
                              n_common=len(local), why='insufficient_poster_overlap')
                         for chain, local in by_chain.items())
            continue
        centers, rotations = map(np.stack, zip(*(front['obs'][k] for k in frames)))
        placement, rel_std, angles, errors = fit_front_placement(
            centers, rotations, seed.trajectory.T_world_imu[frames], front['R_CI'], front['lever'], parameters)
        metric = rel_std is not None and rel_std <= parameters.maximum_relative_scale_std
        offset, full = 0, []
        for chain, frames in by_chain.items():
            local = slice(offset, offset + len(frames))
            offset += len(frames)
            angle = float(np.median(angles[local]))
            rms = float(np.sqrt(np.mean(errors[local]**2)))
            kind = 'dropped'
            if angle <= parameters.other_lens_rotation_deg:
                if metric and rms <= parameters.other_lens_rms_m:
                    placements[chain] = placement
                    full.append(chain)
                    kind = 'full'
                else:
                    placements[chain] = {'R': placement['R'], 'kind': 'rot'}
                    kind = 'rot'
            gates.append(dict(lens='front', chain=list(front['chains'][chain]), kind=kind, n_common=len(frames),
                              ls_scale=placement['s'], scale_rel_std=rel_std, ls_rms_mm=1000 * rms,
                              rot_med_deg=angle))
        if full:
            leader = max(full, key=lambda i: len(by_chain[i]))
            groups.update({i: leader for i in full if i != leader})
    return placements, groups, gates
