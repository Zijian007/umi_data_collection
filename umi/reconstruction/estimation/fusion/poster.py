"""独立图的单板角点约束；复用 KB4 几何，保留分段地图尺度变量。"""
from dataclasses import dataclass, replace

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, V, X

from umi.core.geometry import imu_from_camera, resample_pose_series, select_pose_keyframes
from umi.perception.poster.geometry import (
    PosterGeometryError,
    camera_model,
    estimate_board,
    project_observations,
    residual_metrics,
    sighting_rms_px,
)
from umi.perception.poster.quality import (
    cull_decision,
    cull_threshold,
    culling_record,
    isolated_cull,
    position_support_metrics,
    without_sightings,
)
from umi.reconstruction.estimation.fusion.data import TrajectorySeed
from umi.reconstruction.estimation.poster_factors import (
    add_rear_seat,
    add_tag_depth_barrier_factors,
    add_tag_factors,
    native_factors,
    rear_rotation_evidence,
    solved_rear_extrinsic,
)

BOARD_KEY = gtsam.symbol('h', 0)  # l0 已被独立图的 rear 杆臂占用。
REAR_SEAT_KEY = gtsam.symbol('s', 0)
_MIN_VALID_DEPTH_M = 1e-3


@dataclass(frozen=True)
class PosterWarmStart:
    values: gtsam.Values
    node_times: dict[int, float]

    def apply(self, initial, node_times):
        # Rear exposure nodes can be renumbered when an entire exposure is
        # culled. The map segments, lever arms, board and seat retain their keys.
        seed = gtsam.Values(self.values)
        previous_by_time = {t: k for k, t in self.node_times.items()}
        for k in self.node_times:
            for key in (X(k), V(k), B(k)):
                seed.erase(key)
        for k, t in node_times.items():
            old = previous_by_time.get(t)
            if old is not None:
                seed.insert(X(k), self.values.atPose3(X(old)))
                seed.insert(V(k), self.values.atVector(V(old)))
                seed.insert(B(k), self.values.atConstantBias(B(old)))
        initial.update(seed)


class PosterCullRequest(PosterGeometryError):
    """只重建当前海报图；复用同一本录像的视觉/IMU 对齐与基线。"""

    def __init__(self, audit):
        super().__init__('海报优化后剔除稀疏异常并重求解')
        self.audit = audit
        self.warm_start: PosterWarmStart | None = None


@dataclass(frozen=True)
class PosterInputs:
    board: object
    observations: object
    measured: dict


@dataclass
class PosterContext:
    trajectory_seed: TrajectorySeed
    times: np.ndarray
    points: np.ndarray
    pixels: np.ndarray
    model: object
    camera_from_imu: np.ndarray
    board_pose: np.ndarray
    parameters: object
    evidence: dict
    frame_ids: np.ndarray
    tag_ids: np.ndarray
    selected: np.ndarray
    node_keys: list | None = None

    def augment_nodes(self, nodes, frame_times):
        node_times = {k:float(frame_times[k]) for k in nodes}
        keys = []
        sorted_times = np.array([node_times[k] for k in nodes])
        extra = {}
        for time in self.times:
            right = int(np.searchsorted(sorted_times,time))
            exact = next((i for i in (right,right-1) if 0 <= i < len(nodes) and abs(sorted_times[i]-time) <= 1e-9),None)
            if exact is not None: key = nodes[exact]
            else:
                key = extra.setdefault(float(time), len(frame_times)+len(extra))
                node_times[key] = float(time)
            keys.append(key)
        self.node_keys = keys
        self.evidence['added_nodes'] = len(extra)
        return sorted(node_times,key=node_times.get), node_times

    def add_factors(self, graph, values):
        native = native_factors()
        values.insert(BOARD_KEY,gtsam.Pose3(self.board_pose))
        add_rear_seat(graph, values, REAR_SEAT_KEY, self.parameters)
        add_tag_factors(graph, pose_key=X, board_key=BOARD_KEY, rear_key=REAR_SEAT_KEY,
                        base_ext=self.camera_from_imu, model=self.model, points=self.points,
                        pixels=self.pixels, tag_indices=self.node_keys, parameters=self.parameters, native=native)
        add_tag_depth_barrier_factors(
            graph, pose_key=X, board_key=BOARD_KEY, base_ext=self.camera_from_imu,
            points=self.points, tag_indices=self.node_keys, native=native)

    def measure(self, values, initial_error, final_error):
        board = values.atPose3(BOARD_KEY).matrix()
        nodes, sighting_nodes = np.unique(self.node_keys, return_inverse=True)
        node_poses = np.array([values.atPose3(X(int(node))).matrix() for node in nodes])
        bodies = node_poses[sighting_nodes]
        ext, correction_deg = solved_rear_extrinsic(values, REAR_SEAT_KEY, self.camera_from_imu, self.parameters)
        pixels,points = project_observations(board,bodies @ np.linalg.inv(ext),self.points,self.model)
        metrics = residual_metrics(pixels, self.pixels,
                                   frame_ids=self.frame_ids, tag_ids=self.tag_ids)
        rms = metrics['rms_px']
        depths = points[:, :, 2]
        depth_summary = dict(min_m=float(np.min(depths)),
                             p01_m=float(np.percentile(depths, 1)),
                             median_m=float(np.median(depths)),
                             max_m=float(np.max(depths)),
                             minimum_valid_m=_MIN_VALID_DEPTH_M,
                             positive_fraction=float(np.mean(depths >= _MIN_VALID_DEPTH_M)))
        checks = dict(finite=bool(np.isfinite(pixels).all() and np.isfinite(points).all()),
                      optimized_error_decreased=bool(np.isfinite(final_error)
                                                     and final_error <= initial_error + 1e-5),
                      reprojection_rms_px=float(rms),
                      reprojection_limit_px=float(self.parameters.tag_maximum_reprojection_rms_px),
                      reprojection_passed=bool(np.isfinite(rms)
                                               and rms <= self.parameters.tag_maximum_reprojection_rms_px),
                      positive_depth=bool(np.all(depths >= _MIN_VALID_DEPTH_M)))
        invalid_depth_details = [
            dict(observation=int(row), frame=int(self.frame_ids[row]),
                 tag_id=int(self.tag_ids[row]), corner=int(corner),
                 depth_m=float(depths[row, corner]))
            for row, corner in zip(*np.nonzero(depths < _MIN_VALID_DEPTH_M))
        ]
        diagnostics = dict(metrics=metrics, depth_summary=depth_summary,
                           invalid_depth_observations=int(np.count_nonzero(depths < _MIN_VALID_DEPTH_M)),
                           invalid_depth_details=invalid_depth_details,
                           checks=checks, initial_error=float(initial_error), final_error=float(final_error))
        if not checks['finite'] or not checks['optimized_error_decreased']:
            raise PosterGeometryError('独立海报融合未通过有限性或优化下降检查', metrics=diagnostics)
        reprojection_mask = sighting_rms_px(pixels, self.pixels) > cull_threshold(self.parameters)
        depth_mask = np.any(depths < _MIN_VALID_DEPTH_M, axis=1)
        mask = reprojection_mask | depth_mask
        if np.any(mask):
            if isolated_cull(mask, self.frame_ids, self.parameters.tag_minimum_frames,
                             times=self.times, parameters=self.parameters):
                audit = dict(**culling_record(mask, self.selected, self.frame_ids, self.tag_ids,
                    self.times, self.parameters, stage='optimization'),
                    pre_cull_reprojection_rms_px=rms, pre_cull_reprojection_metrics=metrics)
                audit['pre_cull_invalid_depth_sightings'] = int(np.count_nonzero(depth_mask))
                raise PosterCullRequest(audit)
            raise PosterGeometryError('独立海报优化后异常观测超出剔除预算', metrics=dict(metrics,
                culling=cull_decision(mask, self.frame_ids, self.times, self.parameters)))
        if not checks['reprojection_passed'] or not checks['positive_depth']:
            raise PosterGeometryError('独立海报融合未通过正深度或像素残差检查', metrics=diagnostics)
        support = position_support_metrics(bodies, board, ext, self.points, self.pixels,
                                           self.model, self.times, self.parameters)
        return dict(self.evidence, status='fused', final_reprojection_rms_px=rms,
                    final_reprojection_metrics=metrics, final_depth_summary=depth_summary,
                    final_checks=checks, T_world_board=board.tolist(),
                    T_rear_camera_imu_optimized=ext.tolist(), position_support=support,
                    position_support_video_times_s=[frame['video_time_s'] for frame in support['frames']
                                                    if frame['position_supported']],
                    rear_rotation_correction_deg=correction_deg,
                    rear_rotation_refinement=rear_rotation_evidence(correction_deg, self.parameters))


def eligible_sightings(inputs, video_times_s, parameters, *, exposure_times=None):
    tags = inputs.poster
    observations = tags.observations
    # 独立图的 IMU 已减去 front offset；rear 原片 PTS 需加两镜头 offset 之差。
    times = observations.imu_times_s(tags.measured['cam_time_offset_ns']) - inputs.camera['cam_time_offset_ns']/1e9
    valid = observations.usable & (observations.side_px >= parameters.tag_minimum_side_px)
    valid &= (times >= video_times_s[0]) & (times <= video_times_s[-1])
    if exposure_times is not None:
        valid &= np.isin(times, exposure_times)
    selected = np.flatnonzero(valid)
    return selected, times[selected]


def sample_exposures(inputs, seed: TrajectorySeed, parameters, maximum_dt_s):
    """Select exact rear exposure times, keeping the ends of each visible span."""
    _, observations = eligible_sightings(inputs, seed.video_times_s, parameters)
    times = np.unique(observations)
    if len(times) <= parameters.tag_minimum_frames:
        return times, len(times)
    rear_poses = resample_pose_series(seed.video_times_s, seed.T_world_imu, times) @ imu_from_camera(inputs.poster.measured)
    spans = np.split(np.arange(len(times)), np.flatnonzero(np.diff(times) > maximum_dt_s + 1e-9) + 1)
    indices = np.concatenate([span[select_pose_keyframes(
        times[span], rear_poses[span], maximum_dt_s=maximum_dt_s,
        minimum_translation_m=.03, minimum_rotation_deg=3.)] for span in spans])
    return (times[indices] if len(indices) >= parameters.tag_minimum_frames else times), len(times)


def prepare(inputs, seed: TrajectorySeed, parameters, *, exposure_times=None, _allow_cull=True):
    tags = inputs.poster
    observations = tags.observations
    observations.validate()
    p = parameters.environment_tags
    selected, times = eligible_sightings(inputs, seed.video_times_s, p, exposure_times=exposure_times)
    unique_times, sighting_exposures = np.unique(times, return_inverse=True)
    evidence = dict(status='insufficient_observations',enabled=p.tag_fusion_enabled,layout=tags.board.record(),
                    frames=len(unique_times),observations=len(times),actual_error='not_measured',
                    pose_scope='this_recording',timestamp_policy='exact_rear_imu_nodes',timestamp_tolerance_s=1e-9,
                    map_scale='joint_segment_variables')
    if evidence['frames'] < p.tag_minimum_frames: return None,evidence
    if times[0] < inputs.imu_video_clock[0,0] or times[-1] > inputs.imu_video_clock[-1,0]:
        raise ValueError('IMU 未覆盖环境海报观测时刻')
    model = camera_model(tags.measured)
    camera_from_imu = np.linalg.inv(imu_from_camera(tags.measured))
    points = np.array([tags.board.corners_m[int(i)] for i in observations.tag_ids[selected]])
    pixels = observations.corners_px[selected]
    frame_ids = observations.sample_indices[selected]
    tag_ids = observations.tag_ids[selected]
    exposure_poses = resample_pose_series(seed.video_times_s, seed.T_world_imu, unique_times)
    cameras = (exposure_poses @ np.linalg.inv(camera_from_imu))[sighting_exposures]
    try:
        pose, rms, metrics = estimate_board(
            cameras, points, pixels, model,
            float('inf'), frame_ids=frame_ids, tag_ids=tag_ids,
            return_metrics=True)
    except (ValueError,RuntimeError) as error:
        if p.tag_fusion_enabled:
            raise PosterGeometryError(f'独立海报初始化失败：{error}',
                                      metrics=getattr(error, 'metrics', None)) from error
        rejected = dict(evidence, status='rejected_diagnostics', reason=str(error))
        if getattr(error, 'metrics', None) is not None:
            rejected['reprojection_metrics'] = error.metrics
        return None, rejected
    if _allow_cull and p.tag_fusion_enabled:
        prediction, camera_points = project_observations(pose, cameras, points, model)
        with np.errstate(over='ignore', invalid='ignore'):
            sighting_rms = sighting_rms_px(prediction, pixels)
        invalid_corners = ~np.isfinite(camera_points).all(axis=2) | (camera_points[:, :, 2] <= 0)
        cull_mask = ~np.isfinite(sighting_rms) | (sighting_rms > cull_threshold(p)) | invalid_corners.any(axis=1)
        if isolated_cull(cull_mask, frame_ids, p.tag_minimum_frames, times=times, parameters=p):
            filtered = replace(tags, observations=without_sightings(observations, selected, cull_mask))
            refined, refined_evidence = prepare(
                replace(inputs, poster=filtered), seed, parameters, exposure_times=exposure_times, _allow_cull=False)
            refined_evidence.update(culling_record(cull_mask, selected, frame_ids, tag_ids, times,
                                                   p, stage='initialization'),
                                    pre_cull_reprojection_rms_px=float(rms),
                                    pre_cull_reprojection_metrics=metrics)
            return refined, refined_evidence
    evidence.update(status='diagnostics_only', initial_reprojection_rms_px=rms,
                    initial_reprojection_metrics=metrics,
                    T_world_board_initial=pose.tolist(),
                    first_video_clock_time_s=float(times[0]),last_video_clock_time_s=float(times[-1]),
                    observation_video_times_s=unique_times.tolist())
    if not p.tag_fusion_enabled:
        if rms > p.tag_maximum_reprojection_rms_px:
            evidence.update(status='rejected_diagnostics', reason='独立海报与基线重投影不一致',
                            reprojection_metrics=metrics)
        return None,evidence
    return PosterContext(seed, times, points, pixels, model, camera_from_imu, pose, p,
                         evidence, frame_ids, tag_ids, selected), evidence
