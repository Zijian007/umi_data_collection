"""独立图的单板角点约束；复用 KB4 几何，保留分段地图尺度变量。"""
from dataclasses import dataclass, replace

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X

from software.camera.geometry import imu_from_camera, resample_pose_series
from software.vision.poster.geometry import (
    camera_model,
    estimate_board,
    project_observations,
    residual_metrics,
    PosterGeometryError,
)
from software.vision.poster.quality import (
    position_support_metrics,
    cull_threshold,
    isolated_cull,
    without_sightings,
    culling_record,
    cull_decision,
)
from software.vision.poster.factors import (
    add_rear_seat, add_tag_depth_barrier_factors, add_tag_factors,
    solved_rear_extrinsic, rear_rotation_evidence)

BOARD_KEY = gtsam.symbol('h', 0)  # l0 已被独立图的 rear 杆臂占用。
REAR_SEAT_KEY = gtsam.symbol('s', 0)


class PosterCullRequest(PosterGeometryError):
    """只重建当前海报图；复用同一本录像的视觉/IMU 对齐与基线。"""

    def __init__(self, selected, audit):
        super().__init__('海报优化后剔除稀疏异常并重求解')
        self.selected, self.audit = selected, audit


@dataclass(frozen=True)
class PosterInputs:
    board: object
    observations: object
    measured: dict




@dataclass
class PosterContext:
    baseline: object
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
        values.insert(BOARD_KEY,gtsam.Pose3(self.board_pose))
        add_rear_seat(graph, values, REAR_SEAT_KEY, self.parameters)
        add_tag_factors(graph, pose_key=X, board_key=BOARD_KEY, rear_key=REAR_SEAT_KEY,
                        base_ext=self.camera_from_imu, model=self.model, points=self.points,
                        pixels=self.pixels, tag_indices=self.node_keys, parameters=self.parameters)
        add_tag_depth_barrier_factors(
            graph, pose_key=X, board_key=BOARD_KEY, base_ext=self.camera_from_imu,
            points=self.points, tag_indices=np.asarray(self.node_keys, dtype=int))

    def measure(self, values, initial_error, final_error):
        board = values.atPose3(BOARD_KEY).matrix()
        bodies = np.array([values.atPose3(X(k)).matrix() for k in self.node_keys])
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
                             positive_fraction=float(np.mean(depths > 0)))
        checks = dict(finite=bool(np.isfinite(pixels).all() and np.isfinite(points).all()),
                      optimized_error_decreased=bool(np.isfinite(final_error)
                                                     and final_error <= initial_error + 1e-5),
                      reprojection_rms_px=float(rms),
                      reprojection_limit_px=float(self.parameters.tag_maximum_reprojection_rms_px),
                      reprojection_passed=bool(np.isfinite(rms)
                                               and rms <= self.parameters.tag_maximum_reprojection_rms_px),
                      positive_depth=bool(np.all(depths > 0)))
        diagnostics = dict(metrics=metrics, depth_summary=depth_summary,
                           invalid_depth_observations=int(np.count_nonzero(depths <= 0)),
                           checks=checks, initial_error=float(initial_error), final_error=float(final_error))
        if not checks['finite'] or not checks['optimized_error_decreased']:
            raise PosterGeometryError('独立海报融合未通过有限性或优化下降检查', metrics=diagnostics)
        reprojection_mask = np.sqrt(
            np.mean(np.linalg.norm(pixels-self.pixels, axis=2)**2, axis=1)
        ) > cull_threshold(self.parameters)
        depth_mask = np.any(depths <= 0, axis=1)
        mask = reprojection_mask | depth_mask
        if np.any(mask):
            if (not self.evidence.get('culled_observations') and isolated_cull(mask, self.frame_ids,
                    self.parameters.tag_minimum_frames, times=self.times, parameters=self.parameters)):
                audit = dict(**culling_record(mask, self.selected, self.frame_ids, self.tag_ids,
                    self.times, self.parameters, stage='optimization'),
                    pre_cull_reprojection_rms_px=rms, pre_cull_reprojection_metrics=metrics)
                audit['pre_cull_invalid_depth_sightings'] = int(np.count_nonzero(depth_mask))
                raise PosterCullRequest(self.selected[mask], audit)
            raise PosterGeometryError('独立海报优化后异常观测无法在单轮预算内剔除', metrics=dict(metrics,
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


def prepare(inputs, baseline, parameters, *, _allow_cull=True):
    tags = inputs.poster
    observations = tags.observations
    observations.validate()
    p = parameters.environment_tags
    # 独立图的 IMU 已减去 front offset；rear 原片 PTS 需加两镜头 offset 之差。
    times = observations.imu_times_s(tags.measured['cam_time_offset_ns'])-inputs.camera['cam_time_offset_ns']/1e9
    valid = observations.usable & (observations.side_px >= p.tag_minimum_side_px)
    valid &= (times >= baseline.video_times_s[0]) & (times <= baseline.video_times_s[-1])
    selected = np.flatnonzero(valid)
    times = times[selected]
    evidence = dict(status='insufficient_observations',enabled=p.tag_fusion_enabled,layout=tags.board.record(),
                    frames=len(np.unique(times)),observations=len(times),actual_error='not_measured',
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
    bodies = resample_pose_series(baseline.video_times_s,baseline.T_world_imu,times)
    try:
        pose, rms, metrics = estimate_board(
            bodies @ np.linalg.inv(camera_from_imu), points, pixels, model,
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
    prediction, camera_points = project_observations(
        pose, bodies @ np.linalg.inv(camera_from_imu), points, model)
    with np.errstate(over='ignore', invalid='ignore'):
        errors = np.linalg.norm(prediction - pixels, axis=2)
        sighting_rms = np.sqrt(np.mean(errors ** 2, axis=1))
    invalid_geometry = ((not np.isfinite(camera_points).all())
                        or np.any(camera_points[:, :, 2] <= 0))
    cull_limit = cull_threshold(p)
    cull_mask = (~np.isfinite(sighting_rms) | (sighting_rms > cull_limit))
    if invalid_geometry:
        cull_mask |= (~np.isfinite(camera_points).all(axis=2) | (camera_points[:, :, 2] <= 0)).any(axis=1)
    if (_allow_cull and p.tag_fusion_enabled and isolated_cull(cull_mask, frame_ids,
            p.tag_minimum_frames, times=times, parameters=p)):
        filtered = replace(inputs.poster, observations=without_sightings(observations, selected, cull_mask))
        refined, refined_evidence = prepare(
            replace(inputs, poster=filtered), baseline, parameters, _allow_cull=False)
        refined_evidence = dict(refined_evidence,
                                **culling_record(cull_mask, selected, frame_ids, tag_ids, times,
                                                 p, stage='initialization'),
                                pre_cull_reprojection_rms_px=float(rms),
                                pre_cull_reprojection_metrics=metrics)
        if refined is None:
            return None, refined_evidence
        return replace(refined, evidence=refined_evidence), refined_evidence
    evidence.update(status='diagnostics_only', initial_reprojection_rms_px=rms,
                    initial_reprojection_metrics=metrics,
                    T_world_board_initial=pose.tolist(),
                    first_video_clock_time_s=float(times[0]),last_video_clock_time_s=float(times[-1]),
                    observation_video_times_s=np.unique(times).tolist())
    if not p.tag_fusion_enabled:
        if rms > p.tag_maximum_reprojection_rms_px:
            evidence.update(status='rejected_diagnostics', reason='独立海报与基线重投影不一致',
                            reprojection_metrics=metrics)
        return None,evidence
    return PosterContext(baseline, times, points, pixels, model, camera_from_imu, pose, p,
                         evidence, frame_ids, tag_ids, selected), evidence
