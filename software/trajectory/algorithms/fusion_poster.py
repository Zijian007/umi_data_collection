"""独立图的单板角点约束；复用 KB4 几何，保留分段地图尺度变量。"""
from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X

from software.camera.geometry import imu_from_camera, resample_pose_series
from software.vision.poster_pose import (camera_model, estimate_board, project_observations,
                                        residual_and_jacobians, residual_metrics, PosterGeometryError)

BOARD_KEY = gtsam.symbol('h', 0)  # l0 已被独立图的 rear 杆臂占用。


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
        for key,points,pixels in zip(self.node_keys,self.points,self.pixels):
            radius = np.linalg.norm(pixels-[self.model.cx,self.model.cy],axis=1)
            sigma = np.hypot(self.parameters.tag_corner_sigma_px,
                             self.parameters.tag_sigma_per_radius_px*np.maximum(0.,radius/720.-1.))
            noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(self.parameters.tag_huber),
                        gtsam.noiseModel.Diagonal.Sigmas(np.repeat(sigma,2)))
            def error(factor,state,jacobians,points=points,pixels=pixels):
                residual,jbody,jboard = residual_and_jacobians(state.atPose3(factor.keys()[0]).matrix(),
                    state.atPose3(BOARD_KEY).matrix(),self.camera_from_imu,points,pixels,self.model)
                if jacobians is not None: jacobians[0],jacobians[1] = jbody,jboard
                return residual
            graph.add(gtsam.CustomFactor(noise,[X(key),BOARD_KEY],error))

    def measure(self, values, initial_error, final_error):
        board = values.atPose3(BOARD_KEY).matrix()
        bodies = np.array([values.atPose3(X(k)).matrix() for k in self.node_keys])
        pixels,points = project_observations(board,bodies @ np.linalg.inv(self.camera_from_imu),self.points,self.model)
        metrics = residual_metrics(pixels, self.pixels,
                                   frame_ids=self.frame_ids, tag_ids=self.tag_ids)
        rms = metrics['rms_px']
        if (not np.isfinite(rms) or not np.isfinite(final_error) or final_error > initial_error+1e-5
                or rms > self.parameters.tag_maximum_reprojection_rms_px or np.any(points[:,:,2] <= 0)):
            raise PosterGeometryError('独立海报融合未通过正深度、像素残差或优化下降检查')
        return dict(self.evidence, status='fused', final_reprojection_rms_px=rms,
                    final_reprojection_metrics=metrics, T_world_board=board.tolist())


def prepare(inputs, baseline, parameters):
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
            p.tag_maximum_reprojection_rms_px, frame_ids=frame_ids, tag_ids=tag_ids,
            return_metrics=True)
    except (ValueError,RuntimeError) as error:
        if p.tag_fusion_enabled:
            raise PosterGeometryError(f'独立海报初始化失败：{error}',
                                      metrics=getattr(error, 'metrics', None)) from error
        rejected = dict(evidence, status='rejected_diagnostics', reason=str(error))
        if getattr(error, 'metrics', None) is not None:
            rejected['reprojection_metrics'] = error.metrics
        return None, rejected
    evidence.update(status='diagnostics_only', initial_reprojection_rms_px=rms,
                    initial_reprojection_metrics=metrics,
                    T_world_board_initial=pose.tolist(),
                    first_video_clock_time_s=float(times[0]),last_video_clock_time_s=float(times[-1]))
    if not p.tag_fusion_enabled: return None,evidence
    return PosterContext(baseline, times, points, pixels, model, camera_from_imu, pose, p,
                         evidence, frame_ids, tag_ids), evidence
