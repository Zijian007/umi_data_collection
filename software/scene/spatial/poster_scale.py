"""建图海报尺度：固定视觉地图形状，联合一个尺度和本录像板位姿。"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from software.camera.geometry import imu_from_camera
from software.scene.spatial.fusion import sample_observations
from software.vision.poster_pose import camera_model, estimate_board, project_observations, PosterGeometryError


class PosterScaleConflict(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def estimate_scale(observations, map_lens_measurement, rear_measurement, detections, board, initial_scale, parameters):
    """标签只使用连续地图段内的插值；镜头间米制杆臂不能跟地图一起缩放。"""
    detections.validate()
    evidence = dict(status='insufficient_observations', layout=board.record(), actual_error='not_measured')
    all_times = detections.imu_times_s(rear_measurement['cam_time_offset_ns'])
    selected = np.flatnonzero(detections.usable & (detections.side_px >= parameters.tag_minimum_side_px))
    requested_times = np.unique(all_times[selected])
    sampled = sample_observations(observations, requested_times, parameters.alignment_max_gap_s)
    selected = selected[np.isin(all_times[selected], sampled.imu_times_s)]
    evidence.update(frames=len(sampled.imu_times_s), observations=len(selected),
                    excluded_without_map_support=int(np.sum(detections.usable)-len(selected)))
    if len(sampled.imu_times_s) < parameters.tag_mapping_minimum_frames:
        return evidence
    points = np.array([board.corners_m[int(i)] for i in detections.tag_ids[selected]])
    board_extent = np.ptp(points.reshape(-1,3), axis=0)[:2]
    motion = float(np.linalg.norm(np.ptp(sampled.T_reference_camera[:, :3, 3], axis=0))*initial_scale)
    evidence.update(motion_extent_m=motion, board_extent_m=board_extent.tolist())
    if motion < parameters.tag_mapping_minimum_motion_m or np.min(board_extent) < parameters.tag_mapping_minimum_board_extent_m:
        return dict(evidence, status='insufficient_geometry')

    raw = sampled.T_reference_camera[np.searchsorted(sampled.imu_times_s, all_times[selected])].copy()
    origin = sampled.T_reference_camera[:, :3, 3].mean(axis=0)
    raw[:, :3, 3] -= origin  # 消除原始地图大平移造成的尺度/板平移数值相关性。
    map_camera_from_rear = np.linalg.inv(imu_from_camera(map_lens_measurement)) @ imu_from_camera(rear_measurement)

    def cameras(scale):
        metric = raw.copy()
        metric[:, :3, 3] *= scale
        return metric @ map_camera_from_rear

    model = camera_model(rear_measurement)
    pixels = detections.corners_px[selected]
    radius = np.linalg.norm(pixels-[model.cx, model.cy], axis=2)
    sigma = np.hypot(parameters.tag_corner_sigma_px,
                     parameters.tag_sigma_per_radius_px*np.maximum(0., radius/720.-1.))
    try:
        seed, _ = estimate_board(cameras(initial_scale), points, pixels, model, float('inf'))
    except PosterGeometryError as error:
        raise PosterScaleConflict(str(error), dict(evidence, status='conflict')) from error

    def unpack(state):
        pose = np.eye(4)
        pose[:3,:3] = Rotation.from_rotvec(state[1:4]).as_matrix()
        pose[:3,3] = state[4:7]
        return float(np.exp(state[0])), pose

    def residual(state):
        scale, pose = unpack(state)
        prediction, _ = project_observations(pose, cameras(scale), points, model)
        return ((prediction-pixels)/sigma[:,:,None]).ravel()

    initial = np.r_[np.log(initial_scale), Rotation.from_matrix(seed[:3,:3]).as_rotvec(), seed[:3,3]]
    lower, upper = np.full(7, -np.inf), np.full(7, np.inf)
    lower[0], upper[0] = np.log(initial_scale)-np.log(10.), np.log(initial_scale)+np.log(10.)
    fit = least_squares(residual, initial, bounds=(lower, upper), loss='huber',
                        f_scale=parameters.tag_huber, max_nfev=200)
    scale, pose = unpack(fit.x)
    prediction, camera_points = project_observations(pose, cameras(scale), points, model)
    rms = float(np.sqrt(np.mean(np.sum((prediction-pixels)**2, axis=2))))
    evidence.update(scale=scale, reprojection_rms_px=rms, map_origin_raw=origin.tolist(),
                    T_centered_metric_map_board=pose.tolist(),
                    first_imu_time_s=float(sampled.imu_times_s[0]), last_imu_time_s=float(sampled.imu_times_s[-1]))
    if (not fit.success or not np.isfinite(rms) or np.any(camera_points[:,:,2] <= 0)
            or rms > parameters.tag_maximum_reprojection_rms_px or fit.active_mask[0]):
        raise PosterScaleConflict('海报尺度优化与静态刚性模型不一致或未收敛', dict(evidence, status='conflict'))
    _, singular, vt = np.linalg.svd(fit.jac, full_matrices=False)
    if singular[-1] <= singular[0]*1e-8:
        return dict(evidence, status='unobservable_scale')
    # 统计不确定度使用已声明像素噪声作下限；不能因无噪声合成输入而变成零。
    variance = max(1., float(np.sum(fit.fun**2)/max(len(fit.fun)-7, 1)))
    covariance = (vt.T / singular**2) @ vt * variance
    relative_std = float(np.sqrt(max(covariance[0,0], 0.)))  # log(scale) 的标准差
    evidence['relative_scale_std'] = relative_std
    if not np.isfinite(relative_std) or relative_std > parameters.tag_mapping_maximum_relative_std:
        return dict(evidence, status='uncertain_scale')
    return dict(evidence, status='estimated')


def combine_scales(imu_scale, imu_std, tag_scale, tag_std, parameters):
    values = np.array([imu_scale, imu_std, tag_scale, tag_std])
    if not np.isfinite(values).all() or min(imu_scale, tag_scale) <= 0 or min(imu_std, tag_std) < 0:
        raise ValueError('尺度与统计不确定度必须有限且有效')
    std = np.hypot([imu_std, tag_std], [parameters.tag_mapping_imu_std_floor, parameters.tag_mapping_tag_std_floor])
    logs = np.log([imu_scale, tag_scale])
    difference = float(abs(logs[0]-logs[1]))
    normalized = float(difference/np.linalg.norm(std))
    compatible = (normalized <= parameters.tag_mapping_compatibility_sigma
                  and abs(tag_scale/imu_scale-1.) <= parameters.tag_mapping_maximum_scale_difference)
    weights = 1./std**2
    weights /= weights.sum()
    return dict(compatible=bool(compatible), imu_scale=float(imu_scale), tag_scale=float(tag_scale),
                imu_statistical_std=float(imu_std), tag_statistical_std=float(tag_std),
                imu_std_floor=parameters.tag_mapping_imu_std_floor, tag_std_floor=parameters.tag_mapping_tag_std_floor,
                effective_relative_std=std.tolist(), weights=weights.tolist(), normalized_log_difference=normalized,
                scale=float(np.exp(weights @ logs)), relative_scale_std=float(1./np.linalg.norm(1./std)))


def correct_placement(placement, observations, lens, recording, scene, data):
    """先诊断，显式开启才合并；几何不足保持原尺度，有效来源冲突则拒绝。"""
    from software.scene.spatial.geometry import metric_alignment, MapAlignmentError
    from software.trajectory.algorithms.inertial import estimate_gyro_bias

    if not scene.parameters.tag_mapping_enabled:
        return placement
    parameters = scene.parameters
    report = dict(status='missing_rear_calibration', applied=False, actual_error='not_measured')
    result = deepcopy(placement)
    try:
        if data is not None:
            board, detections = data
            report = estimate_scale(observations, recording.camera.measurements[lens], recording.camera.measurements['rear'],
                                    detections, board, placement['scale'], parameters)
            report['applied'] = False
            if report['status'] == 'estimated':
                merged = combine_scales(placement['scale'], placement['evidence']['relative_scale_std'],
                                        report['scale'], report['relative_scale_std'], parameters)
                report['combination'] = merged
                if not merged['compatible']:
                    raise PosterScaleConflict('海报与 IMU 地图尺度不相容，拒绝加权合并', dict(report, status='conflict'))
                report['status'] = 'diagnostics_only'
                if parameters.tag_mapping_scale_enabled:
                    try:
                        result = metric_alignment(observations, np.linalg.inv(imu_from_camera(recording.camera.measurements[lens])),
                            recording.imu, estimate_gyro_bias(recording.imu[:,1:4], recording.context.parameters.fusion),
                            scene.trajectory.values.fusion.gravity_m_s2, parameters,
                            trajectory_parameters=scene.trajectory.values, fixed_scale=merged['scale'])
                    except ValueError as error:
                        report['inertial_recheck'] = getattr(error, 'diagnostics', {})
                        raise PosterScaleConflict(f'合并尺度未通过惯性复核：{error}', dict(report, status='conflict')) from error
                    result['evidence']['relative_scale_std'] = merged['relative_scale_std']
                    report.update(status='corrected', applied=True)
    except PosterScaleConflict as error:
        report = dict(error.evidence, reason=str(error), applied=False)
        if parameters.tag_mapping_scale_enabled:
            raise MapAlignmentError(str(error), dict(environment_poster=report)) from error
    result['evidence']['environment_poster'] = report
    return result
