"""Task route and publication evidence for joint poster visual-inertial BA."""
from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np

from software.camera.timing import video_to_imu_seconds
from software.camera.geometry import imu_from_camera
from software.infrastructure.concurrency import daemon_map
from software.trajectory.poster_vio.solver import natural_camera
from software.vision.poster.geometry import camera_model
from software.infrastructure.file_store import write_json
from software.scene.models import FusedTrajectory, TaskInputs, ArmExecutionResult
from software.scene.pipeline.tasks.policy import require_poster
from software.scene.pipeline.tasks.results import arm_writer
from software.scene.spatial.fusion import longest_gap
from software.scene.errors import PosterVioQualityError
from software.trajectory.poster_vio import PosterVioPolicy, assess_observations, solve_joint_poster_imu, validate_request
from software.trajectory.reporting.quality import maximum_support_gap
from software.vision.poster.layout import load as load_board
from .poster_vio_inputs import prepare_task_inputs


def execute_task(inputs: TaskInputs, *, fallback=None) -> ArmExecutionResult:
    """Prepare or replay frozen numerical inputs, then publish complete arm tracks."""
    episode = inputs.episode
    require_poster(episode.scene)
    supplied = (inputs.poster_vio_request, inputs.poster_vio_arms, inputs.poster_vio_board_seed)
    generated_inputs = all(item is None for item in supplied)
    if generated_inputs:
        try:
            request, arms, board_seed = prepare_task_inputs(inputs)
        except PosterVioQualityError:
            raise
        except ValueError as error:
            # The automatic route may try this frontend on recordings whose
            # tags, IMU coverage or natural features are insufficient.  Keep
            # explicit/replay requests strict, but let auto continue to the
            # independent poster backend for generated candidate quality.
            if episode.scene.reconstruction_mode != 'auto':
                raise
            raise PosterVioQualityError(f'poster_vio 前端准备质量不足：{error}') from error
    elif any(item is None for item in supplied):
        raise ValueError('poster_vio 冻结请求、数值输入和海报初值必须一起提供')
    else:
        request, arms, board_seed = supplied
    validate_request(request)
    if (set(request.arms) != set(episode.expected_arms) or set(inputs.recordings) != set(request.arms)
            or tuple(item.arm for item in arms) != request.arms):
        raise ValueError('poster_vio 数值输入必须覆盖任务的全部预期使用手')
    board = load_board(episode.scene.tag_layout)
    if request.board_asset_hash != board.asset_hash:
        raise ValueError('poster_vio 冻结请求的海报布局身份不匹配')
    observations = {observation.arm: observation for observation in request.observations}
    for item in arms:
        observation = observations[item.arm]
        recording = inputs.recordings[item.arm]
        if (request.camera_assets[item.arm].get('asset_hashes') != recording.camera.asset_hashes
                or not np.array_equal(item.imu, recording.imu)):
            raise ValueError('poster_vio 冻结请求的相机或 IMU 与任务输入不一致')
        try:
            points = np.array([board.corners_m[int(tag)] for tag in observation.tag_ids])
        except KeyError as error:
            raise ValueError('poster_vio 标签 ID 不在冻结布局中') from error
        if not np.array_equal(item.board_points_m, points):
            raise ValueError('poster_vio 数值角点与冻结布局或 ID 不一致')
        used_cameras = {track.get('camera', 'rear') for track in item.natural_tracks or observation.natural_tracks}
        if len(observation.base_tag_times_s):
            used_cameras.add('front')
        base_spec = getattr(recording.camera, 'base_tag', None)
        expected_spec = (None if base_spec is None else dict(id=base_spec['id'],
            size_m=base_spec['size_mm'] / 1000., dictionary=base_spec.get('dictionary', 'DICT_4X4_100')))
        if request.base_tag_specs.get(item.arm) != expected_spec:
            raise ValueError('poster_vio base tag 配置与冻结任务资产不一致')
        if request.base_tag_specs:
            clock_offset = (float(inputs.offsets_s[item.arm])
                            - recording.camera.measurements['front']['cam_time_offset_ns'] * 1e-9)
            if not np.isclose(request.clock_offsets_s.get(item.arm, np.nan), clock_offset, rtol=0, atol=1e-9):
                raise ValueError('poster_vio base tag 时钟与冻结任务同步不一致')
        for lens in used_cameras:
            calibration = natural_camera(item, lens)
            measured = recording.camera.measurements[lens]
            expected_model = camera_model(measured)
            if (not np.allclose(calibration.T_cam_imu, np.linalg.inv(imu_from_camera(measured)), rtol=0, atol=1e-10)
                    or any(getattr(calibration.model, name, None) != getattr(expected_model, name)
                           for name in ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'))):
                raise ValueError(f'poster_vio {lens} 自然特征标定与冻结任务资产不一致')

    parameters = episode.scene.parameters
    policy = PosterVioPolicy(parameters.poster_vio_minimum_tag_frames,
                             parameters.poster_vio_maximum_observation_gap_s,
                             parameters.poster_vio_minimum_time_span_s)
    quality = assess_observations(request, policy)
    if quality.status != 'ready':
        raise PosterVioQualityError(f'poster_vio 海报观测质量不足：{quality.status}')
    try:
        started = time.monotonic()
        inputs.console.write('poster_vio 联合优化开始', recording=episode.root.name, step=(2, 3))

        def solve(_):
            return solve_joint_poster_imu(
                request, tuple(arms), np.asarray(board_seed, dtype=float),
                progress=lambda message: inputs.console.write(
                    f'poster_vio：{message}', recording=episode.root.name))

        # GTSAM may stay in native code for a long LM iteration and cannot
        # observe Python's SIGINT handler.  Keep the worker daemonized so the
        # outer run boundary can record an interrupted attempt immediately.
        solved = daemon_map(solve, (None,))[0]
        inputs.console.write(f'poster_vio 联合优化完成；耗时 {time.monotonic() - started:.1f} 秒',
                             recording=episode.root.name, state='完成')
    except PosterVioQualityError:
        raise
    except ValueError as error:
        if episode.scene.reconstruction_mode != 'auto' or not generated_inputs:
            raise
        raise PosterVioQualityError(f'poster_vio 联合求解质量不足：{error}') from error
    write_json(inputs.directory / 'poster_vio' / 'quality.json', dict(
        quality=asdict(solved.quality), arms=solved.arm_evidence,
        scene_map_priors_used=False, actual_error='not_measured'))
    if solved.quality.status != 'solved':
        diagnostics = []
        for arm, evidence in solved.arm_evidence.items():
            holdout = evidence.get('holdout', {})
            if holdout.get('status') == 'rejected':
                depth_summary = holdout.get('depth_summary') or {}
                invalid_depth_details = holdout.get('invalid_depth_details') or []
                affected_times = {
                    detail.get('imu_time_s')
                    for detail in invalid_depth_details
                    if detail.get('imu_time_s') is not None
                }
                depth_detail = ''
                if depth_summary.get('min_m') is not None:
                    depth_detail = (
                        f', 最小深度={depth_summary["min_m"]:.6g}m'
                        f', P01={depth_summary.get("p01_m", float("nan")):.6g}m'
                        f', 受影响时刻={len(affected_times)}')
                diagnostics.append(
                    f'{arm} 标签留出拒绝'
                    f'(非正深度={holdout.get("invalid_depth_observations", 0)}'
                    f', RMS={holdout.get("metrics", {}).get("rms_px")}'
                    f'{depth_detail})')
            natural = evidence.get('natural_holdout', {})
            if natural.get('status') == 'rejected':
                diagnostics.append(
                    f'{arm} 自然点留出拒绝'
                    f'(非正深度={natural.get("invalid_depth_observations", 0)}'
                    f', RMS={natural.get("metrics", {}).get("rms_px")})')
        detail = '；'.join(diagnostics) if diagnostics else '详见 poster_vio/quality.json'
        raise PosterVioQualityError(f'poster_vio 联合图质量拒绝：{solved.quality.status}；{detail}')
    coordinate = dict(solved.coordinate_record, scope='task_poster', axes='board_layout', units='metres',
                       reconstruction_backend='poster_vio', quality=asdict(solved.quality))
    if fallback:
        coordinate['fallback'] = dict(fallback)
    results = {}
    for item in arms:
        arm = item.arm
        recording = inputs.recordings[arm]
        output_imu_times = video_to_imu_seconds(recording.video_times_s,
            recording.camera.measurements['front']['cam_time_offset_ns'])
        # Output rows are actual optimized states, never a linear fill across keyframes.
        indices = np.searchsorted(item.times_s, np.round(output_imu_times, 9))
        if (np.any(indices >= len(item.times_s))
                or not np.allclose(item.times_s[indices], output_imu_times, rtol=0, atol=1e-9)):
            raise PosterVioQualityError(f'{arm} poster_vio 缺少工作镜头曝光时刻的优化状态')
        arm_evidence = solved.arm_evidence[arm]
        front_offset = float(recording.camera.measurements['front']['cam_time_offset_ns']) * 1e-9
        observation_times = np.asarray(arm_evidence['observation_imu_times_s']) - front_offset
        support = arm_evidence['position_support']
        frames = [dict(frame, video_time_s=float(frame['video_time_s'] - front_offset))
                  for frame in support['frames']]
        supported_times = np.array([frame['video_time_s'] for frame in frames if frame['position_supported']])
        by_camera = arm_evidence['natural_by_camera']
        boundaries = (recording.video_times_s[:-1] + recording.video_times_s[1:]) / 2
        camera_masks = {}
        for lens, camera_evidence in by_camera.items():
            times = np.asarray(camera_evidence['support_imu_times_s']) - front_offset
            mask = np.zeros(len(output_imu_times), dtype=bool)
            mask[np.searchsorted(boundaries, times, side='right')] = True
            camera_masks[lens] = mask
        tracked = camera_masks['front'] | camera_masks['rear']
        natural_times = np.asarray(arm_evidence['natural_support_imu_times_s']) - front_offset
        # Reporting uses nearest output cells; acceptance keeps exact exposure times.
        gap = maximum_support_gap(recording.video_times_s, np.zeros(len(tracked), dtype=bool),
                                   np.unique(np.r_[natural_times, supported_times]))
        if (support['supported_frames'] < episode.scene.parameters.tag_minimum_frames
                or gap > parameters.poster_vio_maximum_position_support_gap_s):
            raise PosterVioQualityError(f'{arm} poster_vio 海报几何支持不足或共同位置支持缺口过长：{gap:.3f} s')
        poster = dict(status='fused', layout=dict(asset_hash=request.board_asset_hash),
                       final_reprojection_rms_px=arm_evidence['tag_metrics']['rms_px'],
                       final_reprojection_metrics=arm_evidence['tag_metrics'],
                       observation_video_times_s=observation_times.tolist(),
                       position_support_video_times_s=supported_times.tolist(),
                       position_support=dict(support, frames=frames), actual_error='not_measured')
        results[arm] = FusedTrajectory(solved.trajectories[arm][indices], tracked,
            camera_masks['front'], dict(
                mode='poster_vio', reconstruction_backend='poster_vio', scene_map_priors_used=False,
                actual_error='not_measured', poster_vio_quality=asdict(solved.quality),
                poster_vio=arm_evidence, environment_poster=poster,
                maximum_position_support_gap_s=gap,
                maximum_visual_gap_s=longest_gap(recording.video_times_s, tracked),
                frame_provenance=[dict(constraint='full' if observed else 'imu',
                    visual_lens='front' if camera_masks['front'][index] else 'rear' if observed else None,
                    visual_lenses=[lens for lens in ('front', 'rear') if camera_masks[lens][index]])
                    for index, observed in enumerate(tracked)]))
    # Validate both arms before writing either arm into the isolated run directory.
    write_json(inputs.directory / 'reconstruction.json', dict(
        requested_mode=episode.scene.reconstruction_mode, selected_backend='poster_vio',
        scene_map_priors_used=False, coordinate=coordinate, observation_policy=asdict(policy)))
    tracks, width_reports, tag_distance_reports = {}, {}, {}
    for arm, result in results.items():
        track, width, tag_distance = arm_writer.write_arm_result(inputs, arm, result, coordinate)
        tracks[arm] = track
        if width is not None:
            width_reports[arm] = width
        if tag_distance is not None:
            tag_distance_reports[arm] = tag_distance
    return ArmExecutionResult(tracks, width_reports, tag_distance_reports, coordinate)
