"""Poster VIO 任务执行、回放与发布。"""

from __future__ import annotations

import time
from dataclasses import asdict, replace

import numpy as np

from umi.config.collection_schema import require_poster
from umi.core.geometry import imu_from_camera
from umi.core.timing import video_to_imu_seconds
from umi.infra.concurrency import daemon_map
from umi.infra.file_store import write_json
from umi.perception.poster.geometry import camera_model
from umi.perception.poster.layout import load as load_board
from umi.reconstruction.backends import PosterVioQualityError
from umi.reconstruction.backends.poster_vio.initialization import PosterInitializationQualityError
from umi.reconstruction.backends.poster_vio.models import (
    PosterVioPolicy,
    assess_observations,
    validate_problem,
)
from umi.reconstruction.backends.poster_vio.preparation import (
    PosterVioRequest as PosterVioRequest,
    prepare_task_inputs as prepare_task_inputs,
)
from umi.reconstruction.backends.poster_vio.solver import (
    ArmSolveInput,
    PosterSolveQualityError,
    natural_camera,
    solve_joint_poster_imu,
)
from umi.reconstruction.estimation.constraint_quality import longest_gap, maximum_support_gap
from umi.reconstruction.estimation.models import BackendResult, FusedTrajectory
from umi.store.models import TaskInputs
from umi.store.result_io import task_poster_frame


def execute_task(inputs: TaskInputs, *, fallback=None) -> BackendResult:
    """Prepare numerical inputs from the task recordings and solve."""
    scene = inputs.episode.scene
    require_poster(scene)
    try:
        request, arms, board_seed = prepare_task_inputs(inputs)
    except PosterInitializationQualityError as error:
        raise PosterVioQualityError(f'poster_vio 前端准备质量不足：{error}') from error
    try:
        return _solve_task(inputs, request, arms, board_seed, fallback=fallback)
    except PosterVioQualityError as error:
        error.retry_preparation = bool((request.front_map_selection or {}).get('retry_preparation'))
        raise


def replay_task(inputs: TaskInputs, *, request: PosterVioRequest, arms: tuple[ArmSolveInput, ...],
                board_seed: np.ndarray) -> BackendResult:
    """Replay against the already prepared recordings and their frozen time grids."""
    require_poster(inputs.episode.scene)
    if request is None or arms is None or board_seed is None:
        raise ValueError('poster_vio 冻结请求、数值输入和海报初值必须一起提供')
    return _solve_task(inputs, request, arms, board_seed, fallback=None)


def _solve_task(inputs: TaskInputs, request, arms, board_seed, *, fallback) -> BackendResult:
    """校验冻结请求，求解并交付已验收的轨迹。"""
    episode = inputs.episode
    request.validate_assets()
    problem = request.problem()
    validate_problem(problem)
    if (set(request.arms) != set(episode.expected_arms) or set(inputs.recordings) != set(request.arms)
            or tuple(item.arm for item in arms) != request.arms):
        raise ValueError('poster_vio 数值输入必须覆盖任务的全部预期使用手')
    _validate_frozen_request(inputs, request, arms)

    parameters = episode.scene.parameters
    policy = PosterVioPolicy(parameters.poster_vio_minimum_tag_frames,
                             parameters.poster_vio_maximum_observation_gap_s,
                             parameters.poster_vio_minimum_time_span_s)
    quality = assess_observations(problem, policy)
    if quality.status != 'ready':
        raise PosterVioQualityError(f'poster_vio 海报观测质量不足：{quality.status}')
    directory = inputs.directory / 'poster_vio'
    has_maps = any(item.front_maps for item in arms)
    selection = {'selected': 'with_front_maps'} if has_maps else request.front_map_selection
    try:
        solved, results = _validated_solution(inputs, request, problem, arms, board_seed, directory / 'quality.json')
    except PosterVioQualityError as error:
        if not has_maps:
            raise
        rejected = directory / 'front_map_attempt.json'
        quality_path = directory / 'quality.json'
        if quality_path.is_file():
            quality_path.replace(rejected)
        selection = dict(selected=None, rejected_reason=str(error))
        if rejected.exists():
            selection['rejected_evidence'] = rejected.name
        write_json(directory / 'front_map_selection.json', selection)
        inputs.console.write(f'工作镜头地图增强解未通过验收，复用冻结输入重算基础 VIO：{error}',
                             recording=episode.root.name, step=(2, 3))
        base_arms = tuple(replace(item, front_maps=()) for item in arms)
        solved, results = _validated_solution(inputs, request, problem, base_arms, board_seed, quality_path)
        selection['selected'] = 'without_front_maps'
    coordinate = dict(request.coordinate_record, **task_poster_frame(episode, request.board_asset_hash),
                       reconstruction_backend='poster_vio', quality=asdict(solved.quality))
    if fallback:
        coordinate['fallback'] = dict(fallback)
    if selection is not None:
        coordinate['front_map_selection'] = selection
        write_json(directory / 'front_map_selection.json', selection)
    # Validate both arms before writing either arm into the isolated run directory.
    write_json(inputs.directory / 'reconstruction.json', dict(
        requested_mode=episode.scene.reconstruction_mode, selected_backend='poster_vio',
        scene_map_priors_used=False, coordinate=coordinate, observation_policy=asdict(policy)))
    return BackendResult.from_recordings(results, inputs.recordings, coordinate)


def _validated_solution(inputs, request, problem, arms, board_seed, quality_path):
    solved = _run_solver(inputs, problem, arms, board_seed)
    write_json(quality_path, dict(
        quality=asdict(solved.quality), arms=solved.arm_evidence,
        scene_map_priors_used=False, actual_error='not_measured'))
    if solved.quality.status != 'solved':
        detail = _quality_rejection_detail(solved)
        raise PosterVioQualityError(f'poster_vio 联合图质量拒绝：{solved.quality.status}；{detail}')
    return solved, {item.arm: _arm_trajectory(inputs, item, solved, request) for item in arms}


def _validate_frozen_request(inputs: TaskInputs, request, arms) -> None:
    """冻结请求必须与本任务的布局、相机、IMU、基座标签和时钟完全一致（重放时尤其重要）。"""
    episode = inputs.episode
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
        if len(observation.base_tag_times_s) or item.front_maps:
            used_cameras.add('front')
        base_spec = (getattr(recording.camera, 'base_tag', None)
                      if episode.scene.poster_vio_base_tag_enabled else None)
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


def _run_solver(inputs: TaskInputs, problem, arms, board_seed):
    episode = inputs.episode
    try:
        started = time.monotonic()
        inputs.console.write('联合优化', recording=episode.root.name, step=(2, 3))

        def solve(_):
            return solve_joint_poster_imu(
                problem, tuple(arms), np.asarray(board_seed, dtype=float),
                progress=lambda message: inputs.console.write(
                    message, recording=episode.root.name, step=(2, 3),
                    visible=not message.startswith('LM 迭代 ')))

        # GTSAM may stay in native code for a long LM iteration and cannot
        # observe Python's SIGINT handler.  Keep the worker daemonized so the
        # outer run boundary can record an interrupted attempt immediately.
        solved = daemon_map(solve, (None,))[0]
        inputs.console.write(f'联合优化耗时 {time.monotonic() - started:.1f}s',
                             recording=episode.root.name, step=(2, 3))
    except PosterSolveQualityError as error:
        raise PosterVioQualityError(f'poster_vio 联合求解质量不足：{error}') from error
    return solved


def _quality_rejection_detail(solved) -> str:
    """把各手留出验证的拒绝原因汇总成一行，便于终端直接定位。"""
    diagnostics = []
    for arm, evidence in solved.arm_evidence.items():
        for step in evidence.get('refinement', {}).get('rounds', []):
            if step.get('status') == 'rejected' and 'round_limit' in step.get('reasons', []):
                diagnostics.append(f"{arm} 第 {step['round'] + 1} 轮达到剔除上限，仍有 {len(step['removed'])} 个待剔除观测")
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
    return '；'.join(diagnostics) if diagnostics else '详见 poster_vio/quality.json'


def _arm_trajectory(inputs: TaskInputs, item, solved, request) -> FusedTrajectory:
    """一只手：取工作镜头曝光时刻的优化状态，验收位置支持并整理证据。"""
    episode, parameters = inputs.episode, inputs.episode.scene.parameters
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
    map_times = np.asarray([t for segment in arm_evidence.get('front_maps', [])
                            for t in segment.get('position_support', {}).get('supported_imu_times_s', [])]) - front_offset
    map_mask = np.zeros(len(output_imu_times), dtype=bool)
    map_mask[np.searchsorted(boundaries, map_times, side='right')] = True
    camera_masks['front'] |= map_mask
    tracked = camera_masks['front'] | camera_masks['rear']
    natural_times = np.asarray(arm_evidence['natural_support_imu_times_s']) - front_offset
    # Reporting uses nearest output cells; acceptance keeps exact exposure times.
    gap = maximum_support_gap(recording.video_times_s, np.zeros(len(tracked), dtype=bool),
                               np.unique(np.r_[natural_times, supported_times, map_times]))
    if (support['supported_frames'] < episode.scene.parameters.tag_minimum_frames
            or gap > parameters.poster_vio_maximum_position_support_gap_s):
        raise PosterVioQualityError(f'{arm} poster_vio 海报几何支持不足或共同位置支持缺口过长：{gap:.3f} s')
    poster = dict(status='fused', layout=dict(asset_hash=request.board_asset_hash),
                   final_reprojection_rms_px=arm_evidence['tag_metrics']['rms_px'],
                   final_reprojection_metrics=arm_evidence['tag_metrics'],
                   observation_video_times_s=observation_times.tolist(),
                   position_support_video_times_s=supported_times.tolist(),
                   position_support=dict(support, frames=frames), actual_error='not_measured')
    return FusedTrajectory(solved.trajectories[arm][indices], tracked,
        camera_masks['front'], dict(
            mode='poster_vio', reconstruction_backend='poster_vio', scene_map_priors_used=False,
            actual_error='not_measured', poster_vio_quality=asdict(solved.quality),
            poster_vio=arm_evidence, environment_poster=poster,
            maximum_position_support_gap_s=gap,
            maximum_visual_gap_s=longest_gap(recording.video_times_s, tracked),
            frame_provenance=[dict(constraint='full' if observed else 'imu',
                visual_lens='front' if camera_masks['front'][index] else 'rear' if observed else None,
                visual_lenses=[lens for lens in ('front', 'rear') if camera_masks[lens][index]],
                **({'map_role': 'task_front_map'} if map_mask[index] else {}))
                for index, observed in enumerate(tracked)]))
