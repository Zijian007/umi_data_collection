"""Poster VIO 观测准备、执行与回放。"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from umi.config.collection_schema import require_poster
from umi.core.geometry import imu_from_camera
from umi.core.timing import video_to_imu_seconds
from umi.infra.concurrency import daemon_map
from umi.infra.file_store import write_json
from umi.infra.provenance import file_hash
from umi.perception.poster import detection
from umi.perception.poster.geometry import camera_model
from umi.perception.poster.layout import load as load_board
from umi.perception.poster.quality import cull_threshold
from umi.reconstruction.backends import PosterVioQualityError
from umi.reconstruction.backends.poster_vio import base_tags, front_map
from umi.reconstruction.backends.poster_vio.frontend import FeatureCameraInput, track_cameras
from umi.reconstruction.backends.poster_vio.initialization import (
    InertialAlignmentLimitError,
    PosterInitializationQualityError,
    align_inertial,
    estimate_static_tag_extrinsic,
    inertial_tail_seed,
    robust_seed_poses,
    select_keyframes,
    select_observations,
    single_tag_geometry,
)
from umi.reconstruction.backends.poster_vio.models import (
    FeatureTrackingParameters,
    PosterVioObservation,
    PosterVioParameters,
    PosterVioPolicy,
    PosterVioProblem,
    assess_observations,
    validate_problem,
)
from umi.reconstruction.backends.poster_vio.solver import (
    ArmSolveInput,
    NaturalCamera,
    PosterSolveQualityError,
    natural_camera,
    natural_support_nodes,
    natural_training,
    select_holdout,
    solve_joint_poster_imu,
)
from umi.reconstruction.estimation.constraint_quality import longest_gap, maximum_support_gap
from umi.reconstruction.estimation.models import FusedTrajectory
from umi.store import layout
from umi.store.models import ArmExecutionResult, TaskInputs
from umi.store.result_io import task_poster_frame


def execute_task(inputs: TaskInputs, *, fallback=None) -> ArmExecutionResult:
    """Prepare numerical inputs from the task recordings and solve."""
    scene = inputs.episode.scene
    require_poster(scene)
    try:
        request, arms, board_seed = prepare_task_inputs(inputs)
    except PosterInitializationQualityError as error:
        raise PosterVioQualityError(f'poster_vio 前端准备质量不足：{error}') from error
    return _solve_task(inputs, request, arms, board_seed, fallback=fallback)


def replay_task(inputs: TaskInputs, *, request: PosterVioRequest, arms: tuple[ArmSolveInput, ...],
                board_seed: np.ndarray) -> ArmExecutionResult:
    """Replay against the already prepared recordings and their frozen time grids."""
    require_poster(inputs.episode.scene)
    if request is None or arms is None or board_seed is None:
        raise ValueError('poster_vio 冻结请求、数值输入和海报初值必须一起提供')
    return _solve_task(inputs, request, arms, board_seed, fallback=None)


def _solve_task(inputs: TaskInputs, request, arms, board_seed, *, fallback) -> ArmExecutionResult:
    """校验冻结请求 → 观测质量 → 联合求解 → 质量与支持验收 → 写出轨迹。"""
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
    selection = {'selected': 'with_front_maps'} if has_maps else None
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
    return inputs.services.write_task_results(inputs, results, coordinate, parallel=bool(inputs.arm_cpu_sets))


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


@dataclass(frozen=True)
class PosterVioRequest:
    """Frozen inputs shared by all arms in a task-local solve."""

    arms: tuple[str, ...]
    board_asset_hash: str
    observations: tuple[PosterVioObservation, ...]
    camera_assets: dict[str, dict[str, Any]]
    imu_assets: dict[str, dict[str, Any]]
    coordinate_record: dict[str, Any]
    base_tag_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    clock_offsets_s: dict[str, float] = field(default_factory=dict)

    def problem(self) -> PosterVioProblem:
        return PosterVioProblem(self.arms, self.observations, self.base_tag_specs, self.clock_offsets_s)

    def validate_assets(self) -> None:
        if not self.board_asset_hash:
            raise ValueError('poster_vio 必须固定海报布局身份')
        if set(self.camera_assets) != set(self.arms) or set(self.imu_assets) != set(self.arms):
            raise ValueError('poster_vio 相机和 IMU 资产必须覆盖全部使用手')
        record = self.coordinate_record
        if record.get('scope') != 'task_poster' or not str(record.get('frame_id', '')).startswith('poster:'):
            raise ValueError('poster_vio 必须声明 task_poster 坐标系')


@dataclass
class _BaseTagRows:
    """一只手看到的（其他手）夹爪基座标签观测；单手或未声明时为空。"""

    target_arms: tuple[str, ...] = ()
    times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    pixels: np.ndarray = field(default_factory=lambda: np.empty((0, 4, 2), dtype=float))
    frame_polygons: dict = field(default_factory=dict)


@dataclass
class _PreparedArm:
    """一只手完成初始化、等待特征跟踪的全部中间量。"""

    arm: str
    recording: Any
    camera: Any
    streams: tuple[FeatureCameraInput, ...]
    node_times: np.ndarray
    tag_indices: np.ndarray
    ids: np.ndarray
    points: np.ndarray
    pixels: np.ndarray
    raw_times: np.ndarray
    tag_times: np.ndarray
    training_times: np.ndarray
    holdout: np.ndarray
    initialization_mask: np.ndarray
    single_tag_mask: np.ndarray
    base: _BaseTagRows
    seed: np.ndarray
    ext: np.ndarray
    model: Any
    gravity: np.ndarray
    acc_bias: np.ndarray
    node_velocities: np.ndarray
    initial_gyro_bias: np.ndarray
    natural_cameras: dict[str, NaturalCamera]
    inertial: dict
    initialization: dict
    pnp_rms: np.ndarray


def prepare_task_inputs(inputs: TaskInputs):
    """Use only this task's pixels, IMU, layout and frozen camera assets."""
    episode, scene = inputs.episode, inputs.episode.scene
    parameters = scene.parameters.for_algorithm(PosterVioParameters)
    tracking = scene.parameters.for_algorithm(FeatureTrackingParameters)
    require_poster(scene)
    arms = tuple(episode.expected_arms)
    if set(inputs.recordings) != set(arms):
        raise ValueError('poster_vio 需要所有预期使用手的录像')
    base_tag_specs = _base_tag_specs(inputs, arms)
    snapshots, board, owners_by_tag_id, base_tag_cache_dirs = _detect(inputs, arms, base_tag_specs)

    inputs.console.write('海报姿态初始化与 IMU 对齐', recording=episode.root.name, step=(2, 3))
    prepared_arms = [
        _prepare_arm(inputs, arm, snapshot, board, parameters,
                     base_tag_cache_dir=base_tag_cache_dirs.get(arm), owners_by_tag_id=owners_by_tag_id)
        for arm, snapshot in zip(arms, snapshots)]

    tracked = _track_features(inputs, prepared_arms, tracking)
    _recover_partner_coverage(inputs, prepared_arms, tracked, snapshots, board, parameters, tracking,
                              base_tag_cache_dirs, owners_by_tag_id)
    observations, solve_inputs, camera_assets, imu_assets, diagnostics = _assemble(
        inputs, prepared_arms, tracked, parameters)
    solve_inputs, shared_gravity, difference = _share_gravity(solve_inputs, scene)
    coordinate = dict(name=scene.name, session=episode.session.name, version=None,
                       **task_poster_frame(episode, board.asset_hash), gravity_difference_deg=difference)
    # Native IMU -> common task clock. Lens calibration offset must be removed
    # before applying the cross-camera video-clock shift.
    clock_offsets = {arm: float(inputs.offsets_s[arm])
                     - inputs.recordings[arm].camera.measurements['front']['cam_time_offset_ns'] * 1e-9
                     for arm in arms}
    request = PosterVioRequest(arms, board.asset_hash, tuple(observations), camera_assets, imu_assets,
                               coordinate, base_tag_specs=base_tag_specs, clock_offsets_s=clock_offsets)
    solve_inputs = _fit_base_tag_mounts(inputs, request, solve_inputs, observations, base_tag_specs, diagnostics)
    solve_inputs = base_tags.expand_nodes(request.problem(), solve_inputs)
    if scene.parameters.front_map_pixel_fusion:
        inputs.console.write('工作镜头 ORB 地图定位与像素权重', recording=episode.root.name, step=(2, 3))
        solve_inputs = tuple(replace(item, front_maps=front_map.prepare(
            inputs.recordings[item.arm], item, inputs.directory / 'poster_vio' / item.arm / 'front_map',
            scene.trajectory.values.fusion.node_stride)) for item in solve_inputs)
    assets = _write_numerical_inputs(inputs, solve_inputs, observations)
    write_json(inputs.directory / 'poster_vio' / 'inputs.json', dict(
        kind='umi_poster_vio_inputs', schema_version=1,
        board=board.record(), camera_assets=camera_assets, imu_assets=imu_assets,
        assets=assets, diagnostics=diagnostics, shared_gravity_m_s2=shared_gravity.tolist(),
        coordinate=coordinate, base_tag_specs=base_tag_specs,
        clock_offsets_s=clock_offsets,
        scene_map_priors_used=False))
    return request, solve_inputs, np.eye(4)


def _base_tag_specs(inputs: TaskInputs, arms: tuple[str, ...]) -> dict[str, dict]:
    """套件声明的各手夹爪基座标签；批次关闭基座标签时为空。"""
    scene = inputs.episode.scene
    specs = {}
    for arm in arms:
        spec = (getattr(inputs.recordings[arm].camera, 'base_tag', None)
                if scene.poster_vio_base_tag_enabled else None)
        if spec is not None:
            specs[arm] = dict(id=int(spec['id']), size_m=float(spec['size_mm']) / 1000.,
                              dictionary=spec.get('dictionary', 'DICT_4X4_100'))
    return specs


def _detect(inputs: TaskInputs, arms: tuple[str, ...], base_tag_specs: dict):
    """环境海报检测（各手）与双手基座标签检测；返回快照、布局、标签归属和基座缓存目录。"""
    episode, scene = inputs.episode, inputs.episode.scene
    snapshots = inputs.services.prepare_detection_group(inputs, arms, parallel=True)
    for arm, snapshot in zip(arms, snapshots):
        if snapshot is None:
            raise ValueError(f'{arm} poster_vio 缺少环境标签检测结果')
    board = snapshots[0].layout
    owners_by_tag_id = {spec['id']: arm for arm, spec in base_tag_specs.items()}
    if len(owners_by_tag_id) != len(base_tag_specs):
        raise ValueError('双手 base tag ID 必须唯一')
    if len(arms) > 1 and set(owners_by_tag_id) & set(board.ids):
        raise ValueError('base tag ID 不能与环境海报 ID 冲突')

    base_tag_cache_dirs = {}
    if len(arms) > 1 and base_tag_specs:
        def prepare_base_tag_cache(arm: str):
            cache_dir = layout.metadata(episode.root, 'cache/base_tags') / arm
            detection.cached(
                episode.recordings[arm], cache_dir,
                detection.DetectionSettings(border_bits=1, ids=tuple(sorted(owners_by_tag_id)),
                    lens='front', minimum_side_px=scene.parameters.tag_minimum_side_px),
                progress=inputs.console.write, cache_audit=inputs.cache_audit)
            return arm, cache_dir

        base_tag_started = time.monotonic()
        inputs.console.write('基座标签检测（双手并行）', recording=episode.root.name, step=(2, 3))
        base_tag_cache_dirs = dict(daemon_map(prepare_base_tag_cache, arms))
        inputs.console.write(
            f'基座标签检测耗时 {time.monotonic() - base_tag_started:.1f}s',
            recording=episode.root.name, step=(2, 3))
    return snapshots, board, owners_by_tag_id, base_tag_cache_dirs


def _prepare_arm(inputs: TaskInputs, arm: str, snapshot, board, parameters: PosterVioParameters, *,
                 base_tag_cache_dir, owners_by_tag_id: dict, coverage_gap=None) -> _PreparedArm:
    """一只手：选取海报观测 → 留出与鲁棒初值 → 惯性对齐 → 求解采样 → 节点种子 → 特征流。"""
    scene = inputs.episode.scene
    recording = inputs.recordings[arm]
    camera = recording.camera
    if not {'front', 'rear'} <= camera.measurements.keys():
        raise ValueError(f'{arm} poster_vio 缺少 front/rear 相机标定')
    if 'mask' not in camera.files:
        raise ValueError(f'{arm} poster_vio front 自然特征缺少冻结遮手图')
    measured = camera.measurements['rear']
    output_times = video_to_imu_seconds(recording.video_times_s,
                                        camera.measurements['front']['cam_time_offset_ns'])
    if (len(output_times) < 3 or not np.isfinite(output_times).all()
            or np.any(np.diff(output_times) <= 0)
            or recording.imu[0, 0] > output_times[0] or recording.imu[-1, 0] < output_times[-1]):
        raise ValueError(f'{arm} poster_vio IMU 未完整覆盖工作镜头曝光时刻')
    directory = inputs.directory / 'poster_vio' / arm
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(snapshot.directory / 'observations.npz', directory / 'observations.npz')
    detections = snapshot.observations
    _, _, ids, points, pixels, raw_times = select_observations(
        detections, board, measured, parameters,
        start_s=output_times[0], end_s=output_times[-1])
    raw_times = np.round(raw_times, 9)
    tag_times = np.unique(raw_times)
    if len(tag_times) < 3:
        raise PosterVioQualityError('poster_vio 标签时刻至少需要三个')
    base = (_base_tag_rows(directory, base_tag_cache_dir, camera, output_times, owners_by_tag_id)
            if base_tag_cache_dir is not None else _BaseTagRows())

    # 留出部分标签时刻后，用其余时刻做鲁棒 PnP 初值和惯性对齐。
    model = camera_model(measured)
    ext = np.linalg.inv(imu_from_camera(measured))
    holdout = select_holdout(np.round(raw_times, 9), parameters)
    training_times, training_indices = np.unique(np.round(raw_times[~holdout], 9), return_inverse=True)
    if len(training_times) < 3:
        raise PosterVioQualityError('poster_vio 留出后训练标签时刻不足')
    training_times, tag_seeds, pnp_rms, initialization_keep, initialization = robust_seed_poses(
        training_times, training_indices, points[~holdout], pixels[~holdout], ext, model, parameters,
        imu=recording.imu,
        max_rotation_step_deg=scene.trajectory.values.fusion.split_rotation_deg,
        solve_candidates=inputs.poster_pnp_candidates.get(arm))
    initialization_mask = np.ones(len(points), dtype=bool)
    initialization_mask[~holdout] = initialization_keep
    single_tag_mask = np.zeros(len(points), dtype=bool)
    training_rows = np.flatnonzero(~holdout)
    for record in initialization['removed']:
        record['observation'] = int(training_rows[record['observation']])
        record['tag_id'] = int(ids[record['observation']])
        single_tag_mask[record['observation']] = record.get('single_tag_candidate', False)
    trigger = None
    if coverage_gap is None:
        try:
            gravity, acc_bias, velocities, inertial = align_inertial(
                training_times, tag_seeds, recording.imu, scene.trajectory.values.fusion)
        except InertialAlignmentLimitError as limit:
            if not parameters.poster_vio_robust_inertial_seed:
                raise
            trigger = dict(gravity_norm_m_s2=limit.gravity_norm, acc_bias_norm_m_s2=limit.bias_norm)
    else:
        trigger = dict(partner_training_support_gap_s=coverage_gap)
    if trigger is not None:
        from umi.reconstruction.backends.poster_vio.initialization import InertialSeedQualityError, fit_training_seed
        try:
            tag_seeds, gravity, acc_bias, velocities, inertial = fit_training_seed(
                training_times, tag_seeds, recording.imu, scene.trajectory.values.fusion)
        except InertialSeedQualityError as error:
            raise PosterVioQualityError(f'{arm} poster_vio 鲁棒惯性初始化失败：{error}') from error
        inertial['recovery_trigger'] = trigger

    # 联合图只保留按关键帧间隔采样的完整标签帧；种子不受影响。
    solver_keep = _solver_tag_selection(
        raw_times, scene.parameters.poster_vio_keyframe_maximum_dt_s)
    raw_times = raw_times[solver_keep]
    ids = ids[solver_keep]
    points = points[solver_keep]
    pixels = pixels[solver_keep]
    holdout = holdout[solver_keep]
    initialization_mask = initialization_mask[solver_keep]
    single_tag_mask = single_tag_mask[solver_keep]
    tag_times = np.unique(raw_times)
    if len(tag_times) < 3:
        raise PosterVioQualityError('poster_vio 求解采样后标签时刻至少需要三个')

    # 输出时刻与环境镜头特征时刻都成为图节点，即使该时刻看不到标签。
    all_rear_times = video_to_imu_seconds(detections.video_times_s, measured['cam_time_offset_ns'])
    selected = (all_rear_times >= output_times[0]) & (all_rear_times <= output_times[-1])
    rear_times = np.round(all_rear_times[selected], 9)
    held_times = np.unique(np.round(raw_times[holdout], 9))
    feature_selected = selected & ~np.isin(np.round(all_rear_times, 9), held_times)
    node_times = np.unique(np.round(np.r_[output_times, rear_times], 9))
    seed, node_velocities, tail_seed = inertial_tail_seed(training_times, tag_seeds, velocities, node_times,
        recording.imu, gravity, np.asarray(inertial['initial_gyro_bias_rad_s']), acc_bias,
        parameters=parameters, fusion=scene.trajectory.values.fusion)
    if 'recovery_trigger' in inertial:
        from umi.reconstruction.backends.poster_vio.initialization import bridge_training_gaps
        inertial['imu_bridged_nodes'] = bridge_training_gaps(
            training_times, tag_seeds, velocities, node_times, seed, node_velocities, recording.imu,
            gravity, np.asarray(inertial['initial_gyro_bias_rad_s']), acc_bias)
    inertial['tail_seed'] = tail_seed
    tag_indices = np.searchsorted(node_times, np.round(raw_times, 9))
    initialization['single_tag_geometry_rejections'] = _reject_single_tag_geometry(
        single_tag_mask, points, pixels, ids, raw_times, seed[tag_indices], ext, model, parameters)

    natural_cameras = {
        lens: NaturalCamera(np.linalg.inv(imu_from_camera(camera.measurements[lens])),
                             camera_model(camera.measurements[lens]))
        for lens in ('front', 'rear')}
    streams = _feature_streams(scene, recording, camera, detections, feature_selected, all_rear_times,
                               output_times, node_times, seed, ext, model, natural_cameras, base, parameters)
    return _PreparedArm(
        arm=arm, recording=recording, camera=camera, streams=streams, node_times=node_times,
        tag_indices=tag_indices, ids=ids, points=points, pixels=pixels, raw_times=raw_times,
        tag_times=tag_times, training_times=training_times, holdout=holdout,
        initialization_mask=initialization_mask, single_tag_mask=single_tag_mask, base=base,
        seed=seed, ext=ext, model=model, gravity=gravity, acc_bias=acc_bias,
        node_velocities=node_velocities, initial_gyro_bias=np.asarray(inertial['initial_gyro_bias_rad_s']),
        natural_cameras=natural_cameras, inertial=inertial, initialization=initialization, pnp_rms=pnp_rms)


def _base_tag_rows(directory, cache_dir, camera, output_times, owners_by_tag_id) -> _BaseTagRows:
    """复制冻结的基座标签检测，取录像区间内可用的观测并记录其画面多边形（供特征排除）。"""
    shutil.copyfile(cache_dir / 'observations.npz', directory / 'base_tag_observations.npz')
    base_detections = detection.load_observations(directory / 'base_tag_observations.npz')
    base_imu_times = base_detections.imu_times_s(camera.measurements['front']['cam_time_offset_ns'])
    valid = (base_detections.usable & (base_imu_times >= output_times[0])
             & (base_imu_times <= output_times[-1]))
    base_rows = np.flatnonzero(valid)
    base_corners = base_detections.corners_px
    frame_polygons = {}
    for row in base_rows:
        source_frame = int(base_detections.source_frame_indices[base_detections.sample_indices[row]])
        frame_polygons.setdefault(source_frame, []).append(base_corners[row])
    return _BaseTagRows(
        target_arms=tuple(owners_by_tag_id[int(base_detections.tag_ids[row])] for row in base_rows),
        times=np.round(base_imu_times[base_rows], 9), ids=base_detections.tag_ids[base_rows],
        pixels=base_corners[base_rows], frame_polygons=frame_polygons)


def _reject_single_tag_geometry(single_tag_mask, points, pixels, ids, raw_times, tag_seeds, ext, model,
                                parameters) -> list[dict]:
    """单标签因子候选按初值几何复核；不合格者从掩码中移除并返回原因（就地修改掩码）。"""
    rejected = []
    imu_from_rear_rotation = ext[:3, :3].T
    for row in np.flatnonzero(single_tag_mask):
        reason = single_tag_geometry(points[row], pixels[row],
            tag_seeds[row, :3, :3] @ imu_from_rear_rotation, model, cull_threshold(parameters))
        if reason:
            single_tag_mask[row] = False
            rejected.append(dict(tag_id=int(ids[row]), imu_time_s=float(raw_times[row]), reason=reason))
    return rejected


def _feature_streams(scene, recording, camera, detections, feature_selected, all_rear_times, output_times,
                     node_times, seed, ext, model, natural_cameras, base: _BaseTagRows, parameters):
    """工作镜头与环境镜头各一路关键帧特征流；标签区域从自然特征中排除。"""
    frame_polygons = {}
    for sample, corners in zip(detections.sample_indices, detections.corners_px):
        if feature_selected[sample]:
            frame_polygons.setdefault(int(detections.source_frame_indices[sample]), []).append(corners)
    front_selection = select_keyframes(output_times, seed[np.searchsorted(node_times, np.round(output_times, 9))], parameters)
    rear_samples = np.flatnonzero(feature_selected)
    rear_nodes = np.searchsorted(node_times, np.round(all_rear_times[rear_samples], 9))
    rear_selection = select_keyframes(all_rear_times[rear_samples], seed[rear_nodes], parameters)
    rear_samples = rear_samples[rear_selection]
    return (
        FeatureCameraInput('front', getattr(recording, 'source_frame_start', 0) + front_selection * scene.trajectory.values.video.frame_step,
            np.asarray(recording.video_times_s)[front_selection],
            np.searchsorted(node_times, np.round(output_times[front_selection], 9)),
            seed, natural_cameras['front'].T_cam_imu, natural_cameras['front'].model,
            mask_path=camera.files['mask'], excluded_polygons=base.frame_polygons),
        FeatureCameraInput('rear', detections.source_frame_indices[rear_samples],
            detections.video_times_s[rear_samples], rear_nodes[rear_selection],
            seed, ext, model, excluded_polygons=frame_polygons),
    )


def _recover_partner_coverage(inputs, prepared_arms, tracked, snapshots, board, parameters, tracking,
                              base_tag_cache_dirs, owners_by_tag_id):
    """Retry an uncovered partner once, selecting on training observations alone."""
    if not any('recovery_trigger' in item.inertial for item in prepared_arms):
        return
    maximum_gap = inputs.episode.scene.parameters.poster_vio_maximum_position_support_gap_s
    for index, (item, (tracks, _)) in enumerate(zip(prepared_arms, tracked)):
        if 'recovery_trigger' in item.inertial:
            continue
        gap = _potential_support_gap(item, tracks)
        if gap <= maximum_gap:
            continue
        # Even perfect reprojections cannot fill this training gap. Try the
        # other seed once and retain it only if its training coverage suffices.
        inputs.console.write(f'{item.arm} 训练位置支持缺口至少 {gap:.3f}s，尝试备用惯性初值',
                             recording=inputs.episode.root.name, step=(2, 3))
        attempt = dict(original_gap_s=gap)
        item.inertial['partner_coverage_recovery'] = attempt
        try:
            candidate = _prepare_arm(inputs, item.arm, snapshots[index], board, parameters,
                base_tag_cache_dir=base_tag_cache_dirs.get(item.arm), owners_by_tag_id=owners_by_tag_id,
                coverage_gap=gap)
            candidate_tracks = _track_features(inputs, [candidate], tracking)[0]
        except ValueError as error:
            attempt.update(accepted=False, reason=str(error))
            continue
        candidate_gap = _potential_support_gap(candidate, candidate_tracks[0])
        attempt.update(candidate_gap_s=candidate_gap,
                       accepted=candidate_gap <= maximum_gap)
        if attempt['accepted']:
            candidate.inertial['partner_coverage_recovery'] = attempt
            prepared_arms[index], tracked[index] = candidate, candidate_tracks


def _potential_support_gap(item, tracks_by_camera):
    """Lower bound on the final support gap, counting every eligible training tag."""
    tags = (~item.holdout) & (item.initialization_mask | item.single_tag_mask)
    events = [item.node_times[[0, -1]], item.raw_times[tags]]
    for tracks in tracks_by_camera.values():
        if tracks:
            nodes = np.concatenate([np.asarray(t['node_indices'])[natural_training(t)] for t in tracks])
            events.append(item.node_times[natural_support_nodes(nodes)])
    return float(np.diff(np.unique(np.concatenate(events))).max())


def _track_features(inputs: TaskInputs, prepared_arms: list[_PreparedArm], tracking) -> list:
    """各手自然特征跟踪（双手并行）；返回每手 (按镜头的轨迹, 前端诊断)。"""
    episode, parameters = inputs.episode, inputs.episode.scene.parameters

    def track_arm(item: _PreparedArm):
        frontend_diagnostics = {}
        tracks_by_camera = track_cameras(
            episode.recordings[item.arm], item.streams,
            max_features=parameters.poster_vio_maximum_features,
            maximum_tracks=parameters.poster_vio_maximum_tracks,
            minimum_track_length=parameters.poster_vio_minimum_track_length,
            maximum_track_length=parameters.poster_vio_maximum_track_length,
            maximum_observations_per_track=parameters.poster_vio_maximum_observations_per_track,
            parameters=tracking, diagnostics=frontend_diagnostics)
        return tracks_by_camera, frontend_diagnostics

    tracking_started = time.monotonic()
    label = '自然特征跟踪' + ('（双手并行）' if len(prepared_arms) > 1 else '')
    inputs.console.write(label, recording=episode.root.name, step=(2, 3))
    tracked = (daemon_map(track_arm, prepared_arms) if len(prepared_arms) > 1
               else [track_arm(prepared_arms[0])])
    track_count = sum(sum(len(value) for value in result[0].values()) for result in tracked)
    inputs.console.write(
        f'保留特征轨迹 {track_count} 条 · 耗时 {time.monotonic() - tracking_started:.1f}s',
        recording=episode.root.name, step=(2, 3))
    return tracked


def _assemble(inputs: TaskInputs, prepared_arms: list[_PreparedArm], tracked: list, parameters):
    """每手的观测、求解输入、资产记录与诊断。"""
    fusion = inputs.episode.scene.trajectory.values.fusion
    observations, solve_inputs = [], []
    camera_assets, imu_assets, diagnostics = {}, {}, {}
    for item, (tracks_by_camera, frontend_diagnostics) in zip(prepared_arms, tracked):
        natural_tracks = tuple(track for lens in ('front', 'rear') for track in tracks_by_camera[lens])
        observations.append(PosterVioObservation(
            item.arm, item.node_times[item.tag_indices], item.ids, item.pixels, natural_tracks,
            item.base.target_arms, item.base.times, item.base.ids, item.base.pixels))
        solve_inputs.append(ArmSolveInput(
            item.arm, item.node_times, item.recording.imu, item.seed, item.ext, item.model,
            item.points, item.pixels, item.tag_indices, fusion, parameters,
            natural_tracks, gravity_m_s2=item.gravity, initial_acc_bias=item.acc_bias,
            seed_velocities=item.node_velocities, initial_gyro_bias=item.initial_gyro_bias,
            natural_cameras=item.natural_cameras, tag_holdout_mask=item.holdout,
            tag_initialization_mask=item.initialization_mask, single_tag_factor_mask=item.single_tag_mask,
            recovered_inertial_seed='recovery_trigger' in item.inertial))
        camera_assets[item.arm] = dict(asset_hashes=item.camera.asset_hashes)
        imu = item.recording.imu
        imu_assets[item.arm] = dict(sample_count=len(imu), time_start=float(imu[0, 0]), time_end=float(imu[-1, 0]))
        diagnostics[item.arm] = dict(item.inertial, pnp_frame_rms_px=item.pnp_rms.tolist(),
            initialization=item.initialization, frontend=frontend_diagnostics,
            single_tag_factors=int(item.single_tag_mask.sum()),
            feature_tracks=len(natural_tracks), tag_frames=len(item.tag_times),
            base_tag_observations=len(item.base.times), base_tag_self_evidence=None,
            initialization_tag_frames=len(item.training_times),
            holdout_tag_frames=len(np.unique(item.raw_times[item.holdout])),
            feature_tracks_by_camera={lens: len(tracks) for lens, tracks in tracks_by_camera.items()})
    return observations, solve_inputs, camera_assets, imu_assets, diagnostics


def _share_gravity(solve_inputs: list[ArmSolveInput], scene):
    """双手各自的初始化重力须方向一致；求解统一使用其平均方向、标称模长。"""
    gravity_vectors = np.array([item.gravity_m_s2 for item in solve_inputs])
    directions = gravity_vectors / np.linalg.norm(gravity_vectors, axis=1)[:, None]
    difference = float(np.degrees(np.arccos(np.clip(np.min(directions @ directions.T), -1, 1))))
    if difference > scene.parameters.poster_vio_maximum_gravity_difference_deg:
        raise PosterVioQualityError(f'poster_vio 双手海报坐标重力方向不一致：{difference:.3f} deg')
    shared_gravity = np.mean(gravity_vectors, axis=0)
    shared_gravity *= scene.trajectory.values.fusion.gravity_m_s2 / np.linalg.norm(shared_gravity)
    return [replace(item, gravity_m_s2=shared_gravity) for item in solve_inputs], shared_gravity, difference


def _fit_base_tag_mounts(inputs: TaskInputs, request: PosterVioRequest, solve_inputs, observations,
                         base_tag_specs: dict, diagnostics: dict) -> list[ArmSolveInput]:
    """只为被另一只手看到的基座标签拟合安装外参。

    Discover cross-hand use before fitting any mount: self pixels alone do
    not constrain a trajectory with fixed camera calibration and TCP geometry.
    """
    parameters = inputs.episode.scene.parameters
    solve_inputs = list(solve_inputs)
    owners = base_tags.active_owners(request.problem(), solve_inputs)
    for index, item in enumerate(solve_inputs):
        diagnostics[item.arm]['base_tag_mount_status'] = ('enabled' if item.arm in owners
                                                         else 'no_cross_camera_observations')
        if item.arm not in owners:
            continue
        observation = observations[index]
        own_rows = np.array([target == item.arm for target in observation.base_tag_target_arms], dtype=bool)
        if len(np.unique(observation.base_tag_times_s[own_rows])) < parameters.poster_vio_base_tag_minimum_self_frames:
            raise PosterVioQualityError(f'{item.arm} base tag 跨手约束需要足够的自身观测时刻')
        front = item.natural_cameras['front']
        try:
            mount, evidence = estimate_static_tag_extrinsic(
                observation.base_tag_corners_px[own_rows], front.T_cam_imu, front.model,
                base_tag_specs[item.arm]['size_m'],
                maximum_rms_px=parameters.tag_maximum_reprojection_rms_px,
                maximum_cull_fraction=parameters.tag_maximum_cull_fraction,
                minimum_frames=parameters.poster_vio_base_tag_minimum_self_frames)
        except ValueError as error:
            diagnostics[item.arm]['base_tag_self_evidence'] = dict(
                status='rejected', observations=int(np.count_nonzero(own_rows)),
                unique_frames=int(len(np.unique(observation.base_tag_times_s[own_rows]))),
                reason=str(error))
            write_json(inputs.directory / 'poster_vio' / 'inputs_failure.json', dict(
                status='failed', stage='base_tag_mount', arm=item.arm,
                diagnostics=diagnostics, base_tag_specs=base_tag_specs,
                reason=str(error)))
            raise PosterVioQualityError(f'{item.arm} base tag 安装拟合失败：{error}') from error
        solve_inputs[index] = replace(item, T_imu_base_tag=mount)
        diagnostics[item.arm]['base_tag_self_evidence'] = evidence
    return solve_inputs


def _write_numerical_inputs(inputs: TaskInputs, solve_inputs, observations) -> dict:
    """把每手冻结的数值输入写成 numerical_inputs.npz，返回各手输入文件的内容哈希。"""
    assets = {}
    for item in solve_inputs:
        directory = inputs.directory / 'poster_vio' / item.arm
        arrays = dict(times_s=item.times_s, imu=item.imu, seed_poses=item.seed_poses,
                      T_cam_imu=item.T_cam_imu, board_points_m=item.board_points_m, pixels=item.pixels,
                      tag_indices=item.tag_indices, gravity_m_s2=item.gravity_m_s2,
                      initial_acc_bias=item.initial_acc_bias, seed_velocities=item.seed_velocities,
                      initial_gyro_bias=item.initial_gyro_bias, tag_holdout_mask=item.tag_holdout_mask,
                      tag_initialization_mask=item.tag_initialization_mask, single_tag_factor_mask=item.single_tag_factor_mask,
                      base_tag_indices=item.base_tag_indices,
                      base_tag_present=np.array(item.T_imu_base_tag is not None),
                      T_imu_base_tag=item.T_imu_base_tag if item.T_imu_base_tag is not None else np.eye(4))
        observation = next(obs for obs in observations if obs.arm == item.arm)
        arrays.update(base_tag_times_s=observation.base_tag_times_s,
                      base_tag_ids=observation.base_tag_ids,
                      base_tag_target_arms=np.asarray(observation.base_tag_target_arms, dtype='U5'),
                      base_tag_corners_px=observation.base_tag_corners_px)
        for lens, calibration in item.natural_cameras.items():
            arrays[f'natural_{lens}_T_cam_imu'] = calibration.T_cam_imu
            arrays[f'natural_{lens}_intrinsics'] = np.array([
                getattr(calibration.model, name) for name in ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4')])
        for index, segment in enumerate(item.front_maps):
            for key, value in vars(segment).items():
                arrays[f'front_map_{index}_{key}'] = np.asarray(value)
        for track in item.natural_tracks:
            for key in ('node_indices', 'pixels', 'initial_point_m'):
                arrays[f'track_{track["camera"]}_{track["track_id"]}_{key}'] = track[key]
            arrays[f'track_{track["camera"]}_{track["track_id"]}_revisit_count'] = track.get('revisit_count', 0)
            arrays[f'track_{track["camera"]}_{track["track_id"]}_holdout_mask'] = track.get(
                'holdout_mask', np.zeros(len(track['pixels']), dtype=bool))
        np.savez_compressed(directory / 'numerical_inputs.npz', **arrays)
        assets[item.arm] = {name: file_hash(directory / name)
                            for name in ('observations.npz', 'numerical_inputs.npz')}
        if (directory / 'base_tag_observations.npz').is_file():
            assets[item.arm]['base_tag_observations.npz'] = file_hash(directory / 'base_tag_observations.npz')
    return assets


def _solver_tag_selection(times, maximum_dt_s):
    """Select complete tag frames for the joint graph without changing seeds."""
    times = np.asarray(times, dtype=float)
    if not len(times) or not maximum_dt_s:
        return np.ones(len(times), dtype=bool)
    unique = np.unique(np.round(times, 9))
    selected_times = [unique[0]]
    for time_s in unique[1:]:
        if time_s - selected_times[-1] >= maximum_dt_s - 1e-9:
            selected_times.append(time_s)
    if selected_times[-1] != unique[-1]:
        selected_times.append(unique[-1])
    return np.isin(np.round(times, 9), np.asarray(selected_times))
