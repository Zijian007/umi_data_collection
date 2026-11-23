"""Task route and publication evidence for joint poster visual-inertial BA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import time
import numpy as np
from software.camera.timing import video_to_imu_seconds
from software.camera.geometry import imu_from_camera
from software.infrastructure.concurrency import daemon_map
from software.trajectory.algorithms.poster_vio.solver import natural_camera
from software.vision.poster.geometry import camera_model
from software.infrastructure.file_store import write_json
from software.scene.models import TaskInputs, ArmExecutionResult
from software.trajectory.models import FusedTrajectory
from software.scene.management.collection_schema import require_poster
from software.scene.pipeline.tasks import trajectory_writers, measurements
from software.scene.errors import PosterVioQualityError
from software.trajectory.algorithms.poster_vio import ArmSolveInput, PosterVioPolicy, assess_observations, solve_joint_poster_imu, validate_problem, NaturalCamera, PosterVioObservation, PosterVioProblem, base_tags
from software.trajectory.reporting.constraint_quality import longest_gap, maximum_support_gap
from software.vision.poster.layout import load as load_board
from typing import Any
import shutil
from software.infrastructure.provenance import file_hash
from software.scene import layout
from software.trajectory.algorithms.poster_vio.models import FeatureTrackingParameters, PosterVioParameters
from software.trajectory.algorithms.poster_vio.frontend import FeatureCameraInput, track_cameras
from software.trajectory.algorithms.poster_vio.refinement import select_holdout
from software.trajectory.algorithms.poster_vio.initialization import select_keyframes, select_observations, robust_seed_poses, align_inertial, inertial_tail_seed, estimate_static_tag_extrinsic, single_tag_geometry
from software.vision.poster import detection
from software.vision.poster.quality import cull_threshold


def execute_task(inputs: TaskInputs, *, fallback=None) -> ArmExecutionResult:
    """Prepare numerical inputs from the task recordings and solve."""
    scene = inputs.episode.scene
    require_poster(scene)
    allow_quality_fallback = scene.reconstruction_mode == 'auto'
    try:
        request, arms, board_seed = prepare_task_inputs(inputs)
    except PosterVioQualityError:
        raise
    except ValueError as error:
        if not allow_quality_fallback:
            raise
        raise PosterVioQualityError(f'poster_vio 前端准备质量不足：{error}') from error
    return _solve_task(inputs, request, arms, board_seed, fallback=fallback,
                       allow_quality_fallback=allow_quality_fallback)


def replay_task(inputs: TaskInputs, *, request: PosterVioRequest, arms: tuple[ArmSolveInput, ...],
                board_seed: np.ndarray) -> ArmExecutionResult:
    """Replay against the already prepared recordings and their frozen time grids."""
    require_poster(inputs.episode.scene)
    if request is None or arms is None or board_seed is None:
        raise ValueError('poster_vio 冻结请求、数值输入和海报初值必须一起提供')
    return _solve_task(inputs, request, arms, board_seed, fallback=None, allow_quality_fallback=False)


def _solve_task(inputs: TaskInputs, request, arms, board_seed, *, fallback,
                allow_quality_fallback: bool) -> ArmExecutionResult:
    episode = inputs.episode
    request.validate_assets()
    problem = request.problem()
    validate_problem(problem)
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
        base_spec = (getattr(recording.camera, 'base_tag', None)
                      if getattr(episode.scene, 'poster_vio_base_tag_enabled', True) else None)
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
    quality = assess_observations(problem, policy)
    if quality.status != 'ready':
        raise PosterVioQualityError(f'poster_vio 海报观测质量不足：{quality.status}')
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
    except PosterVioQualityError:
        raise
    except ValueError as error:
        if not allow_quality_fallback:
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
    coordinate = dict(request.coordinate_record, scope='task_poster', axes='board_layout', units='metres',
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
    return trajectory_writers.write_task_results(inputs, results, coordinate, parallel=bool(inputs.arm_cpu_sets))


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


def prepare_task_inputs(inputs: TaskInputs):
    """Use only this task's pixels, IMU, layout and frozen camera assets."""
    episode, scene = inputs.episode, inputs.episode.scene
    parameters = scene.parameters.for_algorithm(PosterVioParameters)
    tracking = scene.parameters.for_algorithm(FeatureTrackingParameters)
    require_poster(scene)
    arms = tuple(episode.expected_arms)
    if set(inputs.recordings) != set(arms):
        raise ValueError('poster_vio 需要所有预期使用手的录像')
    observations, solve_inputs = [], []
    camera_assets, imu_assets, diagnostics = {}, {}, {}
    base_tag_specs = {}
    prepared_arms = []
    for arm in arms:
        spec = (getattr(inputs.recordings[arm].camera, 'base_tag', None)
                if getattr(scene, 'poster_vio_base_tag_enabled', True) else None)
        if spec is not None:
            base_tag_specs[arm] = dict(id=int(spec['id']), size_m=float(spec['size_mm']) / 1000.,
                                       dictionary=spec.get('dictionary', 'DICT_4X4_100'))
    snapshots = measurements.prepare_detection_group(inputs, arms, parallel=True)
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

    inputs.console.write('海报姿态初始化与 IMU 对齐', recording=episode.root.name, step=(2, 3))
    for arm, snapshot in zip(arms, snapshots):
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
            raise ValueError('poster_vio 标签时刻至少需要三个')
        base_target_arms = ()
        base_times = np.empty(0, dtype=float)
        base_ids = np.empty(0, dtype=int)
        base_pixels = np.empty((0, 4, 2), dtype=float)
        base_frame_polygons = {}
        if len(arms) > 1 and base_tag_specs:
            base_cache = base_tag_cache_dirs[arm]
            shutil.copyfile(base_cache / 'observations.npz', directory / 'base_tag_observations.npz')
            base_detections = detection.load_observations(directory / 'base_tag_observations.npz')
            base_imu_times = base_detections.imu_times_s(camera.measurements['front']['cam_time_offset_ns'])
            valid = (base_detections.usable & (base_imu_times >= output_times[0])
                     & (base_imu_times <= output_times[-1]))
            base_rows = np.flatnonzero(valid)
            base_target_arms = tuple(owners_by_tag_id[int(base_detections.tag_ids[row])] for row in base_rows)
            base_times = np.round(base_imu_times[base_rows], 9)
            base_ids = base_detections.tag_ids[base_rows]
            base_corners = base_detections.corners_px
            base_pixels = base_corners[base_rows]
            for row in base_rows:
                source_frame = int(base_detections.source_frame_indices[base_detections.sample_indices[row]])
                base_frame_polygons.setdefault(source_frame, []).append(base_corners[row])
        model = camera_model(measured)
        ext = np.linalg.inv(imu_from_camera(measured))
        holdout = select_holdout(np.round(raw_times, 9), parameters)
        training_times, training_indices = np.unique(np.round(raw_times[~holdout], 9), return_inverse=True)
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
        gravity, acc_bias, velocities, inertial = align_inertial(
            training_times, tag_seeds, recording.imu, scene.trajectory.values.fusion)
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
            raise ValueError('poster_vio 求解采样后标签时刻至少需要三个')
        all_rear_times = video_to_imu_seconds(detections.video_times_s, measured['cam_time_offset_ns'])
        selected = (all_rear_times >= output_times[0]) & (all_rear_times <= output_times[-1])
        rear_times = np.round(all_rear_times[selected], 9)
        held_times = np.unique(np.round(raw_times[holdout], 9))
        feature_selected = selected & ~np.isin(np.round(all_rear_times, 9), held_times)
        # Output and rear-feature nodes are included even where no tag is visible.
        node_times = np.unique(np.round(np.r_[output_times, rear_times], 9))
        seed, node_velocities, tail_seed = inertial_tail_seed(training_times, tag_seeds, velocities, node_times,
            recording.imu, gravity, np.asarray(inertial['initial_gyro_bias_rad_s']), acc_bias,
            parameters=parameters, fusion=scene.trajectory.values.fusion)
        inertial['tail_seed'] = tail_seed
        tag_indices = np.searchsorted(node_times, np.round(raw_times, 9))
        rejected_singles = []
        imu_from_rear_rotation = ext[:3, :3].T
        for row in np.flatnonzero(single_tag_mask):
            reason = single_tag_geometry(points[row], pixels[row],
                seed[tag_indices[row], :3, :3] @ imu_from_rear_rotation, model, cull_threshold(parameters))
            if reason:
                single_tag_mask[row] = False
                rejected_singles.append(dict(tag_id=int(ids[row]), imu_time_s=float(raw_times[row]), reason=reason))
        initialization['single_tag_geometry_rejections'] = rejected_singles
        frame_polygons = {}
        for sample, corners in zip(detections.sample_indices, detections.corners_px):
            if feature_selected[sample]:
                frame_polygons.setdefault(int(detections.source_frame_indices[sample]), []).append(corners)
        natural_cameras = {
            lens: NaturalCamera(np.linalg.inv(imu_from_camera(camera.measurements[lens])),
                                 camera_model(camera.measurements[lens]))
            for lens in ('front', 'rear')}
        front_selection = select_keyframes(output_times, seed[np.searchsorted(node_times, np.round(output_times, 9))], parameters)
        rear_samples = np.flatnonzero(feature_selected)
        rear_nodes = np.searchsorted(node_times, np.round(all_rear_times[rear_samples], 9))
        rear_selection = select_keyframes(all_rear_times[rear_samples], seed[rear_nodes], parameters)
        rear_samples = rear_samples[rear_selection]
        streams = (
            FeatureCameraInput('front', getattr(recording, 'source_frame_start', 0) + front_selection * scene.trajectory.values.video.frame_step,
                np.asarray(recording.video_times_s)[front_selection],
                np.searchsorted(node_times, np.round(output_times[front_selection], 9)),
                seed, natural_cameras['front'].T_cam_imu, natural_cameras['front'].model,
                mask_path=camera.files['mask'], excluded_polygons=base_frame_polygons),
            FeatureCameraInput('rear', detections.source_frame_indices[rear_samples],
                detections.video_times_s[rear_samples], rear_nodes[rear_selection],
                seed, ext, model, excluded_polygons=frame_polygons),
        )
        prepared_arms.append(dict(
            arm=arm, recording=recording, streams=streams, node_times=node_times,
            tag_indices=tag_indices, ids=ids, pixels=pixels, base_target_arms=base_target_arms,
            base_times=base_times, base_ids=base_ids, base_pixels=base_pixels,
            seed=seed, ext=ext, model=model, points=points, holdout=holdout,
            initialization_mask=initialization_mask, single_tag_mask=single_tag_mask, gravity=gravity,
            acc_bias=acc_bias, node_velocities=node_velocities,
            initial_gyro_bias=np.asarray(inertial['initial_gyro_bias_rad_s']),
            natural_cameras=natural_cameras, inertial=inertial,
            pnp_rms=pnp_rms, tag_times=tag_times, initialization=initialization,
            camera=camera, base_tag_observations=len(base_times),
            training_times=training_times, raw_times=raw_times))

    def track_arm(item):
        frontend_diagnostics = {}
        tracks_by_camera = track_cameras(
            episode.recordings[item['arm']], item['streams'],
            max_features=scene.parameters.poster_vio_maximum_features,
            maximum_tracks=scene.parameters.poster_vio_maximum_tracks,
            minimum_track_length=scene.parameters.poster_vio_minimum_track_length,
            maximum_track_length=scene.parameters.poster_vio_maximum_track_length,
            maximum_observations_per_track=scene.parameters.poster_vio_maximum_observations_per_track,
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
    for item, (tracks_by_camera, frontend_diagnostics) in zip(prepared_arms, tracked):
        natural_tracks = tuple(track for lens in ('front', 'rear') for track in tracks_by_camera[lens])
        observations.append(PosterVioObservation(
            item['arm'], item['node_times'][item['tag_indices']], item['ids'], item['pixels'], natural_tracks,
            item['base_target_arms'], item['base_times'], item['base_ids'], item['base_pixels']))
        solve_inputs.append(ArmSolveInput(
            item['arm'], item['node_times'], item['recording'].imu, item['seed'], item['ext'], item['model'],
            item['points'], item['pixels'], item['tag_indices'], scene.trajectory.values.fusion, parameters,
            natural_tracks, gravity_m_s2=item['gravity'], initial_acc_bias=item['acc_bias'],
            seed_velocities=item['node_velocities'], initial_gyro_bias=item['initial_gyro_bias'],
            natural_cameras=item['natural_cameras'], tag_holdout_mask=item['holdout'],
            tag_initialization_mask=item['initialization_mask'], single_tag_factor_mask=item['single_tag_mask']))
        camera_assets[item['arm']] = dict(asset_hashes=item['camera'].asset_hashes)
        imu = item['recording'].imu
        imu_assets[item['arm']] = dict(sample_count=len(imu), time_start=float(imu[0, 0]), time_end=float(imu[-1, 0]))
        diagnostics[item['arm']] = dict(item['inertial'], pnp_frame_rms_px=item['pnp_rms'].tolist(),
            initialization=item['initialization'], frontend=frontend_diagnostics,
            single_tag_factors=int(item['single_tag_mask'].sum()),
            feature_tracks=len(natural_tracks), tag_frames=len(item['tag_times']),
            base_tag_observations=item['base_tag_observations'], base_tag_self_evidence=None,
            initialization_tag_frames=len(item['training_times']),
            holdout_tag_frames=len(np.unique(item['raw_times'][item['holdout']])),
            feature_tracks_by_camera={lens: len(tracks) for lens, tracks in tracks_by_camera.items()})
    gravity_vectors = np.array([item.gravity_m_s2 for item in solve_inputs])
    directions = gravity_vectors / np.linalg.norm(gravity_vectors, axis=1)[:, None]
    difference = float(np.degrees(np.arccos(np.clip(np.min(directions @ directions.T), -1, 1))))
    if difference > scene.parameters.poster_vio_maximum_gravity_difference_deg:
        raise ValueError(f'poster_vio 双手海报坐标重力方向不一致：{difference:.3f} deg')
    shared_gravity = np.mean(gravity_vectors, axis=0)
    shared_gravity *= scene.trajectory.values.fusion.gravity_m_s2 / np.linalg.norm(shared_gravity)
    solve_inputs = [replace(item, gravity_m_s2=shared_gravity) for item in solve_inputs]
    coordinate = dict(name=scene.name, session=episode.session.name, version=None,
                       scope='task_poster', frame_id=f'poster:{inputs.directory.name}',
                       axes='board_layout', units='metres', gravity_difference_deg=difference)
    # Native IMU -> common task clock. Lens calibration offset must be removed
    # before applying the cross-camera video-clock shift.
    clock_offsets = {arm: float(inputs.offsets_s[arm])
                     - inputs.recordings[arm].camera.measurements['front']['cam_time_offset_ns'] * 1e-9
                     for arm in arms}
    request = PosterVioRequest(arms, board.asset_hash, tuple(observations), camera_assets, imu_assets,
                               coordinate, base_tag_specs=base_tag_specs, clock_offsets_s=clock_offsets)
    # Discover cross-hand use before fitting any mount: self pixels alone do
    # not constrain a trajectory with fixed camera calibration and TCP geometry.
    owners = base_tags.active_owners(request.problem(), solve_inputs)
    for index, item in enumerate(solve_inputs):
        diagnostics[item.arm]['base_tag_mount_status'] = ('enabled' if item.arm in owners
                                                         else 'no_cross_camera_observations')
        if item.arm not in owners:
            continue
        observation = observations[index]
        own_rows = np.array([target == item.arm for target in observation.base_tag_target_arms], dtype=bool)
        if len(np.unique(observation.base_tag_times_s[own_rows])) < scene.parameters.poster_vio_base_tag_minimum_self_frames:
            raise ValueError(f'{item.arm} base tag 跨手约束需要足够的自身观测时刻')
        front = item.natural_cameras['front']
        try:
            mount, evidence = estimate_static_tag_extrinsic(
                observation.base_tag_corners_px[own_rows], front.T_cam_imu, front.model,
                base_tag_specs[item.arm]['size_m'],
                maximum_rms_px=scene.parameters.tag_maximum_reprojection_rms_px,
                maximum_cull_fraction=scene.parameters.tag_maximum_cull_fraction,
                minimum_frames=scene.parameters.poster_vio_base_tag_minimum_self_frames)
        except ValueError as error:
            diagnostics[item.arm]['base_tag_self_evidence'] = dict(
                status='rejected', observations=int(np.count_nonzero(own_rows)),
                unique_frames=int(len(np.unique(observation.base_tag_times_s[own_rows]))),
                reason=str(error))
            write_json(inputs.directory / 'poster_vio' / 'inputs_failure.json', dict(
                status='failed', stage='base_tag_mount', arm=item.arm,
                diagnostics=diagnostics, base_tag_specs=base_tag_specs,
                reason=str(error)))
            raise ValueError(f'{item.arm} base tag 安装拟合失败：{error}') from error
        solve_inputs[index] = replace(item, T_imu_base_tag=mount)
        diagnostics[item.arm]['base_tag_self_evidence'] = evidence
    solve_inputs = base_tags.expand_nodes(request.problem(), solve_inputs)
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
    write_json(inputs.directory / 'poster_vio' / 'inputs.json', dict(
        kind='umi_poster_vio_inputs', schema_version=1,
        board=board.record(), camera_assets=camera_assets, imu_assets=imu_assets,
        assets=assets, diagnostics=diagnostics, shared_gravity_m_s2=shared_gravity.tolist(),
        coordinate=coordinate, base_tag_specs=base_tag_specs,
        clock_offsets_s=clock_offsets,
        scene_map_priors_used=False))
    return request, solve_inputs, np.eye(4)
