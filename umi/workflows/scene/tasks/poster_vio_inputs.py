"""poster_vio 任务输入准备：本任务的像素、IMU、海报布局与冻结相机资产 → 联合图输入。

阅读顺序与 :func:`prepare_task_inputs` 的步骤一致：
基座标签声明 → 海报与基座标签检测 → 逐手初始化与特征流 → 自然特征跟踪 →
组装观测与求解输入 → 双手共享重力 → 基座标签安装拟合 → 写出可复核的数值输入。
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from umi.config.collection_schema import require_poster
from umi.core.geometry import imu_from_camera
from umi.core.timing import video_to_imu_seconds
from umi.estimation.poster_vio import ArmSolveInput, NaturalCamera, PosterVioObservation, PosterVioProblem, base_tags
from umi.estimation.poster_vio.frontend import FeatureCameraInput, track_cameras
from umi.estimation.poster_vio.initialization import (align_inertial, estimate_static_tag_extrinsic,
                                                      inertial_tail_seed, robust_seed_poses, select_keyframes,
                                                      select_observations, single_tag_geometry)
from umi.estimation.poster_vio.models import FeatureTrackingParameters, PosterVioParameters
from umi.estimation.poster_vio.refinement import select_holdout
from umi.infra.concurrency import daemon_map
from umi.infra.file_store import write_json
from umi.infra.provenance import file_hash
from umi.perception.poster import detection
from umi.perception.poster.geometry import camera_model
from umi.perception.poster.quality import cull_threshold
from umi.store import layout
from umi.workflows.scene.inputs.batch import task_poster_frame
from umi.store.models import TaskInputs
from umi.workflows.scene.tasks import measurements


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
                if getattr(scene, 'poster_vio_base_tag_enabled', True) else None)
        if spec is not None:
            specs[arm] = dict(id=int(spec['id']), size_m=float(spec['size_mm']) / 1000.,
                              dictionary=spec.get('dictionary', 'DICT_4X4_100'))
    return specs


def _detect(inputs: TaskInputs, arms: tuple[str, ...], base_tag_specs: dict):
    """环境海报检测（各手）与双手基座标签检测；返回快照、布局、标签归属和基座缓存目录。"""
    episode, scene = inputs.episode, inputs.episode.scene
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
    return snapshots, board, owners_by_tag_id, base_tag_cache_dirs


def _prepare_arm(inputs: TaskInputs, arm: str, snapshot, board, parameters: PosterVioParameters, *,
                 base_tag_cache_dir, owners_by_tag_id: dict) -> _PreparedArm:
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
        raise ValueError('poster_vio 标签时刻至少需要三个')
    base = (_base_tag_rows(directory, base_tag_cache_dir, camera, output_times, owners_by_tag_id)
            if base_tag_cache_dir is not None else _BaseTagRows())

    # 留出部分标签时刻后，用其余时刻做鲁棒 PnP 初值和惯性对齐。
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
        raise ValueError('poster_vio 求解采样后标签时刻至少需要三个')

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
            tag_initialization_mask=item.initialization_mask, single_tag_factor_mask=item.single_tag_mask))
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
        raise ValueError(f'poster_vio 双手海报坐标重力方向不一致：{difference:.3f} deg')
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
            raise ValueError(f'{item.arm} base tag 跨手约束需要足够的自身观测时刻')
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
            raise ValueError(f'{item.arm} base tag 安装拟合失败：{error}') from error
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
