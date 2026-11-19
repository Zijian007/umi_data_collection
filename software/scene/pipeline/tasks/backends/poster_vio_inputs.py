"""Freeze poster-VIO observations, metric seeds and inertial initialization."""
from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import shutil

import numpy as np

from software.camera.geometry import imu_from_camera
from software.camera.timing import video_to_imu_seconds
from software.infrastructure.file_store import write_json
from software.infrastructure.provenance import file_hash
from software.scene import layout
from software.scene.models import TaskInputs
from software.scene.pipeline.tasks import measurements
from software.scene.pipeline.tasks.policy import require_poster
from software.trajectory.poster_vio import ArmSolveInput, NaturalCamera, PosterVioObservation, PosterVioRequest
from software.trajectory.poster_vio.frontend import FeatureCameraInput, track_cameras
from software.trajectory.poster_vio.refinement import select_holdout
from software.trajectory.poster_vio.keyframes import select_indices
from software.trajectory.poster_vio import base_tags
from software.trajectory.poster_vio.initialization import (
    select_observations, robust_seed_poses, align_inertial, interpolate_seed,
    estimate_static_tag_extrinsic)
from software.vision.poster import detection, layout as tag_board
from software.vision.poster.geometry import camera_model


def prepare_task_inputs(inputs: TaskInputs):
    """Use only this task's pixels, IMU, layout and frozen camera assets."""
    episode, scene = inputs.episode, inputs.episode.scene
    require_poster(scene)
    board = tag_board.load(scene.tag_layout)
    arms = tuple(episode.expected_arms)
    if set(inputs.recordings) != set(arms):
        raise ValueError('poster_vio 需要所有预期使用手的录像')
    observations, solve_inputs = [], []
    camera_assets, imu_assets, diagnostics = {}, {}, {}
    base_tag_specs = {}
    for arm in arms:
        spec = getattr(inputs.recordings[arm].camera, 'base_tag', None)
        if spec is not None:
            base_tag_specs[arm] = dict(id=int(spec['id']), size_m=float(spec['size_mm']) / 1000.,
                                       dictionary=spec.get('dictionary', 'DICT_4X4_100'))
    owners_by_tag_id = {spec['id']: arm for arm, spec in base_tag_specs.items()}
    if len(owners_by_tag_id) != len(base_tag_specs):
        raise ValueError('双手 base tag ID 必须唯一')
    if len(arms) > 1 and set(owners_by_tag_id) & set(board.ids):
        raise ValueError('base tag ID 不能与环境海报 ID 冲突')

    # Each arm has its own source video and cache directory.  Run the shared
    # detection preparation concurrently; this is the dominant cost for a
    # bimanual poster-vio task and does not share mutable numerical state.
    def prepare_arm_detections(arm: str) -> None:
        measurements.prepare_detections(
            episode, arm, inputs.recordings[arm], cache_audit=inputs.cache_audit)

    if len(arms) > 1:
        inputs.console.write('双手环境标签与夹爪检测并行', recording=episode.root.name,
                             step=(2, 3))
        with ThreadPoolExecutor(max_workers=len(arms), thread_name_prefix='poster-detection') as executor:
            list(executor.map(prepare_arm_detections, arms))
    else:
        prepare_arm_detections(arms[0])

    for arm in arms:
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
        cache = layout.metadata(episode.root, 'cache/environment_tags') / arm
        # prepare_detections may skip a lone detector; cached validates source identity
        # in both cases and runs the missing detector when necessary.
        detection.cached(episode.recordings[arm], cache,
            detection.DetectionSettings(border_bits=board.border_bits, ids=board.ids,
                dictionary=board.dictionary, minimum_side_px=scene.parameters.tag_minimum_side_px),
            layout=board, progress=inputs.console.write, cache_audit=inputs.cache_audit)
        directory = inputs.directory / 'poster_vio' / arm
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache / 'observations.npz', directory / 'observations.npz')
        detections = detection.load_observations(directory / 'observations.npz')
        tag_times, _, ids, points, pixels, raw_times = select_observations(
            detections, board, measured, scene.parameters,
            start_s=output_times[0], end_s=output_times[-1])
        tag_times = np.round(tag_times, 9)
        if len(np.unique(tag_times)) != len(tag_times):
            raise ValueError('poster_vio 标签时刻在纳秒舍入后不唯一')
        base_target_arms = ()
        base_times = np.empty(0, dtype=float)
        base_ids = np.empty(0, dtype=int)
        base_pixels = np.empty((0, 4, 2), dtype=float)
        base_frame_polygons = {}
        if len(arms) > 1 and base_tag_specs:
            base_cache = layout.metadata(episode.root, 'cache/base_tags') / arm
            detection.cached(episode.recordings[arm], base_cache,
                detection.DetectionSettings(border_bits=1, ids=tuple(sorted(owners_by_tag_id)),
                    lens='front', minimum_side_px=scene.parameters.tag_minimum_side_px),
                progress=inputs.console.write, cache_audit=inputs.cache_audit)
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
        holdout = select_holdout(np.round(raw_times, 9), scene.parameters)
        training_times, training_indices = np.unique(np.round(raw_times[~holdout], 9), return_inverse=True)
        training_times, tag_seeds, pnp_rms, initialization_keep, initialization = robust_seed_poses(
            training_times, training_indices, points[~holdout], pixels[~holdout], ext, model, scene.parameters)
        initialization_mask = np.ones(len(points), dtype=bool)
        initialization_mask[~holdout] = initialization_keep
        for record in initialization['removed']:
            record['observation'] = int(np.flatnonzero(~holdout)[record['observation']])
            record['tag_id'] = int(ids[record['observation']])
        gravity, acc_bias, velocities, inertial = align_inertial(
            training_times, tag_seeds, recording.imu, scene.trajectory.values.fusion)
        all_rear_times = video_to_imu_seconds(detections.video_times_s, measured['cam_time_offset_ns'])
        selected = (all_rear_times >= output_times[0]) & (all_rear_times <= output_times[-1])
        rear_times = np.round(all_rear_times[selected], 9)
        held_times = np.unique(np.round(raw_times[holdout], 9))
        feature_selected = selected & ~np.isin(np.round(all_rear_times, 9), held_times)
        # Output and rear-feature nodes are included even where no tag is visible.
        node_times = np.unique(np.round(np.r_[output_times, rear_times], 9))
        seed = interpolate_seed(training_times, tag_seeds, node_times)
        node_velocities = np.column_stack([np.interp(node_times, training_times, velocities[:, i]) for i in range(3)])
        tag_indices = np.searchsorted(node_times, np.round(raw_times, 9))
        frame_polygons = {}
        for sample, corners in zip(detections.sample_indices, detections.corners_px):
            if feature_selected[sample]:
                frame_polygons.setdefault(int(detections.source_frame_indices[sample]), []).append(corners)
        natural_cameras = {
            lens: NaturalCamera(np.linalg.inv(imu_from_camera(camera.measurements[lens])),
                                 camera_model(camera.measurements[lens]))
            for lens in ('front', 'rear')}
        front_selection = select_indices(output_times, seed[np.searchsorted(node_times, np.round(output_times, 9))], scene.parameters)
        rear_samples = np.flatnonzero(feature_selected)
        rear_nodes = np.searchsorted(node_times, np.round(all_rear_times[rear_samples], 9))
        rear_selection = select_indices(all_rear_times[rear_samples], seed[rear_nodes], scene.parameters)
        rear_samples = rear_samples[rear_selection]
        streams = (
            FeatureCameraInput('front', front_selection * scene.trajectory.values.video.frame_step,
                np.asarray(recording.video_times_s)[front_selection],
                np.searchsorted(node_times, np.round(output_times[front_selection], 9)),
                seed, natural_cameras['front'].T_cam_imu, natural_cameras['front'].model,
                mask_path=camera.files['mask'], excluded_polygons=base_frame_polygons),
            FeatureCameraInput('rear', detections.source_frame_indices[rear_samples],
                detections.video_times_s[rear_samples], rear_nodes[rear_selection],
                seed, ext, model, excluded_polygons=frame_polygons),
        )
        frontend_diagnostics = {}
        tracks_by_camera = track_cameras(episode.recordings[arm], streams,
            max_features=scene.parameters.poster_vio_maximum_features,
            maximum_tracks=scene.parameters.poster_vio_maximum_tracks,
            minimum_track_length=scene.parameters.poster_vio_minimum_track_length,
            maximum_track_length=scene.parameters.poster_vio_maximum_track_length,
            parameters=scene.parameters, diagnostics=frontend_diagnostics)
        natural_tracks = tuple(track for lens in ('front', 'rear') for track in tracks_by_camera[lens])
        observations.append(PosterVioObservation(
            arm, node_times[tag_indices], ids, pixels, natural_tracks,
            base_target_arms, base_times, base_ids, base_pixels))
        solve_inputs.append(ArmSolveInput(
            arm, node_times, recording.imu, seed, ext, model, points, pixels, tag_indices,
            scene.trajectory.values, scene.parameters, natural_tracks,
            gravity_m_s2=gravity, initial_acc_bias=acc_bias, seed_velocities=node_velocities,
            initial_gyro_bias=np.asarray(inertial['initial_gyro_bias_rad_s']), natural_cameras=natural_cameras,
            tag_holdout_mask=holdout, tag_initialization_mask=initialization_mask))
        camera_assets[arm] = dict(asset_hashes=camera.asset_hashes)
        imu_assets[arm] = dict(sample_count=len(recording.imu), time_start=float(recording.imu[0, 0]),
                               time_end=float(recording.imu[-1, 0]))
        diagnostics[arm] = dict(inertial, pnp_frame_rms_px=pnp_rms.tolist(),
                                initialization=initialization,
                                frontend=frontend_diagnostics,
                                feature_tracks=len(natural_tracks), tag_frames=len(tag_times),
                                base_tag_observations=len(base_times),
                                base_tag_self_evidence=None,
                                initialization_tag_frames=len(training_times),
                                holdout_tag_frames=len(np.unique(raw_times[holdout])),
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
    owners = base_tags.active_owners(request, solve_inputs)
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
        mount, evidence = estimate_static_tag_extrinsic(
            observation.base_tag_corners_px[own_rows], front.T_cam_imu, front.model,
            base_tag_specs[item.arm]['size_m'],
            maximum_rms_px=scene.parameters.tag_maximum_reprojection_rms_px,
            maximum_cull_fraction=scene.parameters.tag_maximum_cull_fraction,
            minimum_frames=scene.parameters.poster_vio_base_tag_minimum_self_frames)
        solve_inputs[index] = replace(item, T_imu_base_tag=mount)
        diagnostics[item.arm]['base_tag_self_evidence'] = evidence
    solve_inputs = base_tags.expand_nodes(request, solve_inputs)
    assets = {}
    for item in solve_inputs:
        directory = inputs.directory / 'poster_vio' / item.arm
        arrays = dict(times_s=item.times_s, imu=item.imu, seed_poses=item.seed_poses,
                      T_cam_imu=item.T_cam_imu, board_points_m=item.board_points_m, pixels=item.pixels,
                      tag_indices=item.tag_indices, gravity_m_s2=item.gravity_m_s2,
                      initial_acc_bias=item.initial_acc_bias, seed_velocities=item.seed_velocities,
                      initial_gyro_bias=item.initial_gyro_bias, tag_holdout_mask=item.tag_holdout_mask,
                      tag_initialization_mask=item.tag_initialization_mask,
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
