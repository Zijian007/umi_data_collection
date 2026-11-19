"""Rigid mount variables, synchronized cross-arm factors and auditable residuals.

The graph structure follows specialist compose.py (see NOTICE.md). KB4
Jacobians, exact asynchronous IMU nodes and cumulative rejection are local.
"""
from dataclasses import replace

import gtsam
import numpy as np

from software.camera.geometry import resample_pose_series
from software.vision.poster import factors
from . import refinement


def matching_node(times, time):
    index = int(np.searchsorted(times, np.round(time, 9)))
    if index == len(times) or abs(times[index] - time) > 1e-9:
        raise ValueError('base tag 观测缺少精确同步 IMU 节点')
    return index


def observation_times(request, observer, target, time):
    return np.round(time + request.clock_offsets_s[observer]
                    - request.clock_offsets_s[target], 9)


def expand_nodes(request, arms):
    """Add exact cross-observation times to both chains before graph selection.

    Values are only initial guesses. Added states have real IMU factors and no
    fabricated visual evidence. Out-of-overlap cross observations stay in the
    request for explicit skipped diagnostics; they are never extrapolated.
    """
    if not any(len(obs.base_tag_times_s) for obs in request.observations):
        return tuple(arms)
    by_arm = {item.arm: item for item in arms}
    extra = {item.arm: list(item.times_s[item.base_tag_indices]) for item in arms}
    for obs in request.observations:
        for target, time in zip(obs.base_tag_target_arms, obs.base_tag_times_s):
            observer = by_arm[obs.arm]
            if not observer.times_s[0] <= time <= observer.times_s[-1]:
                raise ValueError('base tag 观察时刻超出观察者时间范围')
            extra[obs.arm].append(float(time))
            if target != obs.arm:
                owner_time = observation_times(request, obs.arm, target, time)
                owner = by_arm[target]
                if owner.times_s[0] <= owner_time <= owner.times_s[-1]:
                    extra[target].append(float(owner_time))
    result = []
    for item in arms:
        source_times = np.round(item.times_s, 9)
        if np.any(np.diff(source_times) <= 0):
            raise ValueError('base tag 原始状态时刻在纳秒舍入后不唯一')
        times = np.unique(np.round(np.r_[source_times, extra[item.arm]], 9))
        base_indices = np.searchsorted(times, np.unique(np.round(extra[item.arm], 9)))
        if np.array_equal(times, item.times_s) and np.array_equal(base_indices, item.base_tag_indices):
            result.append(item)
            continue
        seeds = resample_pose_series(source_times, item.seed_poses, times)
        velocities = (None if item.seed_velocities is None else np.column_stack([
            np.interp(times, source_times, item.seed_velocities[:, axis]) for axis in range(3)]))
        result.append(replace(item, times_s=times, seed_poses=seeds, seed_velocities=velocities,
            tag_indices=np.searchsorted(times, source_times[item.tag_indices]),
            natural_tracks=tuple(dict(track, node_indices=np.searchsorted(
                times, source_times[track['node_indices']])) for track in item.natural_tracks),
            base_tag_indices=base_indices))
    return tuple(result)


def add_graph(request, arms, graph, values, keys, camera_for):
    """Return mount keys and all used/skipped observations, preserving row IDs."""
    tag_keys, records = {}, []
    by_arm = {item.arm: item for item in arms}
    for arm_index, item in enumerate(arms):
        if item.T_imu_base_tag is None:
            continue
        key = gtsam.symbol('t', arm_index)
        tag_keys[item.arm] = key
        params = item.scene_parameters
        factors.add_base_tag_prior(graph, values, key, item.T_imu_base_tag,
            rotation_sigma_rad=params.poster_vio_base_tag_prior_rotation_sigma_rad,
            translation_sigma_m=params.poster_vio_base_tag_prior_translation_sigma_m)
    for obs in request.observations:
        observer = by_arm[obs.arm]
        camera = camera_for(observer, 'front') if len(obs.base_tag_times_s) else None
        for row, (target, time, pixels) in enumerate(zip(
                obs.base_tag_target_arms, obs.base_tag_times_s, obs.base_tag_corners_px)):
            record = dict(observer=obs.arm, owner=target, observation=row, imu_time_s=float(time),
                          status='used', kind='self' if obs.arm == target else 'cross')
            records.append(record)
            if target not in tag_keys:
                raise ValueError(f'{target} base tag 有观测但缺少经自身观测验证的安装初值')
            owner = by_arm[target]
            owner_time = observation_times(request, obs.arm, target, time)
            if not owner.times_s[0] <= owner_time <= owner.times_s[-1]:
                record.update(status='skipped', reason='outside_common_time_span')
                continue
            spec = request.base_tag_specs[target]
            record.update(pixels=pixels, size_m=spec['size_m'], tag_key=tag_keys[target],
                          camera=camera, owner_imu_time_s=float(owner_time))
            params = dict(size_m=spec['size_m'], scene=observer.scene_parameters)
            if target == obs.arm:
                factors.add_base_tag_self_factor(graph, tag_key=tag_keys[target],
                    T_cam_imu=camera.T_cam_imu, model=camera.model, pixels=pixels, parameters=params)
            else:
                observer_node = matching_node(observer.times_s, time)
                owner_node = matching_node(owner.times_s, owner_time)
                record.update(observer_key=keys[obs.arm].pose(observer_node),
                              owner_key=keys[target].pose(owner_node))
                factors.add_cross_base_tag_factor(graph,
                    observer_key=record['observer_key'], owner_key=record['owner_key'],
                    tag_key=tag_keys[target], T_cam_imu=camera.T_cam_imu,
                    model=camera.model, pixels=pixels, parameters=params)
    return tag_keys, records


def evaluate(arms, tag_keys, records, optimized):
    """Per-arm measurements; only physical gates determine acceptance."""
    evidence = {item.arm: dict(self_errors=[], cross_errors=[], observations=[], corrections=None)
                for item in arms}
    rejected = False
    for item in arms:
        if item.arm not in tag_keys:
            continue
        pose = optimized.atPose3(tag_keys[item.arm])
        delta = gtsam.Pose3(item.T_imu_base_tag).between(pose)
        angle = float(np.degrees(np.linalg.norm(gtsam.Rot3.Logmap(delta.rotation()))))
        shift = float(np.linalg.norm(delta.translation()))
        params = item.scene_parameters
        failed = (not np.isfinite(pose.matrix()).all()
                  or angle > params.poster_vio_base_tag_maximum_rotation_correction_deg
                  or shift > params.poster_vio_base_tag_maximum_translation_correction_m)
        rejected |= failed
        evidence[item.arm]['corrections'] = dict(
            T_imu_base_tag_initial=item.T_imu_base_tag.tolist(),
            T_imu_base_tag_optimized=pose.matrix().tolist(), rotation_deg=angle, translation_m=shift,
            maximum_rotation_deg=params.poster_vio_base_tag_maximum_rotation_correction_deg,
            maximum_translation_m=params.poster_vio_base_tag_maximum_translation_correction_m,
            prior_rotation_sigma_rad=params.poster_vio_base_tag_prior_rotation_sigma_rad,
            prior_translation_sigma_m=params.poster_vio_base_tag_prior_translation_sigma_m,
            prior_source='self_observation_seed', independent_mount_measurement=False,
            status='rejected' if failed else 'within_limits')
    params_by_arm = {item.arm: item.scene_parameters for item in arms}
    for record in records:
        public = {key: record[key] for key in ('observer', 'owner', 'observation', 'imu_time_s', 'kind', 'status')}
        result = evidence[record['observer']]
        if record['status'] != 'used':
            public['reason'] = record['reason']
            public['rms_px'] = None
            result['observations'].append(public)
            continue
        tag = optimized.atPose3(record['tag_key'])
        camera = record['camera']
        corners = factors.base_tag_corners_m(record['size_m'])
        camera_tag = camera.T_cam_imu @ tag.matrix()
        if record['kind'] == 'cross':
            camera_tag = (camera.T_cam_imu
                          @ np.linalg.inv(optimized.atPose3(record['observer_key']).matrix())
                          @ optimized.atPose3(record['owner_key']).matrix() @ tag.matrix())
        points = factors.transform(camera_tag, corners)
        pixels = camera.model.project(points)
        valid = (np.isfinite(points).all() and np.isfinite(pixels).all()
                 and np.all(points[:, 2] > 0))
        errors = np.linalg.norm(pixels - record['pixels'], axis=1) if valid else np.full(4, np.inf)
        rms = float(np.sqrt(np.mean(errors ** 2)))
        # Invalid geometry is a hard rejection, not a removable outlier.
        rejected |= not valid
        public.update(rms_px=rms if valid else None, geometry_valid=bool(valid),
                      owner_imu_time_s=record['owner_imu_time_s'])
        result['observations'].append(public)
        result[record['kind'] + '_errors'].extend(errors.tolist())
    for arm, result in evidence.items():
        for kind in ('self', 'cross'):
            errors = result.pop(kind + '_errors')
            metrics = refinement.residual_summary(errors)
            # Avoid NaN/Inf JSON even when the geometry gate rejected the run.
            if errors and not np.isfinite(errors).all():
                metrics = dict(count=len(errors), rms_px=None, geometry_valid=False)
            result[kind] = metrics
            if metrics['rms_px'] is not None:
                rejected |= metrics['rms_px'] > params_by_arm[arm].tag_maximum_reprojection_rms_px
    return evidence, bool(rejected)
