"""Bounded-time keyframes and conditional IMU recovery at exact output times.

specialist's keyframe/high-rate separation is the reference (see NOTICE.md).
Here missing states are recovered by local inertial optimization with both
endpoint states fixed, rather than distributing endpoint error by interpolation.
"""
from __future__ import annotations

import gtsam
import numpy as np
from scipy.spatial.transform import Rotation

from software.scene.spatial import fusion
from software.trajectory.algorithms.inertial import preintegrate_samples


def select_indices(times, poses, parameters):
    times, poses = np.asarray(times), np.asarray(poses)
    if not len(times):
        return np.empty(0, dtype=int)
    maximum_dt = parameters.poster_vio_keyframe_maximum_dt_s
    if not maximum_dt:
        return np.arange(len(times))
    selected = [0]
    for index in range(1, len(times) - 1):
        previous = selected[-1]
        angle = Rotation.from_matrix(poses[previous, :3, :3].T @ poses[index, :3, :3]).magnitude()
        # Take the preceding sample before an interval would exceed its limit.
        if times[index + 1] - times[previous] > maximum_dt + 1e-9 or (
                np.linalg.norm(poses[index, :3, 3] - poses[previous, :3, 3])
                >= parameters.poster_vio_keyframe_minimum_translation_m
                or np.degrees(angle) >= parameters.poster_vio_keyframe_minimum_rotation_deg):
            selected.append(index)
    if len(times) > 1:
        selected.append(len(times) - 1)
    return np.unique(selected)


def graph_indices(item):
    # Freeze this set BEFORE culling so an X/V/B key keeps its physical time
    # through every warm-started solve, even when observations are removed.
    nodes = [select_indices(item.times_s, item.seed_poses, item.scene_parameters),
             item.tag_indices, item.base_tag_indices]
    nodes += [np.asarray(track['node_indices']) for track in item.natural_tracks]
    return np.unique(np.concatenate(nodes)).astype(int)


def graph_input(item, indices):
    from dataclasses import replace

    holdout_nodes = (None if item.holdout_prior_nodes is None else
                     np.searchsorted(indices, item.holdout_prior_nodes))
    barrier_indices = (None if item.depth_barrier_indices is None else
                       np.searchsorted(indices, item.depth_barrier_indices))
    return replace(item, times_s=item.times_s[indices], seed_poses=item.seed_poses[indices],
        seed_velocities=None if item.seed_velocities is None else item.seed_velocities[indices],
        tag_indices=np.searchsorted(indices, item.tag_indices),
        base_tag_indices=np.searchsorted(indices, item.base_tag_indices),
        holdout_prior_nodes=holdout_nodes,
        depth_barrier_indices=barrier_indices,
        natural_tracks=tuple(dict(track, node_indices=np.searchsorted(indices, track['node_indices']))
                             for track in item.natural_tracks))


def recover_states(item, indices, values, keys):
    """Return all X/V/B states; output and observed timestamps are never snapped."""
    count = len(item.times_s)
    poses = np.empty((count, 4, 4))
    velocities = np.empty((count, 3))
    biases = [None] * count
    for index, node in enumerate(indices):
        poses[node] = values.atPose3(keys.pose(index)).matrix()
        velocities[node] = values.atVector(keys.velocity(index))
        biases[node] = values.atConstantBias(keys.bias(index))
    recovered, maximum_error, zero_residual_intervals = 0, 0., 0
    parameters = item.trajectory_parameters.fusion
    gravity = np.array([0., 0., -parameters.gravity_m_s2]) if item.gravity_m_s2 is None else item.gravity_m_s2
    for start, end in zip(indices[:-1], indices[1:]):
        if end == start + 1:
            continue
        times = item.times_s[start:end + 1]
        initial_poses, initial_velocities = [poses[start].copy()], [velocities[start].copy()]
        bias = biases[start]
        for before, time in zip(times[:-1], times[1:]):
            rotation, velocity, position, _, _, dt = preintegrate_samples(
                item.imu[:, 0], item.imu[:, 1:4], item.imu[:, 4:7], before, time,
                bias.gyroscope(), bias.accelerometer())
            previous_pose, previous_velocity = initial_poses[-1], initial_velocities[-1]
            pose = previous_pose.copy()
            pose[:3, :3] = previous_pose[:3, :3] @ rotation
            pose[:3, 3] += previous_velocity * dt + .5 * gravity * dt ** 2 + previous_pose[:3, :3] @ position
            initial_poses.append(pose)
            initial_velocities.append(previous_velocity + gravity * dt + previous_pose[:3, :3] @ velocity)
        initial_poses, initial_velocities = np.asarray(initial_poses), np.asarray(initial_velocities)
        duration = float(times[-1] - times[0])
        position_error = np.linalg.norm(initial_poses[-1, :3, 3] - poses[end, :3, 3])
        rotation_error = Rotation.from_matrix(initial_poses[-1, :3, :3].T @ poses[end, :3, :3]).magnitude()
        velocity_error = np.linalg.norm(initial_velocities[-1] - velocities[end])
        bias_error = np.linalg.norm(bias.vector() - biases[end].vector())
        normalized_error = max(position_error / max(parameters.acc_noise * duration ** 1.5, 1e-30),
            rotation_error / max(parameters.gyro_noise * np.sqrt(duration), 1e-30),
            velocity_error / max(parameters.acc_noise * np.sqrt(duration), 1e-30),
            bias_error / max(min(parameters.acc_bias_rw, parameters.gyro_bias_rw) * np.sqrt(duration), 1e-30))
        if normalized_error <= 1e-8:
            # This propagation already satisfies both boundary states and the
            # inertial model to numerical precision: no correction to distribute.
            poses[start + 1:end] = initial_poses[1:-1]
            velocities[start + 1:end] = initial_velocities[1:-1]
            biases[start + 1:end] = [bias] * (end - start - 1)
            recovered += end - start - 1
            zero_residual_intervals += 1
            continue
        local_keys = fusion.InertialGraphKeys(0)
        graph, local_values = fusion.build_inertial_graph(times, item.imu,
            Rotation.from_matrix(initial_poses[:, :3, :3]), initial_poses[:, :3, 3], initial_velocities,
            parameters, pose_key_fn=local_keys.pose, velocity_key_fn=local_keys.velocity, bias_key_fn=local_keys.bias,
            gravity_m_s2=gravity, initial_acc_bias=bias.accelerometer(), initial_gyro_bias=bias.gyroscope())
        # Initial V/B priors belong to the global graph. The local conditional
        # problem uses only IMU/random-walk factors and fixed endpoint states.
        conditional = gtsam.NonlinearFactorGraph()
        for index in range(graph.size()):
            factor = graph.at(index)
            if len(factor.keys()) > 1:
                conditional.add(factor)
        for local, global_node in ((0, start), (len(times) - 1, end)):
            pose, velocity, bias = gtsam.Pose3(poses[global_node]), velocities[global_node], biases[global_node]
            local_values.update(local_keys.pose(local), pose)
            local_values.update(local_keys.velocity(local), velocity)
            local_values.update(local_keys.bias(local), bias)
            conditional.add(gtsam.NonlinearEqualityPose3(local_keys.pose(local), pose))
            conditional.add(gtsam.NonlinearEqualityPoint3(local_keys.velocity(local), velocity))
            conditional.add(gtsam.NonlinearEqualityConstantBias(local_keys.bias(local), bias))
        settings = gtsam.LevenbergMarquardtParams()
        settings.setMaxIterations(parameters.lm_iterations)
        initial_error = float(conditional.error(local_values))
        optimized = gtsam.LevenbergMarquardtOptimizer(conditional, local_values, settings).optimize()
        final_error = float(conditional.error(optimized))
        if not np.isfinite(final_error) or final_error > initial_error + 1e-5:
            raise ValueError('poster_vio 关键帧间条件惯性恢复失败')
        maximum_error = max(maximum_error, final_error)
        for local in range(1, len(times) - 1):
            node = start + local
            poses[node] = optimized.atPose3(local_keys.pose(local)).matrix()
            velocities[node] = optimized.atVector(local_keys.velocity(local))
            biases[node] = optimized.atConstantBias(local_keys.bias(local))
            recovered += 1
    if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
        raise ValueError('poster_vio 关键帧恢复产生非有限状态')
    return poses, velocities, biases, dict(global_nodes=len(indices), output_nodes=count,
        recovered_nodes=recovered, recovery=('conditional_inertial_optimization_fixed_endpoints' if recovered
                                             else 'direct_optimized_states'),
        zero_residual_intervals=zero_residual_intervals,
        maximum_interval_final_error=maximum_error, independent_physical_accuracy=False)
