"""Task-local monocular maps with a free metric placement for each map and arm."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import gtsam
import numpy as np
from scipy.spatial.transform import Rotation

from umi.core.geometry import imu_from_camera
from umi.core.timing import video_to_imu_seconds
from umi.perception import slam
from umi.reconstruction.estimation import mapping
from umi.reconstruction.estimation.fusion.graph import jr_inv, logR, skew
from umi.reconstruction.estimation.map_information import camera_covariance


@dataclass(frozen=True)
class FrontMap:
    label: str
    times_s: np.ndarray
    positions: np.ndarray
    body_rotations: np.ndarray
    covariances: np.ndarray
    initial_scale: float
    initial_pose: np.ndarray

    def nodes(self, times):
        indices = np.searchsorted(times, self.times_s)
        if np.any(indices >= len(times)) or not np.array_equal(times[indices], self.times_s):
            raise ValueError('工作镜头地图时刻未对齐到 IMU 图节点')
        return indices


def prepare(recording, item, directory, stride):
    atlas = mapping.build(recording, 'front', directory)
    (localization,) = slam.relocalize_all_maps((atlas,), recording.context, directory,
                                               front_information_stride=stride)
    camera = recording.camera.measurements['front']
    mount = imu_from_camera(camera)
    camera_from_imu = np.linalg.inv(mount)
    times = np.round(video_to_imu_seconds(recording.video_times_s, camera['cam_time_offset_ns']), 9)
    groups = {}
    with localization.frames_csv.open() as stream:
        for row in csv.DictReader(stream):
            if row['ok'] != '1' or row['state'] != '2':
                continue
            covariance = camera_covariance(row['camera_information'])
            if covariance is None:
                continue
            frame = int(row['frame_idx'])
            rotation = Rotation.from_quat([float(row[k]) for k in ('qx', 'qy', 'qz', 'qw')]).as_matrix()
            position = [float(row[k]) for k in ('tx', 'ty', 'tz')]
            groups.setdefault(row['map'], []).append((times[frame], position, rotation, covariance))

    maps = []
    for label, rows in groups.items():
        if len(rows) < 3:
            continue
        t, positions, rotations, covariances = map(np.asarray, zip(*rows))
        nodes = np.searchsorted(item.times_s, t)
        if np.any(nodes >= len(item.times_s)) or not np.array_equal(item.times_s[nodes], t):
            raise ValueError('工作镜头地图定位与输出帧时间不一致')
        seed = item.seed_poses[nodes]
        body_rotations = rotations @ camera_from_imu[:3, :3]
        rotation = Rotation.from_matrix(seed[:, :3, :3] @ body_rotations.transpose(0, 2, 1)).mean().as_matrix()
        target = (seed @ mount)[:, :3, 3]
        centered = (positions - positions.mean(axis=0)) @ rotation.T
        spread = np.sum(centered**2)
        if spread <= 1e-12:
            continue
        scale = float(np.sum(centered * (target - target.mean(axis=0))) / spread)
        if not np.isfinite(scale) or scale <= 0:
            continue
        placement = np.eye(4)
        placement[:3, :3] = rotation
        placement[:3, 3] = np.median(target - scale * positions @ rotation.T, axis=0)
        # This residual uses the optical center directly. Only the rotation residual
        # changes from camera axes to IMU axes; position stays in unscaled map units.
        basis = np.zeros((len(rows), 6, 6))
        basis[:, :3, :3] = mount[:3, :3]
        basis[:, 3:, 3:] = rotations
        transported = basis @ covariances @ basis.transpose(0, 2, 1)
        maps.append(FrontMap(label, t, positions, body_rotations, transported, scale, placement))
    return tuple(maps)


def map_factor(noise, body_key, map_key, scale_key, measured_rotation, measured_position, lever):
    """Rotation in body coordinates and optical-center displacement in unscaled map units."""
    def residual(this, values, jacobians):
        body = values.atPose3(body_key)
        placement = values.atPose3(map_key)
        rotation = placement.rotation().matrix()
        body_rotation = body.rotation().matrix()
        scale = np.exp(values.atVector(scale_key)[0])
        relative = (rotation @ measured_rotation).T @ body_rotation
        angle = logR(relative)
        center = body.translation() + body_rotation @ lever
        position = rotation.T @ (center - placement.translation()) / scale
        if jacobians is not None:
            angular = jr_inv(angle)
            linear = rotation.T @ body_rotation / scale
            body_jac = np.zeros((6, 6))
            body_jac[:3, :3] = angular
            body_jac[3:, :3] = -linear @ skew(lever)
            body_jac[3:, 3:] = linear
            map_jac = np.zeros((6, 6))
            map_jac[:3, :3] = -angular @ relative.T @ measured_rotation.T
            map_jac[3:, :3] = skew(position)
            map_jac[3:, 3:] = -np.eye(3) / scale
            jacobians[0], jacobians[1] = body_jac, map_jac
            jacobians[2] = np.r_[np.zeros(3), -position][:, None]
        return np.r_[angle, position - measured_position]

    return gtsam.CustomFactor(noise, [body_key, map_key, scale_key], residual)


def placement_keys(arm_index, map_index):
    # The joint solver supports up to two arms.
    index = 2 * map_index + arm_index
    return gtsam.symbol('m', index), gtsam.symbol('s', index)


def add_factors(graph, values, item, arm_index, keys):
    lever = np.linalg.inv(item.natural_cameras['front'].T_cam_imu)[:3, 3]
    huber = gtsam.noiseModel.mEstimator.Huber.Create(item.fusion_parameters.huber)
    for i, front_map in enumerate(item.front_maps):
        map_key, scale_key = placement_keys(arm_index, i)
        values.insert(map_key, gtsam.Pose3(front_map.initial_pose))
        values.insert(scale_key, np.array([np.log(front_map.initial_scale)]))
        for n, node in enumerate(front_map.nodes(item.times_s)):
            gaussian = gtsam.noiseModel.Gaussian.Covariance(front_map.covariances[n])
            graph.add(map_factor(gtsam.noiseModel.Robust.Create(huber, gaussian), keys.pose(int(node)),
                                 map_key, scale_key, front_map.body_rotations[n], front_map.positions[n], lever))
