"""Uncertainty of a fixed-map camera pose from matched KB4 pixels."""

import numpy as np


def camera_covariance(field):
    """Decode the ORB driver's symmetric information matrix in local camera coordinates."""
    if not field:
        return None
    packed = np.asarray(field.split(), dtype=float)
    if packed.shape != (21,) or not np.isfinite(packed).all():
        raise ValueError('ORB 相机信息矩阵需要 21 个有限元素')
    info = np.empty((6, 6))
    i, j = np.triu_indices(6)
    info[i, j] = info[j, i] = packed
    eig = np.linalg.eigvalsh(info)
    if eig[0] <= 0 or eig[0] <= 1e-10 * eig[-1]:
        return None
    return np.linalg.inv(info)


def map_factor_covariance(covariance, map_from_camera, camera_from_imu, camera_to_imu_in_camera, scale):
    """Transport a right-local optical pose perturbation to body angle and map displacement."""
    x, y, z = camera_to_imu_in_camera
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    transform = np.zeros((6, 6))
    transform[:3, :3] = camera_from_imu.T
    transform[3:, :3] = -map_from_camera @ cross / scale
    transform[3:, 3:] = map_from_camera
    result = transform @ covariance @ transform.T
    return (result + result.T) * .5
