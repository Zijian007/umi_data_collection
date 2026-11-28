"""Rotation residuals for factors using GTSAM's right-local coordinates."""

import gtsam
import numpy as np

from umi.core.geometry import skew


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    return gtsam.Rot3.Logmap(gtsam.Rot3(rotation))


def right_jacobian_inverse(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec)
    cross = skew(rotvec)
    linear = np.eye(3) + 0.5 * cross
    if angle < 1e-6:
        return linear
    coefficient = 1.0 / angle ** 2 - (1 + np.cos(angle)) / (2 * angle * np.sin(angle))
    return linear + coefficient * cross @ cross
