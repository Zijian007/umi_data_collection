import unittest
from types import SimpleNamespace

import gtsam
import numpy as np

from software.scene.management.inputs.kit import gripper_paths
from software.vision.poster import factors


class _Pinhole:
    fx = fy = 500.0
    cx = cy = 320.0

    def project(self, points):
        points = np.asarray(points, dtype=float)
        return np.column_stack((self.fx * points[:, 0] / points[:, 2] + self.cx,
                                self.fy * points[:, 1] / points[:, 2] + self.cy))

    def project_jacobian(self, points):
        points = np.asarray(points, dtype=float)
        out = np.zeros((len(points), 2, 3))
        out[:, 0, 0] = self.fx / points[:, 2]
        out[:, 0, 2] = -self.fx * points[:, 0] / points[:, 2] ** 2
        out[:, 1, 1] = self.fy / points[:, 2]
        out[:, 1, 2] = -self.fy * points[:, 1] / points[:, 2] ** 2
        return out


class CrossCameraBaseTagTests(unittest.TestCase):
    def test_kit_accepts_optional_base_tag_metadata(self):
        result = gripper_paths({
            'tcp': '/tmp/tcp.json',
            'base_tag': {'id': 20, 'size_mm': 10, 'dictionary': 'DICT_4X4_100'},
        })
        self.assertEqual(result['base_tag']['id'], 20)
        self.assertEqual(result['base_tag']['size_mm'], 10.0)

    def test_cross_factor_is_zero_for_consistent_two_arm_poses(self):
        model = _Pinhole()
        scene = SimpleNamespace(tag_huber=2.0, tag_corner_sigma_px=1.0,
                                tag_sigma_per_radius_px=0.0)
        observer = gtsam.symbol('x', 0)
        owner = gtsam.symbol('x', 1)
        observer_pose = gtsam.Pose3()
        owner_pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 2.0))
        points = factors.base_tag_corners_m(0.1)
        pixels = model.project((owner_pose.matrix()[:3, :3] @ points.T).T
                               + owner_pose.matrix()[:3, 3])
        graph = gtsam.NonlinearFactorGraph()
        factors.add_cross_base_tag_factor(
            graph, observer_key=observer, owner_key=owner, T_cam_imu=np.eye(4),
            model=model, T_imu_tag=np.eye(4), pixels=pixels,
            parameters={'size_m': 0.1, 'scene': scene})
        values = gtsam.Values()
        values.insert(observer, observer_pose)
        values.insert(owner, owner_pose)
        self.assertLess(float(graph.error(values)), 1e-12)


if __name__ == '__main__':
    unittest.main()
