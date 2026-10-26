"""共用外参与逐帧位姿换算：非零杆臂、转动和四元数约定。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from software.camera.geometry import imu_from_camera, camera_from_tcp, compose_pose_series, tcp_frame_definition, transform
from software import config


class CameraGeometryTests(unittest.TestCase):
    def test_optical_parallel_tcp_keeps_fingertip_positions_and_changes_only_basis(self):
        root = Path(__file__).resolve().parents[1]
        new = json.loads((root / 'registry/grippers/handheld_v1/versions/b90726d8a0ed/t_tcp_camera.json').read_text())
        old_path = root / new['origin_geometry']['path']
        self.assertEqual(config.file_hash(old_path), new['origin_geometry']['sha256'])
        old = json.loads(old_path.read_text())
        np.testing.assert_array_equal(new['T_camera_tcp']['t_m'], old['T_camera_tcp']['t_m'])
        positions = np.array([[3., -2., 1.], [-1., 4., .2], [2., 3., 4.]])
        camera_rot = Rotation.from_euler('xyz', [[.1, .2, .3], [-.2, .4, 1.], [2., -1., .5]])
        p_old, q_old = compose_pose_series(positions, camera_rot.as_quat(), camera_from_tcp(old))
        p_new, q_new = compose_pose_series(positions, camera_rot.as_quat(), camera_from_tcp(new))
        np.testing.assert_array_equal(p_new, p_old)
        np.testing.assert_allclose(Rotation.from_quat(q_new).as_matrix(), camera_rot.as_matrix(), atol=1e-14)
        change = Rotation.from_quat(q_old).inv() * Rotation.from_quat(q_new)
        np.testing.assert_allclose(change.magnitude(), np.deg2rad(21.5), atol=1e-8)
        reverse = new['T_tcp_camera']
        np.testing.assert_allclose(camera_from_tcp(new) @ transform(reverse['R_rows'], reverse['t_m']), np.eye(4), atol=1e-14)
        self.assertEqual(tcp_frame_definition(new)['convention'], 'front_optical_parallel_v1')
        self.assertEqual(tcp_frame_definition(old)['convention'], 'asset_defined')
        self.assertIn('jaw opening', tcp_frame_definition(old)['description'])

    def test_robot_tcp_axes_keep_position_and_point_x_toward_image_left(self):
        root = Path(__file__).resolve().parents[1]
        new = json.loads((root / 'software/trajectory/assets/calib/t_tcp_camera_nominal_cad.json').read_text())
        legacy = json.loads((root / new['origin_geometry']['path']).read_text())
        extrinsic = camera_from_tcp(new)
        expected = Rotation.from_euler('x', 21.5, degrees=True).as_matrix() @ np.diag([-1., -1., 1.])
        np.testing.assert_allclose(extrinsic[:3, :3], expected, atol=1e-8)
        np.testing.assert_array_equal(extrinsic[:3, 3], legacy['T_camera_tcp']['t_m'])
        np.testing.assert_allclose(extrinsic[:3, 0], [-1., 0., 0.], atol=1e-14)
        np.testing.assert_allclose(extrinsic[:3, 2], np.asarray(legacy['T_camera_tcp']['R_rows'])[:, 2], atol=1e-8)
        np.testing.assert_allclose(np.cross(extrinsic[:3, 2], extrinsic[:3, 0]), extrinsic[:3, 1], atol=1e-14)
        reverse = new['T_tcp_camera']
        np.testing.assert_allclose(extrinsic @ transform(reverse['R_rows'], reverse['t_m']), np.eye(4), atol=1e-14)
        np.testing.assert_allclose(Rotation.from_quat(reverse['q_xyzw']).as_matrix(), extrinsic[:3, :3].T, atol=1e-14)
        self.assertEqual(new['tcp_frame']['convention'], 'genie_omnipicker_axes_v1')

    def test_pose_chain_matches_homogeneous_transforms_with_nonzero_levers(self):
        measured = dict(q_xyzw=Rotation.from_euler('xyz', [.2, -.4, .7]).as_quat(), p=[.03, -.02, .01])
        tcp = dict(T_camera_tcp=dict(R_rows=Rotation.from_euler('y', -.3).as_matrix(), t_m=[.02, 0., -.15]))
        extrinsic = imu_from_camera(measured) @ camera_from_tcp(tcp)
        positions = np.array([[3., -2., 1.], [4., 1., 2.]])
        quaternions = Rotation.from_euler('xyz', [[.1, .2, .3], [-.2, .4, 1.]]).as_quat()
        p, q = compose_pose_series(positions, -quaternions, extrinsic)
        body = np.tile(np.eye(4), (2, 1, 1))
        body[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
        body[:, :3, 3] = positions
        expected = body @ extrinsic
        np.testing.assert_allclose(p, expected[:, :3, 3], atol=1e-14)
        np.testing.assert_allclose(Rotation.from_quat(q).as_matrix(), expected[:, :3, :3], atol=1e-14)
        self.assertTrue(np.all(q[:, 3] >= 0))
        self.assertGreater(p[0, 0], 2.)  # 换算不会把场景原点移到本段起点。

    def test_independent_export_records_the_selected_tcp_convention(self):
        from software.trajectory import calibration_assets
        from software.pose_io import write_tum
        from software.trajectory.algorithms import postprocess

        root = Path(__file__).resolve().parents[1]
        source = root / 'software/trajectory/assets/calib/t_tcp_camera_nominal_cad.json'
        new = json.loads(source.read_text())
        sources = (
            (root / new['origin_geometry']['path'], 'asset_defined'),
            (root / 'registry/grippers/handheld_v1/versions/b90726d8a0ed/t_tcp_camera.json', 'front_optical_parallel_v1'),
            (source, 'genie_omnipicker_axes_v1'),
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fuse = directory / 'reconstruction/fuse'
            fuse.mkdir(parents=True)
            (fuse / 'fuse_info.json').write_text('{}')
            times = np.arange(4) / 29.97 + 1.
            positions = np.array([[2., 1., .5], [2.1, 1., .5], [2.2, 1.1, .5], [2.3, 1., .5]])
            rotations = Rotation.from_euler('xyz', [[0., .1, 0.], [.1, .2, .3], [-.2, .4, 1.], [.3, -.2, 1.5]])
            write_tum(fuse / 'traj_tum.txt', times, positions, rotations.as_quat(), 'test')
            outputs = []
            for i, (asset, expected) in enumerate(sources):
                output = directory / f'output_{i}'
                output.mkdir()
                with patch.object(postprocess, 'plot'):
                    summary = postprocess.export_trajectories(fuse.parent, {}, output, 'handheld', 'VID.insv',
                        calib=dict(json=str(calibration_assets.SAMPLE_CAMERA_JSON), tcp_json=str(asset)))
                self.assertEqual(summary['tcp_frame']['convention'], expected)
                settings = json.loads((output / 'run_settings.json').read_text())
                self.assertEqual(settings['assets']['camera_tcp_geometry']['tcp_frame'], summary['tcp_frame'])
                self.assertIn(expected, (output / 'traj_tcp.txt').read_text())
                outputs.append(np.loadtxt(output / 'traj_tcp.txt'))
            np.testing.assert_array_equal(outputs[0][:, :4], outputs[1][:, :4])
            np.testing.assert_array_equal(outputs[0][:, :4], outputs[2][:, :4])
            camera = np.loadtxt(output / 'traj_camera.txt')
            np.testing.assert_allclose(Rotation.from_quat(outputs[2][:, 4:]).as_matrix(),
                                       Rotation.from_quat(camera[:, 4:]).as_matrix() @ camera_from_tcp(new)[:3, :3], atol=1e-8)

    def test_shared_geometry_is_fingerprinted_for_both_workflows(self):
        for domain in ('scene', 'trajectory'):
            self.assertIn('software/camera/geometry.py', config.code_identity(domain))
