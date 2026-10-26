"""Fixed-observation numerical regression against the pre-configuration solver.

Reference: kiwi_fuse3s.py SHA256
3dfb5efd392488775936a2d7c245acbad0136f680003b486ad0deac001e0a615.
This bypasses random SLAM: camera observations and IMU are deterministic, GTSAM
runs for real. Selected TUM rows were captured before moving parameter defaults.
"""
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from software import config
from software.trajectory import calibration_assets as assets

GOLDEN = [
    [1, -.000191196, .139494246, .000208187, .000262825, .085124379, -.000089214, .996370294],
    [3.002, .193981223, .023480313, .061265942, .059385556, -.019064951, .064570015, .995962146],
    [5.004, -.088955147, -.131613949, -.105519455, .020255657, -.075656222, -.076064279, .994022168],
    [7.006, -.153162754, -.067974429, .119539077, -.052012749, .054499305, .025283233, .996837629],
    [9.008, .159283156, .108710121, -.099712924, -.037322477, .051651270, .046292405, .996893270],
]


def motion(t):
    t = np.asarray(t)
    rotation = Rot.from_rotvec(np.stack([.12*np.sin(.7*t), .17*np.cos(.9*t), .16*np.sin(1.1*t)], axis=-1)).as_matrix()
    position = np.stack([.2*np.sin(.9*t), .14*np.cos(.7*t), .12*np.sin(1.3*t)], axis=-1)
    acceleration = np.stack([-.2*.9**2*np.sin(.9*t), -.14*.7**2*np.cos(.7*t), -.12*1.3**2*np.sin(1.3*t)], axis=-1)
    return rotation, position, acceleration


@unittest.skipUnless(importlib.util.find_spec('gtsam'), 'GTSAM environment required')
class FusionRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.camera = assets.read_basalt(assets.SAMPLE_CAMERA_JSON)
        times = np.arange(241)/(30000/1001)
        rotation, position, _ = motion(times)
        camera_position = position + np.einsum('nij,j->ni', rotation, self.camera['p'])
        camera_quaternion = Rot.from_matrix(rotation @ Rot.from_quat(self.camera['q_xyzw']).as_matrix()).as_quat()
        self.frames, self.imu = self.root/'frames.csv', self.root/'imu.csv'
        with self.frames.open('w') as out:
            writer = csv.writer(out)
            writer.writerow('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw,map'.split(','))
            for k in range(len(times)):
                writer.writerow([k, times[k], 2, 1, *(camera_position[k]/.7), *camera_quaternion[k], 'm0'])
        ti = np.arange(-.1, times[-1]+.1, 1/300)
        rotation, _, acceleration = motion(ti)
        gyro = Rot.from_matrix(np.swapaxes(rotation,1,2) @ motion(ti+1e-5)[0]).as_rotvec()/1e-5
        specific = np.einsum('nji,nj->ni', rotation, acceleration-np.array([0,0,-9.81]))
        with self.imu.open('w') as out:
            writer = csv.writer(out)
            writer.writerow(['#timestamp', 'wx', 'wy', 'wz', 'ax', 'ay', 'az'])
            for t, g, a in zip(ti, gyro, specific):
                writer.writerow([round((1+t)*1e9)+self.camera['cam_time_offset_ns'], *g, *a])

    def solve(self, parameters, camera=assets.SAMPLE_CAMERA_JSON):
        path = self.root/'parameters.json'
        path.write_text(json.dumps(asdict(parameters)))
        output = self.root/'out'
        command = [sys.executable, str(assets.FUSION_SCRIPT), str(self.frames), str(self.frames), str(self.imu),
                   str(self.camera['cam_time_offset_ns']), '.500794,.498606,-.501155,.499441', str(output),
                   '--lever_prior', '.0147,-.0093,-.0163', '--calibration', str(camera), '--parameters', str(path)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr+result.stdout[-1500:])
        return np.loadtxt(output/'traj_tum.txt'), json.loads((output/'fuse_info.json').read_text())

    def test_defaults_match_original_fixed_observation_result(self):
        trajectory, info = self.solve(config.load('trajectory').values)
        self.assertEqual(trajectory.shape, (241,8))
        np.testing.assert_allclose(trajectory[[0,60,120,180,240]], GOLDEN, rtol=1e-7, atol=2e-7)
        self.assertEqual(info['chains']['front'], [[[0,240], 'full']])
        self.assertEqual(info['stage1']['front']['chains'][0]['k'], [0, 240])
        self.assertTrue(info['stage1']['front']['chains'][0]['good'])
        self.assertEqual(info['nodes'], 49)

    def test_changed_config_and_explicit_camera_reach_the_real_solver(self):
        parameters = config.load('trajectory').values
        parameters = replace(parameters, fusion=replace(parameters.fusion, node_stride=3, gyro_bias_rw=.00003),
                             zupt=replace(parameters.zupt, enabled=False))
        camera = json.loads(assets.SAMPLE_CAMERA_JSON.read_text())
        camera['value0']['T_imu_cam'][0]['px'] += .001
        path = self.root/'camera.json'
        path.write_text(json.dumps(camera))
        _, info = self.solve(parameters, path)
        self.assertEqual(info['nodes'], 81)
        self.assertEqual(info['zupt_frames'], 0)
        self.assertEqual(info['effective_parameters']['fusion']['gyro_bias_rw'], .00003)
        values = assets.read_basalt(path)
        lever = -Rot.from_quat(values['q_xyzw']).as_matrix().T @ values['p']
        np.testing.assert_allclose(info['front_lever_prior'], lever.round(4))

    def test_library_calls_are_isolated_and_do_not_write_outputs(self):
        from software.trajectory.algorithms.fusion import solve
        from software.trajectory.algorithms.fusion_data import load_inputs
        inputs = load_inputs(front_frames=self.frames, rear_frames=self.frames, imu_csv=self.imu,
            camera_json=assets.SAMPLE_CAMERA_JSON, offset_ns=self.camera['cam_time_offset_ns'],
            rear_quaternion=[.500794,.498606,-.501155,.499441], rear_lever=[.0147,-.0093,-.0163], use_rear=False)
        parameters = config.load('trajectory').values
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        quiet = lambda *args, **kwargs: None
        first = solve(inputs, parameters, log=quiet)
        changed = replace(parameters, fusion=replace(parameters.fusion, node_stride=3))
        self.assertEqual(solve(inputs, changed, log=quiet).evidence['nodes'], 81)
        again = solve(inputs, parameters, log=quiet)
        np.testing.assert_array_equal(first.T_world_imu, again.T_world_imu)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_dual_mode_uses_both_physically_consistent_lens_observations(self):
        from software.trajectory.algorithms.fusion import solve
        from software.trajectory.algorithms.fusion_data import load_inputs

        rear_quaternion = [.500794, .498606, -.501155, .499441]
        rear_rotation = Rot.from_quat(rear_quaternion).as_matrix()
        rear_lever = np.array([.0147, -.0093, -.0163])
        times = np.arange(241)/(30000/1001)
        rotation, position, _ = motion(times)
        rear_position = position + np.einsum('nij,j->ni', rotation, -rear_rotation @ rear_lever)
        rear_orientation = Rot.from_matrix(rotation @ rear_rotation).as_quat()
        rear = self.root/'rear.csv'
        with rear.open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw,map'.split(','))
            for k, time in enumerate(times):
                writer.writerow([k, time, 2, 1, *(rear_position[k]/.7), *rear_orientation[k], 'm0'])
        inputs = load_inputs(front_frames=self.frames, rear_frames=rear, imu_csv=self.imu,
            camera_json=assets.SAMPLE_CAMERA_JSON, offset_ns=self.camera['cam_time_offset_ns'],
            rear_quaternion=rear_quaternion, rear_lever=rear_lever, use_rear=True)
        parameters = config.load('trajectory').values
        parameters = replace(parameters, input=replace(parameters.input, mode='dual'))
        solved = solve(inputs, parameters, log=lambda *args, **kwargs: None)
        self.assertEqual(solved.evidence['obs'], dict(rear=49, front=49))
        self.assertIsNotNone(solved.evidence['rear_lever_final'])
        self.assertTrue(np.isfinite(solved.T_world_imu).all())

    def test_negative_csv_vectors_and_legacy_entry_remain_accepted(self):
        path = self.root/'parameters.json'
        path.write_text(json.dumps(asdict(config.load('trajectory').values)))
        output = self.root/'signed'
        old_entry = assets.HERE/'algorithms/kiwi_fuse3s.py'
        command = [sys.executable, str(old_entry), str(self.frames), str(self.frames), str(self.imu),
                   str(self.camera['cam_time_offset_ns']), '-.500794,-.498606,.501155,-.499441', str(output),
                   '--lever_prior', '-.0147,.0093,.0163', '--calibration', str(assets.SAMPLE_CAMERA_JSON), '--parameters', str(path)]
        process = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue((output/'traj_tum.txt').is_file())
