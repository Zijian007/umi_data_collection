"""固定视觉观测的共享尺度回归；不依赖随机 ORB 或外部录像。"""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from software import config
from software.scene.management.inputs import load_scene_parameters
from software.scene.spatial import geometry, mapping


def recording():
    t = np.arange(0, 28.01, .002)
    rotation = Rotation.from_euler('xyz', np.column_stack([
        .45*np.sin(1.1*t), .35*np.cos(1.3*t), .4*np.sin(.8*t)]))
    gyro = np.vstack([(rotation[:-1].inv()*rotation[1:]).as_rotvec()/.002, [0, 0, 0]])
    position = np.column_stack([.4*np.sin(1.2*t), .3*np.sin(1.7*t+.3), .2*np.cos(.9*t)])
    acceleration = np.column_stack([-.4*1.2**2*np.sin(1.2*t), -.3*1.7**2*np.sin(1.7*t+.3), -.2*.9**2*np.cos(.9*t)])
    acc = rotation.inv().apply(acceleration - [0, 0, -9.81])
    imu = np.column_stack([t, gyro, acc])
    selected = np.arange(50, len(t)-50, 17)
    poses = np.tile(np.eye(4), (len(selected), 1, 1))
    poses[:, :3, :3] = rotation[selected].as_matrix()
    poses[:, :3, 3] = position[selected]/2.3
    return geometry.VisualObservations(t[selected], poses, np.arange(len(selected)), 'visual_map'), imu


class SharedAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.parameters = load_scene_parameters(None)
        self.trajectory = config.load('trajectory').values
        self.obs, self.imu = recording()

    def solve(self, obs=None, imu=None):
        return geometry.metric_alignment(self.obs if obs is None else obs, np.eye(4),
            self.imu if imu is None else imu, np.zeros(3), 9.81, self.parameters,
            trajectory_parameters=self.trajectory)

    def test_bad_rotation_segments_do_not_corrupt_shared_scale(self):
        poses = self.obs.T_reference_camera.copy()
        bad = self.obs.imu_times_s < 10
        poses[bad, :3, :3] = Rotation.random(int(bad.sum()), random_state=7).as_matrix()
        obs = replace(self.obs, T_reference_camera=poses)
        result = self.solve(obs)
        self.assertAlmostEqual(result['scale'], 2.3, delta=.01)
        evidence = result['evidence']
        self.assertLess(evidence['relative_scale_std'], self.parameters.maximum_scale_relative_std)
        self.assertGreater(evidence['alignment']['chain_splits']['front']['rot'], 100)
        reliable = [item for item in evidence['alignment']['segments']['chains'] if item['good']]
        self.assertTrue(reliable)
        self.assertTrue(all(self.obs.imu_times_s[item['k'][0]] > 9 for item in reliable))

    def test_same_map_with_conflicting_reliable_scales_is_rejected(self):
        keep = (self.obs.imu_times_s < 12) | (self.obs.imu_times_s > 14)
        poses = self.obs.T_reference_camera[keep].copy()
        times = self.obs.imu_times_s[keep]
        poses[times > 14, :3, 3] *= .5
        obs = geometry.VisualObservations(times, poses, self.obs.indices[keep], 'visual_map')
        with self.assertRaises(geometry.MapAlignmentError) as caught:
            self.solve(obs)
        diagnostics = caught.exception.diagnostics
        scales = [item['scale'] for item in diagnostics['segments']['chains'] if item['good']]
        self.assertEqual(len(scales), 2)
        self.assertAlmostEqual(min(scales), 2.3, delta=.01)
        self.assertAlmostEqual(max(scales), 4.6, delta=.02)
        self.assertIn('共享', str(caught.exception))

    def test_reliable_tail_cannot_publish_a_map_with_an_inconsistent_earlier_segment(self):
        keep = (self.obs.imu_times_s < 8) | (self.obs.imu_times_s > 10)
        times = self.obs.imu_times_s[keep]
        poses = self.obs.T_reference_camera[keep].copy()
        early = times < 8
        poses[early, 0, 3] += .5*np.sin(5*times[early])
        obs = geometry.VisualObservations(times, poses, self.obs.indices[keep], 'visual_map')
        with self.assertRaisesRegex(geometry.MapAlignmentError, '地图整体与固定尺度不一致') as caught:
            self.solve(obs)
        evidence = caught.exception.diagnostics
        self.assertTrue(any(item['good'] for item in evidence['segments']['chains']))
        self.assertTrue(any(item['status'] == 'inconsistent' and item['rms_m'] > self.parameters.maximum_alignment_rms_m
                            for item in evidence['fixed_map_validation']))

    def test_fixed_map_validation_reports_short_chain_as_insufficient(self):
        lens = dict(obs={0: (np.zeros(3), np.eye(3)), 1: (np.ones(3), np.eye(3))},
                    chains=[(0, 1)], R_CI=np.eye(3), lever=np.zeros(3))
        imu = np.array([[0., 0., 0., 0., 0., 0., 9.81],
                        [1., 0., 0., 0., 0., 0., 9.81]])
        result = geometry.fusion_alignment.fixed_map_residuals(
            lens, np.array([0., 1.]), imu, np.zeros(3), 1., np.array([0., 0., -9.81]),
            np.zeros(3), 1, maximum_rms_m=.05)
        self.assertEqual(result, [dict(index=0, k=[0, 1], pairs=0, rms_m=None,
                                       status='insufficient_evidence')])

    def test_separated_consistent_segments_share_one_scale_and_keep_clock(self):
        keep = (self.obs.imu_times_s < 12) | (self.obs.imu_times_s > 14)
        obs = geometry.VisualObservations(self.obs.imu_times_s[keep]+17.25,
            self.obs.T_reference_camera[keep], self.obs.indices[keep], 'visual_map')
        imu = self.imu.copy()
        imu[:, 0] += 17.25
        result = self.solve(obs, imu)
        self.assertAlmostEqual(result['scale'], 2.3, delta=.01)
        shared = result['evidence']['alignment']['shared']['chains']
        self.assertEqual(len(shared), 2)
        self.assertTrue(all(item['good'] for item in shared))
        self.assertEqual(shared[0]['scale'], shared[1]['scale'])
        np.testing.assert_allclose(result['T_scene_metric_map'], np.eye(4), atol=.005)

    def test_failed_candidate_records_segment_evidence_without_modifying_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'map.osa'
            archive.write_bytes(b'unchanged visual atlas')
            diagnostics = dict(segments=dict(chains=[dict(scale=-2., good=False)]))
            camera = SimpleNamespace(measurements={'front': dict(q_xyzw=[0, 0, 0, 1], p=[0, 0, 0])})
            prepared = SimpleNamespace(camera=camera, imu=self.imu)
            scene = SimpleNamespace(trajectory=SimpleNamespace(values=self.trajectory), parameters=self.parameters)
            with patch.object(mapping.slam, 'atlas_candidates', return_value=iter([(0, self.obs)])), \
                    patch.object(mapping.fusion, 'initial_gyro_bias', return_value=np.zeros(3)), \
                    patch.object(mapping, 'metric_alignment', side_effect=geometry.MapAlignmentError('尺度冲突', diagnostics)), \
                    self.assertRaisesRegex(ValueError, '尺度冲突'):
                mapping.choose_map(prepared, 'front', archive, root / 'localize', scene)
            saved = json.loads((root / 'localize/m0/metric_alignment.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['diagnostics'], diagnostics)
            self.assertEqual(archive.read_bytes(), b'unchanged visual atlas')

    def test_map_candidate_selection_records_and_uses_quality_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'map.osa'
            archive.write_bytes(b'unchanged visual atlas')
            camera = SimpleNamespace(measurements={'front': dict(q_xyzw=[0, 0, 0, 1], p=[0, 0, 0])})
            prepared = SimpleNamespace(camera=camera, imu=self.imu,
                                       video_times_s=np.arange(100, dtype=float))
            scene = SimpleNamespace(trajectory=SimpleNamespace(values=self.trajectory), parameters=self.parameters)
            placement_good = dict(scale=2., evidence=dict(rms_m=.01, relative_scale_std=.05, condition=10.,
                alignment=dict(fixed_map_validation=[dict(status='verified', rms_m=.01)])))
            placement_weak = dict(scale=2., evidence=dict(rms_m=.02, relative_scale_std=.1, condition=20.,
                alignment=dict(fixed_map_validation=[dict(status='verified', rms_m=.04)])))
            observations = [self.obs, self.obs]
            with patch.object(mapping.slam, 'atlas_candidates', return_value=iter([(0, observations[0]), (1, observations[1])])), \
                    patch.object(mapping.fusion, 'initial_gyro_bias', return_value=np.zeros(3)), \
                    patch.object(mapping, 'metric_alignment', side_effect=[placement_weak, placement_good]):
                selected, _ = mapping.choose_map(prepared, 'front', archive, root / 'localize', scene)
            self.assertEqual(selected['index'], 1)
            saved = json.loads((root / 'localize/m1/metric_alignment.json').read_text())
            self.assertEqual(saved['status'], 'ok')
            self.assertEqual(saved['selection_score']['observed_frames'], len(self.obs.imu_times_s))

    def test_zero_visual_displacement_keeps_serializable_failure_evidence(self):
        poses = self.obs.T_reference_camera.copy()
        poses[:, :3, 3] = 0
        with self.assertRaises(geometry.MapAlignmentError) as caught:
            self.solve(replace(self.obs, T_reference_camera=poses))
        diagnostics = caught.exception.diagnostics
        self.assertIn('segments', diagnostics)
        self.assertTrue(diagnostics['segments']['chains'])
        json.dumps(diagnostics, allow_nan=False)
