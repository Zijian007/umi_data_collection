"""坐标身份与时钟边界：拒绝未放置/重复缩放和跨场景拼接。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from software import config
from software.camera.timing import (
    canonical_to_video_seconds, euroc_ns_to_seconds,
    video_to_canonical_seconds, video_to_imu_seconds,
)
from software.scene import fusion, inputs, results
from software.scene.geometry import VisualObservations
from software.scene.models import ArmTrajectory


class CoordinateContractTests(unittest.TestCase):
    def test_nonzero_camera_offset_is_applied_once_at_the_clock_boundary(self):
        video_s = np.array([0., .125, 1.5])
        offset_ns = -12_500_000
        canonical_s = video_to_canonical_seconds(video_s)
        imu_ns = canonical_s * 1e9 + offset_ns
        np.testing.assert_array_equal(canonical_to_video_seconds(canonical_s), video_s)
        np.testing.assert_array_equal(euroc_ns_to_seconds(imu_ns, camera_offset_ns=offset_ns), video_s)
        np.testing.assert_allclose(euroc_ns_to_seconds(imu_ns), video_to_imu_seconds(video_s, offset_ns), atol=1e-15)

    def test_map_placement_changes_frame_and_cannot_apply_scale_twice(self):
        poses = np.tile(np.eye(4), (3, 1, 1))
        poses[:, 0, 3] = [1., 2., 3.]
        raw = VisualObservations(np.arange(3.), poses, np.arange(3), 'visual_map')
        placed = raw.placed(dict(scale=2., T_scene_metric_map=np.eye(4)), reference_frame='scene')
        self.assertEqual(placed.reference_frame, 'scene')
        np.testing.assert_array_equal(placed.T_reference_camera[:, 0, 3], [2., 4., 6.])
        with self.assertRaisesRegex(ValueError, '重复'):
            placed.placed(dict(scale=2., T_scene_metric_map=np.eye(4)), reference_frame='scene')
        sampled = fusion.sample_observations(placed, placed.imu_times_s, .2)
        self.assertEqual(sampled.reference_frame, 'scene')

    def test_scene_fusion_rejects_raw_visual_map_coordinates(self):
        times = np.arange(3.)
        obs = VisualObservations(times, np.tile(np.eye(4), (3, 1, 1)), np.arange(3), 'visual_map')
        with self.assertRaisesRegex(ValueError, '场景坐标'):
            fusion.fuse(times, np.zeros((3, 7)), dict(front=obs), dict(front=np.eye(4)),
                        config.load('trajectory').values, inputs.load_scene_parameters(None))

    def test_bimanual_export_rejects_two_different_scene_frames(self):
        track = ArmTrajectory(np.arange(3.), np.tile(np.eye(4), (3, 1, 1)), 'scene:first', {})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, '同一个'):
                results.write_bimanual(dict(left=track, right=replace(track, frame_id='scene:second')), Path(temporary))
            self.assertEqual(list(Path(temporary).iterdir()), [])
