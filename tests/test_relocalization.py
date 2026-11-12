"""Reverse localization must restore original clocks and reject unanchored or inconsistent runs."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from software.scene.management.inputs import load_scene_parameters
from software.scene.spatial import geometry, relocalization, slam
from software import config
from software.vision import slam as visual_slam


def observations(indices, *, shift=0., rotation=0.):
    indices = np.asarray(indices, int)
    poses = np.tile(np.eye(4), (len(indices), 1, 1))
    poses[:, 0, 3] = indices * .01 + shift
    poses[:, :3, :3] = Rotation.from_euler('z', rotation, degrees=True).as_matrix()
    return geometry.VisualObservations(indices / 30 + .007, poses, indices, 'visual_map')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.parameters = load_scene_parameters()

    def merge(self, forward, reverse, scale=1.):
        return relocalization.merge(forward, reverse, metric_scale=scale, parameters=self.parameters)

    def test_prefix_and_internal_gap_recovered_without_changing_forward_poses_or_time(self):
        forward = observations([5, 8, 9, 10, 11])
        reverse = observations(range(10), shift=.001)
        merged, report = self.merge(forward, reverse)
        np.testing.assert_array_equal(merged.indices, np.arange(12))
        np.testing.assert_array_equal(merged.imu_times_s, np.arange(12)/30+.007)
        np.testing.assert_array_equal(merged.T_reference_camera[forward.indices], forward.T_reference_camera)
        np.testing.assert_array_equal(merged.T_reference_camera[:5], reverse.T_reference_camera[:5])
        self.assertEqual(report['added_frames'], 7)
        self.assertEqual(report['segments'][0]['overlap_frames'], 3)

    def test_reverse_segments_require_their_own_overlap_and_consistency(self):
        forward = observations([0, 1, 2, 10, 11, 12, 20, 21])
        reverse = observations([*range(6), *range(8, 15), *range(18, 23)])
        reverse.T_reference_camera[6:13, 0, 3] += .1
        merged, report = self.merge(forward, reverse)
        np.testing.assert_array_equal(merged.indices, [0, 1, 2, 3, 4, 5, 10, 11, 12, 20, 21])
        self.assertEqual([s['status'] for s in report['segments']],
                         ['accepted', 'inconsistent', 'insufficient_overlap'])

    def test_no_overlap_and_no_forward_observations_are_not_used(self):
        for forward in (observations([]), observations([10, 11, 12])):
            merged, report = self.merge(forward, observations(range(5)))
            np.testing.assert_array_equal(merged.indices, forward.indices)
            self.assertEqual(report['added_frames'], 0)

    def test_translation_uses_map_scale_and_rotation_is_checked_independently(self):
        forward = observations([3, 4, 5])
        reverse = observations(range(6), shift=.02)
        self.assertEqual(self.merge(forward, reverse, scale=.5)[1]['added_frames'], 3)
        self.assertEqual(self.merge(forward, reverse, scale=2.)[1]['added_frames'], 0)
        self.assertEqual(self.merge(forward, observations(range(6), rotation=4.))[1]['added_frames'], 0)

    def test_one_conflicting_overlap_rejects_segment_instead_of_averaging_it_away(self):
        forward = observations(range(3, 20))
        reverse = observations(range(20))
        reverse.T_reference_camera[10, 0, 3] += .04
        self.assertEqual(self.merge(forward, reverse)[1]['added_frames'], 0)

    def test_mismatched_times_unsorted_indices_and_bad_scales_are_rejected(self):
        forward, reverse = observations([3, 4, 5]), observations(range(6))
        with self.assertRaisesRegex(ValueError, '时间戳'):
            self.merge(forward, replace(reverse, imu_times_s=reverse.imu_times_s + .1))
        with self.assertRaises(ValueError):
            self.merge(forward, observations([2, 1, 0]))
        for scale in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                self.merge(forward, reverse, scale)

    def test_complete_and_disabled_paths_skip_reverse(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            recording = SimpleNamespace(video_times_s=np.arange(6)/30)
            for parameters, result, status in (
                (self.parameters, observations(range(6)), 'forward_complete'),
                (replace(self.parameters, localization_bidirectional=False), observations([3,4,5]), 'disabled')):
                with patch.object(slam, 'localize', return_value=result) as localize:
                    output = slam.localize_with_recovery(recording, 'front', directory/'map.osa', 0,
                                                         directory, 1., parameters)
                self.assertIs(output, result)
                self.assertEqual(localize.call_count, 1)
                self.assertEqual(json.loads((directory/'recovery.json').read_text())['status'], status)

    def test_reverse_read_restores_original_frame_order_and_camera_imu_offset(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'frames.csv'
            path.write_text('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw\n'
                            '2,0.066666667,2,1,2,0,0,0,0,0,1\n'
                            '1,0.033333333,4,0,,,,,,,\n'
                            '0,0,2,1,0,0,0,0,0,0,1\n')
            result = geometry.read_visual_observations(path, np.array([0., .033, .067]), 7000000)
            np.testing.assert_array_equal(result.indices, [0, 2])
            np.testing.assert_allclose(result.imu_times_s, [.007, .074])

    def test_policy_rejects_insufficient_overlap_and_invalid_boolean(self):
        for data in ({'localization_minimum_overlap_frames':2}, {'localization_bidirectional':1},
                     {'localization_max_position_difference_m':0}, {'localization_max_rotation_difference_deg':181},
                     {'localization_reverse_context_s':0}, {'localization_reverse_context_s':float('nan')}):
            with self.assertRaises(ValueError):
                load_scene_parameters(overrides=data)

    def test_pipeline_runs_reverse_only_after_gap_and_records_accepted_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            recording = SimpleNamespace(video_times_s=np.arange(6)/30)
            with patch.object(slam, 'localize', side_effect=[observations([3, 4, 5]), observations(range(6))]) as localize:
                output = slam.localize_with_recovery(recording, 'front', directory/'map.osa', 0,
                                                     directory, 1., self.parameters)
            np.testing.assert_array_equal(output.indices, np.arange(6))
            self.assertEqual(localize.call_args.kwargs, {'reverse':True})
            self.assertEqual(json.loads((directory/'recovery.json').read_text())['added_frames'], 3)

    def test_window_uses_original_times_and_spans_all_gaps(self):
        times = np.arange(300) / 30.
        times[150:] += .5  # 非均匀时间轴不能按固定 FPS 推算上下文。
        forward = observations(np.delete(np.arange(300), [100, 160]))
        window = relocalization.reverse_window(forward, times, self.parameters)
        self.assertEqual(window, (70, 191))
        self.assertIsNone(relocalization.reverse_window(observations([]), times, self.parameters))
        self.assertIsNone(relocalization.reverse_window(observations(range(300)), times, self.parameters))

    def test_complete_window_skips_full_reverse_and_retains_forward(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            recording = SimpleNamespace(video_times_s=np.arange(300)/30)
            forward = observations(range(10, 300))
            with patch.object(slam, 'localize', side_effect=[forward, observations(range(40), shift=.001)]) as localize:
                output = slam.localize_with_recovery(recording, 'front', directory/'map.osa', 0,
                                                     directory, 1., self.parameters)
            self.assertEqual(localize.call_count, 2)
            self.assertEqual(localize.call_args.kwargs, {'reverse': True, 'frame_range': (0, 40)})
            np.testing.assert_array_equal(output.indices, np.arange(300))
            np.testing.assert_array_equal(output.T_reference_camera[10:], forward.T_reference_camera)
            report = json.loads((directory/'recovery.json').read_text())
            self.assertEqual(report['reverse_processed_frames'], 40)
            self.assertFalse(report['full_reverse_fallback'])

    def test_incomplete_window_retries_full_reverse_against_original_forward_anchors(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            recording = SimpleNamespace(video_times_s=np.arange(300)/30)
            forward = observations(range(10, 300))
            window = observations(range(5, 40), shift=.001)
            whole = observations(range(300), shift=.002)
            with patch.object(slam, 'localize', side_effect=[forward, window, whole]) as localize:
                output = slam.localize_with_recovery(recording, 'front', directory/'map.osa', 0,
                                                     directory, 1., self.parameters)
            self.assertEqual(localize.call_count, 3)
            self.assertEqual(localize.call_args.kwargs, {'reverse': True})
            np.testing.assert_array_equal(output.T_reference_camera[:10], whole.T_reference_camera[:10])
            report = json.loads((directory/'recovery.json').read_text())
            self.assertEqual(report['reverse_processed_frames'], 340)
            self.assertEqual(report['reverse_attempts'][0]['added_frames'], 5)
            self.assertTrue(report['full_reverse_fallback'])


class ReverseRuntimeTests(unittest.TestCase):
    def test_partial_csv_keeps_absolute_indices_and_checks_requested_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'logs').mkdir()
            settings = SimpleNamespace(environment={}, orb_directory=root, vocabulary=root/'vocab')
            context = SimpleNamespace(settings=settings, parameters=config.load('trajectory').values,
                                      frame_count=10, cores='')
            video = SimpleNamespace(lens='front', settings=root/'settings', path=root/'input.mp4')
            for case, reverse, indices in [('reverse', True, [5,4,3]), ('forward', False, [3,4,5]),
                                           ('renumbered', True, [2,1,0]), ('missing', True, [5,3])]:
                output = root/case
                def run(command, log, **kwargs):
                    self.assertEqual(command[command.index('--frame_start') + 1], '3')
                    self.assertEqual(command[command.index('--frame_end') + 1], '6')
                    log.write_text('TRACKED 3 / 3\n')
                    (output/'frames.csv').write_text('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw\n' +
                        ''.join(f'{i},{i/30},2,1,0,0,0,0,0,0,1\n' for i in indices))
                    return 0
                with patch.object(visual_slam.runtime, 'run_logged', side_effect=run):
                    if case in ('reverse', 'forward'):
                        visual_slam.run_orb(video, output, context, root, stage=case,
                                           extra=['--localization_only'], reverse=reverse, frame_range=(3,6))
                        self.assertEqual(json.loads((output/'execution.json').read_text())['rows'], 3)
                    else:
                        with self.assertRaises(visual_slam.ProcessingError):
                            visual_slam.run_orb(video, output, context, root, stage=case,
                                               extra=['--localization_only'], reverse=reverse, frame_range=(3,6))

    def test_reverse_csv_must_have_all_original_frame_indices_in_reverse_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'logs').mkdir()
            settings = SimpleNamespace(environment={}, orb_directory=root, vocabulary=root/'vocab')
            context = SimpleNamespace(settings=settings, parameters=config.load('trajectory').values,
                                      frame_count=3, cores='')
            video = SimpleNamespace(lens='front', settings=root/'settings', path=root/'input.mp4')
            for case, indices in [('valid', [2,1,0]), ('wrong_order', [0,1,2]), ('missing', [2,0])]:
                output = root/case
                def run(command, log, **kwargs):
                    self.assertIn('--reverse', command)
                    log.write_text('TRACKED 3 / 3\n')
                    (output/'frames.csv').write_text('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw\n' +
                        ''.join(f'{i},{i/30},2,1,0,0,0,0,0,0,1\n' for i in indices))
                    return 0
                with patch.object(visual_slam.runtime, 'run_logged', side_effect=run):
                    if case == 'valid':
                        visual_slam.run_orb(video, output, context, root, stage='reverse',
                                            extra=['--localization_only'], reverse=True)
                        self.assertTrue(json.loads((output/'execution.json').read_text())['accepted'])
                    else:
                        with self.assertRaises(visual_slam.ProcessingError):
                            visual_slam.run_orb(video, output, context, root, stage='reverse',
                                                extra=['--localization_only'], reverse=True)
