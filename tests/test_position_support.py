"""海报几何退化、精确事件缺口和训练采样时的来源切换。"""
from dataclasses import replace
import unittest

import numpy as np

from software.configuration import loader as config
from software.dataset import sampling
from software.trajectory.reporting.constraint_quality import maximum_support_gap, position_support
from software.vision.poster.geometry import camera_model, transform
from software.vision.poster.quality import position_support_metrics, cull_decision


class PosterSupportTests(unittest.TestCase):
    def test_culling_budget_cannot_erase_a_large_fraction_or_the_time_span(self):
        parameters = config.load('trajectory').values.environment_tags
        frames, times = np.arange(20), np.arange(20)/10.
        mask = frames < 4
        self.assertEqual(cull_decision(mask, frames, times, parameters)['reason'], 'count_budget')
        mask = frames == 10
        decision = cull_decision(mask, frames, times, parameters)
        self.assertTrue(decision['allowed'])
        self.assertEqual(decision['remaining_frames'], 19)
        self.assertEqual(cull_decision(mask, frames, times,
            replace(parameters, tag_maximum_cull_fraction=0.))['reason'], 'count_budget')
        times = np.r_[0., np.linspace(10., 10.1, 19)]
        self.assertEqual(cull_decision(frames == 0, frames, times, parameters)['reason'], 'time_coverage')
        self.assertEqual(cull_decision(frames > 1, frames, np.arange(20),
            replace(parameters, tag_maximum_cull_fraction=.95))['reason'], 'remaining_frames')

    def geometry(self, *, depth=1., collapsed=False, sigma=1.):
        parameters = config.load('trajectory').values.environment_tags
        parameters = replace(parameters, tag_corner_sigma_px=sigma)
        model = camera_model(dict(camera_type='kb4', resolution=[1440, 1440],
                                  fx=520., fy=520., cx=720., cy=720., k=[0.]*4))
        square = np.array([[-.1, -.1, 0.], [.1, -.1, 0.], [.1, .1, 0.], [-.1, .1, 0.]])
        points = np.array([square + [-.4, 0., 0.], square + [.4, 0., 0.]])
        if collapsed:
            points[:] = 0.
        board = np.eye(4)
        board[2, 3] = depth
        pixels = model.project(transform(board, points).reshape(-1, 3)).reshape(-1, 4, 2)
        result = position_support_metrics(np.array([np.eye(4)]*2), board, np.eye(4),
            points, pixels, model, np.array([.013, .013]), parameters)
        return result['frames'][0]

    def test_perfect_pixels_can_still_have_weak_geometry(self):
        near, far = self.geometry(), self.geometry(depth=12.)
        self.assertTrue(near['position_supported'])
        self.assertLess(near['rms_px'], 1e-9)
        self.assertLess(far['rms_px'], 1e-9)
        self.assertFalse(far['position_supported'])
        self.assertGreater(far['position_std_m'], near['position_std_m']*5)
        noisy = self.geometry(sigma=2.)
        self.assertAlmostEqual(noisy['position_std_m'], near['position_std_m']*2)

    def test_rank_loss_is_unknown_not_zero_uncertainty(self):
        frame = self.geometry(collapsed=True)
        self.assertLess(frame['rank'], 6)
        self.assertIsNone(frame['position_std_m'])
        self.assertIsNone(frame['rotation_std_deg'])
        self.assertFalse(frame['position_supported'])

    def test_exact_events_gate_gap_but_nearest_cells_only_describe_output(self):
        times = np.array([0., 1., 2.])
        visual = np.array([True, False, True])
        poster = dict(status='fused', observation_video_times_s=[.49, .51, 1.4],
                      position_support_video_times_s=[.49, 1.4])
        masks = position_support(times, visual, poster)
        np.testing.assert_array_equal(masks['poster_observed'], [1, 1, 0])
        np.testing.assert_array_equal(masks['poster_position_supported'], [1, 1, 0])
        self.assertAlmostEqual(maximum_support_gap(times, visual, np.array([.49, 1.4])), .91)
        self.assertAlmostEqual(maximum_support_gap(times, visual, np.array([])), 2.)
        unknown = position_support(times, visual, dict(status='fused'))
        self.assertIsNone(unknown['position_supported'])
        self.assertIsNone(unknown['poster_observed'])

    def test_training_combines_sources_before_conservative_bracketing(self):
        times = np.array([0., .1, .2])
        visual, poster = np.array([1, 0, 1], bool), np.array([0, 1, 0], bool)
        support = dict(visual_position_supported=visual, poster_observed=poster,
                       poster_position_supported=poster, position_supported=visual | poster)
        poses = np.tile(np.eye(4), (3, 1, 1))
        arm = sampling.ArmSamples(times, poses, np.arange(3), times, visual, visual, position_support=support)
        arrays, quality = sampling.sample(dict(left=arm, right=arm), 20., .25)
        self.assertTrue(arrays['left_position_supported'].all())
        self.assertFalse(arrays['left_visual_position_supported'][1])
        self.assertFalse(arrays['left_poster_position_supported'][1])
        self.assertTrue(quality['arms']['left']['position_support_available'])
        np.testing.assert_allclose(arrays['T_left_tcp_right_tcp'], np.tile(np.eye(4), (5, 1, 1)))
        legacy = replace(arm, position_support=None)
        arrays, quality = sampling.sample(dict(left=legacy), 20., .25)
        self.assertNotIn('left_poster_position_supported', arrays)
        self.assertFalse(quality['arms']['left']['position_support_available'])

    def test_malformed_or_contradictory_flags_are_rejected(self):
        row = dict(constraint='full', visual_position_supported='1', poster_observed='0',
                   poster_position_supported='1', position_supported='1')
        with self.assertRaisesRegex(ValueError, '观测证据'):
            sampling._read_position_support([row])
        row.update(poster_observed='1', position_supported='0')
        with self.assertRaisesRegex(ValueError, '并集'):
            sampling._read_position_support([row])
        self.assertIsNone(sampling._read_position_support([dict(constraint='full')]))
