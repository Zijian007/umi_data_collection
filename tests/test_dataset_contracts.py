from software.dataset import reader
"""训练质量策略、坐标契约与真实导出包的回归。"""

from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np

from software.dataset import export, sampling
from software.dataset.schema import read_pose_series
from software.scene import cli


def arm_samples(mask, *, legacy=False):
    mask = np.asarray(mask, bool)
    times = np.arange(len(mask)) / 10.
    poses = np.tile(np.eye(4), (len(mask), 1, 1))
    poses[:, 0, 3] = times
    support = None if legacy else dict(position_supported=mask)
    return sampling.ArmSamples(times, poses, np.arange(len(mask)), times,
                               mask, mask, position_support=support)


class QualitySelectionTests(unittest.TestCase):
    def test_default_keeps_samples_strict_selection_splits_at_each_removed_interval(self):
        arm = arm_samples([1, 1, 0, 1, 1])
        default, _ = sampling.sample(dict(left=arm), 20., .25)
        self.assertEqual(len(default['t_episode_s']), 9)
        arrays, quality = sampling.sample(dict(left=arm), 20., .25, quality_policy='position_supported')
        np.testing.assert_allclose(arrays['t_episode_s'], [0., .05, .1, .3, .35, .4])
        np.testing.assert_array_equal(arrays['segment_id'], [0, 0, 0, 1, 1, 1])
        self.assertTrue(arrays['left_position_supported'].all())
        self.assertEqual(quality['selection']['dropped_quality_samples'], 3)
        self.assertEqual(quality['selection']['arms']['left']['unsupported_samples'], 3)

    def test_every_arm_must_have_support_and_all_arrays_follow_the_same_selection(self):
        left, right = arm_samples([1, 1, 1, 1, 1]), arm_samples([1, 1, 0, 1, 1])
        arrays, _ = sampling.sample(dict(left=left, right=right), 20., .25,
                                   quality_policy='position_supported')
        self.assertEqual({len(a) for a in arrays.values()}, {6})
        np.testing.assert_allclose(arrays['T_left_tcp_right_tcp'], np.tile(np.eye(4), (6, 1, 1)))
        self.assertTrue(arrays['right_position_supported'].all())

    def test_legacy_visual_support_is_known_but_missing_poster_evidence_is_unknown(self):
        arm = arm_samples([1, 1, 0, 1, 1], legacy=True)
        arrays, quality = sampling.sample(dict(left=arm), 20., .25, quality_policy='position_supported')
        self.assertEqual(len(arrays['t_episode_s']), 6)
        self.assertNotIn('left_position_supported', arrays)
        self.assertEqual(quality['selection']['arms']['left'], dict(unsupported_samples=0, unknown_samples=3))
        with self.assertRaisesRegex(ValueError, '未知位置支持'):
            sampling.sample(dict(left=arm), 20., .25, quality_policy='position_supported', unknown_support='error')
        # 未知策略不作用于默认保留全部的模式。
        arrays, _ = sampling.sample(dict(left=arm), 20., .25, unknown_support='error')
        self.assertEqual(len(arrays['t_episode_s']), 9)

    def test_quality_and_timestamp_gap_rejections_are_counted_separately(self):
        arm = arm_samples([1, 1, 0, 1, 1])
        arm = replace(arm, times_s=np.array([0., .1, 1., 1.1, 1.2]))
        arrays, quality = sampling.sample(dict(left=arm), 10., .25, quality_policy='position_supported')
        np.testing.assert_allclose(arrays['t_episode_s'], [0., .1, 1.1, 1.2])
        self.assertEqual(quality['dropped_gap_samples'], 8)
        self.assertEqual(quality['selection']['dropped_quality_samples'], 1)
        np.testing.assert_array_equal(arrays['segment_id'], [0, 0, 1, 1])

    def test_no_supported_samples_and_invalid_policy_fail_explicitly(self):
        arm = arm_samples([0, 0, 0])
        for kwargs in (dict(quality_policy='position_supported'), dict(quality_policy='typo'),
                       dict(unknown_support='typo')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sampling.sample(dict(left=arm), 10., .25, **kwargs)

    def test_downsampling_cannot_hide_a_quality_or_timestamp_gap_between_samples(self):
        arm = arm_samples([1, 0, 1, 1, 1])
        arrays, _ = sampling.sample(dict(left=arm), 5., .25, quality_policy='position_supported')
        np.testing.assert_allclose(arrays['t_episode_s'], [0., .2, .4])
        np.testing.assert_array_equal(arrays['segment_id'], [0, 1, 1])
        arm = replace(arm, times_s=np.array([0., .1, 1., 1.1, 1.2]))
        arrays, _ = sampling.sample(dict(left=arm), 1., .25)
        np.testing.assert_allclose(arrays['t_episode_s'], [0., 1.])
        np.testing.assert_array_equal(arrays['segment_id'], [0, 1])


class CoordinateReaderTests(unittest.TestCase):
    def setUp(self):
        self.arrays = dict(t_episode_s=np.array([0., .1]), left_T_scene_tcp=np.tile(np.eye(4), (2, 1, 1)))

    def test_scene_poster_and_legacy_packages_return_explicit_reference_without_copy(self):
        for scene, representation in ((dict(frame_id='scene/map-v1'), 'T_scene_tcp'),
                (dict(frame_id='poster:run', scope='task_poster'), 'T_board_tcp'),
                (dict(frame_id='local:run', scope='local'), 'T_scene_tcp')):
            metadata = dict(arms=['left'], scene=scene)
            poses = read_pose_series(metadata, self.arrays, 'left')
            self.assertIs(poses.T_reference_tcp, self.arrays['left_T_scene_tcp'])
            self.assertEqual(poses.frame_id, scene['frame_id'])
            self.assertEqual(poses.pose_representation, representation)

    def test_conflicting_reference_units_and_invalid_rotations_are_rejected(self):
        metadata = dict(arms=['left'], scene=dict(frame_id='poster:run', scope='task_poster'))
        for change in (dict(frame_id='scene/map'), dict(coordinate_scope='shared_scene'),
                       dict(pose_representation='T_scene_tcp'), dict(position_unit='mm')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                read_pose_series(dict(metadata, **change), self.arrays, 'left')
        self.arrays['left_T_scene_tcp'][0, 0, 0] = -1.
        with self.assertRaisesRegex(ValueError, r'SO\(3\)'):
            read_pose_series(metadata, self.arrays, 'left')


class DatasetContractIntegrationTests(unittest.TestCase):
    def setUp(self):
        from test_dataset import DatasetTests
        self.fixture = DatasetTests('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_strict_cli_exports_portable_report_and_explicit_pose_reader(self):
        import io
        from contextlib import redirect_stdout
        f = self.fixture
        with redirect_stdout(io.StringIO()):
            code = cli.main(['export', str(f.f.episode_path), '--out', str(f.output), '--format', 'training',
                             '--fps', '20', '--quality-policy', 'position_supported', '--unknown-support', 'error'])
        self.assertEqual(code, 0)
        manifest = json.loads((f.output / 'manifest.json').read_text())
        report = json.loads((f.output / 'export_report.json').read_text())
        metadata, arrays = reader.load_episode(f.output, manifest['episodes'][0]['path'])
        self.assertEqual(manifest['quality_policy'], 'position_supported')
        self.assertEqual(report['quality_selection'][0]['quality_policy'], 'position_supported')
        poses = read_pose_series(metadata, arrays, 'left')
        self.assertEqual(poses.frame_id, metadata['frame_id'])
        self.assertTrue(arrays['left_full_constraint'].all())
        from pathlib import Path
        import re
        guide = (Path(__file__).resolve().parents[1] / 'docs/训练数据.md').read_text()
        fields = re.findall(r'^\| `([^`]+)` \|', guide, re.M)
        documented = set()
        for field in fields:
            if '_gripper_' in field:
                continue  # 本样本未配置宽度；宽度通道由现有独立回归覆盖。
            if '<arm>' in field:
                documented.update(field.replace('<arm>', arm) for arm in metadata['arms'])
            elif field in ('t_episode_s', 'segment_id', 'T_left_tcp_right_tcp'):
                documented.add(field)
        self.assertEqual(set(arrays), documented)

    def test_quality_failure_removes_staging_and_never_publishes_partial_package(self):
        f = self.fixture
        with patch.object(sampling, 'read_arm', return_value=arm_samples([0, 0, 0])):
            with self.assertRaises(ValueError):
                export.build([f.f.episode_path], f.output, quality_policy='position_supported')
        self.assertFalse(f.output.exists())
        self.assertFalse(list(f.output.parent.glob('.dataset-*')))
