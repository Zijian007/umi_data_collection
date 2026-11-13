"""相同验收失败停止重建；变化的轨迹和暂时性引擎错误仍保留重试。"""

from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from software.scene.management import inputs
from software.scene.pipeline import map_lifecycle
from software.scene.spatial import mapping, slam
from software.vision.models import ProcessingError

import test_scene as fixtures


class MapRetryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SceneWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def failure(self):
        files = list((self.fixture.scene.map_root / 'versions').glob('*/error.json'))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text())

    def test_identical_rejection_stops_after_two_attempts_and_preserves_published_map(self):
        fixture = self.fixture
        fixture.build_scene()
        pointer = fixture.session_path  # 当前地图只绑定在批次清单
        previous = pointer.read_bytes()
        with patch.object(slam, 'build', side_effect=fixture.build) as build, \
                patch.object(mapping, 'metric_alignment', side_effect=ValueError('固定尺度不一致')):
            with self.assertRaisesRegex(ValueError, '停止重复重建'):
                fixture.build_scene(force=True)
        self.assertEqual(build.call_count, 2)
        self.assertEqual([call.kwargs['refresh'] for call in build.call_args_list], [False, True])
        report = self.failure()['retry']
        self.assertEqual(report['stop_reason'], 'repeated_visual_failure')
        first, second = report['attempts']
        self.assertEqual(first['signature'], second['signature'])
        self.assertEqual(second['repeat_of_attempt'], 1)
        self.assertEqual(first['category'], 'map_acceptance')
        self.assertEqual(pointer.read_bytes(), previous)
        directory, manifest = map_lifecycle.load_published(fixture.scene)
        inputs.validate_assets(directory, manifest['assets'])

    def test_changed_trajectory_can_recover_on_third_attempt(self):
        fixture = self.fixture
        visits = []
        original_alignment = mapping.metric_alignment

        def candidates(recording, lens, archive, directory, **kwargs):
            observed = fixture.observations(recording, lens)
            visits.append(directory)
            poses = observed.T_reference_camera.copy()
            if len(visits) < 3:
                poses[0, 0, 3] += .01 * len(visits)
            yield 0, replace(observed, T_reference_camera=poses)

        def alignment(*args, **kwargs):
            if len(visits) < 3:
                raise ValueError('固定尺度不一致')
            return original_alignment(*args, **kwargs)

        with patch.object(slam, 'atlas_candidates', side_effect=candidates), \
                patch.object(mapping, 'metric_alignment', side_effect=alignment):
            fixture.build_scene()
        self.assertEqual(len(visits), 3)
        _, manifest = map_lifecycle.load_published(fixture.scene)
        report = manifest['retry']
        self.assertEqual(report['stop_reason'], 'accepted')
        self.assertEqual([attempt['status'] for attempt in report['attempts']], ['failed', 'failed', 'accepted'])
        self.assertNotEqual(report['attempts'][0]['signature'], report['attempts'][1]['signature'])

    def test_changed_rejection_reason_keeps_retry_available(self):
        fixture = self.fixture
        original_alignment = mapping.metric_alignment
        visits = []

        def alignment(*args, **kwargs):
            visits.append(None)
            if len(visits) == 1:
                raise ValueError('运动退化')
            if len(visits) == 2:
                raise ValueError('重力不一致')
            return original_alignment(*args, **kwargs)

        with patch.object(mapping, 'metric_alignment', side_effect=alignment):
            fixture.build_scene()
        _, manifest = map_lifecycle.load_published(fixture.scene)
        first, second, accepted = manifest['retry']['attempts']
        self.assertEqual(first['maps'][0]['observations_sha256'], second['maps'][0]['observations_sha256'])
        self.assertNotEqual(first['signature'], second['signature'])
        self.assertEqual(accepted['status'], 'accepted')

    def test_engine_errors_with_same_message_do_not_count_as_identical_trajectories(self):
        with patch.object(slam, 'build', side_effect=ProcessingError('建图', '引擎暂时退出', return_code=1)) as build:
            with self.assertRaisesRegex(ValueError, '场景地图未通过验收'):
                self.fixture.build_scene()
        self.assertEqual(build.call_count, 3)
        report = self.failure()['retry']
        self.assertEqual(report['stop_reason'], 'attempt_limit')
        self.assertTrue(all(attempt['category'] == 'engine' and 'signature' not in attempt
                            for attempt in report['attempts']))

    def test_maximum_one_attempt_does_not_claim_duplicate_evidence(self):
        fixture = self.fixture
        parameters = replace(fixture.scene.trajectory.values,
                             retry=replace(fixture.scene.trajectory.values.retry, max_tries=1))
        scene = replace(fixture.scene, trajectory=replace(fixture.scene.trajectory, values=parameters))
        with patch.object(mapping, 'metric_alignment', side_effect=ValueError('固定尺度不一致')):
            with self.assertRaisesRegex(ValueError, '场景地图未通过验收'):
                map_lifecycle._build_scene(scene)
        report = self.failure()['retry']
        self.assertEqual(report['stop_reason'], 'attempt_limit')
        self.assertEqual(len(report['attempts']), 1)

    def test_all_rejected_submaps_participate_in_comparison(self):
        fixture = self.fixture
        recording = fixture.prepared_recordings[fixture.recording_paths['left']]
        observed = fixture.raw['left']
        errors = []
        for change in (0., .001):
            poses = observed.T_reference_camera.copy()
            poses[0, 1, 3] += change
            candidates = [(0, observed), (1, replace(observed, T_reference_camera=poses))]
            with patch.object(slam, 'atlas_candidates', return_value=iter(candidates)), \
                    patch.object(mapping, 'metric_alignment', side_effect=ValueError('固定尺度不一致')):
                with self.assertRaises(mapping.MapAcceptanceError) as caught:
                    mapping.choose_map(recording, 'front', fixture.root / 'map.osa',
                                       fixture.root / f'candidates-{len(errors)}', fixture.scene)
            errors.append(caught.exception)
        self.assertEqual(errors[0].rejected_maps[0], errors[1].rejected_maps[0])
        self.assertNotEqual(errors[0].retry_signature, errors[1].retry_signature)

    def test_no_candidate_data_cannot_prove_duplicate_trajectory(self):
        with patch.object(slam, 'atlas_candidates', return_value=iter(())):
            with self.assertRaises(ValueError):
                self.fixture.build_scene()
        report = self.failure()['retry']
        self.assertEqual(report['stop_reason'], 'attempt_limit')
        self.assertEqual(len(report['attempts']), 3)
        self.assertTrue(all(attempt['signature'] is None for attempt in report['attempts']))

    def test_fixed_scale_failure_reports_actual_video_interval(self):
        fixture = self.fixture
        recording = fixture.prepared_recordings[fixture.recording_paths['left']]
        diagnostics = dict(fixed_map_validation=[dict(status='inconsistent', rms_m=.933, k=[3, 10])])
        error = mapping.MapAlignmentError('固定尺度不一致', diagnostics)
        with patch.object(mapping, 'metric_alignment', side_effect=error):
            with self.assertRaises(mapping.MapAcceptanceError) as caught:
                mapping.choose_map(recording, 'front', fixture.root / 'map.osa', fixture.root / 'failure', fixture.scene)
        expected = [float(recording.video_times_s[index]) for index in (3, 10)]
        self.assertEqual(caught.exception.rejected_maps[0]['video_interval_s'], expected)
        self.assertIn(f'录像 {expected[0]:.2f}–{expected[1]:.2f} s', str(caught.exception))
