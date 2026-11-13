from software.scene import layout
"""全段快速扫描、按需补检、真实 PTS 与每手有效比例验收。"""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import json
import tempfile
import unittest

import av
import numpy as np

from software.gripper_width import detection, sampled_detection, tag_distance
from software.gripper_width.tag_models import TagDistanceRequest, TagDistanceResult, TagDistanceSeries, TagPoseQuality
from software.scene.management import inputs
from software.storage import write_json
from test_tag_distance import CAMERA, SIZES, projected


def found_pair():
    corners, _ = projected()
    return detection.FrameDetection(tuple(polygon[None] for polygon in corners), np.array([[0], [1]]),
                                    {tag: corners[tag].mean(axis=0) for tag in (0, 1)})


EMPTY = detection.FrameDetection((), None, {})


class SampledTagTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.request = TagDistanceRequest(self.root / 'raw.insv', 'recording', SIZES, 'sizes', CAMERA, 'camera',
                                          np.arange(4) / 30, -.12, TagPoseQuality(2., 2.))
        self.quick_results = {0: found_pair(), 3: found_pair()}
        self.complete_result = found_pair()
        self.quick_calls, self.recovery_calls = [], []

        def frames(video, times, indices):
            for index in np.sort(indices):
                frame = SimpleNamespace(index=int(index))
                yield int(index), float(times[index]), frame

        def detect(frame, detector, roi, converter):
            index = frame.index
            if detector._fast:
                self.quick_calls.append(index)
                return self.quick_results.get(index, EMPTY)
            self.recovery_calls.append(index)
            return self.complete_result

        self.addCleanup(patch.stopall)
        patch.object(sampled_detection, 'sampled_frames', side_effect=frames).start()
        patch.object(sampled_detection, 'detect_frame', side_effect=detect).start()

    def process(self, fraction=.5):
        return tag_distance.process(replace(self.request, min_valid_fraction=fraction), self.root)

    def test_scans_every_output_frame_before_accepting_exactly_fifty_percent(self):
        result = self.process()
        self.assertEqual(self.quick_calls, [0, 1, 2, 3])
        self.assertEqual(self.recovery_calls, [])
        self.assertEqual(result.report['valid_fraction'], .5)
        self.assertTrue(result.report['acceptance']['passed'])
        np.testing.assert_array_equal(result.samples.valid, [True, False, False, True])
        np.testing.assert_allclose(result.samples.times_s, self.request.video_times_s - .12)

    def test_recovery_stops_when_required_count_is_reached(self):
        self.quick_results = {0: found_pair()}
        result = self.process()
        self.assertEqual(len(self.quick_calls), 4)
        self.assertEqual(len(self.recovery_calls), 1)
        self.assertEqual(result.report['acceptance']['valid_rows'], 2)
        self.assertEqual(result.report['detection']['recovery_attempted'], 1)

    def test_increasing_threshold_resumes_recovery_without_rescanning_or_repeating_frames(self):
        self.quick_results = {0: found_pair()}
        self.process()
        recovered_before = list(self.recovery_calls)
        result = self.process(.75)
        self.assertEqual(len(self.quick_calls), 4)
        self.assertEqual(len(self.recovery_calls), 2)
        self.assertNotEqual(self.recovery_calls[-1], recovered_before[0])
        self.assertEqual(result.report['acceptance']['valid_rows'], 3)
        self.assertEqual(result.report['acceptance']['minimum_valid_fraction'], .75)
        warmed = self.process(.75)
        np.testing.assert_array_equal(warmed.samples.distance_mm, result.samples.distance_mm)
        self.assertEqual(len(self.recovery_calls), 2)

    def test_below_threshold_fails_and_reuses_exhausted_detection_cache(self):
        self.quick_results = {}
        self.complete_result = EMPTY
        for _ in range(2):
            with self.assertRaises(tag_distance.TagDistanceAcceptanceError) as raised:
                self.process()
            self.assertEqual(raised.exception.report['valid_fraction'], 0.)
            self.assertIn('0/4', str(raised.exception))
            self.assertFalse(raised.exception.report['acceptance']['passed'])
        self.assertEqual(len(self.quick_calls), 4)
        self.assertEqual(len(self.recovery_calls), 4)

    def test_duplicate_ids_cannot_be_rescued_by_another_pass_missing_one_duplicate(self):
        pair = found_pair()
        duplicated = detection.FrameDetection((pair.corners[0], pair.corners[0] + 50, pair.corners[1]),
                                              np.array([[0], [0], [1]]), {1: pair.centres[1]})
        self.quick_results = {index: duplicated for index in range(4)}
        with self.assertRaises(tag_distance.TagDistanceAcceptanceError) as raised:
            self.process()
        self.assertEqual(self.recovery_calls, [])
        self.assertEqual(raised.exception.report['sources'], {'ambiguous_detection': 4})

    def test_detected_pair_with_invalid_geometry_does_not_count_toward_threshold(self):
        corners, _ = projected()
        corners[0, 1:] = corners[0, 0]
        invalid = detection.FrameDetection(tuple(polygon[None] for polygon in corners), np.array([[0], [1]]),
                                           {tag: corners[tag].mean(axis=0) for tag in (0, 1)})
        self.quick_results = {index: invalid for index in range(4)}
        self.complete_result = invalid
        with self.assertRaises(tag_distance.TagDistanceAcceptanceError) as raised:
            self.process()
        self.assertEqual(raised.exception.report['acceptance']['valid_rows'], 0)
        self.assertNotIn('two_tags', raised.exception.report['sources'])

    def test_changed_output_times_have_a_separate_detection_cache(self):
        self.process()
        self.request = replace(self.request, video_times_s=np.arange(4) / 40)
        self.process()
        self.assertEqual(len(self.quick_calls), 8)

    def test_changed_roi_has_a_separate_corner_cache_and_json_safe_identity(self):
        self.process()
        self.request = replace(self.request, roi_px=(550, 950, 1400, 1300))
        result = self.process()
        self.assertEqual(len(self.quick_calls), 8)
        self.assertEqual(result.report['inputs']['roi_px'], [550, 950, 1400, 1300])

    def test_tampered_quick_cache_is_rejected(self):
        self.process()
        next((self.root / 'sampled_tags').glob('*/quick.npz')).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, '哈希'):
            self.process()

    def test_frame_clock_implementation_change_invalidates_sampled_corner_cache(self):
        self.process()
        original_hash = sampled_detection.artifact_identity.source_hash

        def changed_hash(path):
            return 'new-frame-clock' if str(path).endswith('camera/x3/frame.py') else original_hash(path)

        with patch.object(sampled_detection.artifact_identity, 'source_hash', side_effect=changed_hash):
            self.process()
        self.assertEqual(len(self.quick_calls), 8)

    def test_threshold_uses_ceiling_for_odd_output_count(self):
        self.request = replace(self.request, video_times_s=np.arange(5) / 30)
        result = self.process()
        self.assertEqual(result.report['acceptance']['valid_rows'], 3)
        self.assertEqual(result.report['valid_fraction'], .6)

    def test_decimal_threshold_at_exact_ratio_does_not_require_an_extra_frame(self):
        self.request = replace(self.request, video_times_s=np.arange(50) / 30, max_missing_frames=50)
        self.quick_results = {index: found_pair() for index in range(7)}
        result = self.process(.14)
        self.assertEqual(result.report['valid_fraction'], .14)
        self.assertEqual(self.recovery_calls, [])

    def test_coverage_pass_still_recovers_an_overlong_gap_and_stops_after_bridging_it(self):
        self.request = replace(self.request, video_times_s=np.arange(8) / 30, max_missing_frames=2)
        self.quick_results = {index: found_pair() for index in (0, 1, 6, 7)}
        result = self.process()
        self.assertEqual(len(self.quick_calls), 8)
        self.assertGreater(len(self.recovery_calls), 0)
        self.assertLess(len(self.recovery_calls), 4)
        self.assertLessEqual(result.report['acceptance']['longest_missing_frames'], 2)

    def test_unrecoverable_tail_fails_even_with_more_than_fifty_percent_valid(self):
        self.request = replace(self.request, video_times_s=np.arange(8) / 30, max_missing_frames=2)
        self.quick_results = {index: found_pair() for index in range(5)}
        self.complete_result = EMPTY
        with self.assertRaisesRegex(tag_distance.TagDistanceAcceptanceError, '连续缺失 3 帧.*上限 2 帧') as raised:
            self.process()
        self.assertEqual(raised.exception.report['valid_fraction'], .625)
        self.assertEqual(raised.exception.report['acceptance']['overlong_gaps'][0]['start_row'], 5)

    def test_stricter_gap_limit_resumes_cached_recovery_without_fast_rescan(self):
        self.request = replace(self.request, video_times_s=np.arange(8) / 30, max_missing_frames=4)
        self.quick_results = {index: found_pair() for index in (0, 1, 6, 7)}
        self.process()
        self.assertEqual(self.recovery_calls, [])
        self.request = replace(self.request, max_missing_frames=2)
        result = self.process()
        self.assertEqual(len(self.quick_calls), 8)
        self.assertGreater(len(self.recovery_calls), 0)
        self.assertTrue(result.report['acceptance']['passed'])

    def test_failed_contiguous_window_stops_without_trying_unrelated_missing_frames(self):
        self.request = replace(self.request, video_times_s=np.arange(64) / 30, max_missing_frames=5)
        self.quick_results = {index: found_pair() for index in range(40)}
        self.complete_result = EMPTY
        with self.assertRaises(tag_distance.TagDistanceAcceptanceError) as raised:
            self.process()
        self.assertEqual(len(self.quick_calls), 64)
        self.assertEqual(len(self.recovery_calls), 6)
        self.assertEqual(raised.exception.report['detection']['stop_reason'], 'unrecoverable_gap')
        self.assertEqual(raised.exception.report['acceptance']['longest_missing_frames'], 24)
        # 已有补检证据已足以证明超限，重跑不再做无关补检。
        with self.assertRaises(tag_distance.TagDistanceAcceptanceError):
            self.process()
        self.assertEqual(len(self.recovery_calls), 6)


class SourceFrameSelectionTests(unittest.TestCase):
    def source(self, pts):
        frames = [SimpleNamespace(pts=value, time_base=Fraction(1, 60000)) for value in pts]
        stream = SimpleNamespace(codec_context=SimpleNamespace(width=3840, height=1920),
                                 time_base=Fraction(1, 60000))
        container = MagicMock()
        container.__enter__.return_value = container
        container.streams.video = [stream]
        container.decode.return_value = iter(frames)
        return container, frames

    def test_nearest_pts_ties_use_earlier_frame_and_never_extrapolate(self):
        container, frames = self.source([0, 1001, 2002])
        period = 1001 / 60000
        queries = np.array([-.001, period / 2, period, 2 * period + .0001])
        with patch('av.open', return_value=container):
            selected = list(sampled_detection.sampled_frames(Path('raw.insv'), queries, np.arange(4)))
        self.assertIsNone(selected[0][2])
        self.assertIs(selected[1][2], frames[0])
        self.assertIs(selected[2][2], frames[1])
        self.assertIsNone(selected[3][2])

    def test_video_gap_does_not_assign_a_distant_source_frame(self):
        container, _ = self.source([0, 1001, 6000])
        queries = np.array([.05])
        with patch('av.open', return_value=container):
            selected = list(sampled_detection.sampled_frames(Path('raw.insv'), queries, np.array([0])))
        self.assertIsNone(selected[0][2])

    def test_duplicate_source_pts_are_rejected(self):
        container, _ = self.source([0, 1001, 1001])
        with patch('av.open', return_value=container), self.assertRaisesRegex(ValueError, '非递增'):
            list(sampled_detection.sampled_frames(Path('raw.insv'), np.array([.1]), np.array([0])))

    def test_late_recovery_seeks_before_target_and_preserves_nearest_source_pts(self):
        container, frames = self.source([600000, 601001, 602002])
        query = np.array([601001 / 60000])
        with patch('av.open', return_value=container):
            selected = list(sampled_detection.sampled_frames(Path('raw.insv'), query, np.array([0])))
        requested = container.seek.call_args.args[0]
        self.assertLess(requested, frames[0].pts)
        self.assertTrue(container.seek.call_args.kwargs['backward'])
        self.assertIs(selected[0][2], frames[1])
        self.assertEqual(selected[0][1], query[0])

    def test_recovery_prioritizes_known_tags_and_spreads_queries(self):
        corners = np.full((64, 2, 4, 2), np.nan)
        corners[32:, 0] = 1.
        order = sampled_detection.recovery_order(np.arange(64), corners)
        self.assertEqual(set(order), set(range(64)))
        self.assertTrue((order[:32] >= 32).all())
        self.assertGreater(int(np.ptp(order[:8])), 20)


class TagDistanceAcceptanceTests(unittest.TestCase):
    def test_default_threshold_and_configuration_bounds(self):
        self.assertEqual(inputs.load_scene_parameters().tag_distance_min_valid_fraction, .5)
        self.assertEqual(inputs.load_scene_parameters().tag_distance_max_missing_frames, 15)
        self.assertEqual(inputs.load_scene_parameters().tag_distance_roi_px, [600, 980, 1350, 1250])
        for value in (0., -1., 1.01, float('nan'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                inputs.load_scene_parameters(overrides={'tag_distance_min_valid_fraction': value})
        for value in (-1, 1.5, True):
            with self.subTest(gap=value), self.assertRaises(ValueError):
                inputs.load_scene_parameters(overrides={'tag_distance_max_missing_frames': value})
        self.assertEqual(inputs.load_scene_parameters(overrides={
            'tag_distance_max_missing_frames': 0}).tag_distance_max_missing_frames, 0)
        for roi in ([1, 2, 3], [600., 980, 1350, 1250], [1350, 980, 600, 1250], [0, 0, 2000, 1920]):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                inputs.load_scene_parameters(overrides={'tag_distance_roi_px': roi})

    def test_cli_and_scene_share_the_requested_default_roi(self):
        from software.calibration.cli import build_parser
        for command in ('preview-width', 'detect-width'):
            parsed = build_parser().parse_args([command, 'raw.insv'])
            self.assertEqual(tuple(parsed.roi), (600, 980, 1350, 1250))

    def test_actual_quick_and_recovery_search_do_not_escape_configured_roi(self):
        from test_marker_detection import bright_markers
        for positions, succeeds in [(((0, 700, 1050), (1, 1000, 1050), (0, 100, 100)), True),
                                     (((0, 700, 1050), (1, 1600, 1050)), False)]:
            with self.subTest(positions=positions), tempfile.TemporaryDirectory() as folder:
                half = bright_markers(positions, shape=(1920, 1920))
                pixels = np.column_stack((np.zeros_like(half), half))
                frame = av.VideoFrame.from_ndarray(pixels, format='gray')
                request = TagDistanceRequest(Path('raw.insv'), 'recording', SIZES, 'sizes', CAMERA, 'camera',
                                             np.arange(2) / 30, 0., TagPoseQuality(2., 2.))

                def frames(video, times, indices):
                    return iter((int(index), float(times[index]), frame) for index in np.sort(indices))

                with patch.object(sampled_detection, 'sampled_frames', side_effect=frames):
                    if succeeds:
                        result = tag_distance.process(request, Path(folder))
                        self.assertEqual(result.report['valid_fraction'], 1.)
                        self.assertTrue((result.samples.distance_mm > 20).all())
                    else:
                        with self.assertRaises(tag_distance.TagDistanceAcceptanceError) as raised:
                            tag_distance.process(request, Path(folder))
                        self.assertEqual(raised.exception.report['acceptance']['valid_rows'], 0)

    def test_gap_boundaries_include_leading_trailing_and_all_missing_frames(self):
        for valid, expected in [([False, False, True, False], [[0, 2], [3, 4]]),
                                ([False, False], [[0, 2]]), ([True, True], [])]:
            with self.subTest(valid=valid):
                np.testing.assert_array_equal(tag_distance.missing_runs(np.array(valid)),
                                              np.array(expected, dtype=int).reshape(-1, 2))

    def test_exactly_fifteen_missing_frames_pass_and_sixteen_fail_at_high_coverage(self):
        request = TagDistanceRequest(Path('raw.insv'), 'recording', SIZES, 'sizes', CAMERA, 'camera',
                                     np.arange(40) / 30, 0., TagPoseQuality(2., 2.))
        for count, expected in [(15, True), (16, False)]:
            valid = np.ones(40, dtype=bool)
            valid[:count] = False
            report = tag_distance.acceptance_report(valid, request)
            self.assertEqual(report['passed'], expected)
            self.assertEqual(report['longest_missing_frames'], count)

    def test_each_arm_must_pass_and_failure_preserves_previous_success(self):
        import test_scene
        from software.scene import workflow
        fixture = test_scene.SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build_scene()
        workflow.run_episode(fixture.episode_path)
        output = fixture.task_root / 'results'
        previous = output.resolve()
        original = (previous / 'episode.json').read_bytes()
        for video, camera in list(fixture.cameras.items()):
            path = camera.files['front'].parent / 'tags.json'
            write_json(path, SIZES)
            fixture.cameras[video] = replace(camera, files=dict(camera.files, tag_distance=path),
                                            tag_distance_measurement=SIZES)
        report = dict(valid_fraction=.4, rows=10, sources={'two_tags': 4, 'missing': 6},
                      acceptance=dict(minimum_valid_fraction=.5, valid_rows=4, passed=False,
                                      maximum_missing_frames=15, longest_missing_frames=6,
                                      longest_missing_interval_s=[0., .2]))
        calls = []

        def measure(request, cache, **kwargs):
            calls.append(request)
            if request.video == fixture.recording_paths['right']:
                raise tag_distance.TagDistanceAcceptanceError(report)
            times = request.video_times_s + request.episode_offset_s
            return TagDistanceResult(TagDistanceSeries(times, np.full(len(times), 50.),
                                                       np.full(len(times), 'two_tags')), {})

        with patch.object(tag_distance, 'process', side_effect=measure), \
                self.assertRaisesRegex(ValueError, 'right.*40.0%.*50.0%'):
            workflow.run_episode(fixture.episode_path)
        self.assertEqual(output.resolve(), previous)
        self.assertEqual((previous / 'episode.json').read_bytes(), original)
        self.assertEqual(len(calls), 2)
        diagnostics = list((layout.metadata(output.parent, 'runs')).glob('failed-*/right/tag_distance.json'))
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(json.loads(diagnostics[0].read_text())['acceptance']['passed'], False)
