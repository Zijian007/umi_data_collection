"""海报检测的真实 ArUco 图像、视频时钟、镜头边界和缓存回归。"""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import cv2
import numpy as np

from software import config
from software.scene.cli import build_parser
from software.scene.management.inputs import load_scene_parameters
from software.scene.spatial import fusion
from software.scene.spatial.geometry import VisualObservations
from software.vision import tags


def poster(markers):
    image = np.full((1920, 1920), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    for tag_id, x, y in markers:
        image[y:y+120, x:x+120] = cv2.aruco.generateImageMarker(dictionary, tag_id, 120)
    return image


def recording(path, rear, *, front=None, count=4):
    frame_image = np.concatenate([rear, front if front is not None else np.full_like(rear, 255)], axis=1)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('ffv1', rate=Fraction(60000, 1001))
        stream.width, stream.height, stream.pix_fmt = 3840, 1920, 'gray'
        for i in range(count):
            frame = av.VideoFrame.from_ndarray(frame_image, format='gray')
            frame.pts, frame.time_base = i + 60, Fraction(1001, 60000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class PosterDetectionTests(unittest.TestCase):
    def test_detects_ids_beyond_gripper_dictionary_and_rejects_duplicates(self):
        image = poster([(12, 400, 400), (88, 700, 400), (25, 1000, 400), (25, 1300, 400), (99, 400, 800)])
        found, duplicate, unknown = tags.PosterDetector(tags.DetectionSettings(border_bits=1)).detect(image)
        self.assertEqual([item[0] for item in found], [12, 88])
        self.assertEqual((duplicate, unknown), (1, 1))
        np.testing.assert_allclose(found[0][1], [[400, 400], [519, 400], [519, 519], [400, 519]], atol=1)

    def test_real_video_left_half_pts_partial_visibility_and_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / 'packed.mkv'
            recording(video, poster([(12, 400, 400), (88, 100, 100)]), front=poster([(55, 700, 700)]))
            settings = tags.DetectionSettings(border_bits=1)
            result = tags.detect(video, settings)
            np.testing.assert_array_equal(result.source_frame_indices, [0, 2])
            self.assertGreater(result.video_times_s[0], 1.)
            self.assertEqual(result.tag_ids.tolist(), [12, 88, 12, 88])
            self.assertEqual(result.usable.tolist(), [True, False, True, False])
            np.testing.assert_allclose(result.corners_px[0, 0], [160, 160], atol=1)
            np.testing.assert_allclose(result.imu_times_s(12000000), result.video_times_s[result.sample_indices]+.012)
            report = tags.report(result, settings)
            self.assertEqual(report['sampled_frames'], 2)
            self.assertEqual(report['at_least_one_usable']['frames'], 2)
            self.assertEqual(report['usable_ids_per_frame'], [1, 1])
            self.assertEqual(report['outside_calibrated_crop_fraction'], .5)
            self.assertEqual(report['per_id']['55']['detected']['frames'], 0)

    def test_blank_video_and_sampled_gap_include_start_and_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / 'blank.mkv'
            recording(video, poster([]))
            settings = tags.DetectionSettings(border_bits=1)
            result = tags.detect(video, settings)
            self.assertEqual(result.corners_px.shape, (0, 4, 2))
            report = tags.report(result, settings)
            self.assertEqual(report['at_least_one_usable']['fraction'], 0)
            self.assertIsNone(report['outside_calibrated_crop_fraction'])
            self.assertAlmostEqual(report['at_least_one_usable']['maximum_sampled_gap_s'],
                result.video_times_s[-1]-result.video_times_s[0]+2/tags.FPS_SRC)

    def test_cache_checks_source_parameters_and_payload_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video, output = root / 'source.mkv', root / 'detection'
            recording(video, poster([(88, 500, 500)]))
            settings = tags.DetectionSettings(border_bits=1)
            first = tags.cached(video, output, settings)
            self.assertFalse(first['cache_reused'])
            with patch.object(tags, 'detect', side_effect=AssertionError('cache must avoid decoding')):
                second = tags.cached(video, output, settings)
            self.assertTrue(second['cache_reused'])
            self.assertEqual(first['identity_hash'], second['identity_hash'])
            changed = tags.cached(video, output, replace(settings, minimum_side_px=200.))
            self.assertFalse(changed['cache_reused'])
            self.assertEqual(changed['at_least_one_usable']['fraction'], 0.)
            (output / 'observations.npz').write_bytes(b'broken')
            self.assertFalse(tags.cached(video, output, replace(settings, minimum_side_px=200.))['cache_reused'])
            recording(video, poster([(12, 500, 500)]))
            latest = tags.cached(video, output, settings)
            self.assertFalse(latest['cache_reused'])
            self.assertNotEqual(first['identity_hash'], latest['identity_hash'])
            self.assertEqual(latest['per_id']['88']['usable']['frames'], 0)

    def test_rejects_bad_settings_and_requires_border_on_cli(self):
        for kwargs in (dict(border_bits=0), dict(border_bits=True), dict(border_bits=1, frame_step=0),
                       dict(border_bits=1, minimum_side_px=float('nan')), dict(border_bits=1, ids=(100,)),
                       dict(border_bits=1, ids=(12, 12)), dict(border_bits=1, dictionary='DICT_4X4_50')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                tags.DetectionSettings(**kwargs)
        args = build_parser().parse_args(['detect-tags', 'raw.insv', '--out', '/tmp/poster', '--border-bits', '1'])
        self.assertEqual((args.frame_step, args.border_bits), (2, 1))


class FusionBeforeTagsTests(unittest.TestCase):
    def test_refactoring_preserves_saved_prechange_solution(self):
        with np.load(Path(__file__).parent / 'fixtures/scene_fusion_before_tags.npz', allow_pickle=False) as data:
            times, poses, imu, expected = (data[key] for key in ('times', 'poses', 'imu', 'expected'))
        observation = VisualObservations(times, poses, np.arange(len(times)), 'scene')
        result = fusion.fuse(times, imu, {'front': observation}, {'front': np.eye(4)},
                             config.load('trajectory').values, load_scene_parameters(None))
        np.testing.assert_allclose(result.T_reference_imu, expected, rtol=0, atol=1e-10)
        self.assertTrue(result.tracked.all())
        self.assertEqual(result.evidence['actual_error'], 'not_measured')
