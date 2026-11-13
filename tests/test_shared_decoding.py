"""共享原帧、独立缓存和失败发布；使用真实双镜头视频检测。"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import tempfile
import unittest

import av
import numpy as np

from software import provenance
from software.gripper_width import detection, sampled_detection
from software.scene import layout
from software.scene.management import inputs
from software.scene.pipeline import measurements
from software.storage import write_yaml
from software.vision import tag_board, tags
from software.vision.decoding import VideoDecode
from test_environment_tags import poster, recording
from test_marker_detection import bright_markers


class SharedDecodingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.video = self.root / 'raw.mkv'
        front = bright_markers(((0, 700, 1050), (1, 1000, 1050)), shape=(1920, 1920))
        recording(self.video, poster([(12, 500, 500), (88, 900, 500)]), front=front, count=6)
        with av.open(str(self.video)) as container:
            self.frames = list(container.decode(video=0))
        self.times = np.array([float(frame.pts * frame.time_base) for frame in self.frames])
        self.queries = self.times[::2]
        self.digest = provenance.file_hash(self.video)
        self.settings = tags.DetectionSettings(border_bits=1, ids=(12, 88))

    def prepare(self, cache, *, settings=None, queries=None):
        decoder = VideoDecode(self.video)
        detection.cached(self.video, cache / 'width', self.digest, decoder=decoder)
        sampled = sampled_detection.SampledDetectionCache(
            self.video, cache, self.digest, self.queries if queries is None else queries, None)
        sampled.load(decoder=decoder)
        tags.cached(self.video, cache / 'poster', settings or self.settings, decoder=decoder)
        return decoder, sampled

    def test_shared_outputs_are_byte_identical_to_independent_detection(self):
        cache = self.root / 'shared'
        decoder, sampled = self.prepare(cache)
        with patch('av.open', wraps=av.open) as opened:
            decoder.run()
        self.assertEqual(opened.call_count, 1)
        independent = self.root / 'independent'
        detection.cached(self.video, independent / 'width', self.digest)
        reference = sampled_detection.SampledDetectionCache(self.video, independent, self.digest, self.queries, None)
        reference.load()
        tags.cached(self.video, independent / 'poster', self.settings)
        for actual, expected in (
            (next((cache / 'width').glob('*/detections.npz')), next((independent / 'width').glob('*/detections.npz'))),
            (sampled.directory / 'quick.npz', reference.directory / 'quick.npz'),
            (cache / 'poster/observations.npz', independent / 'poster/observations.npz'),
        ):
            self.assertEqual(actual.read_bytes(), expected.read_bytes())
        self.assertTrue(np.isfinite(sampled.quick_corners).all())
        with patch('av.open', side_effect=AssertionError('warm cache must not open video')):
            warmed, _ = self.prepare(cache)
            self.assertEqual(warmed.run(), [])

    def test_poster_change_does_not_invalidate_gripper_caches(self):
        cache = self.root / 'partial'
        decoder, sampled = self.prepare(cache)
        decoder.run()
        paths = [next((cache / 'width').glob('*/detections.npz')), sampled.directory / 'quick.npz']
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        with patch.object(detection.WidthDetection, 'consume', side_effect=AssertionError('width cache')), \
                patch.object(sampled_detection.QuickDetection, 'consume', side_effect=AssertionError('quick cache')), \
                patch('av.open', wraps=av.open) as opened:
            changed, _ = self.prepare(cache, settings=replace(self.settings, frame_step=3))
            changed.run()
        self.assertEqual(opened.call_count, 1)
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}, before)
        with np.load(cache / 'poster/observations.npz') as observations:
            np.testing.assert_array_equal(observations['source_frame_indices'], [0, 3])

    def test_query_change_only_rescans_sampled_detection(self):
        cache = self.root / 'partial'
        decoder, original = self.prepare(cache)
        decoder.run()
        with patch.object(detection.WidthDetection, 'consume', side_effect=AssertionError('width cache')), \
                patch.object(tags.PosterDetection, 'consume', side_effect=AssertionError('poster cache')), \
                patch('av.open', wraps=av.open) as opened:
            changed, sampled = self.prepare(cache, queries=self.times[1::2])
            changed.run()
        self.assertEqual(opened.call_count, 1)
        self.assertNotEqual(original.directory, sampled.directory)
        np.testing.assert_array_equal(sampled.native_times_s, self.times[1::2])

    def test_shared_nearest_pts_never_extrapolates_and_ties_choose_previous(self):
        # Matroska 时间戳取整到毫秒；使用 16 ms 间隔，使中点在半帧容差内。
        queries = np.array([self.times[0] - .001, (self.times[1] + self.times[2]) / 2,
                            self.times[2], self.times[-1] + .001])
        decoder, sampled = self.prepare(self.root / 'pts', queries=queries)
        decoder.run()
        self.assertTrue(np.isnan(sampled.native_times_s[[0, 3]]).all())
        np.testing.assert_array_equal(sampled.native_times_s[[1, 2]], self.times[[1, 2]])
        self.assertTrue(np.isnan(sampled.quick_corners[[0, 3]]).all())

    def test_decode_failure_does_not_publish_any_detection_cache(self):
        cache = self.root / 'failed'
        decoder, _ = self.prepare(cache)
        container = MagicMock()
        container.__enter__.return_value = container
        container.streams.video = [SimpleNamespace(codec_context=SimpleNamespace(width=3840, height=1920),
                                                  average_rate=tags.FPS_SRC)]

        def broken():
            yield self.frames[0]
            raise RuntimeError('truncated decode')

        container.decode.return_value = broken()
        with patch('av.open', return_value=container), self.assertRaisesRegex(RuntimeError, 'truncated'):
            decoder.run()
        self.assertFalse(list(cache.rglob('*.npz')))
        self.assertFalse(list(cache.rglob('*.json')))

    def test_consumer_validation_failure_does_not_publish_other_caches(self):
        cache = self.root / 'invalid'
        decoder, _ = self.prepare(cache)
        with patch.object(tags.PosterDetection, 'finish', side_effect=ValueError('invalid observations')), \
                self.assertRaisesRegex(ValueError, 'invalid observations'):
            decoder.run()
        self.assertFalse(list(cache.rglob('*.npz')))

    def test_scene_prepares_all_caches_then_consumers_need_no_more_decode(self):
        batch = self.root / 'shelf/batch_001'
        write_yaml(batch / 'batch.yaml', {})
        episode = SimpleNamespace(root=batch / 'take_001', recordings=dict(left=self.video),
            recording_refs=dict(left=dict(sha256=self.digest)),
            scene=SimpleNamespace(tag_layout=tag_board.DEFAULT_LAYOUT, parameters=inputs.load_scene_parameters()))
        prepared = SimpleNamespace(camera=SimpleNamespace(files=dict(width=Path('width.json'), tag_distance=Path('tags.json'))),
                                   video_times_s=self.queries)
        with patch('av.open', wraps=av.open) as opened:
            measurements.prepare_detections(episode, 'left', prepared)
        self.assertEqual(opened.call_count, 1)
        cache = layout.metadata(episode.root, 'cache')
        with patch('av.open', side_effect=AssertionError('already prepared')):
            measurements.prepare_detections(episode, 'left', prepared)
            self.assertIsNotNone(detection.cached(self.video, cache / 'gripper_width', self.digest))
            sampled_detection.SampledDetectionCache(self.video, cache, self.digest, self.queries, None).load()
            board = tag_board.load(tag_board.DEFAULT_LAYOUT)
            result = tags.cached(self.video, cache / 'environment_tags/left',
                tags.DetectionSettings(border_bits=board.border_bits, ids=board.ids, dictionary=board.dictionary,
                                       minimum_side_px=episode.scene.parameters.tag_minimum_side_px), layout=board)
        self.assertTrue(result['cache_reused'])
