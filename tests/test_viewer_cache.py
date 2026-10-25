"""查看器缓存跨重启复用、失效、原子发布和后台任务去重。"""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from software.viewer import cache, maps, media


class ViewerCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        environment = patch.dict('os.environ', {'UMI_VIEW_CACHE': str(self.root/'cache')})
        environment.start(); self.addCleanup(environment.stop)
        self.video = self.root/'source.mp4'
        self.video.write_bytes(b'original video')
        media._decode_cached.cache_clear()
        maps._convert.cache_clear()
        self.addCleanup(media._decode_cached.cache_clear)
        self.addCleanup(maps._convert.cache_clear)

    def test_map_survives_memory_cache_reset_and_invalidates_on_input_change(self):
        first = dict(kind='umi_map_view', schema_version=1, maps=[], note='first')
        second = dict(first, note='second')
        identity = (cache.file_identity(self.video),)
        with patch.object(maps, '_export', return_value=first) as export:
            self.assertEqual(maps._convert(self.root, identity), first)
        export.assert_called_once()
        maps._convert.cache_clear()
        with patch.object(maps, '_export', side_effect=AssertionError('must reuse disk')):
            self.assertEqual(maps._convert(self.root, identity), first)
        self.video.write_bytes(b'updated video')
        with patch.object(maps, '_export', return_value=second) as export:
            self.assertEqual(maps._convert(self.root, (cache.file_identity(self.video),)), second)
        export.assert_called_once()

    def test_corrupt_map_cache_is_regenerated(self):
        data = dict(kind='umi_map_view', schema_version=1, maps=[])
        identity = (cache.file_identity(self.video),)
        target = cache.path('maps', identity, '.json')
        target.parent.mkdir(parents=True)
        target.write_text('{broken')
        with patch.object(maps, '_export', return_value=data):
            self.assertEqual(maps._convert(self.root, identity), data)
        self.assertEqual(json.loads(target.read_text()), data)

    def test_failed_publish_keeps_old_cache_and_removes_partial_file(self):
        target = self.root/'saved.json'
        target.write_bytes(b'complete')
        with self.assertRaises(RuntimeError):
            with cache.atomic_output(target) as temporary:
                temporary.write_bytes(b'partial')
                raise RuntimeError('interrupted')
        self.assertEqual(target.read_bytes(), b'complete')
        self.assertEqual(list(self.root.glob('.building-*')), [])

    def test_first_frame_survives_memory_reset_and_repeated_frames_are_cached(self):
        response = SimpleNamespace(returncode=0, stdout=b'\xff\xd8jpeg', stderr=b'')
        with patch.object(media, 'find_ffmpeg', return_value='/ffmpeg'), patch.object(media.subprocess, 'run', return_value=response) as run:
            media.decode_frame(self.video, 0)
            media._decode_cached.cache_clear()
            media.decode_frame(self.video, 0)
            self.assertEqual(run.call_count, 1)
            media.decode_frame(self.video, 1.5)
            media.decode_frame(self.video, 1.5)
            self.assertEqual(run.call_count, 2)
            self.video.write_bytes(b'changed input')
            media.decode_frame(self.video, 0)
            self.assertEqual(run.call_count, 3)

    def test_preview_work_is_deduplicated_and_ready_file_reused(self):
        entered, release = threading.Event(), threading.Event()
        def encode(video, target):
            entered.set(); release.wait(5)
            with cache.atomic_output(target) as temporary:
                temporary.write_bytes(b'\x1aE\xdf\xa3preview')
        with patch.object(media, 'find_ffmpeg', return_value='/ffmpeg'), patch.object(media, '_encode_preview', side_effect=encode) as encoder:
            try:
                self.assertEqual(media.preview_status(self.video)['status'], 'preparing')
                self.assertTrue(entered.wait(3))
                self.assertEqual(media.preview_status(self.video)['status'], 'preparing')
                self.assertEqual(encoder.call_count, 1)
            finally:
                release.set()
                media._jobs[media._preview_target(self.video)].result(timeout=5)
            state = media.preview_status(self.video)
            self.assertEqual(state['status'], 'ready')
            self.assertEqual(media.preview_file(self.video, state['version']).read_bytes(), b'\x1aE\xdf\xa3preview')
            with self.assertRaises(ValueError):
                media.preview_file(self.video, 'stale-version')
            self.assertEqual(encoder.call_count, 1)

    def test_failed_encoder_never_publishes_partial_preview(self):
        def run(command, **kwargs):
            Path(command[-1]).write_bytes(b'partial')
            return SimpleNamespace(returncode=1, stderr=b'encoder failed')
        with patch.object(media, 'find_ffmpeg', return_value='/ffmpeg'), patch.object(media.subprocess, 'run', side_effect=run):
            target = media._preview_target(self.video)
            with self.assertRaisesRegex(ValueError, 'encoder failed'):
                media._encode_preview(self.video, target)
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.glob('.building-*')), [])
        self.assertEqual(self.video.read_bytes(), b'original video')
