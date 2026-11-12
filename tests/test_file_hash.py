"""大文件摘要按文件系统身份复用；内容或时间变化后必须重算。"""

from pathlib import Path
from unittest.mock import patch
import hashlib
import os
import tempfile
import unittest

from software import provenance


class FileHashTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'recording.bin'
        self.path.write_bytes(b'a' * 64)
        threshold = patch.object(provenance, 'LARGE_FILE_BYTES', 32)
        threshold.start()
        self.addCleanup(threshold.stop)
        self.addCleanup(provenance._large_file_hashes.clear)

    def test_large_file_is_read_once_until_it_changes(self):
        expected = hashlib.sha256(b'a' * 64).hexdigest()
        with patch.object(hashlib, 'file_digest', wraps=hashlib.file_digest) as digest:
            self.assertEqual(provenance.file_hash(self.path), expected)
            self.assertEqual(provenance.file_hash(self.path), expected)
            self.assertEqual(digest.call_count, 1)
            self.path.write_bytes(b'b' * 64)
            stat = self.path.stat()
            os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            self.assertEqual(provenance.file_hash(self.path), hashlib.sha256(b'b' * 64).hexdigest())
            self.assertEqual(digest.call_count, 2)

    def test_small_file_is_always_read(self):
        small = self.path.with_name('kit.json')
        small.write_bytes(b'{}')
        with patch.object(hashlib, 'file_digest', wraps=hashlib.file_digest) as digest:
            provenance.file_hash(small)
            provenance.file_hash(small)
            self.assertEqual(digest.call_count, 2)


class CodeIdentityScopeTests(unittest.TestCase):
    def test_presentation_and_orchestration_files_do_not_invalidate_results(self):
        from software.scene.management import inputs
        trajectory = provenance.code_identity('trajectory')
        self.assertFalse(provenance.PRESENTATION_SOURCES & trajectory.keys())
        self.assertIn('software/trajectory/algorithms/fusion_graph.py', trajectory)
        scene = inputs.algorithm_identity()
        for name in ('processing.py', 'diagnostics.py', 'cli.py'):
            self.assertFalse([path for path in scene if path.endswith('/' + name) and '/scene/' in path], name)
        self.assertIn('software/scene/spatial/fusion.py', scene)

    def test_task_caches_follow_their_stage_sources(self):
        from software.scene.map_cache import source_identity
        prepare, localize = source_identity('prepare'), source_identity('localize')
        self.assertNotIn('software/trajectory/algorithms/fusion_graph.py', localize)
        self.assertLessEqual(prepare.keys(), localize.keys())
        self.assertIn('software/scene/spatial/relocalization.py', localize)


if __name__ == '__main__':
    unittest.main()
