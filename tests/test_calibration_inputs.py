"""Recording layout discovery must preserve stage identity and reject ambiguity."""
import tempfile
import unittest
from pathlib import Path

from software.calibration.inputs import discover_recordings


class RecordingInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def nested(self):
        for stage in "ABCD":
            directory = self.root / stage
            directory.mkdir()
            (directory / "same_original_name.insv").touch()

    def test_nested_layout_uses_stage_names(self):
        self.nested()
        (self.root / "umi_calib_out").mkdir()
        (self.root / "A" / "preview.lrv").touch()
        result = discover_recordings(self.root)
        self.assertEqual(list(result), list("ABCD"))
        for stage, video in result.items():
            self.assertEqual(video.parent.name, stage)

    def test_flat_layout_retains_filename_order(self):
        for name in ("04.insv", "02.insv", "03.insv", "01.INSV"):
            (self.root / name).touch()
        result = discover_recordings(self.root)
        self.assertEqual([p.name for p in result.values()], ["01.INSV", "02.insv", "03.insv", "04.insv"])

    def test_missing_stage_rejected(self):
        (self.root / "A").mkdir()
        (self.root / "A" / "video.insv").touch()
        with self.assertRaisesRegex(ValueError, "缺少 B"):
            discover_recordings(self.root)

    def test_empty_or_multiple_videos_rejected(self):
        self.nested()
        video = self.root / "A" / "same_original_name.insv"
        video.unlink()
        with self.assertRaisesRegex(ValueError, "0 个"):
            discover_recordings(self.root)
        video.touch()
        (self.root / "A" / "retry.insv").touch()
        with self.assertRaisesRegex(ValueError, "2 个"):
            discover_recordings(self.root)

    def test_mixed_layout_rejected(self):
        self.nested()
        (self.root / "extra.insv").touch()
        with self.assertRaisesRegex(ValueError, "同时包含"):
            discover_recordings(self.root)

    def test_missing_folder_rejected(self):
        with self.assertRaisesRegex(ValueError, "不存在"):
            discover_recordings(self.root / "missing")

    def test_flat_incorrect_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "需要正好 4 个"):
            discover_recordings(self.root)


if __name__ == "__main__":
    unittest.main()
