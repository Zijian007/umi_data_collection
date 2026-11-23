"""Portable reader boundary: loading a package must not import the solve stack."""

from __future__ import annotations

import subprocess
import sys
import unittest


class DatasetReaderBoundaryTests(unittest.TestCase):
    def test_reader_import_does_not_load_scene_or_solver_dependencies(self):
        probe = (
            "import sys; import umi.dataset.reader; "
            "print([name for name in ('gtsam', 'cv2', 'scipy', "
            "'umi.workflows.scene.workflow', 'umi.workflows.scene.episodes', "
            "'umi.dataset.export') if name in sys.modules])"
        )
        result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), '[]')


if __name__ == '__main__':
    unittest.main()
