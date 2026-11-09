"""Only batches supply mapping inputs; the CLI has no second scene configuration writer."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from software.scene import cli


class MappingArgumentsTests(unittest.TestCase):
    def test_build_accepts_a_batch_without_duplicate_mapping_arguments(self):
        args = cli.build_parser().parse_args(['build', 'scenes/shelf/morning', '--dry-run'])
        self.assertEqual(args.batch, Path('scenes/shelf/morning'))
        self.assertTrue(args.dry_run)

    def test_old_scene_writers_and_duplicate_mapping_options_are_removed(self):
        cases = [['init', 'scene'], ['session', 'scene', '--name', 'morning', '--kit', 'rig'],
                 ['kit', 'rig', '--left', 'SERIAL'], ['kits'],
                 ['batch', 'scene', '--name', 'morning', '--kit', 'rig'],
                 ['episode', 'batch', '--name', 'take_001'],
                 ['build', 'batch', '--mapping', 'one.insv'], ['build', 'batch', '--mode', 'dual'],
                 ['ingest', 'sd', '--session', 'batch'],
                 ['ingest', 'sd', '--batch', 'batch'],
                 ['task', 'batch', '--name', 'take_001']]
        for command in cases:
            with self.subTest(command=command), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.build_parser().parse_args(command)
            self.assertEqual(error.exception.code, 2)

    def test_build_without_a_batch_does_not_write_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'scene'
            self.assertEqual(cli.main(['build', str(root), '--dry-run']), 2)
            self.assertFalse(root.exists())
