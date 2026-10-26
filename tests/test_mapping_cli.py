"""建图参数在归档原片或写配置前消除歧义。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from software.scene import cli


class MappingArgumentsTests(unittest.TestCase):
    def test_shared_mapping_uses_one_reference_without_assigning_task_handedness(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)/'scene'
            args = cli.build_parser().parse_args(['build', str(root), '--mapping', 'camera.insv'])
            with patch.object(cli.recordings, 'inspect', return_value=dict(serial='CAMERA', sha256='hash')), patch.object(cli.registry, 'pin', return_value={}):
                request = cli.mapping_request(args, root)
            self.assertEqual(request['recording_paths'], {'left': Path('camera.insv')})
            self.assertEqual(request['data']['mode'], 'shared_front')
            self.assertFalse(root.exists())

    def test_ambiguous_or_incomplete_interfaces_fail_before_reading_videos(self):
        cases = [
            ['--mapping', 'one.insv', '--mapping-left', 'two.insv'],
            ['--mapping-left', 'one.insv', '--mapping-right', 'two.insv'],
            ['--mode', 'dual', '--mapping', 'one.insv'],
            ['--mode', 'dual', '--mapping-left', 'one.insv'],
            ['--mapping', 'one.insv', '--rear-left', 'rear.json'],
        ]
        for extra in cases:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as name:
                args = cli.build_parser().parse_args(['build', str(Path(name)/'scene'), *extra])
                with patch.object(cli.recordings, 'inspect') as inspect:
                    with self.assertRaises(ValueError):
                        cli.mapping_request(args, Path(name)/'scene')
                    inspect.assert_not_called()
