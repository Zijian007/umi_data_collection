import tempfile
import unittest
from pathlib import Path

from software.configuration import user_settings as user_config
from software.infrastructure.file_store import write_yaml


class UserConfigTests(unittest.TestCase):
    def test_missing_file_uses_safe_daily_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config = user_config.load(Path(directory) / 'missing.yaml')
        self.assertEqual(config.values['scene']['sync_policy'], 'assumed_simultaneous')
        self.assertEqual(config.values['runtime']['jobs'], 1)
        self.assertIsNone(config.values['viewer']['port'])

    def test_reads_only_user_facing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'user.yaml'
            write_yaml(path, dict(schema_version=1,
                                  scene=dict(sync_policy='audio_auto'),
                                  calibration=dict(intrinsics_source='factory'),
                                  viewer=dict(port=9000),
                                  runtime=dict(jobs=2)))
            config = user_config.load(path)
        self.assertEqual(config.values['scene']['sync_policy'], 'audio_auto')
        self.assertEqual(config.values['calibration']['intrinsics_source'], 'factory')
        self.assertEqual(config.values['viewer']['port'], 9000)
        self.assertEqual(config.values['runtime']['jobs'], 2)

    def test_rejects_algorithm_parameters_in_user_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'user.yaml'
            write_yaml(path, dict(schema_version=1, trajectory=dict(video=dict(frame_step=3))))
            with self.assertRaises(ValueError):
                user_config.load(path)

    def test_v2_feature_switches_normalize_to_internal_choices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'user.yaml'
            write_yaml(path, dict(
                schema_version=2,
                features=dict(
                    audio_sync=dict(enabled=True),
                    factory_intrinsics=dict(enabled=True),
                ),
                runtime=dict(parallel_tasks=dict(enabled=True, jobs=3), cores='0-7'),
                viewer=dict(host='127.0.0.1', port=9000),
            ))
            config = user_config.load(path)
        self.assertEqual(config.values['scene']['sync_policy'], 'audio_auto')
        self.assertEqual(config.values['calibration']['intrinsics_source'], 'factory')
        self.assertEqual(config.values['runtime'], {'jobs': 3, 'cores': '0-7'})
        self.assertEqual(config.values['viewer']['port'], 9000)

    def test_v2_disabled_parallel_tasks_force_one_job(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'user.yaml'
            write_yaml(path, dict(schema_version=2, runtime=dict(
                parallel_tasks=dict(enabled=False, jobs=8), cores='auto')))
            config = user_config.load(path)
        self.assertEqual(config.values['runtime']['jobs'], 1)
