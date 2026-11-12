"""单手回退适配器：真实阶段编排，外部引擎用文件协议替身。"""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from software import config
from software.scene.models import PreparedRecording
from software.scene.pipeline import fallback
from software.scene.management import inputs
from software.trajectory.calibration import assets
from software.trajectory.pipeline import runtime
from software.vision.models import LensVideo, RecordingPaths
from trajectory_fakes import FakeEngines


class SceneFallbackAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.engines = FakeEngines(self.root)
        self.paths = RecordingPaths(self.root / 'prepared')
        self.engines.write_dataset(self.paths.dataset)
        (self.paths.dataset / 'extract_manifest.json').write_text(json.dumps(dict(source=dict(name='right.insv'))))
        self.video = self.root / 'prepared-front.mp4'
        self.video.write_bytes(b'prepared video')
        self.camera = self.root / 'frozen-front.json'
        self.camera.write_bytes(assets.SAMPLE_CAMERA_JSON.read_bytes())
        configuration = config.load('trajectory')
        self.episode = SimpleNamespace(expected_arms=('right',), recordings={'right': self.root / 'right.insv'},
                                       scene=SimpleNamespace(trajectory=configuration,parameters=inputs.load_scene_parameters(),tag_layout=None))
        times = np.loadtxt(self.paths.dataset / 'mav0/cam0/data.csv', delimiter=',', usecols=0) / 1e9 - 1
        self.recording = PreparedRecording(
            SimpleNamespace(files={'front': self.camera, 'mask': runtime.HERE / 'assets/config/front_hand_mask.png'}),
            SimpleNamespace(paths=self.paths, cores='', parameters=configuration.values),
            (LensVideo('front', self.video, self.root / 'scene-settings.yaml', None),), times, np.empty((0, 7)))
        for mocked in (patch.dict(os.environ, self.engines.environment),
                       patch.object(runtime.subprocess, 'run', side_effect=self.engines),
                       patch.object(runtime.console, 'write')):
            mocked.start()
            self.addCleanup(mocked.stop)

    def run_fallback(self):
        return fallback.reconstruct(self.episode, 'right', self.recording, self.root / 'independent')

    def test_reuses_video_frozen_calibration_and_full_original_clock(self):
        result = self.run_fallback()
        self.assertEqual([kind for kind, *_ in self.engines.calls], ['map', 'localize', 'merge', 'fuse'])
        self.assertEqual(len(result.T_reference_imu), len(self.recording.video_times_s))
        np.testing.assert_allclose(result.T_reference_imu[:, 0, 3], np.arange(self.engines.frames) * .01)
        self.assertTrue(result.tracked.all())
        self.assertEqual(result.evidence['mode'], 'independent')
        for kind, command, _ in self.engines.calls:
            if kind in ('map', 'localize'):
                actual_video = Path(command[command.index('--input_video') + 1])
                self.assertTrue(actual_video.samefile(self.video))
                self.assertFalse(actual_video.is_symlink())
            if kind == 'fuse':
                self.assertEqual(Path(command[command.index('--calibration') + 1]), self.camera)
                times = np.loadtxt(command[command.index('--times') + 1], skiprows=1)
                np.testing.assert_allclose(times - 1, self.recording.video_times_s, rtol=0, atol=1e-9)
                self.assertNotIn('--refine-initialization', command)
                parameters = json.loads(Path(command[command.index('--parameters') + 1]).read_text())
                self.assertEqual(parameters['input']['mode'], 'handheld')
                self.assertEqual(parameters['fusion']['gravity_tolerance_m_s2'],
                                 self.episode.scene.trajectory.values.fusion.gravity_tolerance_m_s2)
        directory = self.root / 'independent'
        inputs.validate_assets(directory, inputs.artifact_hashes(directory, ('debug',)))

    def test_independent_quality_failure_is_not_published_as_success(self):
        self.engines.scale_changes = {1: 3.27, 2: 3.27, 3: 3.27}
        with self.assertRaisesRegex(ValueError, '独立模式也失败.*尺度'):
            self.run_fallback()
        record = json.loads((self.root / 'independent/attempts.json').read_text())
        self.assertIsNone(record['chosen'])
        self.assertTrue(all(not attempt['usable'] for attempt in record['attempts']))

    def test_truncated_trajectory_cannot_silently_drop_tail_frames(self):
        def execute(command, **options):
            result = self.engines(command, **options)
            if any(Path(arg).name == 'fusion_cli.py' for arg in command):
                position = next(i for i, arg in enumerate(command) if Path(arg).name == 'fusion_cli.py')
                tum = Path(command[position + 6]) / 'traj_tum.txt'
                tum.write_text(''.join(tum.read_text().splitlines(keepends=True)[:-1]))
            return result

        with patch.object(runtime.subprocess, 'run', side_effect=execute):
            with self.assertRaisesRegex(ValueError, '未完整覆盖原录像时间轴'):
                self.run_fallback()

    def test_bimanual_episode_cannot_use_adapter_with_only_one_recording(self):
        self.episode.expected_arms = ('left', 'right')
        with self.assertRaisesRegex(ValueError, '只支持完整单手任务'):
            self.run_fallback()
        self.assertEqual(self.engines.calls, [])
