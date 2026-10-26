"""训练中间包：真实视频解码、人工验收、时间映射、缺失通道与独立读取。"""
from fractions import Fraction
import io
import json
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch

import av
import numpy as np

from software.dataset import cli, sampling
from software.dataset.export import build, load_episode
from software.scene import episodes, inputs, workflow
from software.storage import write_yaml
import test_scene as fixtures


def encode_video(path, count):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=20)
        stream.width = stream.height = 32
        stream.pix_fmt = 'yuv420p'
        for k in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), k % 255, dtype=np.uint8), format='rgb24')
            frame.pts = k
            frame.time_base = Fraction(1, 20)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        data = inputs.read_yaml(self.f.episode_path)
        data.update(schema_version=5, instruction='双手把盒子放入托盘', synchronization=dict(
            offsets_s=dict(left=0., right=-.12), confirmed=True, note='共同事件，首尾已核验'))
        write_yaml(self.f.episode_path, data)
        for recording in self.f.prepared_recordings.values():
            encode_video(recording.lens_video('front').path, len(recording.video_times_s))
        self.f.build_scene()
        workflow.run_episode(self.f.episode_path)
        episodes.review(self.f.episode_path, accept=True, reviewer='test')
        self.output = self.f.root / 'dataset'

    def test_portable_package_uses_video_indices_and_flags_missing_gripper(self):
        before = self.f.episode_path.read_bytes()
        with redirect_stdout(io.StringIO()) as out:
            code = cli.main(['build', str(self.f.session_path.parent), '--out', str(self.output), '--fps', '20'])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(out.getvalue())['training_ready'])
        manifest = json.loads((self.output / 'manifest.json').read_text())
        entry = manifest['episodes'][0]
        # Removing source files/registries from the reader's reach is unnecessary:
        # these imports must never be consulted when reading an exported package.
        with patch('software.dataset.source.read_accepted', side_effect=AssertionError('source accessed')):
            metadata, arrays = load_episode(self.output, entry['path'])
        self.assertEqual(metadata['instruction'], '双手把盒子放入托盘')
        self.assertEqual(metadata['missing_channels'], ['gripper'])
        self.assertEqual(metadata['kit']['id'], 'test_kit')
        self.assertEqual(set(metadata['tcp_geometry']), {'left', 'right'})
        self.assertEqual(metadata['actual_error'], 'not_measured')
        self.assertFalse(metadata['training_ready'])
        self.assertFalse(any('gripper' in key or 'action' in key for key in arrays))
        times = arrays['t_episode_s']
        self.assertTrue(np.all(np.diff(times) > 0))
        for arm in ('left', 'right'):
            indices = arrays[arm+'_video_frame_index']
            np.testing.assert_allclose(arrays[arm+'_video_pts_s'], indices/20., atol=1e-8)
            offset = metadata['synchronization']['offsets_s'][arm]
            source_times = self.f.prepared_recordings[self.f.recording_paths[arm]].video_times_s
            np.testing.assert_allclose(arrays[arm+'_image_time_s'], source_times[indices]+offset, atol=1e-8)
            np.testing.assert_allclose(arrays[arm+'_image_time_error_s'], arrays[arm+'_image_time_s']-times)
        with self.assertRaisesRegex(ValueError, 'gripper'):
            load_episode(self.output, entry['path'], require_training_ready=True)
        self.assertEqual(self.f.episode_path.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, '已存在'):
            build([self.f.episode_path], self.output)
        # A moved dataset remains readable without the original scene.
        moved = self.f.root / 'moved_dataset'
        self.output.rename(moved)
        self.f.scene.root.rename(self.f.root / 'source_scene_unavailable')
        load_episode(moved, entry['path'])
        video = moved / entry['path'] / 'videos/left.mp4'
        video.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, '哈希'):
            load_episode(moved, entry['path'])

    def test_invalid_annotations_and_review_block_export(self):
        for change, message in ((dict(instruction='  '), 'instruction'),
                                (dict(synchronization=dict(offsets_s=dict(left=0., right=0.), confirmed=False)), 'synchronization')):
            original = inputs.read_yaml(self.f.episode_path)
            data = dict(original, **change)
            write_yaml(self.f.episode_path, data)
            workflow.run_episode(self.f.episode_path)
            episodes.review(self.f.episode_path, accept=True, reviewer='test')
            with self.assertRaisesRegex(ValueError, message):
                build([self.f.episode_path], self.output)
            self.assertFalse(self.output.exists())
            write_yaml(self.f.episode_path, original)
        workflow.run_episode(self.f.episode_path)
        with self.assertRaisesRegex(ValueError, '人工接收'):
            build([self.f.episode_path], self.output)

    def test_corrupt_video_or_source_change_never_publishes_partial_package(self):
        with patch('software.dataset.source.verify_current', side_effect=ValueError('源结果发生变化')):
            with self.assertRaisesRegex(ValueError, '发生变化'):
                build([self.f.episode_path], self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.output.parent.glob('.dataset-*')))
        with patch('software.dataset.sampling.video_timestamps', side_effect=ValueError('视频无法读取')):
            with self.assertRaisesRegex(ValueError, '无法读取'):
                build([self.f.episode_path], self.output)
        self.assertFalse(self.output.exists())

    def test_single_arm_keeps_its_identity_and_needs_no_intercamera_confirmation(self):
        data = inputs.read_yaml(self.f.episode_path)
        data['expected_arms'] = ['right']
        data['recordings'].pop('left')
        data['synchronization'] = dict(offsets_s=dict(right=0.), confirmed=False)
        write_yaml(self.f.episode_path, data)
        workflow.run_episode(self.f.episode_path)
        episodes.review(self.f.episode_path, accept=True, reviewer='test')
        build([self.f.episode_path], self.output, fps=20.)
        entry = json.loads((self.output / 'manifest.json').read_text())['episodes'][0]
        metadata, arrays = load_episode(self.output, entry['path'])
        self.assertEqual(metadata['arms'], ['right'])
        self.assertTrue(metadata['synchronization']['confirmed'])
        self.assertFalse(any(key.startswith('left_') for key in arrays))


class DatasetSamplingTests(unittest.TestCase):
    def test_large_gaps_create_segments_without_extrapolation(self):
        times = np.array([0., .1, 1., 1.1])
        poses = np.tile(np.eye(4), (4, 1, 1))
        poses[:, 0, 3] = times
        arm = sampling.ArmSamples(times, poses, np.arange(4), times, np.ones(4, bool), np.ones(4, bool))
        arrays, quality = sampling.sample(dict(left=arm), 10., .25)
        np.testing.assert_allclose(arrays['t_episode_s'], times)
        np.testing.assert_array_equal(arrays['segment_id'], [0, 0, 1, 1])
        self.assertEqual(quality['segments'], 2)
        np.testing.assert_allclose(arrays['left_T_scene_tcp'][:, 0, 3], times)

    def test_sampling_preserves_scene_translation_and_rotation(self):
        from scipy.spatial.transform import Rotation
        times = np.array([0., .1])
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[:, :3, 3] = [[3., 4., 1.], [3.1, 4., 1.]]
        poses[:, :3, :3] = Rotation.from_euler('z', [[0.], [.2]]).as_matrix()
        arm = sampling.ArmSamples(times, poses, np.arange(2), times, np.ones(2, bool), np.ones(2, bool))
        arrays, _ = sampling.sample(dict(left=arm), 20., .25)
        np.testing.assert_allclose(arrays['left_T_scene_tcp'][1, :3, 3], [3.05, 4., 1.])
        np.testing.assert_allclose(arrays['left_T_scene_tcp'][1, :3, :3], Rotation.from_euler('z', .1).as_matrix())
