"""Result layout must remain usable by trajectory reuse and calibration validation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from umi.workflows.calibration.pipeline import runtime
from umi.workflows.calibration.validation import trajectory as calibration_validation
from umi.workflows.calibration.algorithms import board_pnp
from umi.config import loader as config
from umi.estimation import trajectory_metrics
from umi.workflows.calibration.models import RunPaths, ExecutionOptions
from umi.workflows.segment.reporting import map_data, artifacts as results
from umi.estimation.models import RecordingPaths
from umi.workflows.segment.workflow import REQUIRED_OUTPUTS, reuse_decision


class ResultLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def flat_result(self):
        summary = dict(status='ok', video='VID.insv', run_settings='run_settings.json',
                       duration_s=34.7, rows=1041, tcp_path_length_m=2.96,
                       constraint_fraction={'full': 1.0},
                       run_fingerprint=dict(version=3, recording={}, calibration=dict(assets=dict(tcp_geometry="fixture")), options={}, algorithm_parameters="test", evaluation_hash="test"))
        details = self.root / 'details'
        details.mkdir()
        (details / 'summary.json').write_text(json.dumps(summary))
        for name in REQUIRED_OUTPUTS:
            target = self.root / name if name == 'traj_tcp.txt' else details / name
            target.write_text(name)
        (self.root / 'trajectory.png').write_bytes(b'preview')
        return summary

    def test_camera_only_result_reuses_without_tcp_but_requires_camera_trajectory(self):
        summary = self.flat_result()
        summary['pose_frame'] = 'camera'
        summary['run_fingerprint']['calibration']['assets'].pop('tcp_geometry')
        (self.root / 'details/summary.json').write_text(json.dumps(summary))
        (self.root / 'traj_tcp.txt').unlink()
        fingerprint = summary['run_fingerprint']
        notes = config.load('trajectory').values.notes
        self.assertIsNotNone(reuse_decision(self.root, fingerprint, notes=notes).summary)
        (self.root / 'details/traj_camera.txt').unlink()
        self.assertIsNone(reuse_decision(self.root, fingerprint, notes=notes).summary)

    def test_map_and_video_use_the_same_selected_attempt_and_layout_priority(self):
        summary = {'chosen_try': 2}
        candidates = results.attempt_paths(self.root, summary)
        attempt = candidates[0]
        video = attempt / 'front/front.mp4'
        video.parent.mkdir(parents=True)
        video.write_bytes(str(attempt).encode())
        self.assertEqual(results.front_video_path(self.root, summary), video)
        self.assertEqual(map_data.attempt_directory(self.root, summary), attempt)
        self.assertIsNone(results.front_video_path(self.root, {'chosen_try': 1}))
        self.assertIsNone(map_data.attempt_directory(self.root, {'chosen_try': 1}))
        published = self.root / 'front.mp4'
        published.write_bytes(b'published')
        self.assertEqual(results.front_video_path(self.root, summary), published)

    def test_invalid_attempt_cannot_fall_back_to_another_recording_result(self):
        video = self.root / 'debug/reconstruction/try_1/front/front.mp4'
        video.parent.mkdir(parents=True)
        video.write_bytes(b'first attempt')
        for chosen in (None, True, 0, -1, '2', 1.5):
            with self.subTest(chosen=chosen):
                with self.assertRaisesRegex(ValueError, '重建轮次无效'):
                    results.front_video_path(self.root, {'chosen_try': chosen})
                with self.assertRaisesRegex(map_data.MapDataError, '重建轮次无效'):
                    map_data.attempt_directory(self.root, {'chosen_try': chosen})

    def test_finalize_is_idempotent_and_reuse_uses_current_layout(self):
        summary = self.flat_result()
        before = {name: (self.root / name if name == 'traj_tcp.txt' else self.root / 'details' / name).read_bytes()
                  for name in REQUIRED_OUTPUTS}
        results.finalize_result(self.root, summary)
        self.assertEqual({p.name for p in self.root.iterdir()},
                         {'traj_tcp.txt', 'trajectory.png', 'report.txt', 'details'})
        for name, data in before.items():
            self.assertEqual(results.result_path(self.root, name).read_bytes(), data)
        self.assertEqual(results.load_summary(self.root)['run_settings'], 'details/run_settings.json')
        self.assertIsNotNone(reuse_decision(self.root, summary['run_fingerprint'], notes=config.load('trajectory').values.notes).summary)
        results.finalize_result(self.root, results.load_summary(self.root))  # Repeat safely.
        text = (self.root / 'report.txt').read_text()
        self.assertIn('1041', text)
        self.assertIn('未测量', text)

    def test_root_summary_is_rejected(self):
        summary = self.flat_result()
        (self.root / 'summary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'details/summary.json'):
            results.finalize_result(self.root, summary)
        self.assertTrue((self.root / 'details/summary.json').is_file())

    def test_failure_has_report_and_diagnostics(self):
        summary = dict(status='failed', reason='cannot track', video='VID.insv')
        (self.root / 'debug/logs').mkdir(parents=True)
        results.finalize_result(self.root, summary)
        self.assertIn('cannot track', (self.root / 'report.txt').read_text())
        self.assertEqual(results.load_summary(self.root)['status'], 'failed')
        self.assertTrue((self.root / 'debug/logs').is_dir())

    def test_selected_video_is_published_from_current_attempt(self):
        summary = self.flat_result()
        summary['chosen_try'] = 2
        first_video = self.root / 'debug/reconstruction/try_1/front/front.mp4'
        first_video.parent.mkdir(parents=True)
        first_video.write_bytes(b'lens video')
        winner_video = self.root / 'debug/reconstruction/try_2/front/front.mp4'
        winner_video.parent.mkdir(parents=True)
        winner_video.symlink_to(first_video)
        results.finalize_result(self.root, summary)
        published = self.root / 'front.mp4'
        self.assertFalse(published.is_symlink())
        self.assertEqual(published.read_bytes(), b'lens video')
        self.assertTrue(winner_video.is_symlink())
        self.assertEqual(winner_video.read_bytes(), b'lens video')
        self.assertEqual(winner_video.resolve(), first_video)
        results.finalize_result(self.root, results.load_summary(self.root))
        self.assertEqual(published.read_bytes(), b'lens video')

    def test_cleanup_preserves_published_video_when_winner_shared_first_video(self):
        paths = RecordingPaths(self.root)
        first = paths.attempt(1) / 'front/front.mp4'
        first.parent.mkdir(parents=True)
        first.write_bytes(b'lens video')
        winner = paths.attempt(2) / 'front/front.mp4'
        winner.parent.mkdir(parents=True)
        winner.symlink_to(first)
        paths.dataset.mkdir()
        (paths.dataset / 'imu.csv').write_bytes(b'IMU')
        paths.logs.mkdir()
        (paths.logs / 'run.txt').write_text('log')
        results.finalize_result(self.root, dict(status='ok', chosen_try=2))
        results.cleanup_debug(paths)
        self.assertEqual((self.root / 'front.mp4').read_bytes(), b'lens video')
        self.assertFalse((self.root / 'front.mp4').is_symlink())
        self.assertFalse(paths.reconstruction.exists())
        self.assertFalse(paths.dataset.exists())
        self.assertEqual((paths.logs / 'run.txt').read_text(), 'log')

    def test_cleanup_refuses_to_delete_video_without_publication(self):
        paths = RecordingPaths(self.root)
        first = paths.attempt(1) / 'front/front.mp4'
        first.parent.mkdir(parents=True)
        first.write_bytes(b'lens video')
        with self.assertRaisesRegex(OSError, '尚未发布'):
            results.cleanup_debug(paths)
        self.assertEqual(first.read_bytes(), b'lens video')

    def test_copy_fallback_publishes_independent_video(self):
        paths = RecordingPaths(self.root)
        first = paths.attempt(1) / 'front/front.mp4'
        first.parent.mkdir(parents=True)
        first.write_bytes(b'lens video')
        with patch.object(results.os, 'link', side_effect=OSError('hard links unavailable')):
            results.finalize_result(self.root, dict(status='ok', chosen_try=1))
        results.cleanup_debug(paths)
        self.assertEqual((self.root / 'front.mp4').read_bytes(), b'lens video')

    def test_failed_video_publication_does_not_move_current_data(self):
        summary = self.flat_result()
        video = self.root / 'debug/reconstruction/try_1/front/front.mp4'
        video.parent.mkdir(parents=True)
        video.write_bytes(b'lens video')
        with patch.object(results.os, 'link', side_effect=OSError('no hard links')), patch.object(results.shutil, 'copyfile', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                results.finalize_result(self.root, summary)
        self.assertTrue((self.root / 'details/summary.json').is_file())
        self.assertFalse((self.root / 'front.mp4').exists())
        self.assertEqual(video.read_bytes(), b'lens video')
        self.assertEqual(list(self.root.glob('.front-*')), [])

    def test_calibration_c_check_reads_all_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                video = root / 'VID.insv'
                context = ExecutionOptions(RunPaths(root), '2-3', 1)
                output = context.paths.work / 'C_traj_test/VID'
                output.mkdir(parents=True)
                details = output / 'details'
                details.mkdir()
                (details / 'summary.json').write_text(json.dumps(dict(status='ok', constraint_fraction={'full': 1})))
                (details / 'traj_camera.txt').write_text('0 1 2 3 0 0 0 1\n')
                with patch.object(runtime, 'run_logged', return_value=0) as runner, patch.object(board_pnp, 'read_kb4', return_value=[]), patch.object(board_pnp, 'board_reference', return_value={'idx': [0], 'n_frames': 1}), patch.object(trajectory_metrics, 'compare', return_value={'status':'ok', 'ate_se3_rmse_mm': 5.0}) as compare:
                    check = calibration_validation.trajectory_check(video, {}, context,
                                                                     calibration_json=root / 'calib.json', bundle='factory', tag='test',
                        grid={}, board=config.load('calibration').values.board, trajectory_configuration=config.load('trajectory'))
                self.assertEqual(runner.call_args.args[0][-2:], ['--cores', '2-3'])
                self.assertEqual(runner.call_args.kwargs['environment']['UMI_TRAJ_CALIB'], 'factory')
                self.assertEqual(check['status'], 'ok')
                self.assertEqual(check['ate_se3_rmse_mm'], 5.0)
                np.testing.assert_equal(compare.call_args.args[2], [[1, 2, 3]])


if __name__ == '__main__':
    unittest.main()
