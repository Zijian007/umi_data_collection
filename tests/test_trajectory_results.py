"""Result layout must remain usable by trajectory reuse and calibration validation."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from software.calibration import runtime, validation as calibration_validation
from software.calibration.algorithms import board_pnp
from software import config
from software.calibration.models import RunPaths, StageContext
from software.trajectory import results
from software.trajectory.models import RecordingPaths
from software.trajectory.reuse import REQUIRED_OUTPUTS, reuse_decision


class ResultLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def flat_result(self):
        summary = dict(status='ok', video='VID.insv', run_settings='run_settings.json',
                       duration_s=34.7, rows=1041, tcp_path_length_m=2.96,
                       constraint_fraction={'full': 1.0},
                       run_fingerprint=dict(version=2, recording={}, calibration={}, options={}, algorithm_parameters="test", code={}, engines={}, evaluation_hash="test"))
        (self.root / 'summary.json').write_text(json.dumps(summary))
        for name in REQUIRED_OUTPUTS:
            (self.root / name).write_text(name)
        (self.root / 'trajectory.png').write_bytes(b'preview')
        for name in ('logs', 'work'):
            (self.root / name).mkdir()
            (self.root / name / 'unique.dat').write_bytes(name.encode())
        return summary

    def test_compaction_preserves_artifacts_and_reuse(self):
        summary = self.flat_result()
        before = {name: (self.root / name).read_bytes() for name in REQUIRED_OUTPUTS}
        results.finalize_result(self.root, summary)
        self.assertEqual({p.name for p in self.root.iterdir()},
                         {'traj_tcp.txt', 'trajectory.png', 'report.txt', 'details', 'debug'})
        for name, data in before.items():
            self.assertEqual(results.result_path(self.root, name).read_bytes(), data)
        self.assertEqual((self.root / 'debug/unique.dat').read_bytes(), b'work')
        self.assertEqual((self.root / 'debug/logs/unique.dat').read_bytes(), b'logs')
        self.assertEqual(results.load_summary(self.root)['run_settings'], 'details/run_settings.json')
        self.assertIsNotNone(reuse_decision(self.root, summary['run_fingerprint'], notes=config.load('trajectory').values.notes).summary)
        results.finalize_result(self.root, results.load_summary(self.root))  # Repeat safely.
        text = (self.root / 'report.txt').read_text()
        self.assertIn('1041', text)
        self.assertIn('未测量', text)

    def test_conflicting_layout_is_rejected_before_any_moves(self):
        summary = self.flat_result()
        (self.root / 'diagnostics').mkdir()
        (self.root / 'diagnostics/traj_camera.txt').write_text('different data')
        with self.assertRaisesRegex(ValueError, '布局冲突'):
            results.finalize_result(self.root, summary)
        self.assertTrue((self.root / 'summary.json').is_file())
        self.assertTrue((self.root / 'work/unique.dat').is_file())
        self.assertEqual((self.root / 'diagnostics/traj_camera.txt').read_text(), 'different data')

    def test_failure_has_report_and_diagnostics(self):
        summary = dict(status='failed', reason='cannot track', video='VID.insv')
        (self.root / 'logs').mkdir()
        results.finalize_result(self.root, summary)
        self.assertIn('cannot track', (self.root / 'report.txt').read_text())
        self.assertEqual(results.load_summary(self.root)['status'], 'failed')
        self.assertTrue((self.root / 'debug/logs').is_dir())

    def test_compact_layout_migration_preserves_all_unique_data(self):
        summary = self.flat_result()
        old = self.root / 'diagnostics'
        old.mkdir()
        for name in (*results.AUXILIARY_RESULTS, 'logs', 'work'):
            if (self.root / name).exists():
                shutil.move(str(self.root / name), old / name)
        (old / 'extra.bin').write_bytes(b'unique legacy evidence')
        results.finalize_result(self.root, summary)
        self.assertFalse(old.exists())
        self.assertEqual((self.root / 'debug/unique.dat').read_bytes(), b'work')
        self.assertEqual((self.root / 'debug/legacy/extra.bin').read_bytes(), b'unique legacy evidence')
        self.assertIsNotNone(reuse_decision(self.root, summary['run_fingerprint'], notes=config.load('trajectory').values.notes).summary)

    def test_migration_repairs_selected_video_symlink(self):
        summary = self.flat_result()
        summary['chosen_try'] = 2
        first_video = self.root / 'work/kiwi/front/front.mp4'
        first_video.parent.mkdir(parents=True)
        first_video.write_bytes(b'lens video')
        winner_video = self.root / 'work/kiwi_try2/front/front.mp4'
        winner_video.parent.mkdir(parents=True)
        winner_video.symlink_to(first_video)
        results.finalize_result(self.root, summary)
        published = self.root / 'front.mp4'
        self.assertFalse(published.is_symlink())
        self.assertEqual(published.read_bytes(), b'lens video')
        migrated = self.root / 'debug/reconstruction/try_2/front/front.mp4'
        self.assertTrue(migrated.is_symlink())
        self.assertEqual(migrated.read_bytes(), b'lens video')
        self.assertEqual(migrated.resolve(), self.root / 'debug/reconstruction/try_1/front/front.mp4')
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

    def test_failed_video_publication_does_not_move_or_delete_legacy_data(self):
        summary = self.flat_result()
        video = self.root / 'work/kiwi/front/front.mp4'
        video.parent.mkdir(parents=True)
        video.write_bytes(b'lens video')
        with patch.object(results.os, 'link', side_effect=OSError('no hard links')), patch.object(results.shutil, 'copyfile', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                results.finalize_result(self.root, summary)
        self.assertTrue((self.root / 'summary.json').is_file())
        self.assertFalse((self.root / 'front.mp4').exists())
        self.assertEqual(video.read_bytes(), b'lens video')
        self.assertEqual(list(self.root.glob('.front-*')), [])

    def test_late_debug_conflict_is_rejected_before_any_moves(self):
        summary = self.flat_result()
        (self.root / 'debug/logs').mkdir(parents=True)
        (self.root / 'debug/logs/keep.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError, '布局冲突'):
            results.finalize_result(self.root, summary)
        self.assertTrue((self.root / 'traj_camera.txt').is_file())
        self.assertTrue((self.root / 'work/unique.dat').is_file())
        self.assertEqual((self.root / 'debug/logs/keep.txt').read_text(), 'keep')

    def test_calibration_c_check_reads_all_layouts(self):
        for layout in ('flat', 'diagnostics', 'details'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                video = root / 'VID.insv'
                context = StageContext(RunPaths(root), {}, '2-3', 1, config.load('calibration').values, config.load('trajectory'))
                output = context.paths.work / 'C_traj_test/VID'
                output.mkdir(parents=True)
                details = output if layout == 'flat' else output / layout
                details.mkdir(exist_ok=True)
                (details / 'summary.json').write_text(json.dumps(dict(status='ok', constraint_fraction={'full': 1})))
                (details / 'traj_camera.txt').write_text('0 1 2 3 0 0 0 1\n')
                with patch.object(runtime, 'run_logged', return_value=0) as runner, patch.object(board_pnp, 'read_kb4', return_value=[]), patch.object(board_pnp, 'board_reference', return_value={'idx': [0], 'n_frames': 1}), patch.object(board_pnp, 'compare', return_value={'ate_se3_rmse_mm': 5.0}) as compare:
                    check = calibration_validation.trajectory_check(video, {}, context,
                                                                     calibration_json=root / 'calib.json', bundle='factory', tag='test')
                self.assertEqual(runner.call_args.args[0][-2:], ['--cores', '2-3'])
                self.assertEqual(runner.call_args.kwargs['environment']['UMI_TRAJ_CALIB'], 'factory')
                self.assertEqual(check['status'], 'ok')
                self.assertEqual(check['ate_se3_rmse_mm'], 5.0)
                np.testing.assert_equal(compare.call_args.args[1], [[1, 2, 3]])


if __name__ == '__main__':
    unittest.main()
