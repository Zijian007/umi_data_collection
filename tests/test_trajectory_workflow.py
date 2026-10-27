"""Trajectory orchestration regressions without executing SLAM or fusion."""
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from software.vision import recordings as visual_recordings
from software import config, registry
from software.trajectory import calibration_assets
from software.trajectory import cli, reconstruction, results, runtime, selection, workflow
from software.camera.x3 import extraction
from software.trajectory.models import RecordingPaths, ReconstructionContext, TrajectoryRequest
from software.vision.models import RecordingData
from software.trajectory.reuse import REQUIRED_OUTPUTS, reuse_decision, run_fingerprint
from software.trajectory.selection import CalibrationAssets, SelectedCalibration
from trajectory_fakes import FakeEngines


class RecordingFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.engines = FakeEngines(self.root)
        environment = patch.dict(os.environ, self.engines.environment)
        environment.start()
        self.addCleanup(environment.stop)
        self.video = self.root / 'VID.insv'
        self.video.write_bytes(b'raw recording')
        paths = []
        for name in ('camera.json', 'tcp.json', 'mask.png'):
            path = self.root / name
            path.write_text(name)
            paths.append(path)
        paths[0].write_bytes(calibration_assets.SAMPLE_CAMERA_JSON.read_bytes())
        paths[2].write_bytes((runtime.HERE / 'assets/config/front_hand_mask.png').read_bytes())
        self.selected = SelectedCalibration(CalibrationAssets(*paths), self.root, 'calib-id',
                                            'camera', 'gripper', 'registry', 'registry', 'active', 100)
        self.output = self.root / 'results' / self.video.stem

    def fingerprint(self, request=TrajectoryRequest()):
        return run_fingerprint(self.video, self.selected, request)

    def successful_result(self):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'summary.json').write_text(json.dumps(dict(status='ok', run_fingerprint=self.fingerprint())))
        for name in REQUIRED_OUTPUTS:
            (self.output / name).write_text('output')


class ReuseTests(RecordingFixture):
    def test_unchanged_result_is_reused(self):
        self.successful_result()
        self.assertIsNotNone(reuse_decision(self.output, self.fingerprint(), notes=config.load('trajectory').values.notes).summary)

    def test_each_calibration_asset_change_invalidates_result(self):
        for field in ('camera_json', 'tcp_geometry', 'front_mask'):
            with self.subTest(field=field):
                self.successful_result()
                path = getattr(self.selected.assets, field)
                path.write_bytes(path.read_bytes() + '已变化'.encode())
                result = reuse_decision(self.output, self.fingerprint(), notes=config.load('trajectory').values.notes)
                self.assertIsNone(result.summary)
                self.assertIn('标定或夹爪配置已变化', result.reason)

    def test_recording_change_invalidates_result(self):
        self.successful_result()
        self.video.write_bytes(b'replacement recording')
        self.assertIsNone(reuse_decision(self.output, self.fingerprint(), notes=config.load('trajectory').values.notes).summary)

    def test_processing_options_invalidate_result(self):
        self.successful_result()
        for request in (TrajectoryRequest(mode='dual'), TrajectoryRequest(max_tries=1), TrajectoryRequest(allow_irregular=True)):
            with self.subTest(request=request):
                result = reuse_decision(self.output, self.fingerprint(request), notes=config.load('trajectory').values.notes)
                self.assertIsNone(result.summary)
                self.assertIn('已变化', result.reason)

    def test_retention_and_force_flags_are_not_numerical_inputs(self):
        self.successful_result()
        self.assertIsNotNone(reuse_decision(self.output, self.fingerprint(TrajectoryRequest(keep_work=True, force=True)), notes=config.load('trajectory').values.notes).summary)

    def test_legacy_corrupt_and_failed_summaries_are_not_reused(self):
        self.successful_result()
        for text in ('{"status":"ok"}', '{invalid', '[]', '{"status":"failed"}'):
            with self.subTest(text=text):
                (self.output / 'summary.json').write_text(text)
                self.assertIsNone(reuse_decision(self.output, self.fingerprint(), notes=config.load('trajectory').values.notes).summary)

    def test_missing_output_is_not_reused(self):
        self.successful_result()
        (self.output / 'traj_tcp.txt').unlink()
        self.assertIsNone(reuse_decision(self.output, self.fingerprint(), notes=config.load('trajectory').values.notes).summary)

    def test_relocated_identical_bundle_can_reuse(self):
        self.successful_result()
        relocated = self.root / 'relocated'
        relocated.mkdir()
        paths = []
        for name in ('camera_json', 'tcp_geometry', 'front_mask'):
            source = getattr(self.selected.assets, name)
            destination = relocated / source.name
            destination.write_bytes(source.read_bytes())
            paths.append(destination)
        moved = replace(self.selected, assets=CalibrationAssets(*paths), directory=relocated,
                        selection='UMI_TRAJ_CALIB')
        self.assertIsNotNone(reuse_decision(self.output, run_fingerprint(self.video, moved, TrajectoryRequest()), notes=config.load('trajectory').values.notes).summary)


class SelectionTests(unittest.TestCase):
    def test_recording_serial_is_default(self):
        with patch.dict(os.environ, {'UMI_TRAJ_CALIB': ''}), patch.object(registry, 'read_insv_serial', return_value='actual') as identity, patch.object(registry, 'get', return_value=registry.sample()) as get:
            chosen = selection.select_calibration(Path('video.insv'))
            identity.assert_called_once()
            get.assert_called_once_with('actual')
            self.assertEqual(chosen.selection, 'registry')

    def test_explicit_device_overrides_recording_serial(self):
        with patch.dict(os.environ, {'UMI_TRAJ_CALIB': ''}), patch.object(registry, 'read_insv_serial') as identity, patch.object(registry, 'get', return_value=registry.sample()) as get:
            selection.select_calibration(Path('video.insv'), device='specified')
            get.assert_called_once_with('specified')
            identity.assert_not_called()

    def test_environment_override_has_priority(self):
        with patch.dict(os.environ, {'UMI_TRAJ_CALIB': 'factory'}), patch.object(registry, 'read_insv_identity', return_value={'serial': 'actual'}), patch.object(registry, 'get') as get:
            chosen = selection.select_calibration(Path('video.insv'), device='specified')
            self.assertEqual(chosen.selection, 'UMI_TRAJ_CALIB')
            self.assertEqual(chosen.camera_serial, 'specified')
            get.assert_not_called()
            hashes = chosen.assets.hashes()
            self.assertEqual(chosen.as_postprocess_dict(hashes)['mask'], str(chosen.assets.front_mask))
            self.assertEqual(chosen.time_offset_ns, calibration_assets.read_basalt(chosen.assets.camera_json)['cam_time_offset_ns'])

    def test_invalid_override_is_reported_as_registry_error(self):
        with patch.dict(os.environ, {'UMI_TRAJ_CALIB': '/missing/calibration'}):
            with self.assertRaises(registry.RegistryError):
                selection.select_calibration(Path('video.insv'))

    def test_gripper_override_preserves_camera_and_calibration(self):
        record = dict(registry.sample(), serial='actual', gripper='original')
        replacement = dict(id='replacement', version='geometry_002', tcp_json='/new/tcp.json', mask='/new/mask.png',
                           tcp_sha256='tcp', mask_sha256='mask', dir='/new')
        with patch.dict(os.environ, {'UMI_TRAJ_CALIB': ''}), patch.object(registry, 'get', return_value=record), \
                patch.object(registry, 'gripper_paths', return_value=replacement):
            selected = registry.resolve_insv(Path('video.insv'), device='actual', gripper='replacement')
        self.assertEqual((selected['serial'], selected['id']), ('actual', record['id']))
        self.assertEqual((selected['gripper'], selected['tcp_json']), ('replacement', '/new/tcp.json'))
        self.assertEqual(record['gripper'], 'original')


class WorkflowTests(RecordingFixture):
    def fake_runner(self, command, **kwargs):
        return self.engines(command, **kwargs)

    def fake_postprocess(self, kd, manifest, output, mode, name, calibration, command):
        for filename in REQUIRED_OUTPUTS:
            (output / filename).write_text('processed output')
        return dict(tcp_path_length_m=1.2, constraint_fraction={'full': 1.0}, notes=[])

    def run_recording(self, request=TrajectoryRequest(), *, extraction_error=None, manifest=None):
        from software.trajectory.algorithms import postprocess
        manifest = manifest or {'frames': dict(regular=True, n_frames_used=100, duration_s=3.3)}
        def extract(video, output):
            self.engines.write_dataset(output)
            return manifest

        with patch.object(selection, 'select_calibration', return_value=self.selected), patch.object(runtime, 'git_commit', return_value='test'), patch.object(extraction, 'extract', side_effect=extraction_error or extract), patch.object(runtime.subprocess, 'run', side_effect=self.fake_runner), patch.object(postprocess, 'export_trajectories', side_effect=self.fake_postprocess), patch.object(runtime.console, 'write'):
            return workflow.run_recording(self.video, self.output.parent, request)

    def test_success_skip_and_reprocess_after_calibration_change(self):
        result = self.run_recording()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['calibration_used']['id'], 'calib-id')
        self.assertEqual(result['chosen_try'], 1)
        self.assertEqual((self.output / 'front.mp4').read_bytes(), b'lens video')
        self.assertFalse((self.output / 'debug/reconstruction').exists())
        self.assertFalse(result['debug_retained'])
        self.assertEqual({p.name for p in self.output.iterdir()}, {'traj_tcp.txt', 'front.mp4', 'report.txt', 'details', 'debug'})
        self.assertTrue((self.output / 'details/traj_camera.txt').is_file())
        self.assertEqual(self.run_recording()['status'], 'skipped')
        cal = json.loads(self.selected.assets.camera_json.read_text())
        cal['value0']['intrinsics'][0]['intrinsics']['fx'] += 1
        self.selected.assets.camera_json.write_text(json.dumps(cal))
        self.assertEqual(self.run_recording()['status'], 'ok')
        self.assertEqual(self.run_recording(TrajectoryRequest(force=True))['status'], 'ok')

    def test_display_map_survives_successful_debug_cleanup(self):
        display = dict(kind='umi_map_view', schema_version=1, maps=[])
        with patch('software.trajectory.map_data.export_attempt', return_value=display):
            result = self.run_recording()
        self.assertEqual(result['status'], 'ok')
        self.assertFalse((self.output/'debug/reconstruction').exists())
        self.assertEqual(json.loads((self.output/'details/map.json').read_text()), display)
        self.assertEqual(result['map_visualization'], 'details/map.json')

    def test_invalid_selection_preserves_existing_output(self):
        self.successful_result()
        before = (self.output / 'summary.json').read_bytes()
        with patch.object(selection, 'select_calibration', side_effect=registry.RegistryError('retired')), patch.object(runtime.console, 'write'):
            result = workflow.run_recording(self.video, self.output.parent, TrajectoryRequest())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual((self.output / 'summary.json').read_bytes(), before)

    def test_all_attempts_fail_and_record_diagnostics(self):
        error = reconstruction.ProcessingError('重建质量检查', 'invalid fusion metrics')
        with patch.object(reconstruction, 'assess_try', side_effect=error):
            result = self.run_recording(TrajectoryRequest(max_tries=2))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['tries']), 2)
        self.assertIn('2 次重建均失败', result['reason'])
        self.assertTrue((self.output / 'debug/logs/runner_try2.log').is_file())
        self.assertEqual(result['failed_stage'], '重建质量检查')
        self.assertIn('重建质量检查', (self.output / 'report.txt').read_text())
        self.assertIn('ProcessingError', (self.output / 'debug/logs/stage_error.log').read_text())

    def test_reuse_reads_summary_once(self):
        self.assertEqual(self.run_recording()['status'], 'ok')
        with patch('software.trajectory.reuse.load_summary', wraps=results.load_summary) as load:
            self.assertEqual(self.run_recording()['status'], 'skipped')
        load.assert_called_once_with(self.output)

    def test_force_does_not_read_previous_summary(self):
        self.successful_result()
        with patch('software.trajectory.reuse.load_summary', side_effect=AssertionError('unexpected reuse check')) as load:
            self.assertEqual(self.run_recording(TrajectoryRequest(force=True))['status'], 'ok')
        load.assert_not_called()

    def test_extraction_failure_is_reported_before_running_reconstruction(self):
        with patch.object(reconstruction, 'run_attempts') as reconstruct:
            result = self.run_recording(extraction_error=extraction.Refused('unreadable IMU'))
        reconstruct.assert_not_called()
        self.assertEqual(result['failed_stage'], '读取录像')
        self.assertIn('unreadable IMU', (self.output / 'report.txt').read_text())

    def test_irregular_recording_is_refused_by_default(self):
        manifest = {'frames': dict(regular=False, max_deviation_from_29p97_grid_ms=34, gaps_over_1p5x_median=1)}
        with patch.object(reconstruction, 'run_attempts') as reconstruct:
            result = self.run_recording(manifest=manifest)
        reconstruct.assert_not_called()
        self.assertEqual(result['failed_stage'], '读取录像')
        self.assertIn('--allow-irregular', result['reason'])

    def test_postprocess_failure_retains_fusion_and_does_not_prune(self):
        with patch.object(self, 'fake_postprocess', side_effect=ValueError('invalid pose')), patch.object(results, 'cleanup_debug') as prune:
            result = self.run_recording()
        self.assertEqual(result['failed_stage'], '生成轨迹产物')
        self.assertTrue((self.output / 'debug/reconstruction/try_1/fuse/fuse_info.json').is_file())
        prune.assert_not_called()

    def test_report_is_published_before_pruning(self):
        def prune(paths):
            chosen = paths.attempt(1)
            self.assertTrue((self.output / 'report.txt').is_file())
            self.assertEqual(results.load_summary(self.output)['status'], 'ok')
            self.assertTrue((chosen / 'fuse/fuse_info.json').is_file())

        with patch.object(results, 'cleanup_debug', side_effect=prune) as cleanup:
            self.assertEqual(self.run_recording()['status'], 'ok')
        cleanup.assert_called_once()

    def test_publication_failure_preserves_work(self):
        with patch.object(results, 'finalize_result', side_effect=OSError('disk full')), patch.object(results, 'cleanup_debug') as prune:
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.run_recording()
        prune.assert_not_called()
        self.assertTrue((self.output / 'debug/reconstruction/try_1/fuse/fuse_info.json').is_file())

    def test_keep_work_preserves_unabridged_fusion(self):
        self.assertEqual(self.run_recording(TrajectoryRequest(keep_work=True))['status'], 'ok')
        self.assertTrue((self.output / 'debug/reconstruction/try_1/fuse/fuse_info.json').is_file())

    def test_retry_video_survives_both_retention_modes(self):
        self.engines.bad_tries = {1}
        for keep_work in (False, True):
            with self.subTest(keep_work=keep_work):
                self.engines.calls.clear()
                result = self.run_recording(TrajectoryRequest(keep_work=keep_work, force=True))
            self.assertEqual((result['status'], result['chosen_try'], len(result['tries'])), ('ok', 2, 2))
            self.assertEqual((self.output / 'front.mp4').read_bytes(), b'lens video')
            self.assertFalse((self.output / 'front.mp4').is_symlink())
            self.assertEqual((self.output / 'debug/reconstruction').exists(), keep_work)
            self.assertEqual((self.output / 'debug/lens_videos').exists(), keep_work)
            self.assertEqual(result['debug_retained'], keep_work)
            self.assertEqual([kind for kind, *_ in self.engines.calls].count('video'), 1)
            if keep_work:
                self.assertEqual((self.output / 'debug/reconstruction/try_2/front/front.mp4').read_bytes(), b'lens video')

    def test_cleanup_failure_is_a_warning_on_successful_trajectory(self):
        with patch.object(results, 'cleanup_debug', side_effect=OSError('permission denied')):
            result = self.run_recording()
        self.assertEqual(result['status'], 'ok')
        self.assertIn('permission denied', (self.output / 'report.txt').read_text())
        self.assertEqual(results.load_summary(self.output)['notes'], result['notes'])
        self.assertTrue(results.load_summary(self.output)['debug_retained'])

    def test_interruption_publishes_failure_and_preserves_diagnostics(self):
        with patch.object(results, 'cleanup_debug') as prune:
            with self.assertRaises(KeyboardInterrupt):
                self.run_recording(extraction_error=KeyboardInterrupt())
        prune.assert_not_called()
        summary = results.load_summary(self.output)
        self.assertEqual((summary['status'], summary['failed_stage']), ('failed', '读取录像'))
        self.assertIn('处理已中断', (self.output / 'report.txt').read_text())

    def test_retry_ranks_attempts_and_reuses_prepared_lens_videos(self):
        paths = RecordingPaths(self.root / 'attempt-output')
        paths.logs.mkdir(parents=True)
        qualities = [dict(placements=2, full_fraction=1.0, suspicious=True),
                     dict(placements=1, full_fraction=0.90, suspicious=True),
                     dict(placements=1, full_fraction=0.80, suspicious=True)]
        request = TrajectoryRequest(mode='dual', cores='0-3')
        settings = runtime.reconstruction_settings(dict(os.environ), camera_json=self.selected.assets.camera_json,
                                                   front_mask=self.selected.assets.front_mask,
                                                   output=paths.debug / "settings", parameters=config.resolve_trajectory_request(request).configuration.values)
        context = ReconstructionContext(paths, config.resolve_trajectory_request(request), settings, 100)
        self.engines.write_dataset(paths.dataset)
        times = paths.debug / 'times.csv'
        times.write_text('t_canonical_s\n1.0\n')
        with patch.object(runtime.subprocess, 'run', side_effect=self.fake_runner), patch.object(reconstruction, 'assess_try', side_effect=qualities), patch.object(runtime.console, 'write'):
            videos = visual_recordings.prepare_lens_videos(self.video, context.visual, lenses=('front', 'rear') if context.request.mode == 'dual' else ('front',))
            result = reconstruction.run_attempts(videos, RecordingData({}, paths.dataset / 'mav0/imu0/data.csv', times), context, recording_name=self.video.name)
        self.assertEqual(result.chosen, 2)
        self.assertEqual(result.directory.name, 'try_2')
        self.assertEqual([kind for kind, *_ in self.engines.calls].count('video'), 2)
        for directory in (paths.attempt(1), paths.attempt(2), paths.attempt(3)):
            self.assertEqual((directory / 'front/front.mp4').resolve(), paths.lens_videos / 'front.mp4')
        for kind, command, options in self.engines.calls:
            if kind in ('map', 'localize', 'fuse'):
                self.assertEqual(command[command.index('taskset') + 1:command.index('taskset') + 3], ['-c', '0-3'])
            if kind == 'fuse':
                self.assertIn('--times', command)
                self.assertNotIn('--front_only', command)

    def test_cli_retains_arguments_and_calls_workflow(self):
        with patch.object(workflow, 'run_batch', return_value=0) as run:
            self.assertEqual(cli.main([str(self.video), '--out', str(self.output.parent), '--device', 'camera', '--mode', 'dual']), 0)
            request = run.call_args.args[2]
            self.assertEqual((request.device, request.mode), ('camera', 'dual'))
            self.assertEqual(run.call_args.args[:2], ([str(self.video)], self.output.parent))
            self.assertEqual(run.call_args.kwargs['jobs'], 1)

    def test_batch_rejects_same_named_inputs_before_creating_output(self):
        duplicate = self.root / 'second' / self.video.name
        duplicate.parent.mkdir()
        duplicate.write_bytes(b'another recording')
        with patch.object(workflow, 'run_recording') as run, patch.object(runtime.console, 'write'):
            code = workflow.run_batch([str(self.video), str(duplicate)], self.output.parent, TrajectoryRequest(), jobs=2)
        self.assertEqual(code, 1)
        run.assert_not_called()
        self.assertFalse(self.output.parent.exists())

    def test_missing_input_reports_failure_without_touching_existing_results(self):
        self.output.mkdir(parents=True)
        marker = self.output / 'report.txt'
        marker.write_text('previous result')
        with patch.object(workflow, 'run_recording') as run, patch.object(runtime.console, 'write') as terminal:
            code = workflow.run_batch([str(self.root / 'missing.insv')], self.output.parent, TrajectoryRequest())
        self.assertEqual(code, 1)
        run.assert_not_called()
        self.assertEqual(marker.read_text(), 'previous result')
        self.assertIn('录像路径不存在', terminal.call_args.args[0])
        self.assertEqual(terminal.call_args.kwargs['state'], '失败')

    def test_batch_continues_after_one_recording_fails(self):
        other = self.root / 'other.insv'
        other.write_bytes(b'other')

        def run(video, output_root, request):
            return dict(video=video.name, out=str(output_root / video.stem),
                        status='failed' if video == self.video else 'ok', reason='failed to read')

        with patch.object(workflow, 'run_recording', side_effect=run) as run_one, patch.object(runtime.console, 'write'), patch.object(workflow.reports, 'show_batch') as show:
            code = workflow.run_batch([str(self.video), str(other)], self.output.parent, TrajectoryRequest(), jobs=2)
        self.assertEqual(code, 1)
        self.assertEqual(run_one.call_count, 2)
        self.assertEqual([s['status'] for s in show.call_args.args[0]], ['failed', 'ok'])


class RecordingDataTests(RecordingFixture):
    def test_allowed_irregular_recording_passes_measured_canonical_times_to_runner(self):
        paths = RecordingPaths(self.output)
        manifest = {'frames': dict(regular=False)}

        def extract(video, output):
            cam = output / 'mav0/cam0'
            cam.mkdir(parents=True)
            (cam / 'data.csv').write_text('#timestamp [ns],filename\n1000000000,first.png\n1066733333,second.png\n')
            return manifest

        with patch.object(extraction, 'extract', side_effect=extract):
            data = visual_recordings.prepare(self.video, paths, allow_irregular=True)
        self.assertEqual(data.manifest, manifest)
        self.assertEqual(data.times_csv.read_text(), 't_canonical_s\n1.000000000\n1.066733333\n')


if __name__ == '__main__':
    unittest.main()
