"""单段重建、镜头输入、轨迹和结果目录。"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import gtsam
import numpy as np
from registry_fakes import device_record
from test_calibration_workflow import native
from trajectory_fakes import FakeEngines

from umi.apps import segment as cli
from umi.calibration.models import ExecutionOptions, RunPaths
from umi.calibration.pipeline import runtime as calibration_runtime
from umi.calibration.solve import board_pnp
from umi.calibration.validation import trajectory as calibration_validation
from umi.config import loader as config
from umi.core import frame
from umi.core.geometry import imu_from_camera
from umi.devices import calibration_assets, rear_fallback, registry, selection
from umi.devices import calibration_assets as calibration_source
from umi.devices import selection as trajectory_calibration
from umi.devices.calibration_assets import SAMPLE_CAMERA_JSON, read_basalt
from umi.devices.selection import CalibrationAssets, SelectedCalibration
from umi.devices.x3 import extraction, factory_kb4
from umi.devices.x3 import factory_geometry as factory
from umi.infra import provenance
from umi.infra.file_store import write_json
from umi.perception import recordings as visual_recordings
from umi.perception import runtime as visual_runtime
from umi.perception import slam as visual_slam
from umi.perception.models import ProcessingError, RecordingData
from umi.perception.poster import layout as tag_board
from umi.perception.runtime import prepare_mask
from umi.perception.tools import make_lens_video
from umi.reconstruction.estimation import trajectory_metrics
from umi.reconstruction.estimation.fusion import solver as fusion_solver
from umi.reconstruction.estimation.models import (
    FusionArtifacts,
    ReconstructionContext,
    RecordingPaths,
    TrajectoryRequest,
)
from umi.reconstruction.segment import environment_assets, fusion_runner, map_data, reconstruction, results, workflow
from umi.reconstruction.segment import reconstruction as runtime
from umi.reconstruction.segment import workflow as reuse
from umi.reconstruction.segment.environment_assets import select_environment_assets
from umi.reconstruction.segment.workflow import REQUIRED_OUTPUTS, reuse_decision, run_fingerprint
from umi.scene.inputs import recordings as camera_assets
from umi.store.models import Scene

# Trajectory orchestration regressions without executing SLAM or fusion.


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
        paths[2].write_bytes(selection.DEFAULT_FRONT_MASK.read_bytes())
        self.selected = SelectedCalibration(CalibrationAssets(*paths), self.root, 'calib-id',
                                            'camera', 'registry', 'registry', 'active', 100)
        self.output = self.root / 'results' / self.video.stem

    def fingerprint(self, request=TrajectoryRequest()):
        request = config.resolve_trajectory_request(request)
        selected_assets = select_environment_assets(self.video, self.selected, request)
        return run_fingerprint(self.video, self.selected, request, environment_assets=selected_assets)

    def successful_result(self):
        self.output.mkdir(parents=True, exist_ok=True)
        details = self.output / 'details'
        details.mkdir(exist_ok=True)
        (details / 'summary.json').write_text(json.dumps(dict(status='ok', run_fingerprint=self.fingerprint())))
        for name in REQUIRED_OUTPUTS:
            target = self.output / name if name == 'traj_tcp.txt' else details / name
            target.write_text('output')


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
        for request in (TrajectoryRequest(max_tries=1), TrajectoryRequest(allow_irregular=True)):
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
                (self.output / 'details/summary.json').write_text(text)
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
        self.assertIsNotNone(reuse_decision(self.output, run_fingerprint(self.video, moved, TrajectoryRequest(), environment_assets=select_environment_assets(
            self.video, moved, config.resolve_trajectory_request(TrajectoryRequest()))), notes=config.load('trajectory').values.notes).summary)


class SelectionTests(unittest.TestCase):
    def test_recording_serial_is_default(self):
        with patch.dict(os.environ, {'UMI_TRAJ_CALIB': ''}), patch.object(registry, 'read_insv_serial', return_value='actual') as identity, patch.object(registry, 'get', return_value=device_record()) as get:
            chosen = selection.select_calibration(Path('video.insv'))
            identity.assert_called_once()
            get.assert_called_once_with('actual')
            self.assertEqual(chosen.selection, 'registry')

    def test_explicit_device_overrides_recording_serial(self):
        with patch.dict(os.environ, {'UMI_TRAJ_CALIB': ''}), patch.object(registry, 'read_insv_serial') as identity, patch.object(registry, 'get', return_value=device_record()) as get:
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
            self.assertEqual(chosen.postprocess_record(hashes)['mask'], str(chosen.assets.front_mask))
            self.assertEqual(chosen.time_offset_ns, calibration_assets.read_basalt(chosen.assets.camera_json)['cam_time_offset_ns'])

    def test_invalid_override_fails_at_the_file_boundary(self):
        with patch.dict(os.environ, {'UMI_TRAJ_CALIB': '/missing/calibration'}):
            with self.assertRaises(FileNotFoundError):
                selection.select_calibration(Path('video.insv'))

    def test_explicit_tcp_preserves_camera_selection(self):
        record = dict(device_record(), serial='actual', selection='registry')
        tcp = Path(__file__).parent / 'fixtures/tcp_camera.json'
        with patch.object(registry, 'resolve_insv', return_value=record):
            selected = selection.select_calibration(Path('video.insv'), tcp=tcp)
        self.assertEqual(selected.camera_serial, 'actual')
        self.assertEqual(selected.assets.tcp_geometry, tcp)
        self.assertEqual(selected.geometry, json.loads(tcp.read_text()))
        self.assertNotIn('tcp_json', record)


class WorkflowTests(RecordingFixture):
    def fake_runner(self, command, **kwargs):
        return self.engines(command, **kwargs)

    def fake_fusion(self, request, **kwargs):
        return self.engines.fuse(request, **kwargs)

    def fake_postprocess(self, kd, manifest, output, mode, name, calibration, command, *, parameters=None):
        for filename in REQUIRED_OUTPUTS:
            (output / filename).write_text('processed output')
        return dict(tcp_path_length_m=1.2, constraint_fraction={'full': 1.0}, notes=[])

    def run_recording(self, request=TrajectoryRequest(), *, extraction_error=None, manifest=None):
        from umi.reconstruction.segment import trajectory_results as postprocess
        manifest = manifest or {'frames': dict(regular=True, n_frames_used=100, duration_s=3.3)}
        def extract(video, output, **kwargs):
            self.engines.write_dataset(output)
            return manifest

        with patch.object(selection, 'select_calibration', return_value=self.selected), patch.object(extraction, 'extract', side_effect=extraction_error or extract), patch("subprocess.run", side_effect=self.fake_runner), patch.object(fusion_runner, 'execute', side_effect=self.fake_fusion), patch.object(postprocess, 'write_trajectories', side_effect=self.fake_postprocess), patch.object(runtime.console, 'write'):
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
        with patch('umi.reconstruction.segment.map_data.export_attempt', return_value=display):
            result = self.run_recording()
        self.assertEqual(result['status'], 'ok')
        self.assertFalse((self.output/'debug/reconstruction').exists())
        self.assertEqual(json.loads((self.output/'details/map.json').read_text()), display)
        self.assertEqual(result['map_visualization'], 'details/map.json')

    def test_invalid_selection_preserves_existing_output(self):
        self.successful_result()
        before = (self.output / 'details/summary.json').read_bytes()
        with patch.object(selection, 'select_calibration', side_effect=registry.RegistryError('retired')), patch.object(runtime.console, 'write'):
            result = workflow.run_recording(self.video, self.output.parent, TrajectoryRequest())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual((self.output / 'details/summary.json').read_bytes(), before)

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
        with patch('umi.reconstruction.segment.workflow.load_summary', wraps=results.load_summary) as load:
            self.assertEqual(self.run_recording()['status'], 'skipped')
        load.assert_called_once_with(self.output)

    def test_force_does_not_read_previous_summary(self):
        self.successful_result()
        with patch('umi.reconstruction.segment.workflow.load_summary', side_effect=AssertionError('unexpected reuse check')) as load:
            self.assertEqual(self.run_recording(TrajectoryRequest(force=True))['status'], 'ok')
        load.assert_not_called()

    def test_extraction_failure_is_reported_before_running_reconstruction(self):
        with patch.object(reconstruction, 'run_attempts') as reconstruct:
            result = self.run_recording(extraction_error=extraction.Refused('unreadable IMU'))
        reconstruct.assert_not_called()
        self.assertEqual(result['failed_stage'], '读取录像')
        self.assertIn('unreadable IMU', (self.output / 'report.txt').read_text())

    def test_irregular_recording_is_refused_by_default(self):
        manifest = {'frames': dict(regular=False, max_deviation_from_grid_ms=34, gaps_over_1p5x_median=1)}
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
        qualities = [dict(placements=2, full_fraction=1.0, suspicious=True, usable=True),
                     dict(placements=1, full_fraction=0.90, suspicious=True, usable=True),
                     dict(placements=1, full_fraction=0.80, suspicious=True, usable=True)]
        request = TrajectoryRequest(mode='dual', cores='0-3')
        settings = runtime.reconstruction_settings(dict(os.environ), camera_json=self.selected.assets.camera_json,
                                                   front_mask=self.selected.assets.front_mask,
                                                   output=paths.debug / "settings", parameters=config.resolve_trajectory_request(request).configuration.values)
        context = ReconstructionContext(paths, settings, config.resolve_trajectory_request(request).configuration.values, 100, request.cores)
        self.engines.write_dataset(paths.dataset)
        times = paths.debug / 'times.csv'
        times.write_text('t_canonical_s\n1.0\n')
        with patch("subprocess.run", side_effect=self.fake_runner), patch.object(fusion_runner, 'execute', side_effect=self.fake_fusion), patch.object(reconstruction, 'assess_try', side_effect=qualities), patch.object(runtime.console, 'write'):
            videos = visual_recordings.prepare_lens_videos(self.video, context, lenses=('front', 'rear') if context.parameters.input.mode == 'dual' else ('front',))
            result = reconstruction.run_attempts(videos, RecordingData({}, paths.dataset / 'mav0/imu0/data.csv', times), context, recording_name=self.video.name)
        self.assertEqual(result.chosen, 2)
        self.assertEqual(result.directory.name, 'try_2')
        self.assertEqual([kind for kind, *_ in self.engines.calls].count('video'), 2)
        for directory in (paths.attempt(1), paths.attempt(2), paths.attempt(3)):
            self.assertEqual((directory / 'front/front.mp4').resolve(), paths.lens_videos / 'front.mp4')
        for kind, command, options in self.engines.calls:
            if kind in ('map', 'localize'):
                self.assertEqual(command[command.index('taskset') + 1:command.index('taskset') + 3], ['-c', '0-3'])
            if kind == 'fuse':
                self.assertEqual(options['cores'], '0-3')
                self.assertIsNotNone(command.times)

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

        with patch.object(workflow, 'run_recording', side_effect=run) as run_one, patch.object(runtime.console, 'write'), patch.object(workflow.results, 'show_batch') as show:
            code = workflow.run_batch([str(self.video), str(other)], self.output.parent, TrajectoryRequest(), jobs=2)
        self.assertEqual(code, 1)
        self.assertEqual(run_one.call_count, 2)
        self.assertEqual([s['status'] for s in show.call_args.args[0]], ['failed', 'ok'])


class RecordingDataTests(RecordingFixture):
    def test_allowed_irregular_recording_passes_measured_canonical_times_to_runner(self):
        paths = RecordingPaths(self.output)
        manifest = {'frames': dict(regular=False)}

        def extract(video, output, **kwargs):
            cam = output / 'mav0/cam0'
            cam.mkdir(parents=True)
            (cam / 'data.csv').write_text('#timestamp [ns],filename\n1000000000,first.png\n1066733333,second.png\n')
            return manifest

        with patch.object(extraction, 'extract', side_effect=extract):
            data = visual_recordings.prepare(self.video, paths, allow_irregular=True)
        self.assertEqual(data.manifest, manifest)
        self.assertEqual(data.times_csv.read_text(), 't_canonical_s\n1.000000000\n1.066733333\n')


# Result layout must remain usable by trajectory reuse and calibration validation.


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
                with patch.object(calibration_runtime, 'run_logged', return_value=0) as runner, patch.object(board_pnp, 'read_kb4', return_value=[]), patch.object(board_pnp, 'board_reference', return_value={'idx': [0], 'n_frames': 1}), patch.object(trajectory_metrics, 'compare', return_value={'status':'ok', 'ate_se3_rmse_mm': 5.0}) as compare:
                    check = calibration_validation.trajectory_check(video, {}, context,
                                                                     calibration_json=root / 'calib.json', bundle='factory', tag='test',
                        grid={}, board=config.load('calibration').values.board, trajectory_configuration=config.load('trajectory'))
                self.assertEqual(runner.call_args.args[0][-2:], ['--cores', '2-3'])
                self.assertEqual(runner.call_args.kwargs['environment']['UMI_TRAJ_CALIB'], 'factory')
                self.assertEqual(check['status'], 'ok')
                self.assertEqual(check['ate_se3_rmse_mm'], 5.0)
                np.testing.assert_equal(compare.call_args.args[2], [[1, 2, 3]])


# 重建阶段协议与执行顺序；替代 FFmpeg/SLAM/GTSAM，实际运行定位合并。


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engines = FakeEngines(self.root)
        self.paths = RecordingPaths(self.root / 'output')
        self.paths.logs.mkdir(parents=True)
        environment = dict(os.environ, **self.engines.environment)
        self.settings = runtime.reconstruction_settings(environment, camera_json=calibration_assets.SAMPLE_CAMERA_JSON,
                                                        front_mask=trajectory_calibration.DEFAULT_FRONT_MASK,
                                                        output=self.paths.debug / "settings", parameters=config.load("trajectory").values)
        self.context = ReconstructionContext(self.paths, self.settings, config.load('trajectory').values, 100)
        self.engines.write_dataset(self.paths.dataset)
        self.data = RecordingData(dict(source={'name': 'VID.insv'}), self.paths.dataset / 'mav0/imu0/data.csv')
        self.video = self.root / 'VID.insv'
        self.video.write_bytes(b'raw recording')
        for execute in self.engines.patches():
            execute.start()
            self.addCleanup(execute.stop)
        quiet = patch.object(runtime.console, 'write')
        quiet.start()
        self.addCleanup(quiet.stop)

    def run_stages(self, context=None, data=None):
        context = context or self.context
        videos = visual_recordings.prepare_lens_videos(self.video, context, lenses=('front', 'rear') if context.parameters.input.mode == 'dual' else ('front',))
        return reconstruction.run_attempts(videos, data or self.data, context, recording_name=self.video.name)

    def test_explicit_stage_order_and_forwarded_data(self):
        result = self.run_stages()
        self.assertEqual([kind for kind, *_ in self.engines.calls], ['video', 'map', 'localize', 'merge', 'fuse'])
        self.assertEqual(result.chosen, 1)
        self.assertEqual(result.attempts[0]['full_fraction'], 1.0)
        calls = {kind: (command, options) for kind, command, options in self.engines.calls}
        mapping, _ = calls['map']
        locating, _ = calls['localize']
        fusion, _ = calls['fuse']
        self.assertEqual(mapping[mapping.index('--input_video')+1], locating[locating.index('--input_video')+1])
        self.assertEqual(mapping[mapping.index('--save_map')+1], locating[locating.index('--load_map')+1])
        self.assertEqual(fusion.front_frames, self.paths.attempt(1) / 'front/loc/frames.csv')
        self.assertEqual(fusion.imu_csv, self.data.imu_csv)
        self.assertEqual(json.loads(self.settings.parameters_file.read_text())['input']['mode'], 'handheld')
        self.assertIn('front_setting', (result.directory / 'logs/run.txt').read_text())

    def test_retry_uses_exact_qualified_poster_events_without_hiding_other_failures(self):
        from umi.reconstruction.estimation.constraint_quality import scale_consistency
        directory = self.root/'poster-quality'
        directory.mkdir()
        trajectory = directory/'traj_tum.txt'
        trajectory.write_text(''.join(f'{1+i/10:.9f} 0 0 0 0 0 0 1\n' for i in range(100)))
        info = directory/'fuse_info.json'
        for case in ('good', 'sparse_good', 'legacy', 'weak', 'long_gap', 'disabled', 'scale', 'malformed'):
            with self.subTest(case=case):
                events = [i/10. for i in range(100)] if case == 'good' else [float(i) for i in range(10)]
                poster = dict(status='fused', observation_video_times_s=events,
                              position_support_video_times_s=events.copy())
                if case == 'legacy':
                    poster.pop('position_support_video_times_s')
                elif case == 'weak':
                    poster['position_support_video_times_s'] = []
                elif case == 'long_gap':
                    poster['position_support_video_times_s'] = [0., 1., 2.]
                elif case == 'malformed':
                    poster['position_support_video_times_s'] = [0., 1., 100.]
                evidence = dict(frames=100, init_lens='front', chains=dict(front=[[[0, 9], 'full']]),
                    environment_poster=poster, scale_consistency=scale_consistency([dict(lens='front', chain=0,
                        sid=0, scale_initial=1., scale_final=3. if case == 'scale' else 1.)]))
                info.write_text(json.dumps(evidence))
                configuration = config.load('trajectory', inline=dict(environment_tags=dict(tag_fusion_enabled=True),
                    retry=dict(use_poster_position_support=case != 'disabled')))
                context = replace(self.context, parameters=configuration.values)
                with patch.object(reconstruction, 'run_attempt', return_value=FusionArtifacts(trajectory, info)):
                    result = self.run_stages(context)
                self.assertEqual(len(result.attempts), 1 if case in ('good', 'sparse_good') else 3)
                if case == 'malformed':
                    self.assertIsNone(result.chosen)
                elif case == 'scale':
                    self.assertIsNone(result.chosen)
                    self.assertFalse(result.attempts[0]['usable'])
                else:
                    self.assertIsNotNone(result.chosen)
                    self.assertEqual(result.attempts[0]['full_fraction'], .1)
                if case in ('good', 'sparse_good'):
                    quality = result.attempts[0]
                    self.assertEqual(quality['coverage_basis'], 'visual_and_poster')
                    if case == 'sparse_good':
                        self.assertLess(quality['position_support_fraction'], .95)
                    self.assertAlmostEqual(quality['maximum_position_support_gap_s'], .1 if case == 'good' else 1.)

    def test_measured_lens_settings_are_preserved_without_sample_rear_fallback(self):
        front = calibration_assets.read_basalt(calibration_assets.SAMPLE_CAMERA_JSON)
        rear = dict(front, fx=front['fx']+40., p=[.01, .02, .03])
        texts = {lens: trajectory_calibration.slam_settings_text(camera, self.context.parameters)
                 for lens, camera in dict(front=front, rear=rear).items()}
        output = self.root/'measured-settings'
        settings = visual_runtime.visual_settings({**os.environ, **self.engines.environment, 'UMI_PY_TOOLS': '/tools/python'},
            lens_settings=texts, front_mask=trajectory_calibration.DEFAULT_FRONT_MASK,
            output=output, parameters=self.context.parameters)
        self.assertEqual(settings.front_settings.read_text(), texts['front'])
        self.assertEqual(settings.rear_settings.read_text(), texts['rear'])
        self.assertNotEqual(settings.front_settings.read_text(), settings.rear_settings.read_text())
        self.assertEqual(settings.tools_python, '/tools/python')
        self.assertFalse((output/'parameters.json').exists())
        self.assertEqual(settings.environment['UMI_PY_TOOLS'], '/tools/python')

    def test_output_size_reaches_video_slam_mask_and_run_record(self):
        import cv2

        from umi.reconstruction.segment import results as reports
        from umi.reconstruction.segment import trajectory_results as run_metadata

        configuration = config.load('trajectory', overrides={'video': {'size_px': 960}})
        settings = runtime.reconstruction_settings(dict(os.environ, **self.engines.environment),
            camera_json=calibration_assets.SAMPLE_CAMERA_JSON,
            front_mask=trajectory_calibration.DEFAULT_FRONT_MASK,
            output=self.paths.debug / 'settings-960', parameters=configuration.values)
        context = replace(self.context, parameters=configuration.values, settings=settings)
        videos = visual_recordings.prepare_lens_videos(self.video, context, lenses=('front', 'rear') if context.parameters.input.mode == 'dual' else ('front',))
        video_command = self.engines.calls[0][1]
        video_config = Path(video_command[video_command.index('--config')+1])
        self.assertEqual(config.load('trajectory', video_config).values.video.size_px, 960)
        self.assertEqual(cv2.imread(str(videos[0].mask), 0).shape, (960, 960))
        self.assertEqual(videos[0].settings, settings.front_settings)

        record = run_metadata.collect_run_settings(self.paths.output, {}, 'handheld', 'VID.insv',
            calib=dict(yaml=str(settings.front_settings), mask=str(settings.front_mask),
                       tcp_json=str(Path(__file__).parent / 'fixtures/tcp_camera.json'),
                       source_mask=str(trajectory_calibration.DEFAULT_FRONT_MASK)),
            command=dict(parameters={'configuration': configuration.snapshot()}))
        self.assertEqual(record['frame_geometry']['slam_video_px'], [960, 960])
        self.assertEqual(record['assets']['orbslam3_front_settings']['scalars']['Camera.width'], '960')
        mask_asset = record['assets']['front_hand_mask']
        self.assertEqual(mask_asset['path'], str(settings.front_mask))
        self.assertNotEqual(mask_asset['sha256_12'], mask_asset['source']['sha256_12'])
        (self.paths.output / 'front.mp4').write_bytes(b'video')
        reports.write_result_report(self.paths.output, dict(status='ok', parameters=record['parameters']))
        self.assertIn('960×960', (self.paths.output / 'report.txt').read_text())

    def test_dual_lenses_visit_every_map_and_use_mapping_mask_only(self):
        self.engines.map_count = 2
        request = TrajectoryRequest(mode='dual')
        context = replace(self.context, parameters=config.resolve_trajectory_request(request).configuration.values)
        result = self.run_stages(context)
        kinds = [kind for kind, *_ in self.engines.calls]
        self.assertEqual(kinds, ['video', 'video', 'map', 'map', 'localize', 'localize', 'merge', 'localize', 'localize', 'merge', 'fuse'])
        for kind, command, options in self.engines.calls:
            if kind == 'map':
                front = '/front/' in command[command.index('--input_video')+1]
                self.assertEqual('--mask_img' in command, front)
            elif kind == 'localize':
                self.assertNotIn('--mask_img', command)
                self.assertIn(options['env']['ORB_MAP_INDEX'], ('0', '1'))
            elif kind == 'fuse':
                self.assertEqual(context.parameters.input.mode, 'dual')
                self.assertEqual(command.rear_frames, result.directory / 'rear/loc/frames.csv')
        # 第 1 张图关键帧更多，原合并算法应优先使用 m1。
        self.assertTrue((result.directory / 'front/loc/frames.csv').read_text().splitlines()[1].endswith(',m1'))

    def test_retry_reuses_video_and_reruns_all_three_core_stages(self):
        self.engines.bad_tries = {1}
        result = self.run_stages()
        self.assertEqual(result.chosen, 2)
        kinds = [kind for kind, *_ in self.engines.calls]
        self.assertEqual(kinds.count('video'), 1)
        for kind in ('map', 'localize', 'fuse'):
            self.assertEqual(kinds.count(kind), 2)
        for number in (1, 2):
            self.assertEqual((self.paths.attempt(number) / 'front/front.mp4').resolve(), self.paths.lens_videos / 'front.mp4')

    def failing_alignment(self, request, **options):
        request.output.mkdir()
        (request.output/'alignment_failure.json').write_text(json.dumps(dict(code='NO_USABLE_LENS')))
        self.engines.fuse(request, **options)

    def test_identical_alignment_failure_stops_before_repeating_fusion(self):
        self.engines.failures['fuse'] = 2

        with patch.object(fusion_runner, 'execute', side_effect=self.failing_alignment):
            result = self.run_stages()
        self.assertIsNone(result.chosen)
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.attempts[-1]['stop_reason'], 'identical_alignment_inputs')
        self.assertEqual([kind for kind, *_ in self.engines.calls].count('fuse'), 1)

    def test_changed_visual_input_is_not_treated_as_identical_failure(self):
        self.engines.failures['fuse'] = 2

        def execute(command, **options):
            result = self.engines(command, **options)
            if './x3_mono_slam' in command and '--localization_only' in command:
                path = Path(command[command.index('--output_frames_csv')+1])
                number = next(p.removeprefix('try_') for p in path.parts if p.startswith('try_'))
                path.write_text(path.read_text().replace('0.000000', f'{number}.000000'))
            return result

        with patch("subprocess.run", side_effect=execute), \
                patch.object(fusion_runner, 'execute', side_effect=self.failing_alignment):
            result = self.run_stages()
        self.assertEqual(len(result.attempts), 3)
        self.assertEqual([kind for kind, *_ in self.engines.calls].count('fuse'), 3)
        self.assertTrue(all('stop_reason' not in a for a in result.attempts))

    def test_scale_divergence_retries_despite_full_tracking_and_retains_rejection(self):
        self.engines.scale_changes = {1: 3.27}
        result = self.run_stages()
        self.assertEqual(result.chosen, 2)
        rejected = result.attempts[0]
        self.assertEqual(rejected['full_fraction'], 1.)
        self.assertFalse(rejected['usable'])
        self.assertIn('3.270', rejected['reason'])
        self.assertEqual([kind for kind, *_ in self.engines.calls].count('map'), 2)
        self.assertTrue((self.paths.attempt(1)/'fuse/fuse_info.json').is_file())

    def test_all_divergent_attempts_fail_without_selecting_least_bad(self):
        self.engines.scale_changes = {1: 3.27, 2: .3, 3: 2.1}
        result = self.run_stages()
        self.assertIsNone(result.chosen)
        self.assertEqual(result.failed_stage, '重建质量检查')
        self.assertEqual(len(result.attempts), 3)
        self.assertTrue(all(not q['usable'] for q in result.attempts))
        self.assertIn('尺度', result.reason)

    def test_invalid_scale_cannot_displace_usable_but_fragmented_attempt(self):
        self.engines.bad_tries = {1}
        self.engines.scale_changes = {2: 3., 3: 3.}
        self.assertEqual(self.run_stages().chosen, 1)

    def test_engine_returncode_with_valid_artifact_matches_legacy_policy(self):
        self.engines.map_returncode = 139
        result = self.run_stages()
        self.assertEqual(result.chosen, 1)
        self.assertIn('map: rc=139 TRACKED', (result.directory / 'logs/run.txt').read_text())
        execution = json.loads((result.directory/'front/map/execution.json').read_text())
        self.assertTrue(execution['compatibility_exit'])
        self.assertTrue(execution['accepted'])
        self.assertIsNone(execution['offline_sync'])  # Older engines have no scheduling marker.
        self.assertIsNone(execution['offline_keyframe_interval_s'])
        self.assertEqual(json.loads((result.directory/'logs/run.json').read_text())['status'], 'ok')

    def test_engine_scheduling_mode_is_recorded_for_reproducibility(self):
        def execute(command, **options):
            if './x3_mono_slam' in command:
                options['stdout'].write('offline_sync=1\noffline_keyframe_interval_s=0.2 minimum_frames=6\n')
            return self.engines(command, **options)

        with patch("subprocess.run", side_effect=execute):
            result = self.run_stages()
        for stage in ('map', 'loc_m0'):
            record = json.loads((result.directory / 'front' / stage / 'execution.json').read_text())
            self.assertTrue(record['offline_sync'])
            self.assertEqual(record['offline_keyframe_interval_s'], .2)
            self.assertEqual(record['offline_minimum_keyframe_frames'], 6)
            self.assertTrue(record['accepted'])

    def test_timeout_with_written_artifacts_is_rejected(self):
        self.engines.map_returncode = 124
        result = self.run_stages(replace(self.context, parameters=config.load('trajectory', overrides={'retry': {'max_tries': 1}}).values))
        self.assertIsNone(result.chosen)
        self.assertEqual(result.attempts[0]['rc'], 124)
        record = json.loads((self.paths.attempt(1)/'front/map/execution.json').read_text())
        self.assertFalse(record['accepted'])
        self.assertFalse(record['compatibility_exit'])

    def test_orb_startup_error_includes_stderr_cause_and_log_path(self):
        cause = 'Camera.fps parameter must be an integer number, aborting...'
        self.engines.failures['map'] = 255

        def execute(command, **options):
            if './x3_mono_slam' in command:
                options['stderr'].write('Optional parameter does not exist...\n' + cause + '\n\n')
            return self.engines(command, **options)

        with patch("subprocess.run", side_effect=execute):
            result = self.run_stages(replace(self.context,
                parameters=config.load('trajectory', overrides={'retry': {'max_tries': 1}}).values))
        self.assertIsNone(result.chosen)
        self.assertEqual(result.attempts[0]['rc'], 255)
        self.assertIn(cause, result.reason)
        self.assertIn(str(self.paths.attempt(1) / 'front/map/stderr.txt'), result.reason)
        record = json.loads((self.paths.attempt(1) / 'front/map/execution.json').read_text())
        self.assertFalse(record['accepted'])

    def test_partial_frame_table_is_rejected_even_with_a_zero_exit(self):
        self.engines.frames = 50  # 上下文预期 100 帧，模拟视频提前结束。
        result = self.run_stages(replace(self.context, parameters=config.load('trajectory', overrides={'retry': {'max_tries': 1}}).values))
        self.assertIsNone(result.chosen)
        self.assertIn('完整帧数', result.reason)

    def test_video_failure_stops_before_any_mapping_attempt(self):
        self.engines.failures['video'] = 9
        with self.assertRaises(ProcessingError) as caught:
            self.run_stages()
        self.assertEqual(caught.exception.stage, '准备镜头视频（front）')
        self.assertEqual([kind for kind, *_ in self.engines.calls], ['video'])
        self.assertFalse(self.paths.reconstruction.exists())

    def test_each_core_failure_is_retried_and_attributes_its_stage(self):
        for kind, stage in (('map', '建图'), ('localize', '定位'), ('merge', '合并定位'), ('fuse', 'IMU 融合')):
            with self.subTest(kind=kind):
                paths = RecordingPaths(self.root / f'failed-{kind}')
                paths.logs.mkdir(parents=True)
                self.engines.write_dataset(paths.dataset)
                self.engines.failures = {kind: 9}
                self.engines.calls.clear()
                context = replace(self.context, paths=paths, parameters=config.load('trajectory', overrides={'retry': {'max_tries': 2}}).values)
                data = replace(self.data, imu_csv=paths.dataset / 'mav0/imu0/data.csv')
                result = self.run_stages(context, data)
                self.assertIsNone(result.chosen)
                self.assertEqual(len(result.attempts), 2)
                self.assertIn(stage, result.failed_stage)
                self.assertIn('2 次重建均失败', result.reason)
                self.assertIn(stage, (paths.logs / 'runner_try2.log').read_text())
                self.assertEqual([name for name, *_ in self.engines.calls].count('video'), 1)
                if kind != 'fuse':
                    self.assertNotIn('fuse', [name for name, *_ in self.engines.calls])

    def test_missing_atlas_metadata_is_localization_failure(self):
        self.engines.map_count = 0
        result = self.run_stages(replace(self.context, parameters=config.load('trajectory', overrides={'retry': {'max_tries': 1}}).values))
        self.assertIsNone(result.chosen)
        self.assertIn('定位', result.failed_stage)
        self.assertIn('无法读取图集中的地图列表', result.reason)
        self.assertNotIn('fuse', [kind for kind, *_ in self.engines.calls])

    def test_unreadable_quality_preserves_cause_and_stage(self):
        info = self.root / 'fuse_info.json'
        for content in (None, '{invalid', json.dumps([]), json.dumps({'frames': 'invalid'})):
            with self.subTest(content=content):
                if content is not None:
                    info.write_text(content)
                with self.assertRaises(ProcessingError) as caught:
                    reconstruction.assess_try(FusionArtifacts(self.root / 'traj_tum.txt', info))
                self.assertEqual(caught.exception.stage, '重建质量检查')
                self.assertEqual(caught.exception.return_code, 1)
                self.assertIn(str(info), str(caught.exception))
                self.assertIsNotNone(caught.exception.__cause__)

    def test_real_atlas_log_protocol_is_parsed_into_map_records(self):
        log = self.root / 'map.log'
        log.write_text('noise\natlas_map index 0 id 23 keyframes 81 mappoints 2492\natlas_map index 1 id 42 keyframes 16 mappoints 520\natlas_map selected index 0\n')
        maps = visual_slam.read_atlas_maps(log, lens='front')
        self.assertEqual([(item.index, item.keyframes) for item in maps], [(0, 81), (1, 16)])

    def test_irregular_times_cpu_pinning_and_fusion_parameters_are_preserved(self):
        settings = replace(self.settings, environment=dict(self.settings.environment, ORB3_EXTRA_LD='/engine/extra'))
        context = replace(self.context, cores='0-3', settings=settings)
        times = self.paths.debug / 'times.csv'
        times.write_text('t_canonical_s\n1\n1.1\n')
        self.run_stages(context, replace(self.data, times_csv=times))
        for kind, command, options in self.engines.calls:
            if kind in ('map', 'localize'):
                self.assertEqual(command[:5], ['nice', '-n', '10', 'taskset', '-c'])
                self.assertEqual(command[5], '0-3')
                self.assertEqual(command[6:8], ['timeout', '348'])
                self.assertTrue(options['env']['LD_LIBRARY_PATH'].endswith(':/engine/extra'))
            if kind == 'fuse':
                self.assertEqual(options['cores'], '0-3')
                self.assertEqual(command.times, times)
                actual = json.loads(command.parameters.read_text())
                expected = context.parameters
                self.assertEqual(actual['fusion']['step'], expected.fusion.step)
                self.assertEqual(actual['fusion']['node_stride'], expected.fusion.node_stride)
                self.assertEqual(actual['fusion']['split_rotation_deg'], expected.fusion.split_rotation_deg)
                self.assertTrue(actual['fusion']['map_residual'])
                self.assertTrue(actual['fusion']['adaptive_sigma'])
                self.assertTrue(actual['zupt']['enabled'])
                self.assertEqual(command.calibration, self.settings.camera_json)


# 可配置裁切、取帧与编码：验证真实画面、相机几何、时钟和资产不变性。


PROJECT = Path(__file__).resolve().parents[1]
TOOLS_PATH = str(Path(os.environ.get('UMI_DEPS', PROJECT / '.deps')) / 'env/bin') + os.pathsep + os.environ.get('PATH', '')
FFMPEG = shutil.which('ffmpeg', path=TOOLS_PATH)
FFPROBE = shutil.which('ffprobe', path=TOOLS_PATH)

class VideoParametersTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def video(self, **values):
        return config.load('trajectory', overrides={'video': values}).values.video

    def test_crop_resize_preserves_camera_rays_for_both_templates(self):
        import cv2
        video = self.video(crop=dict(size_px=1200, offset_x_px=-37, offset_y_px=51), size_px=600, frame_step=4)
        camera = read_basalt(SAMPLE_CAMERA_JSON)
        parameters = config.load('trajectory', overrides={'video': {'crop': dict(size_px=1200, offset_x_px=-37, offset_y_px=51), 'size_px': 600, 'frame_step': 4}}).values
        native_pixel = [875., 413.]
        x, y, size, _ = frame.calibrated_crop_xywh(video.crop)
        for rear in (False, True):
            base = cv2.FileStorage(selection.slam_settings_text(camera, config.load('trajectory').values, rear=rear),
                                  cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
            generated = cv2.FileStorage(selection.slam_settings_text(camera, parameters, rear=rear),
                                       cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
            for pixel, offset, focal, centre in zip(native_pixel, (x, y), ('fx', 'fy'), ('cx', 'cy')):
                output_pixel = (pixel-offset+.5)*video.size_px/size-.5
                old_pixel = (pixel+.5)*720/1440-.5
                expected_ray = (old_pixel-base.getNode('Camera1.'+centre).real())/base.getNode('Camera1.'+focal).real()
                actual_ray = (output_pixel-generated.getNode('Camera1.'+centre).real())/generated.getNode('Camera1.'+focal).real()
                self.assertAlmostEqual(actual_ray, expected_ray, places=10)
            self.assertEqual(generated.getNode('Camera.fps').real(), 15)
            self.assertEqual([generated.getNode(f'Camera1.k{k}').real() for k in range(1, 5)],
                             [base.getNode(f'Camera1.k{k}').real() for k in range(1, 5)])

    def test_orb_requires_integer_nominal_fps_while_frame_clock_stays_exact(self):
        import cv2
        camera = read_basalt(SAMPLE_CAMERA_JSON)
        for step, expected in ((1, 60), (2, 30), (4, 15), (120, 1)):
            parameters = config.load('trajectory', overrides={'video': {'frame_step': step}}).values
            for rear in (False, True):
                with self.subTest(step=step, rear=rear):
                    settings = cv2.FileStorage(selection.slam_settings_text(camera, parameters, rear=rear),
                                              cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
                    node = settings.getNode('Camera.fps')
                    self.assertTrue(node.isInt())
                    self.assertEqual(node.real(), expected)
            self.assertEqual(frame.output_frame_rate(step), Fraction(60000, 1001 * step))

    def test_crop_mask_uses_same_region_and_preserves_source(self):
        import cv2
        source = self.root / 'mask.png'
        mask = np.zeros((1440, 1440), dtype=np.uint8)
        mask[600:, 700:] = 255
        cv2.imwrite(str(source), mask)
        before = source.read_bytes()
        video = self.video(crop=dict(size_px=1000, offset_x_px=20, offset_y_px=-30), size_px=500)
        x, y, size, _ = frame.calibrated_crop_xywh(video.crop)
        target = prepare_mask(source, self.root / 'processed.png', video.size_px, crop=video.crop)
        expected = cv2.resize(mask[y:y+size, x:x+size], (500, 500), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(cv2.imread(str(target), cv2.IMREAD_GRAYSCALE), expected)
        self.assertEqual(source.read_bytes(), before)

    def test_invalid_combinations_are_rejected_before_processing(self):
        cases = [dict(frame_step=0), dict(frame_step=True), dict(crop=dict(size_px=1500)),
                 dict(crop=dict(offset_x_px=1)), dict(crop=dict(size_px=1200, offset_y_px=-121)),
                 dict(crop=dict(size_px=600), size_px=720), dict(codec='shell-command'),
                 dict(pixel_format='rgb'), dict(resize_filter='unknown'), dict(keep_audio='yes'),
                 dict(audio_bitrate_kbps=0)]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.video(**values)

    def test_extracted_timestamps_use_configured_step(self):
        raw = self.root / 'source.insv'
        raw.write_bytes(b'raw')
        times = np.arange(-.05, 1.1, .005)
        imu = dict(t_s=times, gyr=np.zeros((len(times), 3)), acc=np.zeros((len(times), 3)),
                   acc_range_g=16, gyro_range_dps=2000)
        info = dict(video_streams=[dict(w=3840, h=1920)], average_rate=60000/1001)
        for step in (1, 3, 4):
            with self.subTest(step=step), patch.object(extraction, 'video_stream_info', return_value=info), \
                    patch.object(extraction, 'read_imu_and_meta', return_value=imu), \
                    patch.object(extraction, 'presentation_pts', return_value=(list(range(12)), 1001/60000)):
                output = self.root / f'frames-{step}'
                manifest = extraction.extract(raw, output, frame_step=step)
                stored = np.loadtxt(output / 'mav0/cam0/data.csv', delimiter=',', comments='#', usecols=0)/1e9-1
                np.testing.assert_allclose(stored, np.arange(0, 12, step)*1001/60000, atol=1e-9)
                self.assertEqual(manifest['frames']['step'], step)
                self.assertTrue(manifest['frames']['regular'])
                self.assertAlmostEqual(manifest['frames']['fps_used'], 60000/1001/step)

    def test_video_changes_invalidate_computation_identity(self):
        original = config.load('trajectory')
        for values in (dict(frame_step=4), dict(crop=dict(size_px=1200)), dict(codec='libx265'),
                       dict(pixel_format='yuv444p'), dict(resize_filter='lanczos'), dict(keep_audio=True)):
            with self.subTest(values=values):
                changed = config.load('trajectory', overrides={'video': values})
                self.assertNotEqual(changed.computation_hash, original.computation_hash)

    @unittest.skipUnless(FFMPEG and FFPROBE, 'FFmpeg required')
    def test_real_mp4_frame_rate_dimensions_and_audio_follow_yaml(self):
        import av
        environment = patch.dict(os.environ, PATH=TOOLS_PATH)
        environment.start()
        self.addCleanup(environment.stop)
        raw = self.root / 'source.insv'
        generated = subprocess.run([FFMPEG, '-v', 'error', '-f', 'lavfi', '-i',
            'color=gray:size=3840x1920:rate=60000/1001', '-f', 'lavfi', '-i',
            'sine=frequency=440:sample_rate=16000', '-frames:v', '12', '-t', '0.2002',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', '-f', 'mp4', str(raw)],
            capture_output=True, text=True, timeout=40)
        self.assertEqual(generated.returncode, 0, generated.stderr)
        for step, audio in ((1, True), (4, False)):
            video = self.video(frame_step=step, crop=dict(size_px=1200, offset_x_px=11, offset_y_px=-9),
                               size_px=64, resize_filter='lanczos', keep_audio=audio)
            mav = self.root / f'mav-{step}/cam0'
            mav.mkdir(parents=True)
            count = len(range(0, 12, step))
            (mav / 'data.csv').write_text('#timestamp,filename\n'+'1000000000,frame.png\n'*count)
            target = self.root / f'front-{step}.mp4'
            with redirect_stdout(io.StringIO()):
                make_lens_video.render(str(raw), str(mav.parent), str(target), 'right', video=video)
            with av.open(str(target)) as container:
                stream = container.streams.video[0]
                self.assertEqual((stream.width, stream.height), (64, 64))
                self.assertEqual(stream.average_rate, Fraction(60000, 1001*step))
                self.assertEqual(bool(container.streams.audio), audio)
                decoded = list(container.decode(video=0))
                self.assertEqual(len(decoded), count)
            manifest = json.loads(target.with_name(target.stem+'_manifest.json').read_text())
            self.assertEqual(manifest['crop_xywh'], list(frame.crop_xywh('right', video.crop)))
            self.assertEqual(manifest['parameters']['keep_audio'], audio)


# 正式 rear 不依赖海报：预检、冻结、子进程传递和数值时间轴。


class RearAssetsTests(RecordingFixture):
    def setUp(self):
        super().setUp()
        raw = json.loads(calibration_assets.SAMPLE_CAMERA_JSON.read_text())
        raw.update(kind='umi_rear_calibration', camera_serial='camera', lens='rear', validation=dict(verdict='PASS'))
        self.rear_file = self.root/'rear.json'
        write_json(self.rear_file, raw)
        override = patch.object(registry, 'CAMERAS', self.root/'cameras')
        override.start(); self.addCleanup(override.stop)
        self.version = 'rear_' + provenance.file_hash(self.rear_file)[:12]
        registry.store_version(registry.CAMERAS/'camera/calibrations', self.version,
            {'rear_calibration.json': self.rear_file}, kind='umi_rear_calibration', serial='camera')
        self.request = config.resolve_trajectory_request(TrajectoryRequest(mode='dual', rear_calibration=self.version))

    def test_formal_dual_has_independent_identity_and_no_poster_dependency(self):
        selected_assets = environment_assets.select_environment_assets(self.video, self.selected, self.request)
        rear = selected_assets.rear
        self.assertIsNone(selected_assets.poster)
        fingerprint = self.fingerprint(self.request)
        self.assertEqual(fingerprint['calibration']['assets']['rear_calibration'], rear.identity.sha256)
        self.assertNotIn('environment_poster', fingerprint['calibration']['assets'])
        self.assertNotIn('rear_geometry', fingerprint['calibration']['assets'])
        manifest = environment_assets.freeze_rear_calibration(rear, self.output/'devices/rear')
        self.assertEqual(environment_assets.load_frozen_rear(manifest)['cam_time_offset_ns'], calibration_assets.read_basalt(rear.path)['cam_time_offset_ns'])
        self.successful_result()
        write_json(self.output/'details/summary.json', dict(status='ok', run_fingerprint=fingerprint))
        self.assertIsNotNone(reuse.reuse_decision(self.output, fingerprint, notes=self.request.configuration.values.notes).summary)
        (manifest.parent/'calibration.json').write_text('{}')
        self.assertIsNone(reuse.reuse_decision(self.output, fingerprint, notes=self.request.configuration.values.notes).summary)

    def test_missing_rear_refuses_before_force_can_remove_old_result(self):
        self.successful_result()
        before = (self.output/'details/summary.json').read_bytes()
        with patch.object(selection, 'select_calibration', return_value=self.selected):
            result = workflow.run_recording(self.video, self.output.parent, TrajectoryRequest(mode='dual', force=True))
        self.assertEqual(result['status'], 'failed')
        self.assertIn('无法读取 INSV 工厂参数', result['reason'])
        self.assertEqual(before, (self.output/'details/summary.json').read_bytes())

    def test_rejects_unused_rear_and_frozen_source_changes(self):
        with self.assertRaisesRegex(ValueError, '未使用'):
            environment_assets.select_environment_assets(self.video, self.selected, replace(self.request, mode='handheld'))
        selected_assets = environment_assets.select_environment_assets(self.video, self.selected, self.request)
        rear = selected_assets.rear
        rear.path.write_text('{}')
        with self.assertRaisesRegex(ValueError, '预检后'):
            environment_assets.freeze_rear_calibration(rear, self.output/'devices/rear')

    def test_workflow_passes_frozen_rear_without_triggering_poster_detection(self):
        # Use the established fake external engines but the real orchestration.
        helper = WorkflowTests('test_success_skip_and_reprocess_after_calibration_change')
        helper.setUp(); self.addCleanup(helper.doCleanups)
        commands = []
        original = helper.fake_fusion
        def fuse(request, **kwargs):
            self.assertTrue(request.rear_calibration.is_file())
            environment_assets.load_frozen_rear(request.rear_calibration)
            self.assertIsNone(request.poster)
            commands.append(request)
            return original(request, **kwargs)
        with patch.object(helper, 'fake_fusion', side_effect=fuse), patch.object(environment_assets, 'prepare_poster') as poster:
            result = helper.run_recording(self.request)
        self.assertEqual(result['status'], 'ok', result)
        self.assertTrue(commands)
        self.assertEqual(result['rear_calibration']['version'], self.version)
        self.assertTrue((helper.output/'devices/rear/calibration.json').is_file())
        poster.assert_not_called()

    def test_wrong_source_rejected_by_same_validator_as_scene(self):
        with self.assertRaisesRegex(ValueError, '来源'):
            environment_assets.select_environment_assets(self.video, replace(self.selected, intrinsics_source='factory'), self.request)

    def test_poster_and_fingerprint_share_one_explicit_asset_selection(self):
        from umi.perception.poster import layout as tag_board
        request = replace(self.request, tag_layout=tag_board.DEFAULT_LAYOUT)
        with patch.object(registry, 'rear_path', wraps=registry.rear_path) as resolve_rear:
            selected_assets = environment_assets.select_environment_assets(self.video, self.selected, request)
            resolve_rear.assert_called_once_with('camera', self.version)
        with patch.object(registry, 'rear_path', side_effect=AssertionError('must not reselect')):
            fingerprint = reuse.run_fingerprint(self.video, self.selected, request, environment_assets=selected_assets)
        board_record = selected_assets.poster.board.record()
        expected_poster_hash = provenance.canonical_hash(dict(
            layout=board_record, rear_sha256=selected_assets.rear.identity.sha256, rear_version=self.version))
        self.assertEqual(fingerprint['calibration']['assets']['environment_poster'], expected_poster_hash)
        with self.assertRaisesRegex(ValueError, '已选 rear'):
            environment_assets.EnvironmentAssets(poster=selected_assets.poster)


class RearNumericalTests(unittest.TestCase):
    def test_dual_without_poster_matches_diagnostic_only_baseline(self):
        from test_poster import independent_poster_fixture as fixture

        from umi.core.geometry import imu_from_camera, resample_pose_series
        inputs, parameters = fixture()
        poses = np.tile(np.eye(4), (len(inputs.video_times_s), 1, 1))
        for k, (position, rotation) in inputs.observations['front'].items():
            poses[k, :3, :3] = rotation
            poses[k, :3, 3] = position * 2.3
        shifted = inputs.video_times_s[:-1] + .013
        rear_poses = resample_pose_series(inputs.video_times_s, poses, shifted) @ imu_from_camera(inputs.poster.measured)
        inputs = replace(inputs, observations=dict(inputs.observations,
            rear={k: (pose[:3, 3]/1.7, pose[:3, :3]) for k, pose in enumerate(rear_poses)}))
        parameters = replace(parameters, input=replace(parameters.input, mode='dual'),
                             environment_tags=replace(parameters.environment_tags, tag_fusion_enabled=False))
        diagnosed = fusion_solver.solve(inputs, parameters, log=lambda *a, **k: None)
        standalone = fusion_solver.solve(replace(inputs, rear_camera=inputs.poster.measured, poster=None), parameters, log=lambda *a, **k: None)
        np.testing.assert_array_equal(standalone.T_world_imu, diagnosed.T_world_imu)
        self.assertGreater(standalone.evidence['obs']['rear'], 0)
        self.assertNotIn('environment_poster', standalone.evidence)
        # Exercise the actual fusion CLI with independent rear assets and no poster.
        import csv
        import io
        import tempfile
        from contextlib import redirect_stdout
        from dataclasses import asdict

        from scipy.spatial.transform import Rotation
        from test_poster import write_camera

        from umi.reconstruction.segment import fusion_runner
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            front, rear = root/'front.json', root/'rear.json'
            write_camera(front, inputs.camera)
            write_camera(rear, inputs.poster.measured, rear=True)
            selected = environment_assets.SelectedRearCalibration(rear, environment_assets.RearCalibrationIdentity(
                serial='test-camera', version='rear_test', intrinsics_source='measured', sha256=provenance.file_hash(rear)))
            manifest = environment_assets.freeze_rear_calibration(selected, root/'frozen')
            for lens in ('front', 'rear'):
                with (root/f'{lens}.csv').open('w') as stream:
                    writer = csv.writer(stream)
                    writer.writerow('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw,map'.split(','))
                    for k, (position, rotation) in inputs.observations[lens].items():
                        writer.writerow([k, inputs.video_times_s[k], 2, 1, *position,
                                         *Rotation.from_matrix(rotation).as_quat(), 'map0'])
            with (root/'imu.csv').open('w') as stream:
                writer = csv.writer(stream); writer.writerow(['#t', 'wx', 'wy', 'wz', 'ax', 'ay', 'az'])
                for row in inputs.imu_video_clock:
                    writer.writerow([round((row[0]+1)*1e9)+17000000, *row[1:]])
            (root/'times.csv').write_text('t_canonical_s\n'+''.join(f'{t+1:.12f}\n' for t in inputs.video_times_s))
            write_json(root/'parameters.json', asdict(parameters))
            with redirect_stdout(io.StringIO()):
                fusion_runner.fuse(fusion_runner.FusionRequest(root/'rear.csv', root/'front.csv', root/'imu.csv',
                    17000000, (0, 0, 0, 1), (0, 0, 0), root/'out', front, root/'parameters.json',
                    times=root/'times.csv', rear_calibration=manifest))
            evidence = json.loads((root/'out/fuse_info.json').read_text())
            self.assertNotIn('environment_poster', evidence)
            self.assertGreater(evidence['obs']['rear'], 0)
            np.testing.assert_allclose(np.loadtxt(root/'out/traj_tum.txt')[:, 1:4],
                                       standalone.T_world_imu[:, :3, 3], atol=2e-6)

    def test_conflicting_poster_and_rear_measurements_are_rejected(self):
        from test_poster import independent_poster_fixture as fixture
        inputs, parameters = fixture()
        with self.assertRaisesRegex(ValueError, 'rear 标定不一致'):
            fusion_solver.solve(replace(inputs, rear_camera=dict(inputs.poster.measured, cam_time_offset_ns=0)), parameters)


# Factory rear geometry, immutable run assets, and both selection paths.


class RearFallbackTests(RecordingFixture):
    @classmethod
    def setUpClass(cls):
        cls.forward = factory_kb4.fit_forward(factory.lens_model(native()['lenses'][0], 1440),
                                             config.load('calibration').values.factory)

    def setUp(self):
        super().setUp()
        self.native = native()
        # Nonzero translation on both sides catches rotating index0's translation.
        self.native['lenses'][0].update(yaw=.581, pitch=.714, roll=90.712, tx=.002, ty=-.001, tz=.003)
        self.native['lenses'][1].update(yaw=-.518, pitch=.410, roll=90.417, tx=-.001542, ty=-.00001, tz=-.026248)
        for override in (patch.object(factory, 'read_factory', return_value=self.native),
                         patch.object(factory_kb4, 'fit_forward', return_value=self.forward)):
            override.start(); self.addCleanup(override.stop)
        self.request = config.resolve_trajectory_request(TrajectoryRequest(tag_layout=tag_board.DEFAULT_LAYOUT))

    def derive(self):
        return rear_fallback.derive(self.video, front_path=self.selected.assets.camera_json,
                                   serial='camera', front_source='measured')

    def test_geometry_matches_specialist_and_composes_front_b_and_clock(self):
        # Independent GTSAM equations from specialist.record.offset_v3.
        body_from_lens = []
        for lens in self.native['lenses']:
            rotation = gtsam.Rot3.Ypr(*np.deg2rad([lens['yaw'], lens['pitch'], lens['roll']]))
            translation = -rotation.rotate(np.array([lens['tx'], lens['ty'], lens['tz']]))
            if lens['index'] == 0:
                rotation = gtsam.Rot3.Rz(np.pi).compose(rotation)
            body_from_lens.append(gtsam.Pose3(rotation, translation))
        expected_relative = body_from_lens[1].inverse().compose(body_from_lens[0]).matrix()
        np.testing.assert_allclose(factory.working_from_environment(self.native), expected_relative, atol=1e-14)
        front = calibration_assets.read_basalt(self.selected.assets.camera_json)
        derived = self.derive()
        np.testing.assert_allclose(imu_from_camera(derived.measurement),
                                   imu_from_camera(front) @ expected_relative, atol=1e-14)
        self.assertEqual(derived.measurement['cam_time_offset_ns'], front['cam_time_offset_ns'])
        self.assertNotEqual(derived.measurement['fx'], front['fx'])
        self.assertEqual(derived.measurement['provenance']['actual_error'], 'not_measured')
        self.assertEqual(json.loads(derived.serialized)['validation']['verdict'], 'NOT_EVALUATED')

    def test_independent_preflight_is_read_only_freezes_and_checks_hashes(self):
        before = set(self.root.rglob('*'))
        selected = environment_assets.select_environment_assets(self.video, self.selected, self.request)
        self.assertEqual(set(self.root.rglob('*')), before)
        self.assertIsNone(selected.rear.path)
        self.assertTrue(selected.rear.identity.version.startswith('factory_fallback_'))
        manifest = environment_assets.freeze_rear_calibration(selected.rear, self.output/'devices/rear')
        measured = environment_assets.load_frozen_rear(manifest)
        self.assertEqual(measured, selected.rear.factory.measurement)
        self.assertEqual(provenance.file_hash(manifest.parent/'calibration.json'), selected.rear.identity.sha256)
        with self.assertRaisesRegex(ValueError, '通过验收'):
            calibration_source.read_rear(manifest.parent/'calibration.json', serial='camera')
        (manifest.parent/'calibration.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, '哈希'):
            environment_assets.load_frozen_rear(manifest)

    def test_formal_binding_is_authoritative_even_when_broken(self):
        broken = self.root/'broken.json'; write_json(broken, {})
        with patch.object(registry, 'rear_path', return_value=broken), \
             patch.object(rear_fallback, 'derive', side_effect=AssertionError('must not fall back')):
            with self.assertRaisesRegex(ValueError, 'camera_serial'):
                environment_assets.select_environment_assets(self.video, self.selected,
                    replace(self.request, rear_calibration='explicit'))

    def test_fallback_result_reuse_verifies_the_frozen_snapshot(self):
        request = config.resolve_trajectory_request(TrajectoryRequest(mode='dual'))
        with patch('umi.infra.orb_install.require_forward_kb4'):
            selected = environment_assets.select_environment_assets(self.video, self.selected, request)
        fingerprint = reuse.run_fingerprint(self.video, self.selected, request, environment_assets=selected)
        self.successful_result()
        manifest = environment_assets.freeze_rear_calibration(selected.rear, self.output/'devices/rear')
        write_json(self.output/'details/summary.json', dict(status='ok', run_fingerprint=fingerprint))
        self.assertIsNotNone(reuse.reuse_decision(self.output, fingerprint, notes=request.configuration.values.notes).summary)
        (manifest.parent/'calibration.json').write_text('{}')
        self.assertIsNone(reuse.reuse_decision(self.output, fingerprint, notes=request.configuration.values.notes).summary)

    def test_source_identity_and_front_changes_affect_fallback_hash(self):
        first = self.derive()
        document = json.loads(self.selected.assets.camera_json.read_text())
        document['value0']['cam_time_offset_ns'] += 1000000
        write_json(self.selected.assets.camera_json, document)
        self.assertNotEqual(first.sha256, self.derive().sha256)
        self.native['serial'] = 'another-camera'
        with self.assertRaisesRegex(ValueError, '序列号'):
            self.derive()
        with self.assertRaisesRegex(ValueError, '样例'):
            rear_fallback.derive(self.video, front_path=self.selected.assets.camera_json,
                                 serial='camera', front_source='sample')

    def test_invalid_factory_metadata_does_not_remove_previous_result(self):
        from umi.reconstruction.segment import workflow
        self.successful_result()
        before = (self.output/'details/summary.json').read_bytes()
        with patch.object(factory, 'read_factory', side_effect=ValueError('unsupported v6')), \
             patch('umi.devices.selection.select_calibration', return_value=self.selected):
            result = workflow.run_recording(self.video, self.output.parent, replace(self.request, force=True))
        self.assertEqual(result['status'], 'failed')
        self.assertIn('unsupported v6', result['reason'])
        self.assertEqual(before, (self.output/'details/summary.json').read_bytes())

    def test_shared_scene_and_dual_share_fallback_and_frozen_asset_identity(self):
        from umi.scene.collection import configuration as scene_inputs
        front = calibration_assets.read_basalt(self.selected.assets.camera_json)
        record = dict(serial='camera', status='active', json=str(self.selected.assets.camera_json),
            dir=str(self.selected.directory), source='registry', cam_time_offset_ns=front['cam_time_offset_ns'],
            measurement=front, intrinsics_source='measured')
        scene = Scene(self.root/'collection.yaml', 'fallback', 'shared_front', 'left',
            {'left': self.video}, {}, config.load('trajectory'), scene_inputs.load_scene_parameters(None),
            {'camera': dict(calibration='front-version')}, tag_layout=tag_board.DEFAULT_LAYOUT)
        with patch.object(registry, 'read_insv_serial', return_value='camera'), \
             patch.object(registry, 'get', return_value=record), \
             patch('umi.infra.orb_install.require_forward_kb4') as capability:
            shared = camera_assets.select_camera(self.video, scene)
            capability.assert_not_called()
            dual = camera_assets.select_camera(self.video, replace(scene, mode='dual'))
            capability.assert_called_once()
            self.assertEqual(shared.asset_hashes['rear'], dual.asset_hashes['rear'])
            frozen = camera_assets.snapshot_cameras({'left': shared}, self.output)['left']
            self.assertIsNone(frozen.factory_rear)
            self.assertEqual(shared.asset_hashes, frozen.asset_hashes)
            self.assertEqual(json.loads(frozen.files['rear'].read_text())['kind'], rear_fallback.KIND)
            broken = self.root/'broken.json'; write_json(broken, {})
            with self.assertRaisesRegex(ValueError, 'camera_serial'):
                camera_assets.select_camera(self.video, replace(scene, rear_calibrations={'camera': broken}))

    def test_dual_workflow_passes_factory_snapshot_to_fusion(self):
        helper = WorkflowTests('test_success_skip_and_reprocess_after_calibration_change')
        helper.setUp(); self.addCleanup(helper.doCleanups)
        with patch('umi.infra.orb_install.require_forward_kb4'), \
             patch.object(factory, 'read_factory', return_value=self.native):
            result = helper.run_recording(TrajectoryRequest(mode='dual'))
        self.assertEqual(result['status'], 'ok', result)
        self.assertTrue(result['rear_calibration']['version'].startswith('factory_fallback_'))
        measured = environment_assets.load_frozen_rear(helper.output/'devices/rear/manifest.json')
        self.assertEqual(measured['calibration_kind'], rear_fallback.KIND)
        from umi.reconstruction.segment.trajectory_results import collect_run_settings
        settings = collect_run_settings(helper.output, {}, 'dual', helper.video.name,
            calib=dict(json=str(helper.selected.assets.camera_json),
                       rear_calibration_manifest=str(helper.output/'devices/rear/manifest.json')))
        self.assertEqual(settings['frame_geometry']['rear_lens']['calibration_source'], 'insv_factory_fallback')
