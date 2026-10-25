"""Configuration boundaries, precedence, provenance and safe workflow behavior."""
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from software import config, registry
from software.calibration import inputs, reports, workflow as calibration_workflow
from software.calibration.models import CalibrationRequest, SegmentPaths
from software.trajectory import calibration_assets, cli, reconstruction, results, reuse, runtime, selection, workflow
from software.trajectory.models import FusionResult, TrajectoryRequest
import test_calibration_workflow as calibration_tests
import test_trajectory_workflow as trajectory_tests


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def overlay(self, text, name='custom.yaml'):
        path = self.root / name
        path.write_text(text)
        return path

    def test_defaults_are_typed_immutable_and_preserve_operational_values(self):
        p = config.load('trajectory').values
        self.assertEqual(p.video.size_px, 720)
        self.assertEqual((p.retry.max_tries, p.retry.minimum_full_fraction), (3, .95))
        self.assertEqual((p.fusion.step, p.fusion.node_stride, p.fusion.refine_chain_pairs), (30, 5, 8))
        self.assertEqual((p.fusion.group_scale_ratio, p.fusion.split_rotation_deg), (1.25, 15))
        self.assertEqual((p.fusion.acc_noise, p.fusion.gyro_noise, p.fusion.gyro_bias_rw), (.002, .0002, .00002))
        c = config.load('calibration').values
        self.assertEqual((c.board.intrinsics_step, c.board.camera_imu_step, c.board.validation_step), (5, 15, 1))
        self.assertEqual((c.checks.ate_fail_mm, c.checks.scale_warn, c.checks.scale_fail), (30, .03, .1))
        with self.assertRaises(FrozenInstanceError):
            p.fusion.step = 10

    def test_cli_only_overrides_explicit_fields(self):
        path = self.overlay("input:\n  mode: \"dual\"\n  allow_irregular: true\nretry:\n  max_tries: 7\n")
        with patch.object(workflow, 'run_batch', return_value=0) as run:
            cli.main(['video.insv', '--config', str(path)])
            request = config.resolve_trajectory_request(run.call_args.args[2])
            self.assertEqual((request.mode, request.max_tries, request.allow_irregular), ('dual', 7, True))
            cli.main(['video.insv', '--config', str(path), '--mode', 'handheld', '--max-tries', '1', '--no-allow-irregular'])
            request = config.resolve_trajectory_request(run.call_args.args[2])
            self.assertEqual((request.mode, request.max_tries, request.allow_irregular), ('handheld', 1, False))
            self.assertEqual(request.configuration.snapshot()['cli_overrides'], dict(input=dict(mode='handheld', allow_irregular=False), retry=dict(max_tries=1)))

    def test_strict_validation_rejects_unknown_wrong_type_nonfinite_and_bad_ranges(self):
        bad = ["fusoin:\n  step: 5", "fusion:\n  stepp: 5", "fusion:\n  step: true", "fusion:\n  node_stride: 1.5",
               "fusion:\n  acc_noise: 0", "fusion:\n  gyro_noise: .nan", "fusion:\n  position_sigma_m: .inf",
               "video:\n  size_px: 719", "video:\n  size_px: 1442", "video:\n  size_px: 0",
               "video:\n  size_px: true", "video:\n  size_px: 720.0",
               "video:\n  crf: 52", "video:\n  preset: \"made-up\"", "input:\n  mode: \"rear\"",
               "retry:\n  minimum_full_fraction: 1.1", "retry:\n  max_tries: 0",
               "fusion:\n  group_scale_ratio: 0.5", "fusion:\n  initial_bias_min_samples: 3000",
               "orb:\n  scale_factor: 1.0", "orb:\n  minimum_fast: 30", "schema_version: 2",
               "fusion:\n  step: \"10\"", "fusion: 10", "zupt:\n  enabled: 1"]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                config.load('trajectory', self.overlay(text))
        for text in ["checks:\n  scale_warn: 0.2", "checks:\n  ate_warn_mm: 31",
                     "gravity:\n  distinct_direction_cosine: 1", "board:\n  minimum_bearing_z: 2",
                     "board:\n  minimum_pnp_tags: 2", "gravity:\n  minimum_window_span_fraction: 1.1"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                config.load('calibration', self.overlay(text))

    def test_invalid_yaml_missing_file_and_invalid_snapshot_fail_cleanly(self):
        for path in (self.root / 'missing', self.overlay('[fusion\n')):
            with self.assertRaises(ValueError):
                config.load('trajectory', path)
        for data in ([], {}, {'schema_version': 1}):
            with self.assertRaises(ValueError):
                config.from_dict('trajectory', data)

    def test_yaml_rejects_duplicate_keys_at_any_level(self):
        for text in ('schema_version: 1\nschema_version: 1\n',
                     'fusion:\n  step: 10\n  step: 20\n'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'duplicate key'):
                config.load('trajectory', self.overlay(text))

    def test_yaml_requires_mapping_roots_and_string_keys(self):
        for text in ('', '# empty', 'null', '[]', '42', '1: value',
                     'fusion:\n  1: 2\n', 'fusion: []', '{}\n---\n{}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                config.load('trajectory', self.overlay(text))
        self.assertEqual(config.load('trajectory', self.overlay('{}')).values,
                         config.load('trajectory').values)

    def test_yaml_boolean_semantics_and_safe_tags(self):
        for word in ('on', 'off', 'yes', 'no'):
            with self.subTest(word=word), self.assertRaises(ValueError):
                config.load('trajectory', self.overlay(f'input:\n  allow_irregular: {word}'))
        for text in ('%YAML 1.1\n---\ninput:\n  allow_irregular: on\n',
                     'fusion:\n  step: !!python/object/apply:builtins.int ["7"]',
                     'fusion:\n  step: !custom 7'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                config.load('trajectory', self.overlay(text))

    def test_yaml_scientific_notation_and_complete_snapshot_roundtrip(self):
        resolved = config.load('trajectory', self.overlay('fusion:\n  gyro_noise: 2e-4\n  linear_tolerance: 1e-12'))
        self.assertEqual(resolved.values.fusion.gyro_noise, .0002)
        self.assertEqual(resolved.values.fusion.linear_tolerance, 1e-12)
        for kind in ('calibration', 'trajectory'):
            original = config.load(kind)
            path = self.root / f'{kind}.yaml'
            config.write_yaml(path, original.values)
            restored = config.load(kind, path)
            self.assertEqual(original.values, restored.values)
            self.assertEqual(original.computation_hash, restored.computation_hash)
            self.assertEqual(original.evaluation_hash, restored.evaluation_hash)

    def test_comments_key_order_location_and_integer_float_spelling_do_not_change_hashes(self):
        a = config.load('trajectory', self.overlay("fusion:\n  position_sigma_m: 1\n  step: 15\n", 'a.yaml'))
        b = config.load('trajectory', self.overlay("# comment\nfusion:\n  step: 15\n  position_sigma_m: 1.0\n", 'b.yaml'))
        self.assertEqual((a.computation_hash, a.evaluation_hash), (b.computation_hash, b.evaluation_hash))
        self.assertNotEqual(a.sources, b.sources)

    def test_numerical_and_evaluation_hashes_are_independent(self):
        default = config.load('trajectory')
        notes = config.load('trajectory', self.overlay("notes:\n  minimum_full_fraction: 0.8"))
        retry = config.load('trajectory', self.overlay("retry:\n  minimum_full_fraction: 0.8"))
        self.assertEqual(notes.computation_hash, default.computation_hash)
        self.assertNotEqual(notes.evaluation_hash, default.evaluation_hash)
        self.assertNotEqual(retry.computation_hash, default.computation_hash)
        self.assertEqual(retry.evaluation_hash, default.evaluation_hash)
        c = config.load('calibration')
        changed = config.load('calibration', self.overlay("checks:\n  ate_fail_mm: 20"))
        self.assertEqual(changed.computation_hash, c.computation_hash)
        self.assertNotEqual(changed.evaluation_hash, c.evaluation_hash)

    def test_related_trajectory_config_resolves_relative_to_explicit_calibration_file(self):
        self.overlay("fusion:\n  node_stride: 3", 'trajectory.yaml')
        c = config.load('calibration', self.overlay("trajectory_validation:\n  config: \"trajectory.yaml\""))
        self.assertEqual(c.load_trajectory_validation().values.fusion.node_stride, 3)
        config.write_yaml(self.root / 'resolved.yaml', c.load_trajectory_validation().values)
        self.assertEqual(config.load('trajectory', self.root / 'resolved.yaml').values, c.load_trajectory_validation().values)

    def test_retry_gate_changes_quality_assessment(self):
        info = self.root / 'fuse_info.json'
        info.write_text(json.dumps(dict(frames=100, init_lens='front', chains=dict(front=[[[0,89],'full']]))))
        result = FusionResult(self.root / 'traj.txt', info)
        p = config.load('trajectory').values.retry
        self.assertTrue(reconstruction.assess_try(result, p)['suspicious'])
        self.assertFalse(reconstruction.assess_try(result, replace(p, minimum_full_fraction=.8))['suspicious'])

    def test_slam_generation_uses_both_measurement_and_algorithm_config(self):
        p = config.load('trajectory', self.overlay("orb:\n  features: 777\n  initial_fast: 21")).values
        camera = calibration_assets.read_basalt(calibration_assets.FACTORY_JSON)
        yaml = calibration_assets.slam_settings_text(camera, p)
        self.assertIn('ORBextractor.nFeatures: 777', yaml)
        self.assertIn('ORBextractor.iniThFAST: 21', yaml)
        self.assertIn('Camera1.fx: %.12f' % (camera['fx']/2), yaml)
        self.assertNotIn('IMU.NoiseGyro', yaml)  # ORB wrapper is pure monocular.

    def test_slam_geometry_scales_both_lenses_without_changing_distortion(self):
        import cv2
        from software.camera.x3.frame import scale_intrinsics

        camera = calibration_assets.read_basalt(calibration_assets.FACTORY_JSON)
        for rear in (False, True):
            baseline = cv2.FileStorage(calibration_assets.slam_settings_text(
                camera, config.load('trajectory').values, rear=rear), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
            original = {key: baseline.getNode(f'Camera1.{key}').real() for key in ('fx', 'fy', 'cx', 'cy')}
            distortion = [baseline.getNode(f'Camera1.k{k}').real() for k in range(1, 5)]
            extrinsics = baseline.getNode('IMU.T_b_c1').mat()
            for size in (360, 960, 1440):
                with self.subTest(rear=rear, size=size):
                    p = config.load('trajectory', overrides={'video': {'size_px': size}}).values
                    generated = cv2.FileStorage(calibration_assets.slam_settings_text(camera, p, rear=rear),
                                                cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
                    self.assertEqual(generated.getNode('Camera.width').real(), size)
                    self.assertEqual(generated.getNode('Camera.height').real(), size)
                    for key, value in scale_intrinsics(original, 720, size).items():
                        self.assertAlmostEqual(generated.getNode(f'Camera1.{key}').real(), value, places=9)
                    self.assertEqual([generated.getNode(f'Camera1.k{k}').real() for k in range(1, 5)], distortion)
                    self.assertTrue((generated.getNode('IMU.T_b_c1').mat() == extrinsics).all())
                    self.assertEqual(generated.getNode('Camera.fps').real(), baseline.getNode('Camera.fps').real())

    def test_resized_mask_preserves_categories_and_source_asset(self):
        import cv2
        import numpy as np

        original = self.root / 'mask.png'
        mask = np.zeros((720, 720), dtype=np.uint8)
        mask[360:, 360:] = 255
        cv2.imwrite(str(original), mask)
        source_bytes = original.read_bytes()
        for size in (360, 720, 960, 1440):
            with self.subTest(size=size):
                output = calibration_assets.prepare_mask(original, self.root / f'mask-{size}.png', size)
                resized = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
                self.assertEqual(resized.shape, (size, size))
                self.assertEqual(set(np.unique(resized)), {0, 255})
                self.assertTrue((resized[:size//2] == 0).all())
                self.assertTrue((resized[size//2:, size//2:] == 255).all())
        self.assertEqual(original.read_bytes(), source_bytes)

    def test_child_physical_check_uses_explicit_values_without_reloading_defaults(self):
        from software.calibration.tools import imu_calib_physical_check as physical
        argv = ['--calib', str(calibration_assets.FACTORY_JSON), '--euroc', str(self.root/'dataset'),
                '--out', str(self.root/'check'), '--window-s', '.7', '--gyro-max-dps', '3',
                '--acc-std-max', '.06', '--min-window-samples', '25', '--gravity', '9.81',
                '--distinct-dir-cos', '.9', '--min-window-span-fraction', '.85']
        evidence = dict(status=physical.STATUS_REPORT_ONLY, exit_code=0)
        with patch.object(config, 'load', side_effect=AssertionError('unexpected defaults reload')), patch.object(physical, 'analyze', return_value=evidence) as analyze, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(physical.main(argv), 0)
        self.assertEqual(analyze.call_args.args[2:], (.7,3,.06,25,9.81,None,.9,.85))

    def test_child_video_tool_uses_explicit_values_without_reloading_defaults(self):
        from types import SimpleNamespace
        from software.trajectory.tools import make_lens_video
        cam = self.root/'mav0/cam0'
        cam.mkdir(parents=True)
        (cam/'data.csv').write_text('#timestamp,filename\n1000000000,frame.png\n')
        video = self.root/'front.mp4'
        video.write_bytes(b'video')
        with patch.object(config, 'load', side_effect=AssertionError('unexpected defaults reload')), patch.object(make_lens_video.subprocess, 'run', return_value=SimpleNamespace(stdout='{"streams":[{"nb_read_frames":"1"}]}')) as run, contextlib.redirect_stdout(io.StringIO()):
            make_lens_video.render('source.insv', str(cam.parent), str(video), 'right', size_px=960, crf=11, preset='slow')
        command = run.call_args_list[0].args[0]
        self.assertEqual(command[command.index('-crf')+1], '11')
        self.assertEqual(command[command.index('-preset')+1], 'slow')
        self.assertIn('scale=960:960:flags=area', command[command.index('-vf')+1])
        manifest = json.loads(video.with_name('front_manifest.json').read_text())
        self.assertEqual(manifest['size'], 960)

    def test_v2_bundle_has_only_measurement_and_provenance(self):
        bundle = self.root / 'bundle'
        description = calibration_assets.make_bundle(calibration_assets.FACTORY_JSON, bundle)
        self.assertEqual({p.name for p in bundle.iterdir()}, {'calibration_r1440.json', 'bundle.json'})
        self.assertEqual(description['bundle_schema_version'], 2)
        self.assertEqual(Path(description['fuser']), calibration_assets.FACTORY_FUSER)
        camera = bundle / 'calibration_r1440.json'
        camera.write_text(camera.read_text()+' ')
        with self.assertRaises(calibration_assets.CalibrationError):
            calibration_assets.describe(bundle, 'user')

    def test_legacy_bundle_is_read_only_and_never_executes_its_script(self):
        source = Path(registry.sample()['dir'])
        target = self.root / 'legacy'
        shutil.copytree(source, target)
        script = target / 'kiwi_fuse3s.py'
        script.write_text(script.read_text()+'\nraise RuntimeError("must never execute")\n')
        before = {p.name: p.read_bytes() for p in target.iterdir()}
        record = calibration_assets.describe(target, 'user')
        self.assertEqual(record['bundle_schema_version'], 1)
        self.assertEqual(Path(record['fuser']), calibration_assets.FACTORY_FUSER)
        self.assertEqual(before, {p.name: p.read_bytes() for p in target.iterdir()})


class SafetyTests(trajectory_tests.RecordingFixture):
    def test_invalid_configuration_with_force_preserves_existing_results(self):
        self.successful_result()
        before = {p.name:p.read_bytes() for p in self.output.iterdir()}
        path = self.root / 'bad.yaml'
        path.write_text("fusion:\n  acc_noise: -1")
        with patch.object(selection, 'select_calibration') as select, patch.object(runtime.console, 'write'):
            self.assertEqual(workflow.run_batch([str(self.video)], self.output.parent,
                             TrajectoryRequest(config=path, force=True)), 1)
        select.assert_not_called()
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.output.iterdir()})

    def test_dry_run_with_force_creates_no_output_and_does_not_run_engines(self):
        with patch.object(selection, 'select_calibration', return_value=self.selected), patch.object(runtime, 'run_logged') as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(workflow.run_batch([str(self.video)], self.output.parent,
                             TrajectoryRequest(force=True, dry_run=True)), 0)
        execute.assert_not_called()
        self.assertFalse(self.output.parent.exists())
        self.successful_result()
        before = {p.name:p.read_bytes() for p in self.output.iterdir()}
        with patch.object(selection, 'select_calibration', return_value=self.selected), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(workflow.run_batch([str(self.video)], self.output.parent,
                             TrajectoryRequest(force=True, dry_run=True)), 0)
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.output.iterdir()})

    def test_missing_engine_with_force_preserves_old_output(self):
        self.successful_result()
        marker = self.output / 'summary.json'
        before = marker.read_bytes()
        with patch.object(selection, 'select_calibration', return_value=self.selected), patch.dict(os.environ, {'ORB3_DIR':'/missing/engine'}), patch.object(runtime.console, 'write'):
            summary = workflow.run_recording(self.video, self.output.parent, TrajectoryRequest(force=True))
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(marker.read_bytes(), before)

    def test_calibration_dry_run_does_not_prepare_runtime_or_touch_output(self):
        segments = SegmentPaths(*(self.root / f'{name}.insv' for name in 'ABCD'))
        for _, path in segments.items():
            path.write_bytes(b'recording')
        request = CalibrationRequest(None, segments, self.output, 24, force=True, dry_run=True)
        with patch.object(registry, 'read_insv_serial', return_value='camera'), patch.object(calibration_workflow.runtime, 'prepare') as prepare, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(calibration_workflow.run(request), 0)
        prepare.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_calibration_config_validated_before_runtime_and_cleanup(self):
        path = self.root / 'bad.yaml'
        path.write_text("checks:\n  ate_warn_mm: 99")
        request = CalibrationRequest(None, None, self.output, 24, config=path, force=True)
        with patch.object(calibration_workflow.runtime, 'prepare') as prepare, patch.object(calibration_workflow.runtime.console, 'write'):
            self.assertEqual(calibration_workflow.run(request), 2)
        prepare.assert_not_called()

    def test_algorithm_and_engine_changes_invalidate_cache(self):
        self.successful_result()
        path = self.root / 'new.yaml'
        path.write_text("fusion:\n  node_stride: 2")
        fp = self.fingerprint(TrajectoryRequest(config=path))
        self.assertIn('算法参数已变化', reuse.reuse_decision(self.output, fp, notes=config.load('trajectory').values.notes).reason)
        engine = self.engines.orb / 'lib/libORB_SLAM3.so'
        engine.write_text('changed binary')
        self.assertIn('引擎', reuse.reuse_decision(self.output, self.fingerprint(), notes=config.load('trajectory').values.notes).reason)

    def test_video_size_change_invalidates_cached_trajectory(self):
        self.successful_result()
        path = self.root / 'resolution.yaml'
        path.write_text('video:\n  size_px: 960\n')
        fingerprint = self.fingerprint(TrajectoryRequest(config=path))
        self.assertIn('算法参数已变化', reuse.reuse_decision(
            self.output, fingerprint, notes=config.load('trajectory').values.notes).reason)

    def test_missing_evidence_for_re_evaluation_requires_recomputation(self):
        self.successful_result()
        fp = self.fingerprint()
        fp['evaluation_hash'] = 'changed'
        self.assertIn('缺少', reuse.reuse_decision(self.output, fp, notes=config.load('trajectory').values.notes).reason)


class EvaluationTests(unittest.TestCase):
    def test_calibration_recheck_updates_only_acceptance_and_refuses_algorithm_change(self):
        helper = calibration_tests.FullWorkflowTests()
        with tempfile.TemporaryDirectory() as temp, helper.simulation(Path(temp)) as (args, commands, install):
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            output = Path(args.out)
            camera = output / 'bundle/calibration_r1440.json'
            before = camera.read_bytes()
            cfg = Path(temp) / 'strict.yaml'
            cfg.write_text("checks:\n  ate_fail_mm: 2.0\n  ate_warn_mm: 1.0")
            with patch.object(calibration_workflow.runtime, 'run_logged') as execute:
                self.assertEqual(calibration_workflow.recheck(output, cfg), 1)
            execute.assert_not_called()
            install.assert_not_called()
            self.assertEqual(camera.read_bytes(), before)
            report = reports.load_report(output)
            self.assertEqual(report['verdict'], 'FAIL')
            self.assertEqual(report['evaluation_history'][0]['verdict'], 'PASS')
            report_before = (output / 'diagnostics/report.json').read_bytes()
            cfg.write_text("board:\n  validation_step: 2")
            self.assertEqual(calibration_workflow.recheck(output, cfg), 2)
            self.assertEqual(report_before, (output / 'diagnostics/report.json').read_bytes())

    def test_calibration_overrides_reach_solver_and_physical_check_commands(self):
        helper = calibration_tests.FullWorkflowTests()
        with tempfile.TemporaryDirectory() as temp, helper.simulation(Path(temp)) as (args, commands, install):
            path = Path(temp) / 'custom.yaml'
            path.write_text("imu:\n  gyro_noise: 0.0004\ngravity:\n  window_s: 0.7\n  minimum_window_span_fraction: 0.9\nboard:\n  intrinsics_step: 7\n  camera_imu_step: 11\n  validation_step: 2\n")
            args.config = path
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            imu_command = next(command for command in commands if '--gyro-noise-std' in command)
            self.assertEqual(float(imu_command[imu_command.index('--gyro-noise-std')+1]), .0004)
            gravity_command = next(command for command in commands if '--max-gravity-residual' in command)
            self.assertEqual(float(gravity_command[gravity_command.index('--window-s')+1]), .7)
            self.assertEqual(float(gravity_command[gravity_command.index('--min-window-span-fraction')+1]), .9)
            from software.calibration.algorithms import board_pnp
            self.assertEqual([call.kwargs['step'] for call in board_pnp.detect_all.call_args_list], [7,11,2])
            report = reports.load_report(Path(args.out))
            self.assertEqual(report['parameters']['configuration']['values']['imu']['gyro_noise'], .0004)
            install.assert_not_called()

    def test_trajectory_notes_update_reuses_trajectory_and_updates_parameter_records(self):
        helper = trajectory_tests.WorkflowTests()
        helper.setUp()
        try:
            helper.assertEqual(helper.run_recording()['status'], 'ok')
            summary = results.load_summary(helper.output)
            summary.update(quality_notes=[], quality_evidence=dict(full_fraction=1.0, map_position_starts_at_s=0.0),
                           notes=[], frames={'regular': True},
                           camera_map_chains=1, map_position_starts_at_s=0,
                           tracking={'front_map_build_tracked':[100,100]})
            results.finalize_result(helper.output, summary)
            settings = helper.output / 'details/run_settings.json'
            settings.write_text('{}')
            trajectory = helper.output / 'traj_tcp.txt'
            before = trajectory.read_bytes()
            cfg = helper.root / 'notes.yaml'
            cfg.write_text("notes:\n  minimum_full_fraction: 0.8")
            with patch.object(reconstruction, 'run_attempts') as reconstruct:
                self.assertEqual(helper.run_recording(TrajectoryRequest(config=cfg))['status'], 'skipped')
            reconstruct.assert_not_called()
            self.assertEqual(trajectory.read_bytes(), before)
            effective = json.loads(settings.read_text())['parameters']['configuration']
            self.assertEqual(effective['values']['notes']['minimum_full_fraction'], .8)
            self.assertEqual(results.load_summary(helper.output)['parameters']['configuration'], effective)
        finally:
            helper.doCleanups()


if __name__ == '__main__':
    unittest.main()
