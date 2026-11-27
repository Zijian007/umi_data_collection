"""配置读取、结构校验和项目默认值。"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, fields, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import test_calibration_workflow as calibration_tests
import test_segment as segment_tests
from collection_defaults_fakes import defaults_document, patched_defaults
from registry_fakes import legacy_bundle

from umi.apps import segment as cli
from umi.apps.viewer import cache, maps, server
from umi.apps.viewer import video as media
from umi.calibration import result_io
from umi.calibration import workflow as calibration_workflow
from umi.calibration.models import CalibrationRequest, SegmentPaths
from umi.calibration.pipeline import inputs
from umi.config import collection_defaults
from umi.config import loader as config
from umi.config import operational_settings as settings
from umi.config import reconstruction as reconstruction_config
from umi.config.collection_schema import normalize_collection, validate_collection_defaults
from umi.devices import calibration_assets, registry
from umi.devices import selection as trajectory_calibration
from umi.infra.file_store import read_yaml, write_yaml
from umi.perception import runtime as visual_runtime
from umi.perception.gripper import distance as tag_distance
from umi.perception.gripper import frame_detection, width_estimation
from umi.perception.gripper.marker_detection import MarkerDetector
from umi.perception.runtime import prepare_mask
from umi.reconstruction.estimation.models import FusionArtifacts, TrajectoryRequest
from umi.reconstruction.segment import reconstruction, results, workflow
from umi.reconstruction.segment import reconstruction as runtime
from umi.reconstruction.segment import workflow as reuse
from umi.scene.inputs import recordings
from umi.store.models import SceneParameters

# Configuration boundaries, precedence, provenance and safe workflow behavior.


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
        self.assertEqual((c.board.intrinsics_step, c.board.validation_step), (5, 1))
        self.assertEqual((c.checks.ate_fail_mm, c.checks.scale_warn, c.checks.scale_fail), (30, .03, .1))
        with self.assertRaises(FrozenInstanceError):
            p.fusion.step = 10

    def test_environment_tag_defaults_have_one_shared_source(self):
        from umi.config import reconstruction
        from umi.scene.collection.configuration import load_scene_parameters

        trajectory = config.load('trajectory')
        scene = load_scene_parameters()
        self.assertEqual(trajectory.sources, (str(reconstruction.DEFAULT_PATH),))
        shared = [(path, targets) for path, targets in reconstruction.FIELDS.items()
                  if path[0] == 'poster' and len(targets) == 2 and targets[0][0] == 'trajectory']
        self.assertEqual(len(shared), 13)
        defaults = reconstruction.defaults()
        for path, (trajectory_target, scene_target) in shared:
            with self.subTest(path='.'.join(path)):
                self.assertEqual(getattr(trajectory.values.environment_tags, trajectory_target[-1]),
                                 getattr(scene, scene_target[1]))
                self.assertEqual(getattr(scene, scene_target[1]), defaults.scene[scene_target[1]])
        calibration = config.load('calibration').values.imu
        for name in config.CALIBRATION_IMU_FIELDS:
            self.assertEqual(getattr(calibration, name), getattr(trajectory.values.fusion, name))

    def test_calibration_yaml_maps_its_public_check_names_to_the_snapshot(self):
        path = self.overlay('checks:\n  intrinsics_mean_reprojection_px: 1.1\n  camera_imu_mean_reprojection_px: 1.3\n')
        resolved = config.load('calibration', path)
        expected = config.load('calibration', inline={
            'checks': {'intrinsics_reprojection_px': 1.1, 'camera_imu_reprojection_px': 1.3}})
        self.assertEqual(resolved.values, expected.values)
        self.assertEqual(resolved.evaluation_hash, expected.evaluation_hash)
        with self.assertRaisesRegex(ValueError, '未知字段'):
            config.load('calibration', self.overlay('checks:\n  intrinsics_reprojection_px: 1.1\n'))

    def test_cli_only_overrides_explicit_fields(self):
        path = self.overlay("input:\n  mode: \"dual\"\n  allow_irregular_frames: true\nretry:\n  maximum_tries: 7\n")
        with patch.object(workflow, 'run_batch', return_value=0) as run:
            cli.main(['video.insv', '--config', str(path)])
            request = config.resolve_trajectory_request(run.call_args.args[2])
            self.assertEqual((request.mode, request.max_tries, request.allow_irregular), ('dual', 7, True))
            cli.main(['video.insv', '--config', str(path), '--mode', 'handheld', '--max-tries', '1', '--no-allow-irregular'])
            request = config.resolve_trajectory_request(run.call_args.args[2])
            self.assertEqual((request.mode, request.max_tries, request.allow_irregular), ('handheld', 1, False))
            self.assertEqual(request.configuration.snapshot()['cli_overrides'], dict(input=dict(mode='handheld', allow_irregular=False), retry=dict(max_tries=1)))

    def test_strict_validation_rejects_unknown_wrong_type_nonfinite_and_bad_ranges(self):
        bad = ["fusoin:\n  solver: {}", "fusion:\n  solver:\n    stride: 5",
               "fusion:\n  alignment:\n    step_frames: true", "fusion:\n  solver:\n    node_stride: 1.5",
               "fusion:\n  imu:\n    acc_noise: 0", "fusion:\n  imu:\n    gyro_noise: .nan",
               "fusion:\n  visual:\n    position_sigma_m: .inf",
               "video:\n  size_px: 719", "video:\n  size_px: 1442", "video:\n  size_px: 0",
               "video:\n  size_px: true", "video:\n  size_px: 720.0",
               "video:\n  crf: 52", "video:\n  preset: \"made-up\"", "input:\n  mode: \"rear\"",
               "retry:\n  target_localized_fraction: 1.1", "retry:\n  maximum_tries: 0",
               "fusion:\n  chains:\n    group_scale_ratio: 0.5",
               "fusion:\n  initial_gyro_bias:\n    minimum_samples: 3000",
               "vision:\n  orb:\n    scale_factor: 1.0", "vision:\n  orb:\n    minimum_fast: 30", "schema_version: 2",
               "fusion:\n  alignment:\n    step_frames: \"10\"", "fusion: 10", "fusion:\n  zupt:\n    enabled: 1",
               "localization:\n  minimum_localized_fraction: 0.4", "trajectory:\n  fusion: {}"]
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
                config.decode_parameters('trajectory', data)

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
                config.load('trajectory', self.overlay(f'input:\n  allow_irregular_frames: {word}'))
        for text in ('%YAML 1.1\n---\ninput:\n  allow_irregular_frames: on\n',
                     'fusion:\n  alignment:\n    step_frames: !!python/object/apply:builtins.int ["7"]',
                     'fusion:\n  alignment:\n    step_frames: !custom 7'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                config.load('trajectory', self.overlay(text))

    def test_yaml_scientific_notation_and_complete_snapshot_roundtrip(self):
        resolved = config.load('trajectory', self.overlay('fusion:\n  imu:\n    gyro_noise: 2e-4\n  solver:\n    linear_tolerance: 1e-12'))
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
        a = config.load('trajectory', self.overlay(
            "fusion:\n  visual:\n    position_sigma_m: 1\n  alignment:\n    step_frames: 15\n", 'a.yaml'))
        b = config.load('trajectory', self.overlay(
            "# comment\nfusion:\n  alignment:\n    step_frames: 15\n  visual:\n    position_sigma_m: 1.0\n", 'b.yaml'))
        self.assertEqual((a.computation_hash, a.evaluation_hash), (b.computation_hash, b.evaluation_hash))
        self.assertNotEqual(a.sources, b.sources)

    def test_numerical_and_evaluation_hashes_are_independent(self):
        default = config.load('trajectory')
        notes = config.load('trajectory', self.overlay("reporting:\n  warnings:\n    minimum_localized_fraction: 0.8"))
        retry = config.load('trajectory', self.overlay("retry:\n  target_localized_fraction: 0.8"))
        self.assertEqual(notes.computation_hash, default.computation_hash)
        self.assertNotEqual(notes.evaluation_hash, default.evaluation_hash)
        self.assertNotEqual(retry.computation_hash, default.computation_hash)
        self.assertEqual(retry.evaluation_hash, default.evaluation_hash)
        c = config.load('calibration')
        changed = config.load('calibration', self.overlay("checks:\n  ate_fail_mm: 20"))
        self.assertEqual(changed.computation_hash, c.computation_hash)
        self.assertNotEqual(changed.evaluation_hash, c.evaluation_hash)

    def test_c_inherited_path_is_relative_to_the_calibration_defaults(self):
        defaults = self.root / 'algorithms'
        defaults.mkdir()
        document = (config.ALGORITHM_DIRECTORY / 'calibration.yaml').read_text()
        (defaults / 'calibration.yaml').write_text(document.replace('config: ""', 'config: validation.yaml', 1))
        (defaults / 'validation.yaml').write_text('fusion:\n  solver:\n    node_stride: 3\n')
        expert = self.overlay('checks:\n  ate_fail_mm: 20\n')
        with patch.object(config, 'ALGORITHM_DIRECTORY', defaults):
            calibration = config.load('calibration', expert)
        self.assertEqual(calibration.load_trajectory_validation().values.fusion.node_stride, 3)

    def test_c_applies_relative_trajectory_overrides_with_calibration_constraints(self):
        tuning = self.overlay(
            'vision:\n  initial_tag:\n    size_m: 0.024\nfusion:\n  solver:\n    node_stride: 3\n'
            'poster:\n  standalone_fusion: true\ninput:\n  mode: dual\n', 'trajectory-tuning.yaml')
        standalone = config.load('trajectory', tuning).values
        self.assertTrue(standalone.environment_tags.tag_fusion_enabled)
        self.assertEqual(standalone.input.mode, 'dual')
        c = config.load('calibration', self.overlay('trajectory_validation:\n  config: trajectory-tuning.yaml\n'))
        original = c.load_trajectory_validation()
        measured = c.load_trajectory_validation(tag_size_m=.0357)
        self.assertEqual(original.values.orb.initial_tag_size_m, .024)
        self.assertEqual(measured.values.orb.initial_tag_size_m, .0357)
        self.assertEqual(measured.values.fusion.node_stride, 3)
        self.assertFalse(measured.values.environment_tags.tag_fusion_enabled)
        self.assertEqual(measured.values.input.mode, 'handheld')
        self.assertNotEqual(original.computation_hash, measured.computation_hash)
        self.assertEqual(measured.snapshot()['cli_overrides'], {'orb': {'initial_tag_size_m': .0357},
            'environment_tags': {'tag_fusion_enabled': False}, 'input': {'mode': 'handheld'}})
        config.write_yaml(self.root / 'child.yaml', measured.values)
        child = config.load('trajectory', self.root / 'child.yaml')
        self.assertEqual(child.values, measured.values)
        from umi.reconstruction.segment.environment_assets import select_environment_assets
        request = config.resolve_trajectory_request(TrajectoryRequest(configuration=child))
        selected_assets = select_environment_assets(None, None, request)
        self.assertIsNone(selected_assets.rear)
        self.assertIsNone(selected_assets.poster)

    def test_fixed_precision_and_geometry_fields_are_no_longer_tuning_options(self):
        for field, value in [('time_match_tolerance_ns', 1000), ('minimum_trajectory_spread_m', .000001),
                             ('coverage_radius_px', 960.)]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, '未知配置项'):
                config.load('calibration', self.overlay(f'board:\n  {field}: {value}\n'))

    def test_retry_gate_changes_quality_assessment(self):
        info = self.root / 'fuse_info.json'
        info.write_text(json.dumps(dict(frames=100, init_lens='front', chains=dict(front=[[[0,89],'full']]),
            scale_consistency=dict(status='valid', maximum_change_ratio=1.0))))
        result = FusionArtifacts(self.root / 'traj.txt', info)
        p = config.load('trajectory').values.retry
        self.assertTrue(reconstruction.assess_try(result, p)['suspicious'])
        self.assertFalse(reconstruction.assess_try(result, replace(p, minimum_full_fraction=.8))['suspicious'])

    def test_slam_generation_uses_both_measurement_and_algorithm_config(self):
        p = config.load('trajectory', self.overlay("vision:\n  orb:\n    features: 777\n    initial_fast: 21")).values
        camera = calibration_assets.read_basalt(calibration_assets.SAMPLE_CAMERA_JSON)
        yaml = trajectory_calibration.slam_settings_text(camera, p)
        self.assertIn('ORBextractor.nFeatures: 777', yaml)
        self.assertIn('ORBextractor.iniThFAST: 21', yaml)
        self.assertIn('Camera1.fx: %.12f' % (camera['fx']/2), yaml)
        self.assertNotIn('IMU.NoiseGyro', yaml)  # ORB wrapper is pure monocular.

    def test_slam_geometry_scales_both_lenses_without_changing_distortion(self):
        import cv2

        from umi.core.frame import scale_intrinsics

        camera = calibration_assets.read_basalt(calibration_assets.SAMPLE_CAMERA_JSON)
        for rear in (False, True):
            baseline = cv2.FileStorage(trajectory_calibration.slam_settings_text(
                camera, config.load('trajectory').values, rear=rear), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
            original = {key: baseline.getNode(f'Camera1.{key}').real() for key in ('fx', 'fy', 'cx', 'cy')}
            distortion = [baseline.getNode(f'Camera1.k{k}').real() for k in range(1, 5)]
            extrinsics = baseline.getNode('IMU.T_b_c1').mat()
            for size in (360, 960, 1440):
                with self.subTest(rear=rear, size=size):
                    p = config.load('trajectory', overrides={'video': {'size_px': size}}).values
                    generated = cv2.FileStorage(trajectory_calibration.slam_settings_text(camera, p, rear=rear),
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
                output = prepare_mask(original, self.root / f'mask-{size}.png', size)
                resized = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
                self.assertEqual(resized.shape, (size, size))
                self.assertEqual(set(np.unique(resized)), {0, 255})
                self.assertTrue((resized[:size//2] == 0).all())
                self.assertTrue((resized[size//2:, size//2:] == 255).all())
        self.assertEqual(original.read_bytes(), source_bytes)

    def test_child_physical_check_uses_explicit_values_without_reloading_defaults(self):
        from umi.calibration.tools import imu_physical_check as physical
        argv = ['--calib', str(calibration_assets.SAMPLE_CAMERA_JSON), '--euroc', str(self.root/'dataset'),
                '--out', str(self.root/'check'), '--window-s', '.7', '--gyro-max-dps', '3',
                '--acc-std-max', '.06', '--min-window-samples', '25', '--gravity', '9.81',
                '--distinct-dir-cos', '.9', '--min-window-span-fraction', '.85']
        evidence = dict(status=physical.STATUS_REPORT_ONLY, exit_code=0)
        with patch.object(config, 'load', side_effect=AssertionError('unexpected defaults reload')), patch.object(physical, 'analyze', return_value=evidence) as analyze, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(physical.main(argv), 0)
        self.assertEqual(analyze.call_args.args[2:], (.7,3,.06,25,9.81,None,.9,.85))

    def test_child_video_tool_uses_explicit_values_without_reloading_defaults(self):
        from types import SimpleNamespace

        from umi.perception.tools import make_lens_video
        cam = self.root/'mav0/cam0'
        cam.mkdir(parents=True)
        (cam/'data.csv').write_text('#timestamp,filename\n1000000000,frame.png\n')
        video = self.root/'front.mp4'
        video.write_bytes(b'video')
        video_parameters = config.load('trajectory', overrides={'video': {'size_px': 960, 'crf': 11, 'preset': 'slow'}}).values.video
        with patch.object(config, 'load', side_effect=AssertionError('unexpected defaults reload')), patch.object(make_lens_video.subprocess, 'run', return_value=SimpleNamespace(stdout='{"streams":[{"nb_read_frames":"1"}]}')) as run, contextlib.redirect_stdout(io.StringIO()):
            make_lens_video.render('source.insv', str(cam.parent), str(video), 'right', video=video_parameters)
        command = run.call_args_list[0].args[0]
        self.assertEqual(command[command.index('-crf')+1], '11')
        self.assertEqual(command[command.index('-preset')+1], 'slow')
        self.assertIn('scale=960:960:flags=area', command[command.index('-vf')+1])
        manifest = json.loads(video.with_name('front_manifest.json').read_text())
        self.assertEqual(manifest['size'], 960)

    def test_v2_bundle_preserves_measurement_and_rejects_tampering(self):
        bundle = legacy_bundle(self.root / 'bundle')
        description = calibration_assets.describe(bundle, 'user')
        self.assertEqual(description['bundle_schema_version'], 2)
        self.assertEqual(description['measurement'], calibration_assets.read_basalt(calibration_assets.SAMPLE_CAMERA_JSON))
        camera = bundle / 'calibration_r1440.json'
        camera.write_text(camera.read_text()+' ')
        with self.assertRaises(calibration_assets.CalibrationError):
            calibration_assets.describe(bundle, 'user')

    def test_v1_bundle_is_rejected(self):
        target = legacy_bundle(self.root / 'legacy')
        metadata = json.loads((target / 'bundle.json').read_text())
        metadata['schema_version'] = 1
        (target / 'bundle.json').write_text(json.dumps(metadata))
        with self.assertRaises(calibration_assets.CalibrationError):
            calibration_assets.describe(target, 'user')


class SafetyTests(segment_tests.RecordingFixture):
    @staticmethod
    def tree_snapshot(root):
        return {str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob('*') if path.is_file()}

    def test_invalid_configuration_with_force_preserves_existing_results(self):
        self.successful_result()
        before = self.tree_snapshot(self.output)
        path = self.root / 'bad.yaml'
        path.write_text("fusion:\n  imu:\n    acc_noise: -1")
        with patch.object(trajectory_calibration, 'select_calibration') as select, patch.object(runtime.console, 'write'):
            self.assertEqual(workflow.run_batch([str(self.video)], self.output.parent,
                             TrajectoryRequest(config=path, force=True)), 1)
        select.assert_not_called()
        self.assertEqual(before, self.tree_snapshot(self.output))

    def test_dry_run_with_force_creates_no_output_and_does_not_run_engines(self):
        with patch.object(trajectory_calibration, 'select_calibration', return_value=self.selected), patch.object(visual_runtime, 'run_logged') as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(workflow.run_batch([str(self.video)], self.output.parent,
                             TrajectoryRequest(force=True, dry_run=True)), 0)
        execute.assert_not_called()
        self.assertFalse(self.output.parent.exists())
        self.successful_result()
        before = self.tree_snapshot(self.output)
        with patch.object(trajectory_calibration, 'select_calibration', return_value=self.selected), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(workflow.run_batch([str(self.video)], self.output.parent,
                             TrajectoryRequest(force=True, dry_run=True)), 0)
        self.assertEqual(before, self.tree_snapshot(self.output))

    def test_missing_engine_with_force_preserves_old_output(self):
        self.successful_result()
        marker = self.output / 'details/summary.json'
        before = marker.read_bytes()
        with patch.object(trajectory_calibration, 'select_calibration', return_value=self.selected), patch.dict(os.environ, {'ORB3_DIR':'/missing/engine'}), patch.object(runtime.console, 'write'):
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
            camera = output / 'calibration.json'
            before = camera.read_bytes()
            cfg = Path(temp) / 'strict.yaml'
            cfg.write_text("checks:\n  ate_fail_mm: 2.0\n  ate_warn_mm: 1.0")
            with patch.object(calibration_workflow.runtime, 'run_logged') as execute:
                self.assertEqual(calibration_workflow.recheck(output, cfg), 1)
            execute.assert_not_called()
            install.assert_not_called()
            self.assertEqual(camera.read_bytes(), before)
            report = result_io.load_result(output)
            self.assertEqual(report['evaluation']['decision'], 'rejected')
            self.assertEqual(report['evaluation_history'][0]['decision'], 'accepted')
            report_before = (output / 'run/record.json').read_bytes()
            cfg.write_text("board:\n  validation_step: 2")
            self.assertEqual(calibration_workflow.recheck(output, cfg), 2)
            self.assertEqual(report_before, (output / 'run/record.json').read_bytes())

    def test_calibration_overrides_reach_solver_and_physical_check_commands(self):
        helper = calibration_tests.FullWorkflowTests()
        with tempfile.TemporaryDirectory() as temp, helper.simulation(Path(temp)) as (args, commands, install):
            path = Path(temp) / 'custom.yaml'
            path.write_text("imu:\n  gyro_noise: 0.0004\ngravity:\n  window_s: 0.7\n  minimum_window_span_fraction: 0.9\nboard:\n  intrinsics_step: 7\n  validation_step: 2\n")
            args.config = path
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            imu_command = next(command for command in commands if '--gyro-noise-std' in command)
            self.assertEqual(float(imu_command[imu_command.index('--gyro-noise-std')+1]), .0004)
            gravity_command = next(command for command in commands if any('umi.calibration.tools.imu_physical_check' in part for part in command))
            self.assertEqual(float(gravity_command[gravity_command.index('--window-s')+1]), .7)
            self.assertEqual(float(gravity_command[gravity_command.index('--min-window-span-fraction')+1]), .9)
            from umi.calibration.pipeline import board_detection
            self.assertEqual({call.args[0].name: call.kwargs['step'] for call in board_detection.detect_all.call_args_list},
                             {'A': 7, 'C': 2})
            report = result_io.load_result(Path(args.out))
            self.assertEqual(report['computation']['configuration']['values']['imu']['gyro_noise'], .0004)
            install.assert_not_called()

    def test_trajectory_notes_update_reuses_trajectory_and_updates_parameter_records(self):
        helper = segment_tests.WorkflowTests()
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
            cfg.write_text("reporting:\n  warnings:\n    minimum_localized_fraction: 0.8")
            with patch.object(reconstruction, 'run_attempts') as reconstruct:
                self.assertEqual(helper.run_recording(TrajectoryRequest(config=cfg))['status'], 'skipped')
            reconstruct.assert_not_called()
            self.assertEqual(trajectory.read_bytes(), before)
            effective = json.loads(settings.read_text())['parameters']['configuration']
            self.assertEqual(effective['values']['notes']['minimum_full_fraction'], .8)
            self.assertEqual(results.load_summary(helper.output)['parameters']['configuration'], effective)
        finally:
            helper.doCleanups()


# 操作配置的冻结、覆盖、资源预算与缓存边界。


class OperationalSettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for kind in settings.SCHEMA:
            (self.root / f'{kind}.yaml').write_bytes((settings.DIRECTORY / f'{kind}.yaml').read_bytes())
        self.defaults = {kind: dict(settings.load(kind, directory=self.root, environment={}).values)
                         for kind in settings.SCHEMA}
        directory = patch.object(settings, 'DIRECTORY', self.root)
        directory.start()
        self.addCleanup(directory.stop)
        environment = patch.dict(os.environ, dict(os.environ))
        environment.start()
        self.addCleanup(environment.stop)
        for name in (settings.SNAPSHOT_ENV, 'UMI_OPENCV_THREADS'):
            os.environ.pop(name, None)
        settings.current.cache_clear()
        self.addCleanup(settings.current.cache_clear)
        media.shutdown()
        self.addCleanup(media.shutdown)

    def overlay(self, kind, **values):
        write_yaml(self.root / f'{kind}.yaml', {**self.defaults[kind], **values})
        settings.current.cache_clear()

    def test_local_override_files_are_not_read(self):
        write_yaml(self.root / 'runtime.local.yaml', {'decode_threads': 2})
        configured = settings.load('runtime', environment={})
        self.assertEqual(configured.values['decode_threads'], self.defaults['runtime']['decode_threads'])
        self.assertEqual(configured.sources, (str(self.root / 'runtime.yaml'),))

    def test_defaults_environment_cli_and_origins(self):
        self.overlay('runtime', opencv_threads=3, decode_threads=2)
        configured = settings.load('runtime', environment={'UMI_OPENCV_THREADS': '6'})
        self.assertEqual(configured.values['opencv_threads'], 6)
        self.assertEqual(configured.values['decode_threads'], 2)
        self.assertEqual(configured.origins['opencv_threads'], 'environment:UMI_OPENCV_THREADS')
        self.assertTrue(configured.origins['decode_threads'].endswith('runtime.yaml'))
        with self.assertRaises(TypeError):
            configured.values['decode_threads'] = 7
        self.overlay('viewer', port=8000)
        configured = settings.load('viewer', overrides={'port': 0})
        self.assertEqual(configured.values['port'], 0)
        self.assertEqual(configured.origins['port'], 'command_line:--port')
        self.assertEqual(self.defaults['viewer']['port'], 9092)

    def test_invalid_fields_types_ranges_duplicates_and_relations(self):
        cases = [('runtime', {'decode_threads': True}), ('runtime', {'decode_threads': 0}),
                 ('runtime', {'decode_threads': 1.5}), ('runtime', {'decode_threads': None}),
                 ('runtime', {'unknown': 2}), ('viewer', {'port': 65536}),
                 ('viewer', {'preview_workers': 3, 'preview_pending': 2}),
                 ('viewer', {'preview_width': 721}), ('viewer', {'map_max_points': 50001}),
                 ('viewer', {'host': ''}), ('viewer', {'preview_timeout_s': float('inf')}),
                 ('gripper_tag_detection', {'recovery_batch_size': -1})]
        for kind, overlay in cases:
            with self.subTest(kind=kind, overlay=overlay):
                self.overlay(kind, **overlay)
                with self.assertRaises(ValueError):
                    settings.load(kind, environment={})
                write_yaml(self.root / f'{kind}.yaml', self.defaults[kind])
        (self.root / 'viewer.yaml').write_text('port: 9092\nport: 9093\n')
        with self.assertRaisesRegex(ValueError, 'viewer.yaml'):
            settings.load('viewer')

    def test_snapshot_is_frozen_in_process_and_in_child(self):
        first = settings.current('runtime')
        write_yaml(self.root / 'runtime.yaml', {**self.defaults['runtime'], 'decode_threads': 2})
        self.assertIs(settings.current('runtime'), first)
        env = settings.child_environment()
        # The child uses the snapshot even though the YAML now differs.
        code = "from umi.config import operational_settings as settings; print(settings.values('runtime')['decode_threads'])"
        child = subprocess.run([sys.executable, '-c', code], env=env, text=True,
                               capture_output=True, check=True)
        self.assertEqual(int(child.stdout), first.values['decode_threads'])
        self.assertEqual(settings.load('runtime').values['decode_threads'], 2)

    def test_budget_respects_cpu_affinity_cli_and_custom_threshold(self):
        self.overlay('runtime', calibration_max_workers=6, calibration_reserved_cpus=2, arm_min_cpus=3)
        with patch.object(settings.os, 'sched_getaffinity', return_value=set(range(16))):
            report = settings.execution_report(cores='2-7', workers=20, calibration=True)
            self.assertEqual(report['resources']['available_cpus'], list(range(2, 8)))
            self.assertEqual(report['resources']['calibration_workers'], 6)
            self.assertEqual(report['resources']['opencv_threads'], 1)
            self.assertEqual(settings.calibration_workers(cores='2-7'), 4)
            self.assertEqual(recordings.arm_cores(('left', 'right'), '2-7', parallel=True),
                             {'left': '2,4,6', 'right': '3,5,7'})
            self.assertFalse(settings.execution_report(jobs=2)['resources']['dual_arm_eligible'])
            with self.assertRaisesRegex(ValueError, '--cores'):
                settings.execution_report(cores='40-42')

    def test_runtime_changes_reuse_detection_but_detection_changes_invalidate_final_inputs(self):
        original = [function() for function in (frame_detection.identity, width_estimation.identity, tag_distance.identity)]
        self.overlay('runtime', decode_threads=2, slam_timeout_base_s=900)
        self.assertEqual(original, [function() for function in (frame_detection.identity, width_estimation.identity, tag_distance.identity)])
        self.overlay('gripper_tag_detection', candidate_limit=48, full_search_interval=10)
        for before, function in zip(original, (frame_detection.identity, width_estimation.identity, tag_distance.identity)):
            self.assertNotEqual(before, function())
        detector = MarkerDetector()
        self.assertEqual(detector._candidate_limit, 48)
        self.assertEqual(detector._full_search_interval, 10)

    def test_gripper_roi_default_is_shared_by_scene_width_commands_and_detection(self):
        from umi.apps import calib
        from umi.perception.gripper import video_tools
        from umi.scene.collection.configuration import load_scene_parameters

        roi = (620, 990, 1330, 1240)
        prior_identity = frame_detection.identity()
        self.overlay('gripper_tag_detection', roi_px=list(roi))
        self.assertEqual(load_scene_parameters().tag_distance_roi_px, list(roi))
        self.assertEqual(load_scene_parameters(overrides=dict(tag_distance_roi_px=[1, 2, 3, 4])).tag_distance_roi_px,
                         [1, 2, 3, 4])
        self.assertNotEqual(frame_detection.identity(), prior_identity)
        self.assertEqual(frame_detection.identity()['default_roi_px'], list(roi))
        for command, function in (('preview-width', 'preview'), ('detect-width', 'monitor')):
            with self.subTest(command=command), patch.object(video_tools, function, return_value={}) as called, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                for value, extra in ((roi, []), ([5, 6, 7, 8], ['--roi', '5', '6', '7', '8'])):
                    self.assertEqual(calib.main([command, 'recording.insv', *extra]), 0)
                    selected_roi = called.call_args.kwargs['roi'] if function == 'preview' else called.call_args.args[1]
                    self.assertEqual(selected_roi, value)

    def test_viewer_content_changes_invalidate_only_related_artifacts(self):
        video = self.root / 'video.mp4'
        video.write_bytes(b'video')
        with patch.object(media, 'find_ffmpeg', return_value='/ffmpeg'):
            preview, frame = media._preview_target(video), media.video_version(video)
            self.overlay('viewer', preview_workers=2, preview_pending=3, preview_timeout_s=900,
                         preview_encode_threads=2, frame_cache_entries=1)
            self.assertEqual(media._preview_target(video), preview)
            self.assertEqual(media.video_version(video), frame)
            self.overlay('viewer', preview_crf=20)
            self.assertNotEqual(media._preview_target(video), preview)
            self.assertEqual(media.video_version(video), frame)
            self.overlay('viewer', frame_quality=8)
            self.assertEqual(media._preview_target(video), preview)
            self.assertNotEqual(media.video_version(video), frame)

    def test_viewer_configured_cache_and_single_resource_bundle(self):
        self.overlay('viewer', frame_cache_entries=1, preview_workers=2, preview_pending=3,
                     decode_concurrency=3)
        with ThreadPoolExecutor(max_workers=8) as threads:
            bundles = list(threads.map(lambda _: media._resources(), range(20)))
        self.assertTrue(all(bundle is bundles[0] for bundle in bundles))
        semaphore, executor = bundles[0]
        self.assertEqual(executor._max_workers, 2)
        for _ in range(3):
            self.assertTrue(semaphore.acquire(blocking=False))
        self.assertFalse(semaphore.acquire(blocking=False))
        for _ in range(3):
            semaphore.release()
        calls = []
        @cache.bounded('frame_cache_entries')
        def invoke(key):
            calls.append(key)
            return key
        for key in (1, 1, 2, 1):
            invoke(key)
        self.assertEqual(calls, [1, 2, 1])

    def test_display_budget_does_not_reject_or_mutate_published_map(self):
        self.overlay('viewer', map_max_points=2)
        document = dict(kind='umi_map_view', schema_version=1, maps=[dict(index=0,
            coordinates='map', alignment=dict(status='unaligned'), points=[[i, 0, 0] for i in range(5)],
            keyframes=[], bounds=None)])
        self.assertEqual(len(maps.validate(document)['maps'][0]['points']), 5)
        with patch.object(maps, '_load', return_value=document):
            self.assertEqual(len(maps.load(self.root, {})['maps'][0]['points']), 2)
        self.assertEqual(len(document['maps'][0]['points']), 5)

    def test_viewer_show_config_uses_yaml_and_cli_without_resources(self):
        self.overlay('viewer', port=8000, preview_workers=2, preview_pending=3)
        with patch.object(server, 'RecordingCatalog') as catalog, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(server.main(['--show-config', '--port', '0']), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report['effective']['port'], 0)
        self.assertEqual(report['effective']['preview_workers'], 2)
        self.assertEqual(report['map_export']['timeout_s'], 180)
        self.assertEqual(report['configuration']['origins']['port'], 'command_line:--port')
        self.assertIsNone(media._resource_pair)
        catalog.assert_not_called()

    def test_invalid_config_is_reported_before_listen_and_help_still_works(self):
        self.overlay('viewer', port='oops')
        with patch.object(server, 'ThreadingHTTPServer') as listen, redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as error:
                server.main([])
        self.assertEqual(error.exception.code, 2)
        self.assertIn('viewer.yaml', output.getvalue())
        listen.assert_not_called()
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as help_exit:
            server.main(['--help'])
        self.assertEqual(help_exit.exception.code, 0)


class CollectionDefaultsTests(unittest.TestCase):
    def test_missing_file_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                collection_defaults.load(Path(directory) / 'missing.yaml')

    def test_project_file_uses_the_collection_layout(self):
        config = collection_defaults.load()
        self.assertEqual(config.path, collection_defaults.DEFAULT_PATH.resolve())
        self.assertEqual(set(config.values), {'audio_sync', 'map', 'backend'})
        self.assertEqual(config.report(), dict(path=str(config.path), values=dict(config.values)))

    def test_normalizes_declared_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'collection_defaults.yaml'
            write_yaml(path, dict(sync=dict(mode='assumed_simultaneous', algorithm='kiwi'),
                                  map=dict(layout='shared', reference_arm='right'),
                                  backend=dict(mode='poster_vio', auto_order=['poster_vio', 'scene_map'],
                                               poster_vio=dict(base_tag=False))))
            config = collection_defaults.load(path)
        self.assertEqual(config.values, dict(audio_sync=dict(enabled=False, algorithm='kiwi'),
                                             map=dict(layout='shared', reference_arm='right'),
                                             backend=dict(mode='poster_vio', auto_order=('poster_vio', 'scene_map'),
                                                          base_tag_enabled=False)))

    def test_invalid_missing_and_out_of_scope_fields_are_rejected(self):
        default = read_yaml(collection_defaults.DEFAULT_PATH)
        for path, value in (
                (('sync', 'mode'), 'manual'),
                (('sync', 'mode'), True),
                (('sync', 'algorithm'), 'bogus'),
                (('map', 'layout'), 'bogus'),
                (('map', 'reference_arm'), 'both'),
                (('map', 'version'), 'map-v1'),
                (('map', 'poster_scale'), 'correct'),
                (('backend', 'mode'), 'bogus'),
                (('backend', 'auto_order'), []),
                (('backend', 'auto_order'), ['poster_vio', 'poster_vio']),
                (('backend', 'auto_order'), ['auto']),
                (('backend', 'poster_vio', 'base_tag'), 1),
                (('backend', 'poster_vio', 'base_tag'), {'enabled': True}),
                (('backend', 'poster_vio', 'robust_inertial_seed'), True),
                (('backend', 'scene_map'), {}),
                (('kit',), 'kit_001'),
                (('tasks',), {}),
                (('schema_version',), 2)):
            document = deepcopy(default)
            section = document
            for key in path[:-1]:
                section = section[key]
            section[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_collection_defaults(document)
        for path in (('sync',), ('sync', 'mode'), ('sync', 'algorithm'), ('map', 'layout'), ('backend', 'mode'),
                     ('backend', 'auto_order'), ('backend', 'poster_vio', 'base_tag')):
            document = deepcopy(default)
            section = document
            for key in path[:-1]:
                section = section[key]
            section.pop(path[-1])
            with self.subTest(missing=path), self.assertRaises(ValueError):
                validate_collection_defaults(document)


class CollectionSchemaTests(unittest.TestCase):
    def base(self):
        return dict(
            kit='kit_001',
            arms=['left', 'right'],
            description='shelf',
            poster=dict(layout='poster.json', detection=True),
            backend=dict(mode='auto', scene_map=dict(poster_constraint='staged')))

    def test_reads_project_defaults_and_honors_explicit_overrides(self):
        with TemporaryDirectory() as directory:
            options = defaults_document()
            options['backend']['mode'] = 'poster_vio'
            options['map'] = dict(layout='per_arm', reference_arm='right')
            with patched_defaults(directory, options):
                document = dict(kit='kit', arms=['left'], mapping={})
                inherited = normalize_collection(document)
                overridden = normalize_collection({**document, 'backend': {'mode': 'auto'},
                                                   'map': {'layout': 'shared'}})
        self.assertEqual(inherited['reconstruction']['backend'], 'poster_vio')
        self.assertEqual(inherited['scene']['layout'], 'per_arm')
        self.assertEqual(inherited['scene']['map_reference_arm'], 'right')
        self.assertEqual(overridden['reconstruction']['backend'], 'auto')
        self.assertEqual(overridden['scene']['layout'], 'shared')
        self.assertEqual(overridden['scene']['map_reference_arm'], 'right')

    def test_normalizes_map_poster_reconstruction_and_task_override(self):
        document = self.base()
        document['backend']['poster_vio'] = dict(base_tag=False)
        document.update(map=dict(layout='per_arm', reference_arm='right', version='map-v1'),
                        tasks=dict(take_001=dict(
                            arms=['right'], sync='assumed_simultaneous',
                            map=dict(epoch='epoch_001'),
                            backend=dict(mode='poster_vio', poster_vio=dict(base_tag=True)))))
        normalized = normalize_collection(document)
        self.assertNotIn('schema_version', normalized)
        self.assertEqual(normalized['scene']['layout'], 'per_arm')
        self.assertEqual(normalized['scene']['map_reference_arm'], 'right')
        self.assertEqual(normalized['scene']['description'], 'shelf')
        self.assertEqual(normalized['map_version'], 'map-v1')
        self.assertEqual(normalized['tasks']['take_001']['expected_arms'], ['right'])
        self.assertEqual(normalized['tasks']['take_001']['map_epoch'], 'epoch_001')
        self.assertEqual(normalized['tasks']['take_001']['synchronization'], 'assumed_simultaneous')
        self.assertEqual(normalized['tasks']['take_001']['reconstruction']['backend'], 'poster_vio')
        self.assertFalse(normalized['reconstruction']['base_tag']['enabled'])
        self.assertTrue(normalized['tasks']['take_001']['reconstruction']['base_tag']['enabled'])

    def test_omitted_poster_depends_on_backends_instead_of_organizer_layout(self):
        document = dict(kit='kit_001', arms=['left'])
        inherited = normalize_collection(document)
        explicit = {**document, 'backend': {'mode': 'auto'}}
        self.assertEqual(normalize_collection(explicit), inherited)
        self.assertEqual(normalize_collection({**explicit, 'mapping': {}}), inherited)
        self.assertNotIn('tags', normalize_collection({**explicit, 'poster': None})['scene'])
        map_only = {**explicit, 'backend': {'mode': 'auto', 'auto_order': ['scene_map']}}
        self.assertNotIn('tags', normalize_collection(map_only)['scene'])
        map_only['tasks'] = {'take_001': {'backend': {'mode': 'poster_vio'}}}
        self.assertIn('tags', normalize_collection(map_only)['scene'])

    def test_task_keys_mirror_the_batch_keys(self):
        document = self.base()
        document.update(instruction='batch instruction', tasks=dict(
            take_001={}, take_002=dict(instruction='own instruction', sync=dict(
                mode='manual', offsets_s=dict(left=0., right=.1), confirmed=True, note='checked'))))
        tasks = normalize_collection(document)['tasks']
        self.assertEqual(tasks['take_001']['instruction'], 'batch instruction')
        self.assertEqual(tasks['take_002']['instruction'], 'own instruction')
        self.assertEqual(tasks['take_002']['synchronization']['offsets_s'], dict(left=0., right=.1))

    def test_task_base_tag_switch_can_inherit_batch_setting(self):
        document = self.base()
        document['backend']['poster_vio'] = dict(base_tag=False)
        document['tasks'] = dict(take_001=dict(backend=dict(mode='poster_vio')))
        normalized = normalize_collection(document)
        self.assertNotIn('base_tag', normalized['tasks']['take_001']['reconstruction'])

    def test_arms_are_required_and_tasks_cannot_expand_them(self):
        document = self.base()
        document.pop('arms')
        with self.assertRaisesRegex(ValueError, '需要 arms'):
            normalize_collection(document)
        document['arms'] = ['right']
        document['tasks'] = dict(take_001=dict(arms=['left']))
        with self.assertRaisesRegex(ValueError, 'tasks.take_001.arms'):
            normalize_collection(document)
        for value in (None, [], ['right', 'right'], [['right']]):
            document['tasks']['take_001']['arms'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_collection(document)

    def test_batch_sync_mode_selects_the_default_for_two_arm_tasks(self):
        document = self.base()
        self.assertNotIn('audio_sync_enabled', normalize_collection(document))
        for mode, enabled in (('audio_auto', True), ('assumed_simultaneous', False)):
            document['sync'] = dict(mode=mode)
            self.assertIs(normalize_collection(document)['audio_sync_enabled'], enabled)
        document['sync'] = dict(algorithm='kiwi')
        self.assertEqual(normalize_collection(document)['audio_sync_algorithm'], 'kiwi')
        for record in (dict(mode='manual'), dict(mode=True), dict(enabled=True), dict(algorithm='other')):
            document['sync'] = record
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, r'sync'):
                normalize_collection(document)

    def test_poster_scale_replaces_the_mapping_switches(self):
        document = self.base()
        expected = dict(off=(False, False), diagnose=(True, False), correct=(True, True))
        for scale, (mapping, correction) in expected.items():
            document['map'] = dict(poster_scale=scale)
            tags = normalize_collection(document)['scene']['tags']
            with self.subTest(scale=scale):
                self.assertEqual((tags['mapping']['enabled'], tags['mapping']['scale']['enabled']),
                                 (mapping, correction))
        document['map'] = dict(poster_scale='on')
        with self.assertRaisesRegex(ValueError, 'map.poster_scale'):
            normalize_collection(document)
        document['map'] = dict(poster_scale='diagnose')
        document['poster']['detection'] = False
        with self.assertRaisesRegex(ValueError, '冲突'):
            normalize_collection(document)
        document.pop('poster')
        document['backend'] = dict(mode='scene_map')
        with self.assertRaisesRegex(ValueError, 'poster.layout'):
            normalize_collection(document)

    def test_poster_constraint_is_one_choice(self):
        document = self.base()
        for value, policy in (('staged', dict(enabled=True, method='staged')),
                              ('joint', dict(enabled=True, method='joint')), ('off', dict(enabled=False))):
            document['backend']['scene_map'] = dict(poster_constraint=value)
            with self.subTest(value=value):
                self.assertEqual(normalize_collection(document)['reconstruction']['scene_map_poster_constraint'],
                                 policy)

    def test_rejects_schema_version_and_conflicting_task_map_selectors(self):
        with self.assertRaisesRegex(ValueError, '未知字段.*schema_version'):
            normalize_collection(dict(schema_version=1))
        document = self.base()
        document['tasks'] = dict(take_001=dict(map=dict(epoch='e', version='v')))
        with self.assertRaisesRegex(ValueError, '不能同时'):
            normalize_collection(document)

    def test_unsupported_collection_fields_use_the_same_validation(self):
        for document, field in (
                (dict(kit='kit_001', recording=dict(arms=['left'])), 'recording'),
                (dict(kit='kit_001', arms=['left'], scene=dict(description='x')), 'scene'),
                (dict(kit='kit_001', arms=['left'], environment_poster=dict(layout_path='p.json')),
                 'environment_poster'),
                (dict(kit='kit_001', arms=['left'], map=dict(selection=dict(version='v'))), 'map.selection'),
                (dict(kit='kit_001', arms=['left'], backend=dict(mode='auto', poster_edge_trim={})),
                 'backend.poster_edge_trim'),
                (dict(kit='kit_001', arms=['left'], backend=dict(
                    mode='auto', poster_vio=dict(initialization=dict(max_observation_gap_s=3.)))),
                 'backend.poster_vio.initialization'),
                (dict(kit='kit_001', arms=['left'], tasks=dict(take_001=dict(synchronization='audio_auto'))),
                 'tasks.take_001.synchronization')):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, rf'未知字段.*{field.rsplit(".", 1)[-1]}'):
                normalize_collection(document)

    def test_rejects_unknown_backend_poster_and_task_fields(self):
        document = self.base()
        document['reconstruction'] = dict(backend='auto')
        with self.assertRaisesRegex(ValueError, '未知字段.*reconstruction'):
            normalize_collection(document)
        document.pop('reconstruction')
        for field, value in (('task_constraint', True), ('layout_path', 'p.json')):
            document['poster'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'poster 含未知字段'):
                normalize_collection(document)
            document['poster'].pop(field)
        document['tasks'] = dict(take_001=dict(reconstruction={}))
        with self.assertRaisesRegex(ValueError, r'tasks\.take_001 含未知字段'):
            normalize_collection(document)

    def test_backend_reports_invalid_options_at_the_public_paths(self):
        document = dict(kit='kit_001', arms=['right'])
        for section, options, field in (
            ('scene_map', dict(poster_constraint='other'), 'poster_constraint'),
            ('poster_vio', dict(base_tag=None), 'base_tag'),
            ('poster_vio', dict(maximum_observation_gap_s=float('inf')), 'maximum_observation_gap_s'),
            ('poster_independent', dict(maximum_position_support_gap_s=False), 'maximum_position_support_gap_s'),
            ('poster_independent', dict(maximum_gravity_difference_deg=181), 'maximum_gravity_difference_deg'),
            ('edge_trim', dict(minimum_missing_duration_s=0), 'minimum_missing_duration_s'),
        ):
            document['backend'] = {'mode': 'auto', section: options}
            with self.subTest(section=section, field=field), self.assertRaisesRegex(ValueError, field):
                normalize_collection(document)

    def test_rejects_invalid_backend_declarations(self):
        for record in (dict(mode=True), dict(mode='auto', extra=True), 'auto'):
            document = dict(kit='kit_001', arms=['right'], backend=record)
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, 'backend'):
                normalize_collection(document)
        with self.assertRaisesRegex(ValueError, 'reconstruction'):
            normalize_collection(dict(kit='kit_001', arms=['right'], reconstruction=dict(mode='auto')))


# reconstruction.yaml 公开参数树与内部轨迹/场景字段之间的契约测试。


def _leaf_paths(values: dict, prefix=()):
    for name, value in values.items():
        if isinstance(value, dict):
            yield from _leaf_paths(value, (*prefix, name))
        else:
            yield (*prefix, name)


class ReconstructionSchemaTests(unittest.TestCase):
    def test_every_internal_field_has_exactly_one_public_source(self):
        trajectory = {('trajectory', *path) for path in _leaf_paths(asdict(config.load('trajectory').values))}
        trajectory -= {('trajectory', 'schema_version')}
        scene = {('scene', field.name) for field in fields(SceneParameters)}
        # 内部版本号、collection.yaml 推导的策略开关和来自运行配置的 ROI 不属于算法参数文件。
        scene -= {('scene', name) for name in ('schema_version', 'tag_distance_roi_px', *reconstruction_config.POSTER_POLICY_DEFAULTS)}
        targets = [target for path, values in reconstruction_config.FIELDS.items() if path not in reconstruction_config.OPTIONAL
                   for target in values]
        self.assertEqual(len(targets), len(set(targets)))
        self.assertEqual(set(targets), trajectory | scene)

    def test_project_defaults_are_complete_and_strict(self):
        document = read_yaml(reconstruction_config.DEFAULT_PATH)
        self.assertEqual(document['schema_version'], reconstruction_config.SCHEMA_VERSION)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reconstruction.yaml'
            for mutate, message in (
                    (lambda d: d['fusion']['solver'].pop('node_stride'), '缺少字段'),
                    (lambda d: d['fusion']['solver'].update(stride=5), '未知字段'),
                    (lambda d: d.update(schema_version=2), 'schema_version'),
                    (lambda d: d.update(schema_version=3.0), 'schema_version'),
                    (lambda d: d.pop('schema_version'), 'schema_version'),
                    (lambda d: d.update(trajectory={}), 'trajectory 是未知字段')):
                broken = copy.deepcopy(document)
                mutate(broken)
                write_yaml(path, broken)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    reconstruction_config.split(read_yaml(path), label=str(path), complete=True)

    def test_public_tree_round_trips_the_resolved_values(self):
        from umi.scene.collection.configuration import load_scene_parameters
        trajectory = asdict(config.load('trajectory').values)
        scene = asdict(load_scene_parameters())
        tree = reconstruction_config.public_tree(trajectory=trajectory, scene=scene)
        self.assertEqual(tree['schema_version'] if 'schema_version' in tree else None, None)
        document = read_yaml(reconstruction_config.DEFAULT_PATH)
        document.pop('schema_version')
        for path in reconstruction_config.OPTIONAL:
            tree_section = tree
            for key in path[:-1]:
                tree_section = tree_section[key]
            tree_section.pop(path[-1])
        self.assertEqual(tree, document)

    def test_shared_poster_fields_reach_both_engines(self):
        result = reconstruction_config.split(dict(poster=dict(acceptance=dict(maximum_position_std_m=.02))), label='x')
        self.assertEqual(result.trajectory, dict(environment_tags=dict(tag_maximum_position_std_m=.02)))
        self.assertEqual(result.scene, dict(tag_maximum_position_std_m=.02))

    def test_backend_gap_overrides_the_shared_default_regardless_of_yaml_order(self):
        document = dict(backends=dict(poster_independent=dict(maximum_position_support_gap_s=4.)),
                        poster=dict(acceptance=dict(maximum_position_support_gap_s=3.)))
        origins = {}
        values = reconstruction_config.split(document, label='expert.yaml', gap_sources=origins).scene
        self.assertEqual(values['poster_maximum_position_support_gap_s'], 4.)
        self.assertEqual(values['poster_vio_maximum_position_support_gap_s'], 3.)
        self.assertEqual(origins['poster_maximum_position_support_gap_s'],
                         'expert.yaml:backends.poster_independent.maximum_position_support_gap_s')
        self.assertEqual(origins['poster_vio_maximum_position_support_gap_s'],
                         'expert.yaml:poster.acceptance.maximum_position_support_gap_s')

    def test_segment_overlays_reject_scene_only_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'segment.yaml'
            write_yaml(path, dict(localization=dict(minimum_localized_fraction=.4)))
            with self.assertRaisesRegex(ValueError, '只用于场景流程'):
                reconstruction_config.read_overlay(path, trajectory_only=True)
            write_yaml(path, dict(poster=dict(acceptance=dict(maximum_position_std_m=.02))))
            self.assertEqual(reconstruction_config.read_overlay(path, trajectory_only=True).scene, {})
