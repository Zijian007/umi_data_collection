"""正式 rear 不依赖海报：预检、冻结、子进程传递和数值时间轴。"""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from software import config, provenance, registry
from software.storage import write_json
from software.trajectory.algorithms import fusion
from software.trajectory.calibration import assets, selection
from software.trajectory.models import TrajectoryRequest
from software.trajectory.pipeline import calibration_assets, poster_assets, reuse
from software.trajectory import workflow
from test_trajectory_workflow import RecordingFixture
import test_trajectory_workflow as workflow_tests


class RearAssetsTests(RecordingFixture):
    def setUp(self):
        super().setUp()
        raw = json.loads(assets.SAMPLE_CAMERA_JSON.read_text())
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
        selected_assets = calibration_assets.select_environment_assets(self.video, self.selected, self.request)
        rear = selected_assets.rear
        self.assertIsNone(selected_assets.poster)
        fingerprint = self.fingerprint(self.request)
        self.assertEqual(fingerprint['calibration']['assets']['rear_calibration'], rear.identity.sha256)
        self.assertNotIn('environment_poster', fingerprint['calibration']['assets'])
        self.assertNotIn('rear_geometry', fingerprint['calibration']['assets'])
        manifest = calibration_assets.freeze_rear_calibration(rear, self.output/'devices/rear')
        self.assertEqual(calibration_assets.load_frozen_rear(manifest)['cam_time_offset_ns'], assets.read_basalt(rear.path)['cam_time_offset_ns'])
        self.successful_result()
        write_json(self.output/'summary.json', dict(status='ok', run_fingerprint=fingerprint))
        self.assertIsNotNone(reuse.reuse_decision(self.output, fingerprint, notes=self.request.configuration.values.notes).summary)
        (manifest.parent/'calibration.json').write_text('{}')
        self.assertIsNone(reuse.reuse_decision(self.output, fingerprint, notes=self.request.configuration.values.notes).summary)

    def test_missing_rear_refuses_before_force_can_remove_old_result(self):
        self.successful_result()
        before = (self.output/'summary.json').read_bytes()
        with patch.object(selection, 'select_calibration', return_value=self.selected):
            result = workflow.run_recording(self.video, self.output.parent, TrajectoryRequest(mode='dual', force=True))
        self.assertEqual(result['status'], 'failed')
        self.assertIn('无法读取 INSV 工厂参数', result['reason'])
        self.assertEqual(before, (self.output/'summary.json').read_bytes())

    def test_rejects_unused_rear_and_frozen_source_changes(self):
        with self.assertRaisesRegex(ValueError, '未使用'):
            calibration_assets.select_environment_assets(self.video, self.selected, replace(self.request, mode='handheld'))
        selected_assets = calibration_assets.select_environment_assets(self.video, self.selected, self.request)
        rear = selected_assets.rear
        rear.path.write_text('{}')
        with self.assertRaisesRegex(ValueError, '预检后'):
            calibration_assets.freeze_rear_calibration(rear, self.output/'devices/rear')

    def test_workflow_passes_frozen_rear_without_triggering_poster_detection(self):
        # Use the established fake external engines but the real orchestration.
        helper = workflow_tests.WorkflowTests('test_success_skip_and_reprocess_after_calibration_change')
        helper.setUp(); self.addCleanup(helper.doCleanups)
        commands = []
        original = helper.fake_runner
        def run(command, **kwargs):
            if '--lever_prior' in command:
                manifest = Path(command[command.index('--rear-calibration')+1])
                self.assertTrue(manifest.is_file())
                calibration_assets.load_frozen_rear(manifest)
                self.assertNotIn('--poster', command)
                commands.append(command)
            return original(command, **kwargs)
        with patch.object(helper, 'fake_runner', side_effect=run), patch.object(poster_assets, 'prepare') as poster:
            result = helper.run_recording(self.request)
        self.assertEqual(result['status'], 'ok', result)
        self.assertTrue(commands)
        self.assertEqual(result['rear_calibration']['version'], self.version)
        self.assertTrue((helper.output/'devices/rear/calibration.json').is_file())
        poster.assert_not_called()

    def test_wrong_source_rejected_by_same_validator_as_scene(self):
        with self.assertRaisesRegex(ValueError, '来源'):
            calibration_assets.select_environment_assets(self.video, replace(self.selected, intrinsics_source='factory'), self.request)

    def test_poster_and_fingerprint_share_one_explicit_asset_selection(self):
        from software.vision import tag_board
        request = replace(self.request, tag_layout=tag_board.DEFAULT_LAYOUT)
        with patch.object(registry, 'rear_path', wraps=registry.rear_path) as resolve_rear:
            selected_assets = calibration_assets.select_environment_assets(self.video, self.selected, request)
            resolve_rear.assert_called_once_with('camera', self.version)
        with patch.object(registry, 'rear_path', side_effect=AssertionError('must not reselect')):
            fingerprint = reuse.run_fingerprint(self.video, self.selected, request, environment_assets=selected_assets)
        board_record = selected_assets.poster.board.record()
        expected_poster_hash = provenance.canonical_hash(dict(
            layout=board_record, rear_sha256=selected_assets.rear.identity.sha256, rear_version=self.version))
        self.assertEqual(fingerprint['calibration']['assets']['environment_poster'], expected_poster_hash)
        with self.assertRaisesRegex(ValueError, '已选 rear'):
            calibration_assets.EnvironmentAssets(poster=selected_assets.poster)


class RearNumericalTests(unittest.TestCase):
    def test_dual_without_poster_matches_diagnostic_only_baseline(self):
        from test_independent_poster import fixture
        from software.camera.geometry import imu_from_camera, resample_pose_series
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
        diagnosed = fusion.solve(inputs, parameters, log=lambda *a, **k: None)
        standalone = fusion.solve(replace(inputs, rear_camera=inputs.poster.measured, poster=None), parameters, log=lambda *a, **k: None)
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
        from software.trajectory.algorithms import fusion_cli
        from test_independent_poster import write_camera
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            front, rear = root/'front.json', root/'rear.json'
            write_camera(front, inputs.camera)
            write_camera(rear, inputs.poster.measured, rear=True)
            selected = calibration_assets.SelectedRearCalibration(rear, calibration_assets.RearCalibrationIdentity(
                serial='test-camera', version='rear_test', intrinsics_source='measured', sha256=provenance.file_hash(rear)))
            manifest = calibration_assets.freeze_rear_calibration(selected, root/'frozen')
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
                code = fusion_cli.main([str(root/'rear.csv'), str(root/'front.csv'), str(root/'imu.csv'),
                    '17000000', '0,0,0,1', str(root/'out'), '--lever_prior', '0,0,0',
                    '--calibration', str(front), '--rear-calibration', str(manifest),
                    '--parameters', str(root/'parameters.json'), '--times', str(root/'times.csv')])
            self.assertEqual(code, 0)
            evidence = json.loads((root/'out/fuse_info.json').read_text())
            self.assertNotIn('environment_poster', evidence)
            self.assertGreater(evidence['obs']['rear'], 0)
            np.testing.assert_allclose(np.loadtxt(root/'out/traj_tum.txt')[:, 1:4],
                                       standalone.T_world_imu[:, :3, 3], atol=2e-6)

    def test_conflicting_poster_and_rear_measurements_are_rejected(self):
        from test_independent_poster import fixture
        inputs, parameters = fixture()
        with self.assertRaisesRegex(ValueError, 'rear 标定不一致'):
            fusion.solve(replace(inputs, rear_camera=dict(inputs.poster.measured, cam_time_offset_ns=0)), parameters)


if __name__ == '__main__':
    unittest.main()
