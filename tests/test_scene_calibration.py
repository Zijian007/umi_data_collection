"""后镜头裁切与 A/B/D/C 流程；求解器替代，C 的尺度/融合和独立误差计算使用真实数值代码。"""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from software.calibration.pipeline import board_detection
from software.calibration.pipeline import recordings
from software.calibration import cli, workflow, result_io
from software.calibration.solve import basalt
from software.calibration.validation import gravity
from software.calibration.algorithms import board_pnp
from software.calibration.models import CalibrationInputs, Decision, CameraImuResult, GravityResult, IntrinsicsResult, SegmentPaths
from software.scene.spatial import rear_calibration
from software.scene.management import inputs as scene_inputs
from software.storage import write_json
from software.vision.models import LensVideo
import test_scene as scene_fixtures


class RearCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = scene_fixtures.SceneWorkflowTests('test_multitask_bimanual_metric_world_and_reuse')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.video = self.fixture.recording_paths['left']
        self.front = self.fixture.cameras[self.video]
        recordings = SegmentPaths(self.video, self.video, self.video, self.video)
        self.inputs = CalibrationInputs(recordings, self.front.serial, dict(tagCols=6, tagRows=6, tagSize=.024, tagSpacing=.3))
        self.poses = scene_fixtures.motion()[1]

    def solve_b(self, dataset, grid, a, parameters, context, *, prepared_bag=None):
        write_json(context.paths.calibration, json.loads(self.front.files['front'].read_text()))
        return CameraImuResult(context.paths.calibration, self.front.measurements['front'], self.metrics())

    @staticmethod
    def metrics():
        return dict(corners_per_frame=1000., mean_reprojection_px=.1, rms_reprojection_px=.12)

    @staticmethod
    def coverage():
        return dict(reach_radius_percentile=1., coverage_percentile=99.0, coverage_reach_threshold=0.5, frames_reaching_by_direction=dict(right=50, left=50, down=50, up=50),
                    board_pose_fraction=1., board_distance_m=[.2, .5])

    def prepare(self, video, camera, *args, **kwargs):
        self.assertAlmostEqual(args[0].trajectory.values.orb.initial_tag_size_m, self.inputs.grid['tagSize'])
        self.assertFalse(args[0].parameters.tag_fusion_enabled)
        self.assertFalse(args[0].parameters.tag_mapping_enabled)
        self.assertFalse(args[0].parameters.tag_mapping_scale_enabled)
        rec = self.fixture.prepared_recordings[video]
        return replace(rec, camera=camera, lens_videos=(LensVideo('rear', rec.lens_videos[0].path, rec.lens_videos[0].settings, None),))

    def run_rear(self, scale, *, tag_size_mm=24., install=False):
        self.inputs = replace(self.inputs, grid=dict(self.inputs.grid, tagSize=tag_size_mm/1000.))
        output = self.fixture.root / 'rear_output'
        def dataset(video, target, **kwargs):
            target.mkdir(parents=True)
        a = IntrinsicsResult(self.front.files['front'], [350., 350., 720., 720., 0., 0., 0., 0.], self.metrics(), self.coverage())
        reference = dict(video_time_s=self.fixture.prepared_recordings[self.video].video_times_s, idx=np.arange(len(self.poses)), p=scale*self.poses[:, :3, 3], n_frames=len(self.poses), reproj_deg_med=[])
        with patch.object(rear_calibration.registry, 'pin', return_value=self.fixture.device_pins[self.front.serial]), \
             patch.object(rear_calibration.calibration_inputs, 'prepare_inputs', return_value=self.inputs), \
             patch.object(rear_calibration.camera_assets, 'select_camera', return_value=self.front), \
             patch.object(workflow.runtime, 'prepare', return_value=True), \
             patch.object(recordings, 'build', side_effect=dataset) as build, \
             patch.object(recordings, 'imu_stats', return_value=dict(gyro_norm_p95_rad_s=1.)), \
             patch.object(basalt, 'estimate_intrinsics', return_value=a), \
             patch.object(basalt, 'prepare_bag', return_value=(output/'run/work/camera_imu.bag', dict(elapsed_s=0.))), \
             patch.object(basalt, 'estimate_camera_imu', side_effect=self.solve_b), \
             patch.object(gravity, 'validate_gravity', return_value=GravityResult('REPORT_ONLY', 6, 6, .01)), \
             patch.object(rear_calibration.recordings, 'prepare_recording', side_effect=self.prepare), \
             patch.object(board_detection, 'detect_cached', return_value={}), \
             patch.object(workflow, '_coverage', return_value=self.coverage()), \
             patch.object(board_pnp, 'board_reference', return_value=reference):
            arguments = ['run', str(self.fixture.source/'rear'), '--out', str(output), '--tag-size-mm', str(tag_size_mm)]
            code = cli.main(arguments + (['--install'] if install else []))
            doc = result_io.load_result(output)
            result = dict(status=Decision(doc['evaluation']['decision']).label, exit_code=code)
        self.assertEqual(build.call_count, 4)
        self.assertTrue(all(call.kwargs['lens'] == 'rear' for call in build.call_args_list))
        self.assertEqual([call.kwargs['with_images'] for call in build.call_args_list], [True, True, True, False])
        return result, output

    def test_publishes_matching_rear_measurement_only_after_independent_C_pass(self):
        result, output = self.run_rear(1.)
        self.assertTrue(result['status'].startswith('PASS'))
        measured = json.loads((output / 'rear_calibration.json').read_text())
        self.assertEqual(measured['camera_serial'], self.front.serial)
        self.assertEqual(measured['lens'], 'rear')
        self.assertLess(measured['validation']['C']['ate_se3_rmse_mm'], 2.)
        self.assertTrue((output / 'report.txt').is_file())
        self.assertNotIn('样例标定', (output / 'report.txt').read_text())

    def test_rear_C_isolates_global_scene_poster_switches(self):
        enabled = replace(scene_inputs.load_scene_parameters(),
            tag_fusion_enabled=True, tag_mapping_enabled=True, tag_mapping_scale_enabled=True)
        with patch.object(scene_inputs, 'load_scene_parameters', return_value=enabled):
            result, _ = self.run_rear(1.)
        self.assertTrue(result['status'].startswith('PASS'))

    def test_C_scale_error_prevents_rear_measurement_publication(self):
        result, output = self.run_rear(1.5)
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse((output / 'rear_calibration.json').exists())
        self.assertTrue((output / 'run/work').exists())
        self.assertEqual(result_io.load_result(output)['evaluation']['decision'], 'rejected')

    def test_rear_validation_passes_measured_a3_size_to_scene_preparation(self):
        result, output = self.run_rear(1., tag_size_mm=35.7)
        self.assertTrue(result['status'].startswith('PASS'))
        snapshot = result_io.load_result(output)['computation']['trajectory_validation']
        self.assertAlmostEqual(snapshot['values']['orb']['initial_tag_size_m'], .0357)

    def test_rear_auto_install_uses_rear_measurement_after_acceptance(self):
        with patch.object(workflow.devices, 'install_rear', return_value=0) as install_rear, \
             patch.object(workflow.devices, 'install_bundle') as install_front:
            result, output = self.run_rear(1., install=True)
        self.assertEqual(result['exit_code'], 0)
        install_rear.assert_called_once_with(output/'rear_calibration.json')
        install_front.assert_not_called()

    def test_failed_rear_validation_does_not_install(self):
        with patch.object(workflow.devices, 'install_rear') as install:
            result, _ = self.run_rear(1.5, install=True)
        self.assertEqual(result['exit_code'], 1)
        install.assert_not_called()


class RearPixelsTests(unittest.TestCase):
    def test_front_and_rear_calibration_decode_the_correct_half(self):
        executable = Path(__file__).resolve().parents[1] / '.deps/env/bin/ffmpeg'
        if not executable.is_file():
            self.skipTest('project FFmpeg is not installed')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / 'packed.insv'
            subprocess.run([str(executable), '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                '-i', 'color=black:size=3840x1920:rate=60000/1001,drawbox=x=1920:y=0:w=1920:h=1920:color=white:t=fill',
                '-frames:v', '2', '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-f', 'mp4', str(video)], check=True)
            def extract(source, out, **kwargs):
                path = out / 'mav0/cam0'
                path.mkdir(parents=True)
                (path / 'data.csv').write_text('#timestamp,filename\n1000000000,1000000000.png\n')
                imu = out / 'mav0/imu0'
                imu.mkdir()
                (imu / 'data.csv').write_text('#t,wx,wy,wz,ax,ay,az\n1000000000,0,0,0,0,0,9.81\n')
                (out / 'extract_manifest.json').write_text('{}')
            with patch.object(recordings.extraction, 'extract', side_effect=extract):
                recordings.build(video, root / 'front', lens='front')
                recordings.build(video, root / 'rear', lens='rear')
            front = cv2.imread(str(root / 'front/mav0/cam0/data/1000000000.png'), cv2.IMREAD_GRAYSCALE)
            rear = cv2.imread(str(root / 'rear/mav0/cam0/data/1000000000.png'), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(front.shape, (1440, 1440))
            self.assertGreater(float(front.mean()), 220.)
            self.assertLess(float(rear.mean()), 30.)


if __name__ == '__main__':
    unittest.main()
