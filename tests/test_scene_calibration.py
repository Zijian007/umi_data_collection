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

from software.calibration import recording_data
from software.calibration.models import CalibrationInputs, CameraImuResult, GravityResult, IntrinsicsResult, SegmentPaths
from software.scene import rear_calibration
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

    def solve_b(self, dataset, a, context):
        write_json(context.paths.calibration, json.loads(self.front.files['front'].read_text()))
        return CameraImuResult(context.paths.calibration, self.front.measurements['front'], self.metrics(), self.coverage())

    @staticmethod
    def metrics():
        return dict(corners_per_frame=1000., mean_reprojection_px=.1)

    @staticmethod
    def coverage():
        return dict(reach_r_p99=1., frames_reaching_r0p5_by_direction=dict(right=50, left=50, down=50, up=50),
                    board_pose_fraction=1., board_distance_m=[.2, .5])

    def prepare(self, video, camera, *args, **kwargs):
        rec = self.fixture.prepared_recordings[video]
        return replace(rec, camera=camera, lens_videos=(LensVideo('rear', rec.lens_videos[0].path, rec.lens_videos[0].settings, None),))

    def run_rear(self, scale):
        output = self.fixture.root / 'rear_output'
        def dataset(video, target, **kwargs):
            target.mkdir(parents=True)
        a = IntrinsicsResult(self.front.files['front'], [350., 350., 720., 720., 0., 0., 0., 0.], self.metrics(), self.coverage())
        reference = dict(idx=np.arange(len(self.poses)), p=scale*self.poses[:, :3, 3], n_frames=len(self.poses), reproj_deg_med=[])
        with patch.object(rear_calibration.registry, 'pin', return_value=self.fixture.device_pins[self.front.serial]), \
             patch.object(rear_calibration.calibration_inputs, 'prepare_inputs', return_value=self.inputs), \
             patch.object(rear_calibration.inputs, 'select_camera', return_value=self.front), \
             patch.object(rear_calibration.runtime, 'prepare', return_value=True), \
             patch.object(rear_calibration.recording_data, 'build', side_effect=dataset) as build, \
             patch.object(rear_calibration.recording_data, 'imu_stats', return_value=dict(gyro_norm_p95_rad_s=1.)), \
             patch.object(rear_calibration.intrinsics, 'solve_intrinsics', return_value=a), \
             patch.object(rear_calibration.camera_imu, 'solve_camera_imu', side_effect=self.solve_b), \
             patch.object(rear_calibration.validation, 'validate_gravity', return_value=GravityResult('CONSISTENT', 6, 6, .01)), \
             patch.object(rear_calibration.recordings, 'prepare_recording', side_effect=self.prepare), \
             patch.object(rear_calibration.board_pnp, 'detect_all', return_value={}), \
             patch.object(rear_calibration.board_pnp, 'board_reference', return_value=reference):
            result = rear_calibration.run(self.fixture.source, output=output)
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

    def test_C_scale_error_prevents_rear_measurement_publication(self):
        result, output = self.run_rear(1.5)
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse((output / 'rear_calibration.json').exists())
        self.assertTrue((output / 'diagnostics/work').exists())
        self.assertEqual(json.loads((output / 'report.json').read_text())['status'], 'FAIL')


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
            with patch.object(recording_data.extraction, 'extract', side_effect=extract):
                recording_data.build(video, root / 'front', lens='front')
                recording_data.build(video, root / 'rear', lens='rear')
            front = cv2.imread(str(root / 'front/mav0/cam0/data/1000000000.png'), cv2.IMREAD_GRAYSCALE)
            rear = cv2.imread(str(root / 'rear/mav0/cam0/data/1000000000.png'), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(front.shape, (1440, 1440))
            self.assertGreater(float(front.mean()), 220.)
            self.assertLess(float(rear.mean()), 30.)


if __name__ == '__main__':
    unittest.main()
