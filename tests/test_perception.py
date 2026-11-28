"""相机模型、帧处理、标签检测与共享解码。"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import av
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from test_poster import poster_layout_fixture, poster_recording

from umi.config import operational_settings as settings
from umi.config.collection_schema import normalize_collection
from umi.config.loader import load
from umi.core import frame
from umi.core.frame import output_frame_rate
from umi.core.geometry import (
    camera_from_tcp,
    compose_pose_series,
    describe_tcp_frame,
    imu_from_camera,
    tcp_frame_definition,
)
from umi.devices.x3 import extraction
from umi.infra import provenance
from umi.infra.file_store import read_tum, write_tum, write_yaml
from umi.infra.terminal import Console
from umi.perception.decoding import VideoDecode
from umi.perception.gripper import frame_detection
from umi.perception.gripper import frame_detection as sampled_detection
from umi.perception.gripper.frame_detection import FrameDetector, crop_gray
from umi.perception.gripper.marker_detection import CascadeMarkerDetector, FrameDetection, MarkerDetector
from umi.perception.models import RecordingData, RecordingPaths
from umi.perception.poster import detection as poster_detection
from umi.perception.poster import layout as tag_board
from umi.perception.tools import make_lens_video, merge_frames
from umi.reconstruction.backends import PosterVioQualityError
from umi.reconstruction.estimation.fixed_map.fusion import SceneFusionQualityError
from umi.scene import measurements
from umi.scene import routing as estimation
from umi.scene.collection import configuration as scene_inputs
from umi.scene.inputs import trimming as trim
from umi.scene.inputs import recordings
from umi.scene.maps import binding as estimation_binding
from umi.store import layout
from umi.store.cache import StageCache
from umi.reconstruction.estimation.models import BackendResult
from umi.store.models import TaskInputs

# 固定裁剪是既有标定的像素约定，简化几何实现时仍需严格保持。


class LocalizationMergeTests(unittest.TestCase):
    def test_pixel_information_follows_selected_map_across_tracking_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for label, priority, validity in [('m0', 10, [1, 0, 0, 1, 1]), ('m1', 5, [1, 1, 0, 1, 0])]:
                path = root / f'{label}.csv'
                sources.append((label, priority, path))
                with path.open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    extra = ['camera_information'] if label == 'm0' else []
                    writer.writerow('frame_idx t state ok tx ty tz qx qy qz qw'.split() + extra)
                    for i, ok in enumerate(validity):
                        writer.writerow([i, i / 30, 2 if ok else 1, ok, 0, 0, 0, 0, 0, 0, 1]
                                        + ([f'info{i}'] if extra else []))
            output = root / 'merged.csv'
            merge_frames.merge(sources, output)
            with output.open() as stream:
                result = [(r['map'], r['camera_information']) for r in csv.DictReader(stream)]
            self.assertEqual(result, [('m0', 'info0'), ('m1', ''), ('', ''), ('m1', ''), ('m0', 'info4')])


class CameraFrameTests(unittest.TestCase):
    def test_calibrated_crop_coordinates_are_preserved(self):
        self.assertEqual(frame.crop_xywh('right'), (2160, 240, 1440, 1440))
        self.assertEqual(frame.crop_xywh('left'), (240, 240, 1440, 1440))
        with self.assertRaises(ValueError):
            frame.crop_xywh('auto')

    def test_recording_mode_recognition_is_preserved(self):
        cases = [
            ([{'w': 3840, 'h': 1920}], frame.PROFILE),
            ([(3840, 1920)], frame.PROFILE),
            ([(5760, 2880)], 'unknown'),
            ([(1920, 960)], 'unknown'),
            ([{'w': 2880, 'h': 2880}], 'x3_360_5k7_30'),
            ([{'w': 1920, 'h': 1080}], 'unknown'),
            ([{'w': 0, 'h': 1920}], 'unknown'),
            ([{'w': 2880, 'h': 2880}] * 2, 'unknown'),
            ([], 'unknown'),
        ]
        for streams, expected in cases:
            with self.subTest(streams=streams):
                self.assertEqual(frame.detect_profile_from_streams(streams), expected)

    def test_pixel_centre_scaling_preserves_normalized_rays(self):
        intrinsics = dict(fx=510.0, fy=512.0, cx=723.0, cy=720.0, k1=.03)
        pixel = (875.0, 413.0)
        for size in (360, 720, 960, 1440):
            with self.subTest(size=size):
                scaled = frame.scale_intrinsics(intrinsics, 1440, size)
                for coordinate, focal, centre in zip(pixel, ('fx', 'fy'), ('cx', 'cy')):
                    resized_coordinate = (coordinate + .5) * size / 1440 - .5
                    self.assertAlmostEqual((resized_coordinate-scaled[centre])/scaled[focal],
                                           (coordinate-intrinsics[centre])/intrinsics[focal])
                self.assertEqual(scaled['k1'], intrinsics['k1'])
        self.assertEqual(frame.scale_intrinsics(intrinsics, 1440, 1440), intrinsics)

    def test_geometry_record_uses_fixed_crop_and_configured_output_size(self):
        geometry = frame.geometry_record(960, mode='dual')
        self.assertEqual(geometry['slam_video_px'], [960, 960])
        self.assertEqual(geometry['front_lens']['crop_xywh'], [2160, 240, 1440, 1440])
        self.assertEqual(geometry['frame_step'], 2)
        self.assertTrue(geometry['rear_lens']['used'])
        self.assertAlmostEqual(geometry['fps_used'], 30000/1001)

    def test_unsupported_recording_is_refused_before_reading_imu(self):
        for width, height in ((2880, 2880), (5760, 2880), (1920, 960)):
            with self.subTest(resolution=(width, height)), \
                    patch.object(extraction, 'video_stream_info', return_value={'video_streams': [{'w': width, 'h': height}]}), \
                    patch.object(extraction, 'read_imu_and_meta') as read_imu:
                with self.assertRaises(extraction.Refused):
                    extraction.extract(Path('unsupported.insv'), Path('/unused'))
                read_imu.assert_not_called()


# 共用外参与逐帧位姿换算：非零杆臂、转动和四元数约定。


class CameraGeometryTests(unittest.TestCase):
    def test_display_uses_asset_fields_even_for_a_known_convention(self):
        frame = dict(convention='genie_omnipicker_axes_v1', origin='measured origin',
                     axes=dict(x='measured x direction', y='measured y direction', z='measured z direction'))
        text = describe_tcp_frame(frame)
        for value in (frame['origin'], *frame['axes'].values()):
            self.assertIn(value, text)
        frame['axes']['x'] = 'updated x direction'
        self.assertIn('updated x direction', describe_tcp_frame(frame))
        self.assertNotIn('measured x direction', describe_tcp_frame(frame))
        self.assertEqual(describe_tcp_frame(dict(convention='asset_defined', description='legacy axes')), 'legacy axes')

    def test_camera_extrinsics_do_not_require_or_invent_tcp(self):
        from umi.reconstruction.segment import trajectory_results as run_metadata
        extrinsics = run_metadata.load_extrinsics()
        self.assertIn('T_imu_cam', extrinsics)
        self.assertNotIn('T_cam_tcp', extrinsics)

    def test_optical_parallel_tcp_keeps_fingertip_positions_and_changes_only_basis(self):
        root = Path(__file__).resolve().parents[1]
        new = json.loads((root / 'tests/fixtures/grippers/handheld_v1/002/t_tcp_camera.json').read_text())
        old_path = root / 'tests/fixtures/grippers/handheld_v1/001/t_tcp_camera.json'
        old = json.loads(old_path.read_text())
        np.testing.assert_array_equal(new['T_camera_tcp']['t_m'], old['T_camera_tcp']['t_m'])
        positions = np.array([[3., -2., 1.], [-1., 4., .2], [2., 3., 4.]])
        camera_rot = Rotation.from_euler('xyz', [[.1, .2, .3], [-.2, .4, 1.], [2., -1., .5]])
        p_old, q_old = compose_pose_series(positions, camera_rot.as_quat(), camera_from_tcp(old))
        p_new, q_new = compose_pose_series(positions, camera_rot.as_quat(), camera_from_tcp(new))
        np.testing.assert_array_equal(p_new, p_old)
        np.testing.assert_allclose(Rotation.from_quat(q_new).as_matrix(), camera_rot.as_matrix(), atol=1e-14)
        change = Rotation.from_quat(q_old).inv() * Rotation.from_quat(q_new)
        np.testing.assert_allclose(change.magnitude(), np.deg2rad(21.5), atol=1e-8)
        self.assertEqual(tcp_frame_definition(new)['convention'], 'front_optical_parallel_v1')
        self.assertEqual(tcp_frame_definition(old)['convention'], 'asset_defined')
        self.assertIn('jaw opening', tcp_frame_definition(old)['description'])

    def test_robot_tcp_axes_keep_position_and_point_x_toward_image_left(self):
        root = Path(__file__).resolve().parents[1]
        new = json.loads((root / 'tests/fixtures/grippers/handheld_v1/003/t_tcp_camera.json').read_text())
        legacy = json.loads((root / 'tests/fixtures/grippers/handheld_v1/001/t_tcp_camera.json').read_text())
        extrinsic = camera_from_tcp(new)
        expected = Rotation.from_euler('x', 21.5, degrees=True).as_matrix() @ np.diag([-1., -1., 1.])
        np.testing.assert_allclose(extrinsic[:3, :3], expected, atol=1e-8)
        np.testing.assert_array_equal(extrinsic[:3, 3], legacy['T_camera_tcp']['t_m'])
        np.testing.assert_allclose(extrinsic[:3, 0], [-1., 0., 0.], atol=1e-14)
        np.testing.assert_allclose(extrinsic[:3, 2], np.asarray(legacy['T_camera_tcp']['R_rows'])[:, 2], atol=1e-8)
        np.testing.assert_allclose(np.cross(extrinsic[:3, 2], extrinsic[:3, 0]), extrinsic[:3, 1], atol=1e-14)
        self.assertEqual(new['tcp_frame']['convention'], 'genie_omnipicker_axes_v1')

    def test_pose_chain_matches_homogeneous_transforms_with_nonzero_levers(self):
        measured = dict(q_xyzw=Rotation.from_euler('xyz', [.2, -.4, .7]).as_quat(), p=[.03, -.02, .01])
        tcp = dict(T_camera_tcp=dict(R_rows=Rotation.from_euler('y', -.3).as_matrix(), t_m=[.02, 0., -.15]))
        extrinsic = imu_from_camera(measured) @ camera_from_tcp(tcp)
        positions = np.array([[3., -2., 1.], [4., 1., 2.]])
        quaternions = Rotation.from_euler('xyz', [[.1, .2, .3], [-.2, .4, 1.]]).as_quat()
        p, q = compose_pose_series(positions, -quaternions, extrinsic)
        body = np.tile(np.eye(4), (2, 1, 1))
        body[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
        body[:, :3, 3] = positions
        expected = body @ extrinsic
        np.testing.assert_allclose(p, expected[:, :3, 3], atol=1e-14)
        np.testing.assert_allclose(Rotation.from_quat(q).as_matrix(), expected[:, :3, :3], atol=1e-14)
        self.assertTrue(np.all(q[:, 3] >= 0))
        self.assertGreater(p[0, 0], 2.)  # 换算不会把场景原点移到本段起点。

    def test_independent_result_records_the_selected_tcp_convention(self):
        from umi.devices import calibration_assets
        from umi.reconstruction.segment import trajectory_results as postprocess

        root = Path(__file__).resolve().parents[1]
        source = root / 'tests/fixtures/grippers/handheld_v1/003/t_tcp_camera.json'
        new = json.loads(source.read_text())
        sources = (
            (root / 'tests/fixtures/grippers/handheld_v1/001/t_tcp_camera.json', 'asset_defined'),
            (root / 'tests/fixtures/grippers/handheld_v1/002/t_tcp_camera.json', 'front_optical_parallel_v1'),
            (source, 'genie_omnipicker_axes_v1'),
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fuse = directory / 'reconstruction/fuse'
            fuse.mkdir(parents=True)
            (fuse / 'fuse_info.json').write_text('{}')
            times = np.arange(4) / 29.97 + 1.
            positions = np.array([[2., 1., .5], [2.1, 1., .5], [2.2, 1.1, .5], [2.3, 1., .5]])
            rotations = Rotation.from_euler('xyz', [[0., .1, 0.], [.1, .2, .3], [-.2, .4, 1.], [.3, -.2, 1.5]])
            write_tum(fuse / 'traj_tum.txt', times, positions, rotations.as_quat(), 'test')
            outputs = []
            for i, (asset, expected) in enumerate(sources):
                output = directory / f'output_{i}'
                output.mkdir()
                with patch.object(postprocess, 'plot'):
                    summary = postprocess.write_trajectories(fuse.parent, {}, output, 'handheld', 'VID.insv',
                        calib=dict(json=str(calibration_assets.SAMPLE_CAMERA_JSON), tcp_json=str(asset)))
                self.assertEqual(summary['tcp_frame']['convention'], expected)
                settings = json.loads((output / 'run_settings.json').read_text())
                self.assertEqual(settings['assets']['camera_tcp_geometry']['tcp_frame'], summary['tcp_frame'])
                self.assertIn(expected, (output / 'traj_tcp.txt').read_text())
                outputs.append(np.loadtxt(output / 'traj_tcp.txt'))
            np.testing.assert_array_equal(outputs[0][:, :4], outputs[1][:, :4])
            np.testing.assert_array_equal(outputs[0][:, :4], outputs[2][:, :4])
            camera = np.loadtxt(output / 'traj_camera.txt')
            np.testing.assert_allclose(Rotation.from_quat(outputs[2][:, 4:]).as_matrix(),
                                       Rotation.from_quat(camera[:, 4:]).as_matrix() @ camera_from_tcp(new)[:3, :3], atol=1e-8)

    def test_camera_only_result_matches_camera_poses_without_tcp_artifacts(self):
        from umi.reconstruction.segment import results as reports
        from umi.reconstruction.segment import trajectory_results as postprocess
        from umi.reconstruction.segment.trajectory_results import load_extrinsics

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fuse = directory / 'reconstruction/fuse'
            fuse.mkdir(parents=True)
            (fuse / 'fuse_info.json').write_text('{}')
            times = np.arange(4) / 29.97 + 1.
            positions = np.array([[2., 1., .5], [2.1, 1., .5], [2.2, 1.1, .5], [2.3, 1., .5]])
            rotations = Rotation.from_euler('xyz', [[0., .1, 0.], [.1, .2, .3], [-.2, .4, 1.], [.3, -.2, 1.5]])
            write_tum(fuse / 'traj_tum.txt', times, positions, rotations.as_quat(), 'test')
            with patch.object(postprocess, 'plot'):
                summary = postprocess.write_trajectories(fuse.parent, {}, directory, 'handheld', 'camera.insv')
            _, camera_positions, camera_quaternions = read_tum(directory / 'traj_camera.txt')
            expected_positions, expected_quaternions = compose_pose_series(
                positions, rotations.as_quat(), load_extrinsics()['T_imu_cam'])
            np.testing.assert_allclose(camera_positions, expected_positions, atol=5.1e-7)  # TUM 位置保存六位小数。
            np.testing.assert_allclose(camera_quaternions, expected_quaternions, atol=1e-8)
            self.assertFalse((directory / 'traj_tcp.txt').exists())
            self.assertEqual(summary['pose_frame'], 'camera')
            self.assertIn('camera_path_length_m', summary)
            self.assertNotIn('tcp_frame', summary)
            settings = json.loads((directory / 'run_settings.json').read_text())
            self.assertNotIn('camera_tcp_geometry', settings['assets'])
            reports.show_recording(dict(summary, wall_s=1., out=str(directory)))
            reports.write_result_report(directory, summary)
            self.assertIn('相机', (directory / 'report.txt').read_text())


# 裁剪后灰度转换的数值、范围、颜色语义与原始坐标。


class ConversionTrace:
    def __init__(self):
        self.converter = av.video.reformatter.VideoReformatter()
        self.frames = []

    def reformat(self, frame, **kwargs):
        self.frames.append((frame.width, frame.height, frame.format.name, frame.colorspace, frame.color_range))
        return self.converter.reformat(frame, **kwargs)


class FrameProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pixels = np.random.default_rng(42).integers(0, 256, (1920, 3840, 3), dtype=np.uint8)
        cls.rgb = av.VideoFrame.from_ndarray(pixels, format='rgb24')

    def test_planar_semiplanar_and_packed_formats_crop_before_gray_with_identical_pixels(self):
        roi = (601, 981, 1349, 1249)
        for name in ('yuv420p', 'yuvj420p', 'yuv422p', 'yuv444p', 'gray', 'rgb24',
                     'bgr24', 'rgba', 'bgr0', 'nv12', 'p010le', 'yuyv422', 'rgb565le'):
            with self.subTest(format=name):
                frame = self.rgb.reformat(format=name)
                expected = frame.to_ndarray(format='gray')[981:1249, 2521:3269]
                trace = ConversionTrace()
                actual = crop_gray(frame, roi, trace)
                np.testing.assert_array_equal(actual, expected)
                width, height, _, _, _ = trace.frames[0]
                self.assertLessEqual(width, 762)
                self.assertLessEqual(height, 282)
                self.assertEqual(actual.shape, (268, 748))

    def test_default_roi_is_the_only_region_sent_to_gray_converter(self):
        frame = self.rgb.reformat(format='yuv420p')
        trace = ConversionTrace()
        actual = crop_gray(frame, (600, 980, 1350, 1250), trace)
        self.assertEqual(trace.frames[0][:2], (750, 270))
        self.assertEqual(actual.shape, (270, 750))

    def test_colorspace_range_and_source_planes_are_preserved(self):
        frame = self.rgb.reformat(format='yuv420p')
        frame.colorspace, frame.color_range = 1, 1
        original_planes = [bytes(plane) for plane in frame.planes]
        trace = ConversionTrace()
        actual = crop_gray(frame, (600, 980, 1350, 1250), trace)
        self.assertEqual(trace.frames[0][3:], (frame.colorspace, frame.color_range))
        np.testing.assert_array_equal(actual, frame.to_ndarray(format='gray')[980:1250, 2520:3270])
        self.assertEqual([bytes(plane) for plane in frame.planes], original_planes)

    def test_crop_never_requests_a_full_source_image_conversion(self):
        class SourceFrame:
            def __init__(self, frame):
                self.frame = frame

            def __getattr__(self, name):
                return getattr(self.frame, name)

            def to_ndarray(self, *args, **kwargs):
                raise AssertionError('source frame must be cropped before conversion')

        pixels = crop_gray(SourceFrame(self.rgb), (600, 980, 1350, 1250), ConversionTrace())
        self.assertEqual(pixels.shape, (270, 750))

    def test_roi_at_last_pixel_keeps_exact_bounds_with_chroma_alignment(self):
        frame = self.rgb.reformat(format='yuv420p')
        roi = (1917, 1917, 1920, 1920)
        actual = crop_gray(frame, roi, ConversionTrace())
        expected = frame.to_ndarray(format='gray')[1917:1920, 3837:3840]
        np.testing.assert_array_equal(actual, expected)

    def test_detection_restores_native_coordinates_and_ignores_outside_duplicate(self):
        half = bright_markers(((0, 700, 1050), (1, 1000, 1050), (0, 100, 100)), shape=(1920, 1920))
        frame = av.VideoFrame.from_ndarray(np.column_stack((np.zeros_like(half), half)), format='gray')
        for fast in (False, True):
            with self.subTest(fast=fast):
                detector = FrameDetector((600, 980, 1350, 1250), mode='fast' if fast else 'full')
                detected = detector.detect(frame)
                self.assertAlmostEqual(detected.distance_px, 300., delta=.3)
                np.testing.assert_allclose(detected.centres[0], [719.5, 1069.5], atol=.3)
                np.testing.assert_allclose(detected.centres[1], [1019.5, 1069.5], atol=.3)


# 小尺寸暗标签补检、原始坐标与歧义拒绝。


FULL_SEARCH_INTERVAL = settings.values('gripper_tag_detection')['full_search_interval']


def dark_markers(tags: tuple[int, ...], *, size=(24, 14), sigma=.4, positions=(20, 120, 220)) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    gray = np.full((240, 320), 8, np.uint8)
    width, height = size
    for tag, left in zip(tags, positions):
        marker = cv2.aruco.generateImageMarker(dictionary, tag, 96)
        patch = np.pad(marker, 16, constant_values=255)
        patch = np.rint(patch.astype(float) * 17 / 255 + 8).astype(np.uint8)
        patch = cv2.resize(patch, size, interpolation=cv2.INTER_AREA)
        patch = cv2.GaussianBlur(patch, (0, 0), sigma)
        gray[100:100 + height, left:left + width] = patch
    return gray


def bright_markers(positions: tuple[tuple[int, int, int], ...], *, shape=(1200, 1600)) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    gray = np.full(shape, 255, np.uint8)
    for tag, left, top in positions:
        gray[top:top + 40, left:left + 40] = cv2.aruco.generateImageMarker(dictionary, tag, 40)
    return gray


class CascadeDetectionTests(unittest.TestCase):
    def test_unique_fast_pair_skips_full_detection_and_preserves_native_coordinates(self):
        detector = CascadeMarkerDetector((550, 650, 900, 800))
        detector._fallback._detect_region = Mock(side_effect=AssertionError('unnecessary fallback'))
        found = detector.detect(bright_markers(((0, 600, 700), (1, 800, 700))))
        self.assertAlmostEqual(found.distance_px, 200., delta=.3)
        np.testing.assert_allclose(found.centres[0], [619.5, 719.5], atol=.3)
        detector._fallback._detect_region.assert_not_called()

    def test_faint_pair_uses_full_fallback_without_reusing_previous_frame(self):
        detector = CascadeMarkerDetector()
        detector._fallback._detect_region = Mock(wraps=detector._fallback._detect_region)
        found = detector.detect(dark_markers((0, 1), size=(20, 12), sigma=.3, positions=(40, 180)))
        self.assertAlmostEqual(found.distance_px, 140., delta=.4)
        self.assertEqual(detector._fallback._detect_region.call_count, 1)
        self.assertFalse(np.isfinite(detector.detect(np.zeros((240, 320), np.uint8)).distance_px))

    def test_duplicate_detected_by_fast_pass_cannot_be_hidden_by_fallback(self):
        detector = CascadeMarkerDetector()
        detector._fallback._detect_region = Mock(side_effect=AssertionError('ambiguity must be retained'))
        found = detector.detect(bright_markers(((0, 100, 100), (0, 200, 100), (1, 300, 100))))
        self.assertFalse(np.isfinite(found.distance_px))
        detector._fallback._detect_region.assert_not_called()

    def test_disagreeing_positions_or_ids_between_passes_are_rejected(self):
        polygon = np.array([[[20., 30.], [40., 30.], [40., 50.], [20., 50.]]], dtype=np.float32)
        quick = FrameDetection((polygon,), np.array([[0]]), {0: polygon[0].mean(axis=0)})
        for corners in ((polygon + 50, polygon + 100), (polygon + 100, polygon)):
            complete = FrameDetection(corners, np.array([[0], [1]]),
                                      {tag: p[0].mean(axis=0) for tag, p in enumerate(corners)})
            with self.subTest(corners=corners), patch.object(MarkerDetector, '_detect_region', side_effect=[quick, complete]):
                found = CascadeMarkerDetector().detect(np.zeros((240, 320), np.uint8))
                self.assertFalse(np.isfinite(found.distance_px))


class MarkerDetectionTests(unittest.TestCase):
    def test_fast_recovers_faint_noisy_tags_without_moving_native_centres(self):
        for brightness, gain in ((.25, 2), (.4, 1)):
            gray = cv2.convertScaleAbs(dark_markers((0, 1)), alpha=brightness)
            noise = np.random.default_rng(42).normal(0, .6, gray.shape)
            gray = np.rint(np.clip(gray.astype(float) + noise, 0, 255)).astype(np.uint8)
            expected = np.array([[31.5, 106.5], [131.5, 106.5]])
            for rotation in range(4):
                with self.subTest(brightness=brightness, gain=gain, rotation=rotation):
                    found = MarkerDetector(fast=True, contrast=gain).detect(gray)
                    self.assertAlmostEqual(found.distance_px, 100., delta=.7)
                    for tag in (0, 1):
                        np.testing.assert_allclose(found.centres[tag], expected[tag], atol=.7)
                width = gray.shape[1]
                gray = np.rot90(gray).copy()
                expected = np.column_stack((expected[:, 1], width - 1 - expected[:, 0]))

    def test_fast_low_threshold_rejects_unrelated_codes_and_dark_noise(self):
        negatives = [dark_markers((2, 3)), dark_markers((48, 49))]
        negatives.extend(np.random.default_rng(seed).integers(0, 30, size=(240, 320), dtype=np.uint8)
                         for seed in range(4))
        for gain in (1, 2):
            for index, gray in enumerate(negatives):
                with self.subTest(gain=gain, negative=index):
                    found = MarkerDetector(fast=True, contrast=gain).detect(gray)
                    self.assertEqual(found.centres, {})
                    self.assertFalse(np.isfinite(found.distance_px))

    def test_sharpened_recovery_cannot_relabel_an_existing_physical_marker(self):
        polygon = np.array([[[20., 30.], [40., 30.], [40., 50.], [20., 50.]]], dtype=np.float32)
        for fast in (False, True):
            with self.subTest(fast=fast):
                detector = MarkerDetector(fast=fast)
                detector._primary = Mock()
                detector._primary.detectMarkers.return_value = ([polygon], np.array([[1]], dtype=np.int32), [])
                detector._recovery = Mock()
                detector._recovery.detectMarkers.return_value = ([polygon.copy()], np.array([[0]], dtype=np.int32), [])
                detector._recover_candidates = Mock(return_value=iter(()))
                found = detector.detect(np.zeros((100, 100), np.uint8))
                self.assertEqual(found.ids.flatten().tolist(), [1])
                self.assertNotIn(0, found.centres)
                self.assertFalse(np.isfinite(found.distance_px))

    def test_noisy_small_tags_keep_native_centres_under_rotation(self):
        gray = cv2.convertScaleAbs(dark_markers((0, 1)), alpha=.25)
        noise = np.random.default_rng(42).normal(0, .3, gray.shape)
        gray = np.rint(np.clip(gray.astype(float) + noise, 0, 255)).astype(np.uint8)
        expected = np.array([[31.5, 106.5], [131.5, 106.5]])
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                found = MarkerDetector(fast=True, contrast=2).detect(gray)
                self.assertAlmostEqual(found.distance_px, 100., delta=.4)
                for tag in (0, 1):
                    np.testing.assert_allclose(found.centres[tag], expected[tag], atol=.4)
            width = gray.shape[1]
            gray = np.rot90(gray).copy()
            expected = np.column_stack((expected[:, 1], width - 1 - expected[:, 0]))

    def test_contrast_recovers_underexposed_tags_without_changing_coordinates(self):
        gray = cv2.convertScaleAbs(dark_markers((0, 1)), alpha=.25)
        original = gray.copy()
        self.assertFalse(np.isfinite(MarkerDetector(fast=True).detect(gray).distance_px))
        detected = MarkerDetector(fast=True, contrast=2).detect(gray)
        self.assertAlmostEqual(detected.distance_px, 100., delta=.3)
        np.testing.assert_allclose(detected.centres[0], [31.5, 106.5], atol=.3)
        np.testing.assert_allclose(detected.centres[1], [131.5, 106.5], atol=.3)
        np.testing.assert_array_equal(gray, original)

    def test_contrast_preserves_ambiguity_and_does_not_create_missing_measurements(self):
        duplicated = cv2.convertScaleAbs(dark_markers((0, 0, 1)), alpha=.25)
        detected = MarkerDetector(fast=True, contrast=2).detect(duplicated)
        self.assertEqual(detected.ids.flatten().tolist().count(0), 2)
        self.assertFalse(np.isfinite(detected.distance_px))
        for gray in (np.zeros((240, 320), np.uint8),
                     np.random.default_rng(58).integers(0, 15, size=(240, 320), dtype=np.uint8)):
            detected = MarkerDetector(fast=True, contrast=2).detect(gray)
            self.assertFalse(np.isfinite(detected.distance_px))

    def test_invalid_contrast_is_rejected(self):
        for gain in (0, .5, float('inf'), float('nan')):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                MarkerDetector(contrast=gain)

    def test_fixed_roi_always_crops_and_ignores_targets_outside_declared_range(self):
        detector = MarkerDetector((550, 650, 900, 800))
        native_detect = detector._detect_region
        detector._detect_region = Mock(wraps=native_detect)
        gray = bright_markers(((0, 100, 100), (0, 600, 700), (1, 800, 700)))
        for _ in range(FULL_SEARCH_INTERVAL + 1):
            detected = detector.detect(gray)
            self.assertAlmostEqual(detected.distance_px, 200., delta=.3)
            np.testing.assert_allclose(detected.centres[0], [619.5, 719.5], atol=.3)
        self.assertTrue(all(call.args[0].shape == (150, 350) for call in detector._detect_region.call_args_list))
        self.assertEqual(detector._search_bounds, (550, 650, 900, 800))

    def test_fixed_roi_rejects_empty_negative_and_out_of_image_bounds(self):
        for roi in ((20, 20, 20, 40), (-1, 0, 50, 50), (0, 0, 400, 250)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                MarkerDetector(roi).detect(np.zeros((240, 320), np.uint8))

    def test_fast_preview_skips_expensive_candidate_enlargement(self):
        detector = MarkerDetector(fast=True)
        detector._recover_candidates = Mock()
        detected = detector.detect(dark_markers((0, 1), size=(20, 12), sigma=.3))
        detector._recover_candidates.assert_not_called()
        self.assertFalse(np.isfinite(detected.distance_px))

    def test_learned_region_reduces_search_and_retains_native_coordinates(self):
        detector = MarkerDetector()
        initial = bright_markers(((0, 600, 700), (1, 800, 700)))
        self.assertTrue(np.isfinite(detector.detect(initial).distance_px))
        native_detect = detector._detect_region
        detector._detect_region = Mock(wraps=native_detect)
        moved = bright_markers(((0, 620, 710), (1, 810, 710)))
        detected = detector.detect(moved)
        searched = detector._detect_region.call_args.args[0]
        self.assertLess(searched.size, initial.size / 4)
        np.testing.assert_allclose(detected.centres[0], [639.5, 729.5], atol=.3)
        np.testing.assert_allclose(detected.centres[1], [829.5, 729.5], atol=.3)
        self.assertAlmostEqual(detected.distance_px, 190., delta=.3)

    def test_periodic_full_search_recovers_targets_that_leave_region(self):
        detector = MarkerDetector()
        detector.detect(bright_markers(((0, 200, 200), (1, 400, 200))))
        moved = bright_markers(((0, 1200, 900), (1, 1400, 900)))
        for _ in range(FULL_SEARCH_INTERVAL - 1):
            missing = detector.detect(moved)
            self.assertEqual(missing.centres, {})
            self.assertFalse(np.isfinite(missing.distance_px))
        recovered = detector.detect(moved)
        np.testing.assert_allclose(recovered.centres[0], [1219.5, 919.5], atol=.3)
        self.assertAlmostEqual(recovered.distance_px, 200., delta=.3)
        self.assertTrue(np.isfinite(detector.detect(moved).distance_px))

    def test_duplicate_target_inside_search_region_remains_ambiguous(self):
        detector = MarkerDetector()
        detector.detect(bright_markers(((0, 200, 200), (1, 400, 200))))
        detected = detector.detect(bright_markers(((0, 200, 200), (0, 300, 200), (1, 400, 200))))
        self.assertEqual(detected.ids.flatten().tolist().count(0), 2)
        self.assertFalse(np.isfinite(detected.distance_px))

    def test_frame_size_change_restarts_full_search(self):
        detector = MarkerDetector()
        detector.detect(bright_markers(((0, 600, 700), (1, 800, 700))))
        resized = bright_markers(((0, 100, 200), (1, 300, 200)), shape=(500, 700))
        detected = detector.detect(resized)
        np.testing.assert_allclose(detected.centres[0], [119.5, 219.5], atol=.3)
        self.assertAlmostEqual(detected.distance_px, 200., delta=.3)

    def test_candidate_enlargement_recovers_tags_missing_from_both_full_frame_passes(self):
        gray = dark_markers((0, 1), size=(20, 12), sigma=.3, positions=(40, 180))
        detector = MarkerDetector()
        self.assertIsNone(detector._primary.detectMarkers(gray)[1])
        sharpened = cv2.addWeighted(gray, 2, cv2.GaussianBlur(gray, (0, 0), 1), -1, 0)
        self.assertIsNone(detector._recovery.detectMarkers(sharpened)[1])
        detected = detector.detect(gray)
        self.assertEqual(sorted(detected.ids.flatten().tolist()), [0, 1])
        np.testing.assert_allclose(detected.centres[0], [49.5, 105.5], atol=.4)
        np.testing.assert_allclose(detected.centres[1], [189.5, 105.5], atol=.4)
        self.assertAlmostEqual(detected.distance_px, 140., delta=.4)

    def test_candidate_coordinates_handle_crops_clipped_at_image_edge(self):
        detected = MarkerDetector().detect(dark_markers((0, 1), size=(20, 12), sigma=.3, positions=(5, 180)))
        np.testing.assert_allclose(detected.centres[0], [14.5, 105.5], atol=.4)
        self.assertAlmostEqual(detected.distance_px, 175., delta=.4)

    def test_candidate_cannot_relabel_an_existing_physical_marker(self):
        detector = MarkerDetector()
        polygon = np.array([[[20., 30.], [40., 30.], [40., 50.], [20., 50.]]], dtype=np.float32)
        detector._primary = Mock()
        detector._primary.detectMarkers.return_value = ([polygon], np.array([[1]], dtype=np.int32), [])
        detector._recovery = Mock()
        detector._recovery.detectMarkers.return_value = ([], None, [])
        detector._recover_candidates = Mock(return_value=iter([(polygon.copy(), 0)]))
        detected = detector.detect(np.zeros((100, 100), np.uint8))
        self.assertEqual(detected.ids.flatten().tolist(), [1])
        self.assertFalse(np.isfinite(detected.distance_px))

    def test_local_duplicate_tags_remain_ambiguous_after_overlapping_crops(self):
        detected = MarkerDetector().detect(dark_markers((0, 0, 1), size=(20, 12), sigma=.3))
        self.assertEqual(detected.ids.flatten().tolist().count(0), 2)
        self.assertNotIn(0, detected.centres)
        self.assertFalse(np.isfinite(detected.distance_px))

    def test_local_search_does_not_measure_unrelated_ids_or_noise(self):
        for gray in (dark_markers((2, 3), size=(20, 12), sigma=.3),
                     np.random.default_rng(58).integers(0, 30, size=(240, 320), dtype=np.uint8)):
            with self.subTest(mean=float(gray.mean())):
                detected = MarkerDetector().detect(gray)
                self.assertEqual(detected.centres, {})
                self.assertFalse(np.isfinite(detected.distance_px))

    def test_dark_small_tags_are_recovered_in_native_coordinates(self):
        detector = MarkerDetector()
        gray = dark_markers((0, 1))
        original = gray.copy()
        self.assertIsNone(detector._primary.detectMarkers(gray)[1])
        detected = detector.detect(gray)
        self.assertAlmostEqual(detected.distance_px, 100., delta=.2)
        np.testing.assert_allclose(detected.centres[0], [31.5, 106.5], atol=.2)
        np.testing.assert_allclose(detected.centres[1], [131.5, 106.5], atol=.2)
        np.testing.assert_array_equal(gray, original)

    def test_recovery_keeps_duplicate_target_ids_ambiguous(self):
        detected = MarkerDetector().detect(dark_markers((0, 0, 1)))
        self.assertEqual(detected.ids.flatten().tolist().count(0), 2)
        self.assertNotIn(0, detected.centres)
        self.assertFalse(np.isfinite(detected.distance_px))

    def test_missing_tags_are_not_carried_forward(self):
        detector = MarkerDetector()
        self.assertTrue(np.isfinite(detector.detect(dark_markers((0, 1))).distance_px))
        blank = detector.detect(np.full((240, 320), 8, np.uint8))
        self.assertIsNone(blank.ids)
        self.assertEqual(blank.centres, {})
        self.assertFalse(np.isfinite(blank.distance_px))

    def test_successful_native_detection_is_preserved_without_recovery(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        gray = np.full((240, 320), 255, np.uint8)
        for tag, left in ((0, 40), (1, 180)):
            gray[100:160, left:left + 60] = cv2.aruco.generateImageMarker(dictionary, tag, 60)
        detector = MarkerDetector()
        corners, ids, _ = detector._primary.detectMarkers(gray)
        detector._recovery = Mock()
        detected = detector.detect(gray)
        detector._recovery.detectMarkers.assert_not_called()
        np.testing.assert_array_equal(detected.ids.flatten(), ids.flatten())
        for actual, original in zip(detected.corners, corners):
            np.testing.assert_array_equal(actual, original)


# 共享原帧、独立缓存和失败发布；使用真实双镜头视频检测。


class SharedDecodingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.video = self.root / 'raw.mkv'
        front = bright_markers(((0, 700, 1050), (1, 1000, 1050)), shape=(1920, 1920))
        poster_recording(self.video, poster_layout_fixture([(12, 500, 500), (88, 900, 500)]), front=front, count=6)
        with av.open(str(self.video)) as container:
            self.frames = list(container.decode(video=0))
        self.times = np.array([float(frame.pts * frame.time_base) for frame in self.frames])
        self.queries = self.times[::2]
        self.digest = provenance.file_hash(self.video)
        self.settings = poster_detection.DetectionSettings(border_bits=1, ids=(12, 88))

    def prepare(self, cache, *, settings=None, queries=None):
        decoder = VideoDecode(self.video)
        frame_detection.cached_width(self.video, cache / 'width', self.digest, decoder=decoder)
        sampled = sampled_detection.SampledDetectionCache(
            self.video, cache, self.digest, self.queries if queries is None else queries, None)
        sampled.load(decoder=decoder)
        poster_detection.cached(self.video, cache / 'poster', settings or self.settings, decoder=decoder)
        return decoder, sampled

    def test_shared_outputs_are_byte_identical_to_independent_detection(self):
        cache = self.root / 'shared'
        decoder, sampled = self.prepare(cache)
        with patch('av.open', wraps=av.open) as opened:
            decoder.run()
        self.assertEqual(opened.call_count, 1)
        independent = self.root / 'independent'
        frame_detection.cached_width(self.video, independent / 'width', self.digest)
        reference = sampled_detection.SampledDetectionCache(self.video, independent, self.digest, self.queries, None)
        reference.load()
        poster_detection.cached(self.video, independent / 'poster', self.settings)
        for actual, expected in (
            (next((cache / 'width').glob('*/detections.npz')), next((independent / 'width').glob('*/detections.npz'))),
            (sampled.directory / 'quick.npz', reference.directory / 'quick.npz'),
            (cache / 'poster/observations.npz', independent / 'poster/observations.npz'),
        ):
            self.assertEqual(actual.read_bytes(), expected.read_bytes())
        self.assertTrue(np.isfinite(sampled.quick_corners).all())
        with patch('av.open', side_effect=AssertionError('warm cache must not open video')):
            warmed, _ = self.prepare(cache)
            self.assertEqual(warmed.run(), [])

    def test_poster_change_does_not_invalidate_gripper_caches(self):
        cache = self.root / 'partial'
        decoder, sampled = self.prepare(cache)
        decoder.run()
        paths = [next((cache / 'width').glob('*/detections.npz')), sampled.directory / 'quick.npz']
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        with patch.object(frame_detection.WidthDetection, 'consume', side_effect=AssertionError('width cache')), \
                patch.object(sampled_detection.QuickDetection, 'consume', side_effect=AssertionError('quick cache')), \
                patch('av.open', wraps=av.open) as opened:
            changed, _ = self.prepare(cache, settings=replace(self.settings, frame_step=3))
            changed.run()
        self.assertEqual(opened.call_count, 1)
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}, before)
        with np.load(cache / 'poster/observations.npz') as observations:
            np.testing.assert_array_equal(observations['source_frame_indices'], [0, 3])

    def test_query_change_only_rescans_sampled_detection(self):
        cache = self.root / 'partial'
        decoder, original = self.prepare(cache)
        decoder.run()
        with patch.object(frame_detection.WidthDetection, 'consume', side_effect=AssertionError('width cache')), \
                patch.object(poster_detection.PosterDetection, 'consume', side_effect=AssertionError('poster cache')), \
                patch('av.open', wraps=av.open) as opened:
            changed, sampled = self.prepare(cache, queries=self.times[1::2])
            changed.run()
        self.assertEqual(opened.call_count, 1)
        self.assertNotEqual(original.directory, sampled.directory)
        np.testing.assert_array_equal(sampled.native_times_s, self.times[1::2])

    def test_shared_nearest_pts_never_extrapolates_and_ties_choose_previous(self):
        # Matroska 时间戳取整到毫秒；使用 16 ms 间隔，使中点在半帧容差内。
        queries = np.array([self.times[0] - .001, (self.times[1] + self.times[2]) / 2,
                            self.times[2], self.times[-1] + .001])
        decoder, sampled = self.prepare(self.root / 'pts', queries=queries)
        decoder.run()
        self.assertTrue(np.isnan(sampled.native_times_s[[0, 3]]).all())
        np.testing.assert_array_equal(sampled.native_times_s[[1, 2]], self.times[[1, 2]])
        self.assertTrue(np.isnan(sampled.quick_corners[[0, 3]]).all())

    def test_decode_failure_does_not_publish_any_detection_cache(self):
        cache = self.root / 'failed'
        decoder, _ = self.prepare(cache)
        container = MagicMock()
        container.__enter__.return_value = container
        container.streams.video = [SimpleNamespace(codec_context=SimpleNamespace(width=3840, height=1920),
                                                  average_rate=poster_detection.FPS_SRC)]

        def broken():
            yield self.frames[0]
            raise RuntimeError('truncated decode')

        container.decode.return_value = broken()
        with patch('av.open', return_value=container), self.assertRaisesRegex(RuntimeError, 'truncated'):
            decoder.run()
        self.assertFalse(list(cache.rglob('*.npz')))
        self.assertFalse(list(cache.rglob('*.json')))

    def test_consumer_validation_failure_does_not_publish_other_caches(self):
        cache = self.root / 'invalid'
        decoder, _ = self.prepare(cache)
        with patch.object(poster_detection.PosterDetection, 'finish', side_effect=ValueError('invalid observations')), \
                self.assertRaisesRegex(ValueError, 'invalid observations'):
            decoder.run()
        self.assertFalse(list(cache.rglob('*.npz')))

    def test_background_detection_failure_does_not_publish_other_caches(self):
        cache = self.root / 'background-failed'
        decoder, _ = self.prepare(cache)
        with patch.object(poster_detection.PosterDetection, 'consume', side_effect=ValueError('worker failed')), \
                self.assertRaisesRegex(ValueError, 'worker failed'):
            decoder.run()
        self.assertFalse(list(cache.rglob('*.npz')))
        self.assertFalse(list(cache.rglob('*.json')))

    def test_scene_prepares_all_caches_then_consumers_need_no_more_decode(self):
        batch = self.root / 'shelf/batch_001'
        write_yaml(batch / 'collection.yaml', dict(
            kit='fixture', arms=['left'], map=dict(layout='shared'), backend=dict(mode='scene_map')))
        episode = SimpleNamespace(root=batch / 'take_001', recordings=dict(left=self.video),
            recording_refs=dict(left=dict(sha256=self.digest)),
            scene=SimpleNamespace(tag_layout=tag_board.DEFAULT_LAYOUT,
                                  environment_tag_detection=True,
                                  parameters=scene_inputs.load_scene_parameters()))
        prepared = SimpleNamespace(camera=SimpleNamespace(files=dict(width=Path('width.json'), tag_distance=Path('tags.json'))),
                                   video_times_s=self.queries)
        output = []
        inputs = TaskInputs(episode, {'left': prepared}, {'left': 0.}, batch, None, None,
                            Console('场景', line_sink=output.append))
        with patch('av.open', wraps=av.open) as opened:
            snapshot, = measurements.prepare_detection_group(inputs, ('left',), parallel=True)
        self.assertEqual(sum('检测完成' in line for line in output), 1)
        self.assertIn('左 3/3 帧（海报 6 次）', output[-1])
        self.assertFalse(any('快速检测' in line for line in output))
        output.clear()
        with patch('av.open', side_effect=AssertionError('already prepared')):
            cached_snapshot, = measurements.prepare_detection_group(
                replace(inputs, detections={}), ('left',), parallel=True)
        self.assertIn('左 3/3 帧（海报 6 次，缓存）', output[-1])
        self.assertTrue(cached_snapshot.report['cache_reused'])
        self.assertEqual(opened.call_count, 1)
        self.assertTrue(len(snapshot.observations.video_times_s))
        cache = layout.metadata(episode.root, 'cache')
        full = sampled_detection.SampledDetectionCache(self.video, cache, self.digest, self.queries, None)
        clipped = sampled_detection.SampledDetectionCache(self.video, cache, self.digest, self.queries[1:], None)
        with patch('av.open', side_effect=AssertionError('clipped quick scan should reuse source corners')), \
             patch.object(measurements, 'prepare_detections', side_effect=AssertionError('duplicate stage')):
            self.assertIs(measurements.prepare_task_detections(
                replace(inputs, directory=batch / 'fallback', detections={'left': snapshot}), 'left'),
                          snapshot)
            full.load()
            clipped.load(quick_source=full)
        np.testing.assert_array_equal(clipped.quick_corners, full.quick_corners[1:])
        np.testing.assert_array_equal(clipped.native_times_s, full.native_times_s[1:])
        with patch('av.open', side_effect=AssertionError('already prepared')):
            measurements.prepare_detections(episode, 'left', prepared)
            self.assertIsNotNone(frame_detection.cached_width(self.video, cache / 'gripper_width', self.digest))
            sampled_detection.SampledDetectionCache(self.video, cache, self.digest, self.queries, None).load()
            board = tag_board.load(tag_board.DEFAULT_LAYOUT)
            result = poster_detection.cached(self.video, cache / 'environment_tags/left',
                poster_detection.DetectionSettings(border_bits=board.border_bits, ids=board.ids, dictionary=board.dictionary,
                                       minimum_side_px=episode.scene.parameters.tag_minimum_side_px), layout=board)
        self.assertTrue(result['cache_reused'])

    def test_scene_map_without_poster_does_not_decode_tags_before_fallback(self):
        scene = SimpleNamespace(tag_layout=tag_board.DEFAULT_LAYOUT, environment_tag_detection=False,
                                parameters=scene_inputs.load_scene_parameters())
        episode = SimpleNamespace(root=self.root / 'take_001', recordings={'left': self.video},
                                  recording_refs={'left': {'sha256': self.digest}}, scene=scene)
        recording = SimpleNamespace(camera=SimpleNamespace(files={}), video_times_s=self.times)
        inputs = TaskInputs(episode, {'left': recording}, {'left': 0.}, self.root, None, None, None)
        with patch('av.open', side_effect=AssertionError('map without poster should not decode')):
            self.assertIsNone(measurements.prepare_task_detections(inputs, 'left'))
        scene.environment_tag_detection = True
        with patch('av.open', wraps=av.open) as opened:
            snapshot = measurements.prepare_task_detections(inputs, 'left')
            self.assertIs(measurements.prepare_task_detections(replace(inputs, detections={'left': snapshot}), 'left'),
                          snapshot)
        self.assertEqual(opened.call_count, 1)
        self.assertGreater(len(snapshot.observations.tag_ids), 0)


# 首尾规则、同步交集、原片帧映射和真实视频裁剪回归。


class TagTrimmingTests(unittest.TestCase):
    def test_configuration_defaults_and_invalid_values(self):
        doc = dict(kit='rig', arms=['right'], backend=dict(mode='poster_vio'))
        self.assertNotIn('poster_edge_trim', normalize_collection(doc)['reconstruction'])
        doc['trim'] = dict(enabled=True)
        with self.assertRaisesRegex(ValueError, '未知字段'):
            normalize_collection(doc)
        doc.pop('trim')
        doc['backend']['edge_trim'] = dict(enabled=True)
        self.assertEqual(normalize_collection(doc)['reconstruction']['poster_edge_trim'], dict(enabled=True))
        for value in (False, 0, -1, float('nan'), float('inf'), '1'):
            doc['backend']['edge_trim']['minimum_missing_duration_s'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_collection(doc)
        doc['backend']['edge_trim'] = dict(enabled='true')
        with self.assertRaises(ValueError):
            normalize_collection(doc)

    def test_stable_boundaries_ignore_isolated_detections_and_keep_internal_gap(self):
        self.assertEqual(trim.stable_interval([.1, 2., 2.1, 2.2, 7., 7.1, 7.2, 9.]), (2., 7.2))
        for times in ([], [1, 1.1], [1, 2, 3], [1, float('nan')]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                trim.stable_interval(times)
        self.assertEqual(trim.retained_interval((0., 10.), (.5, 9.5), 1.), (0., 10.))
        self.assertEqual(trim.retained_interval((0., 10.), (2., 8.), 1.), (2., 8.))

    def test_geometry_boundary_uses_two_tags_and_lens_clock_offsets(self):
        from scipy.spatial.transform import Rotation
        from test_poster import poster_fusion_fixture as fixture

        _, times, imu, _, extrinsics, trajectory, parameters, observations, board, measured, _, _ = fixture()
        imu_rear = np.linalg.inv(extrinsics['rear'])
        rear = dict(measured, q_xyzw=Rotation.from_matrix(imu_rear[:3, :3]).as_quat().tolist(),
                    p=imu_rear[:3, 3].tolist())
        front = dict(rear, cam_time_offset_ns=23000000)
        recording = SimpleNamespace(camera=SimpleNamespace(measurements=dict(front=front, rear=rear)),
                                    video_times_s=times, imu=imu,
                                    context=SimpleNamespace(parameters=trajectory))
        accepted, rejected = trim.reliable_times(observations, board, recording, parameters)
        self.assertGreater(len(accepted), 3)
        np.testing.assert_allclose(accepted, observations.video_times_s - .01, atol=1e-9)
        # One tag cannot start/end the work interval, regardless of its pixel quality.
        single = replace(observations, usable=observations.usable & (np.arange(len(observations.tag_ids)) % 3 == 0))
        accepted, rejected = trim.reliable_times(single, board, recording, parameters)
        self.assertEqual(len(accepted), 0)
        self.assertEqual(rejected['insufficient_tags'], len(observations.video_times_s))

    def test_trim_and_vio_share_exact_pnp_hypotheses(self):
        from test_poster import poster_fusion_fixture as fixture

        from umi.perception.poster import pnp
        from umi.perception.poster.geometry import camera_model

        _, _, _, _, _, _, parameters, observations, board, measured, _, _ = fixture()
        rows = observations.sample_indices == 0
        points = np.array([board.corners_m[tag] for tag in observations.tag_ids[rows]]).reshape(-1, 3)
        pixels = observations.corners_px[rows].reshape(-1, 2)
        solve = pnp.FrameCandidateCache()
        with patch.object(pnp, 'solve_frame_candidates', wraps=pnp.solve_frame_candidates) as solver:
            pnp.solve_frame(points, pixels, camera_model(measured),
                            parameters.tag_maximum_reprojection_rms_px, solve_candidates=solve)
            pnp.solve_frame(points.copy(), pixels.copy(), camera_model(measured),
                            2 * parameters.tag_maximum_reprojection_rms_px, solve_candidates=solve)
            solver.assert_called_once()

    def test_pnp_branch_jump_does_not_form_stable_boundary(self):
        from test_poster import poster_fusion_fixture as fixture
        _, times, imu, _, _, trajectory, parameters, observations, board, measured, _, _ = fixture()
        measured = dict(measured, q_xyzw=[0., 0., 0., 1.], p=[0., 0., 0.])
        recording = SimpleNamespace(camera=SimpleNamespace(measurements=dict(front=measured, rear=measured)),
                                    video_times_s=times, imu=imu,
                                    context=SimpleNamespace(parameters=trajectory))
        bad = np.eye(4)
        bad[:3, :3] = np.diag([-1., -1., 1.])
        candidates = [(np.eye(4) if i % 2 == 0 else bad, 0.) for i in range(len(observations.video_times_s))]
        with patch.object(trim, 'solve_frame', side_effect=candidates):
            accepted, rejected = trim.reliable_times(observations, board, recording, parameters)
        self.assertGreater(rejected['pnp_branch_jump'], 0)
        with self.assertRaises(ValueError):
            trim.stable_interval(accepted)

    def inputs(self, root, mode='poster_vio', arms=('left', 'right')):
        scene = SimpleNamespace(recording_trim=dict(enabled=True, minimum_missing_duration_s=1.),
                                scene_map_edge_trim=False, scene_map_poster_constraint_enabled=False,
                                auto_order=('scene_map', 'poster_vio', 'poster_independent'),
                                environment_tag_detection=True,
                                reconstruction_mode=mode, tag_layout='board',
                                parameters=SimpleNamespace(tag_minimum_side_px=6.))
        episode = SimpleNamespace(root=root / 'take_001', scene=scene,
                                  recordings={arm: root / f'{arm}.insv' for arm in arms})
        recordings = {arm: SimpleNamespace(video_times_s=np.arange(101) / 10., camera=SimpleNamespace(files={}))
                      for arm in arms}
        return TaskInputs(episode, recordings, {arm: (0. if arm == 'left' else .5) for arm in arms},
                          root, None, None, SimpleNamespace(write=lambda *a, **kw: None))

    def test_two_hands_use_sync_intersection_and_preserve_internal_missing_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self.inputs(Path(temporary))
            board = SimpleNamespace(ids=(12, 13), border_bits=1, dictionary='DICT_4X4_100')
            # Left 2..8, right 1..9 in native front clocks => common 2..8.
            times = [np.array([2., 2.1, 2.2, 7.8, 7.9, 8.]),
                     np.array([1., 1.1, 1.2, 8.8, 8.9, 9.])]
            snapshots = {arm: SimpleNamespace(layout=board, observations=None) for arm in inputs.recordings}
            with patch.object(trim.measurements, 'prepare_task_detections',
                              side_effect=lambda current, arm, **kwargs: snapshots[arm]), \
                 patch.object(trim, 'reliable_times', side_effect=[(t, {}) for t in times]), \
                 patch.object(trim, 'crop_recording', side_effect=lambda r, a, b, **kwargs: (a, b)):
                result, report = trim.prepare(inputs)
            self.assertEqual(result.recordings, dict(left=(20, 81), right=(15, 76)))
            self.assertEqual(report['common_episode_interval_s'], [2., 8.])
            self.assertEqual(inputs.recordings['left'].video_times_s.size, 101)
            self.assertEqual(report['status'], 'applied')
            self.assertEqual(json.loads((Path(temporary) / 'tag_trim.json').read_text()), report)

    def test_disabled_requires_no_detector_and_scene_map_bypasses_trim(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self.inputs(Path(temporary), mode='scene_map')
            inputs.episode.scene.recording_trim = {}
            self.assertEqual(trim.prepare(inputs), (inputs, None))
            inputs.episode.scene.recording_trim = dict(enabled=True)
            result = BackendResult({}, {}, {})
            with patch.object(trim, 'prepare', side_effect=AssertionError('must not trim')), \
                 patch.dict(estimation.ESTIMATORS, scene_map=lambda *args: result):
                self.assertIs(estimation.execute(inputs, estimation_binding.MapSelection.without_map()).coordinate_record,
                              result.coordinate_record)

    def test_auto_prepares_trim_for_each_poster_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self.inputs(Path(temporary), mode='auto')
            recordings = {'right': object()}
            candidates = {'right': object()}
            result = BackendResult({}, {}, {})
            seen = []
            def vio(current, selection):
                seen.append(current)
                raise PosterVioQualityError('quality')
            def independent(current, selection):
                seen.append(current)
                return result
            with patch.object(trim, 'prepare', side_effect=lambda current: (
                    replace(current, recordings=recordings, poster_pnp_candidates=candidates), dict(status='applied'))) as prepare, \
                 patch.dict(estimation.ESTIMATORS, poster_vio=vio, poster_independent=independent):
                estimation.execute(inputs, estimation_binding.MapSelection.without_map())
            self.assertEqual([call.args[0].directory.name for call in prepare.call_args_list],
                             ['00_poster_vio', '01_poster_independent'])
            self.assertEqual(len(seen), 2)
            for current in seen:
                self.assertIs(current.recordings, recordings)
                self.assertIs(current.poster_pnp_candidates, candidates)
            self.assertEqual(result.coordinate_record['recording_trim']['status'], 'applied')

    def test_auto_trims_scene_map_when_poster_constraint_is_enabled(self):
        for constraint in (True, False):
            with self.subTest(constraint=constraint), tempfile.TemporaryDirectory() as temporary:
                inputs = self.inputs(Path(temporary), mode='auto')
                scene = inputs.episode.scene
                scene.scene_map_edge_trim = True
                scene.scene_map_poster_constraint_enabled = constraint
                cropped = replace(inputs, recordings={'left': object(), 'right': object()})
                result = BackendResult({}, {}, {})
                seen = {}

                def scene_map(current, selection):
                    seen['map'] = current.recordings
                    raise SceneFusionQualityError('map coverage')

                def vio(current, selection):
                    seen['vio'] = current.recordings
                    return result

                with patch.object(trim, 'prepare', side_effect=lambda current: (
                        replace(current, recordings=cropped.recordings), dict(status='applied'))) as prepare, \
                     patch.dict(estimation.ESTIMATORS, scene_map=scene_map, poster_vio=vio):
                    estimation.execute(inputs, estimation_binding.MapSelection.available(Path(temporary), {}))
                self.assertEqual(prepare.call_count, 2 if constraint else 1)
                self.assertIs(seen['map'], cropped.recordings if constraint else inputs.recordings)
                self.assertIs(seen['vio'], cropped.recordings)
                self.assertEqual(result.coordinate_record['recording_trim']['status'], 'applied')

    def test_output_frame_provenance_and_viewer_clock_after_head_trim(self):
        from test_scene import SceneWorkflowTests

        from umi.reconstruction.estimation.models import FusedTrajectory
        from umi.scene.publication import write_arm_trajectory
        from umi.store.result_io import read_arm_settings, read_arm_summary

        fixture = SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        original = fixture.prepared_recordings[fixture.recording_paths['right']]
        step = original.context.parameters.video.frame_step
        recording = replace(original, video_times_s=original.video_times_s[3:9], source_frame_start=3 * step)
        coordinate = dict(scope='local', frame_id='local:test', version=None)
        result = FusedTrajectory(np.tile(np.eye(4), (6, 1, 1)), np.ones(6, bool), np.ones(6, bool),
                                 dict(maximum_visual_gap_s=0.))
        directory = fixture.root / 'output/right'
        track = write_arm_trajectory(recording, result, .5, coordinate, directory)
        with (directory / 'details/frames.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row['video_frame']) for row in rows], list(range(6)))
        self.assertEqual([int(row['source_video_frame']) for row in rows], list(range(3 * step, 9 * step, step)))
        np.testing.assert_allclose([float(row['t_video_s']) for row in rows], recording.video_times_s)
        manifest = dict(kind='umi_local_episode', schema_version=5, status='local_ok', scene=coordinate,
            map=None, arms=dict(right='right'), arm_results=dict(right=track.metadata),
            tcp_frames=dict(right=track.tcp_frame), synchronization=dict(offsets_s=dict(right=.5)),
            inputs=dict(cameras=dict(right=dict(serial='SERIAL_right')), trajectory_parameters=dict(video={})))
        expected = float(recording.video_times_s[0] + .5)
        self.assertEqual(read_arm_summary(directory, manifest)['video_time_offset_s'], expected)
        self.assertEqual(read_arm_settings(directory, manifest)['video_time_offset_s'], expected)
        self.assertEqual(manifest['synchronization']['offsets_s']['right'], .5)

    def test_cropped_vio_features_keep_original_insv_indices(self):
        from test_poster_vio import PosterVioWorkflowTests

        from umi.reconstruction.backends.poster_vio import preparation as poster_vio_inputs
        fixture = PosterVioWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for recording in fixture.inputs.recordings.values():
            recording.source_frame_start = 60
        captured = []
        def capture(video, streams, *args, **kwargs):
            captured.extend(streams)
            return dict(front=(), rear=())
        with patch.object(poster_vio_inputs, 'track_cameras', side_effect=capture):
            poster_vio_inputs.prepare_task_inputs(fixture.inputs)
        fronts = [item for item in captured if item.camera == 'front']
        self.assertEqual(len(fronts), 2)
        for item in fronts:
            self.assertGreaterEqual(int(item.source_frames[0]), 60)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires ffmpeg')
    def test_real_video_crop_preserves_original_clock_imu_and_source_frame_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = RecordingPaths(root / 'prepared')
            paths.lens_videos.mkdir(parents=True)
            cam = paths.dataset / 'mav0/cam0'
            cam.mkdir(parents=True)
            parameters = load('trajectory').values
            parameters = replace(parameters, video=replace(parameters.video, frame_step=2,
                codec='libx264', crf=12, preset='ultrafast', keep_audio=False))
            rate = output_frame_rate(parameters.video.frame_step)
            times = 5. + np.arange(12) / float(rate)
            rows = [f'{round((t + 1) * 1e9)},unused.png' for t in times]
            (cam / 'data.csv').write_text('# timestamp,filename\n' + '\n'.join(rows) + '\n')
            imu_file = paths.dataset / 'mav0/imu0/data.csv'
            imu_file.parent.mkdir()
            imu_file.write_text('1000000000,0,0,0,0,0,9.81\n11000000000,0,0,0,0,0,9.81\n')
            manifest = dict(source=dict(serial='fixture'), frames=dict(n_frames_used=12, duration_s=1.))
            (paths.dataset / 'extract_manifest.json').write_text(json.dumps(manifest))
            video = root / 'source.insv'
            with av.open(str(video), 'w', format='mp4') as container:
                stream = container.add_stream('libx264', rate=rate * 2)
                stream.width = stream.height = 32
                stream.pix_fmt = 'yuv420p'
                for index in range(24):
                    frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 9, np.uint8), format='rgb24')
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            original_bytes = video.read_bytes()
            def render(source, context, *, lenses, source_frame_start=0):
                self.assertEqual(lenses, ('front',))
                target = context.paths.lens_videos / 'front.mp4'
                target.parent.mkdir(parents=True)
                with patch.object(make_lens_video, 'crop_xywh', return_value=(0, 0, 32, 32)), \
                     patch.object(make_lens_video, 'validate_video_parameters'):
                    make_lens_video.render(str(source), str(context.paths.dataset / 'mav0'), str(target), 'right',
                                           video=replace(parameters.video, size_px=32),
                                           source_frame_start=source_frame_start)

            camera = SimpleNamespace(serial='fixture', measurements={'front': {}, 'rear': {}},
                                     selected=SimpleNamespace(assets=SimpleNamespace(front_mask=None)))
            scene = SimpleNamespace(trajectory=SimpleNamespace(values=parameters), reconstruction_mode='poster_vio')
            settings = SimpleNamespace(front_settings=None, rear_settings=None, front_mask=None)
            cache = StageCache(root / 'cache')
            with patch.object(recordings.recording_data, 'prepare', return_value=RecordingData(manifest, imu_file)), \
                 patch.object(recordings.runtime, 'visual_settings', return_value=settings), \
                 patch.object(recordings.selection, 'slam_settings_text', return_value=''), \
                 patch.object(recordings.recording_data, 'prepare_lens_videos', side_effect=render) as encode:
                original = recordings.prepare_recording(video, camera, scene, paths.output,
                                                         cache=cache, defer_lens_videos=True)
                cropped = trim.crop_recording(original, 3, 9)
                self.assertFalse(cropped.context.paths.lens_videos.exists())
                self.assertIs(cropped.full_video_times_s, original.video_times_s)
                encode.assert_not_called()
                output_video = cropped.lens_video('front').path
                self.assertEqual(cropped.lens_video('front').path, output_video)
                rerun = recordings.prepare_recording(video, camera, scene, root / 'rerun',
                                                      cache=cache, defer_lens_videos=True)
                cached_video = trim.crop_recording(rerun, 3, 9).lens_video('front').path
                self.assertEqual(cached_video.read_bytes(), output_video.read_bytes())
                encode.assert_called_once()
            self.assertFalse((cropped.context.paths.lens_videos / 'rear.mp4').exists())
            np.testing.assert_allclose(cropped.video_times_s, times[3:9], atol=1e-9, rtol=0)
            self.assertIs(cropped.imu, original.imu)
            self.assertEqual(cropped.source_frame_start, 6)
            self.assertEqual(cropped.context.frame_count, 6)
            self.assertEqual(video.read_bytes(), original_bytes)
            self.assertEqual(json.loads((paths.dataset / 'extract_manifest.json').read_text()), manifest)
            with av.open(str(output_video)) as container:
                frames = list(container.decode(video=0))
                self.assertEqual(len(frames), 6)
                self.assertAlmostEqual(float(frames[0].pts * frames[0].time_base), 0.)
                for index, frame in enumerate(frames):
                    self.assertAlmostEqual(frame.to_ndarray(format='rgb24').mean(), (index + 3) * 18, delta=3)
            selected = (cropped.context.paths.dataset / 'mav0/cam0/data.csv').read_text().splitlines()[1:]
            self.assertEqual(selected, rows[3:9])
