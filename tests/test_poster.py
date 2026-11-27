"""海报检测、场景约束和独立海报重建。"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import cv2
import gtsam
import numpy as np
from scipy.spatial.transform import Rotation

from umi.config import loader as config
from umi.core.geometry import resample_pose_series
from umi.devices import registry
from umi.infra import provenance
from umi.infra.file_store import read_yaml, write_yaml
from umi.perception.poster import detection as poster_detection
from umi.perception.poster import geometry as poster_pose
from umi.perception.poster import layout as tag_board
from umi.perception.poster.layout import DEFAULT_LAYOUT, load, resolve_layout_path
from umi.reconstruction.backends import PosterVioQualityError
from umi.reconstruction.backends.poster_independent import run as independent_reconstruction
from umi.reconstruction.backends.poster_independent import run as poster_independent
from umi.reconstruction.backends.poster_vio import run as poster_vio
from umi.reconstruction.backends.poster_vio.initialization import PosterInitializationQualityError
from umi.reconstruction.backends.scene_map import run as scene_reconstruction
from umi.reconstruction.estimation import poster_factors
from umi.reconstruction.estimation.fixed_map import fusion, poster
from umi.reconstruction.estimation.fixed_map.fusion import SceneFusionQualityError
from umi.reconstruction.estimation.fixed_map.geometry import VisualObservations
from umi.reconstruction.estimation.fixed_map.parameters import (
    FixedMapParameters,
    MapInertialParameters,
    MappedPosterParameters,
)
from umi.reconstruction.estimation.fusion import poster as fusion_poster
from umi.reconstruction.estimation.fusion import solver as fusion_solver
from umi.reconstruction.estimation.fusion.data import FusionInputs
from umi.reconstruction.estimation.models import FusedTrajectory
from umi.scene import measurements
from umi.scene import run as workflow
from umi.scene.collection import batch as batches
from umi.scene.collection import batch as episode_inputs
from umi.scene.collection.configuration import SceneOverrides, load_scene_parameters
from umi.scene.errors import MapCompatibilityError, MapUnavailableError
from umi.scene.maps import binding as map_lifecycle_binding
from umi.scene.maps import binding as map_repository_binding
from umi.scene.maps import versions as map_repository
from umi.scene.maps.binding import MapSelection
from umi.store import result_io as asset_identity
from umi.store.models import TaskInputs
from umi.store.result_io import read_arm_summary, result_paths, scene_reference

# PDF 海报布局的几何、身份与坏资产检查，不靠猜测 ID 生成运行时角点。


class TagBoardTests(unittest.TestCase):
    def test_existing_default_layout_path_resolves_to_moved_asset(self):
        legacy = DEFAULT_LAYOUT.parents[1] / 'calibration_targets' / DEFAULT_LAYOUT.name
        self.assertEqual(resolve_layout_path(legacy), DEFAULT_LAYOUT)
        self.assertEqual(load(legacy).asset_hash, load().asset_hash)

    def test_pdf_asset_dimensions_corner_order_and_source(self):
        board = load()
        self.assertEqual(board.ids, tuple(range(12, 89)))
        self.assertEqual((board.dictionary, board.border_bits), ('DICT_4X4_100', 1))
        self.assertEqual(board.geometry_status, 'pdf_design')
        np.testing.assert_allclose(board.corners_m[12], [[.06,.05,0],[.16,.05,0],[.16,.15,0],[.06,.15,0]])
        np.testing.assert_allclose(board.corners_m[88], [[.84,1.35,0],[.94,1.35,0],[.94,1.45,0],[.84,1.45,0]])
        self.assertEqual(provenance.file_hash(DEFAULT_LAYOUT.parent / board.source_pdf['path']), board.source_pdf['sha256'])

    def test_pdf_render_decodes_all_77_ids_in_declared_positions_and_orientations(self):
        renderer = shutil.which('pdftoppm')
        if renderer is None:
            self.skipTest('PDF rasterizer not installed')
        board = load()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'poster'
            subprocess.run([renderer, '-scale-to', '2400', '-png', '-singlefile',
                            str(DEFAULT_LAYOUT.parent / board.source_pdf['path']), str(output)], check=True,
                           capture_output=True, timeout=30)
            image = cv2.imread(str(output.with_suffix('.png')), cv2.IMREAD_GRAYSCALE)
        parameters = cv2.aruco.DetectorParameters()
        parameters.markerBorderBits = board.border_bits
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100), parameters)
        corners, ids, _ = detector.detectMarkers(image)
        self.assertIsNotNone(ids)
        self.assertEqual(sorted(ids.ravel().tolist()), list(board.ids))
        scale = np.array([image.shape[1], image.shape[0]]) / board.sheet_size_m
        for tag_id, detected in zip(ids.ravel().tolist(), corners):
            np.testing.assert_allclose(detected.reshape(4,2), board.corners_m[tag_id][:,:2]*scale, atol=2., rtol=0)

    def test_invalid_layouts_are_rejected(self):
        original = json.loads(DEFAULT_LAYOUT.read_text())
        cases = []
        for key, value in [('units','mm'), ('dictionary','DICT_4X4_50'), ('border_bits',True),
                           ('sheet_size_m',[1.,float('nan')]), ('corner_order','unknown')]:
            doc = copy.deepcopy(original); doc[key] = value; cases.append(doc)
        doc=copy.deepcopy(original);doc['markers'][1]['id']=12;cases.append(doc)
        doc=copy.deepcopy(original);doc['markers'][0]['corners_m'][1][0]+=.01;cases.append(doc)
        doc=copy.deepcopy(original);doc['markers'][0]['corners_m'].reverse();cases.append(doc)
        doc=copy.deepcopy(original);doc['markers'][0]['corners_m'][0][2]=.001;cases.append(doc)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'layout.json'
            for i, doc in enumerate(cases):
                with self.subTest(i=i):
                    path.write_text(json.dumps(doc))
                    with self.assertRaises(ValueError): load(path)

    def test_geometry_identity_ignores_marker_serialization_order_but_tracks_dimensions(self):
        doc=json.loads(DEFAULT_LAYOUT.read_text())
        original=load()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'layout.json'
            doc['markers'].reverse()
            path.write_text(json.dumps(doc))
            self.assertEqual(load(path).geometry_hash,original.geometry_hash)
            for marker in doc['markers']:
                marker['corners_m']=(np.asarray(marker['corners_m'])*.99).tolist()
            path.write_text(json.dumps(doc))
            self.assertNotEqual(load(path).geometry_hash,original.geometry_hash)


# 海报检测的真实 ArUco 图像、视频时钟、镜头边界和缓存回归。


def poster_layout_fixture(markers):
    image = np.full((1920, 1920), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    for tag_id, x, y in markers:
        image[y:y+120, x:x+120] = cv2.aruco.generateImageMarker(dictionary, tag_id, 120)
    return image


def poster_recording(path, rear, *, front=None, count=4):
    frame_image = np.concatenate([rear, front if front is not None else np.full_like(rear, 255)], axis=1)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('ffv1', rate=Fraction(60000, 1001))
        stream.width, stream.height, stream.pix_fmt = 3840, 1920, 'gray'
        for i in range(count):
            frame = av.VideoFrame.from_ndarray(frame_image, format='gray')
            frame.pts, frame.time_base = i + 60, Fraction(1001, 60000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class PosterDetectionTests(unittest.TestCase):
    def test_shared_observation_reader_preserves_arrays_and_rejects_bad_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video, archive = root / 'source.mkv', root / 'observations.npz'
            poster_recording(video, poster_layout_fixture([(12, 400, 400)]))
            observations = poster_detection.detect(video, poster_detection.DetectionSettings(border_bits=1))
            payload = {name: getattr(observations, name) for name in observations.__dataclass_fields__}
            np.savez_compressed(archive, **payload)
            loaded = poster_detection.load_observations(archive)
            for name, expected in payload.items():
                np.testing.assert_array_equal(getattr(loaded, name), expected)
                self.assertEqual(getattr(loaded, name).dtype, expected.dtype)
            incomplete = dict(payload)
            incomplete.pop('sample_indices')
            np.savez_compressed(archive, **incomplete)
            with self.assertRaises(KeyError):
                poster_detection.load_observations(archive)
            np.savez_compressed(archive, **dict(payload, sample_indices=np.full_like(
                observations.sample_indices, len(observations.video_times_s))))
            with self.assertRaisesRegex(ValueError, '索引越界'):
                poster_detection.load_observations(archive)
            np.savez_compressed(archive, **dict(payload, tag_ids=np.array([object()], dtype=object)))
            with self.assertRaises(ValueError):
                poster_detection.load_observations(archive)

    def test_detects_ids_beyond_gripper_dictionary_and_rejects_duplicates(self):
        image = poster_layout_fixture([(12, 400, 400), (88, 700, 400), (25, 1000, 400), (25, 1300, 400), (99, 400, 800)])
        found, duplicate, unknown = poster_detection.PosterDetector(poster_detection.DetectionSettings(border_bits=1)).detect(image)
        self.assertEqual([item[0] for item in found], [12, 88])
        self.assertEqual((duplicate, unknown), (1, 1))
        np.testing.assert_allclose(found[0][1], [[400, 400], [519, 400], [519, 519], [400, 519]], atol=1)

    def test_real_video_left_half_pts_partial_visibility_and_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / 'packed.mkv'
            poster_recording(video, poster_layout_fixture([(12, 400, 400), (88, 100, 100)]), front=poster_layout_fixture([(55, 700, 700)]))
            settings = poster_detection.DetectionSettings(border_bits=1)
            result = poster_detection.detect(video, settings)
            np.testing.assert_array_equal(result.source_frame_indices, [0, 2])
            self.assertGreater(result.video_times_s[0], 1.)
            self.assertEqual(result.tag_ids.tolist(), [12, 88, 12, 88])
            self.assertEqual(result.usable.tolist(), [True, False, True, False])
            np.testing.assert_allclose(result.corners_px[0, 0], [160, 160], atol=1)
            np.testing.assert_allclose(result.imu_times_s(12000000), result.video_times_s[result.sample_indices]+.012)
            report = poster_detection.report(result, settings)
            self.assertEqual(report['sampled_frames'], 2)
            self.assertEqual(report['at_least_one_usable']['frames'], 2)
            self.assertEqual(report['usable_ids_per_frame'], [1, 1])
            self.assertEqual(report['outside_calibrated_crop_fraction'], .5)
            self.assertEqual(report['per_id']['55']['detected']['frames'], 0)

    def test_blank_video_and_sampled_gap_include_start_and_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / 'blank.mkv'
            poster_recording(video, poster_layout_fixture([]))
            settings = poster_detection.DetectionSettings(border_bits=1)
            result = poster_detection.detect(video, settings)
            self.assertEqual(result.corners_px.shape, (0, 4, 2))
            report = poster_detection.report(result, settings)
            self.assertEqual(report['at_least_one_usable']['fraction'], 0)
            self.assertIsNone(report['outside_calibrated_crop_fraction'])
            self.assertAlmostEqual(report['at_least_one_usable']['maximum_sampled_gap_s'],
                result.video_times_s[-1]-result.video_times_s[0]+2/poster_detection.FPS_SRC)

    def test_cache_checks_source_parameters_and_payload_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video, output = root / 'source.mkv', root / 'detection'
            poster_recording(video, poster_layout_fixture([(88, 500, 500)]))
            settings = poster_detection.DetectionSettings(border_bits=1)
            first = poster_detection.cached(video, output, settings)
            self.assertFalse(first['cache_reused'])
            with patch.object(poster_detection, 'detect', side_effect=AssertionError('cache must avoid decoding')):
                second = poster_detection.cached(video, output, settings)
            self.assertTrue(second['cache_reused'])
            self.assertEqual(first['identity_hash'], second['identity_hash'])
            changed = poster_detection.cached(video, output, replace(settings, minimum_side_px=200.))
            self.assertFalse(changed['cache_reused'])
            self.assertEqual(changed['at_least_one_usable']['fraction'], 0.)
            (output / 'observations.npz').write_bytes(b'broken')
            self.assertFalse(poster_detection.cached(video, output, replace(settings, minimum_side_px=200.))['cache_reused'])
            poster_recording(video, poster_layout_fixture([(12, 500, 500)]))
            latest = poster_detection.cached(video, output, settings)
            self.assertFalse(latest['cache_reused'])
            self.assertNotEqual(first['identity_hash'], latest['identity_hash'])
            self.assertEqual(latest['per_id']['88']['usable']['frames'], 0)

    def test_rejects_bad_settings(self):
        for kwargs in (dict(border_bits=0), dict(border_bits=True), dict(border_bits=1, frame_step=0),
                       dict(border_bits=1, full_frame_every=0),
                       dict(border_bits=1, minimum_side_px=float('nan')), dict(border_bits=1, ids=(100,)),
                       dict(border_bits=1, ids=(12, 12)), dict(border_bits=1, dictionary='DICT_4X4_50')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                poster_detection.DetectionSettings(**kwargs)

    def test_follow_window_is_periodically_reset_by_full_frame_scan(self):
        settings = poster_detection.DetectionSettings(border_bits=1, full_frame_every=2)
        consumer = poster_detection.PosterDetection(settings)
        polygon = np.array([[400., 400.], [519., 400.], [519., 519.], [400., 519.]])
        with patch.object(consumer.detector, 'detect', side_effect=[
                ([(12, polygon)], 0, 0), ([(12, polygon)], 0, 0),
                ([(12, polygon)], 0, 0), ([(12, polygon)], 0, 0)]) as detect:
            for index in range(3):
                frame = type('Frame', (), {
                    'pts': index, 'time_base': 1,
                    'to_ndarray': lambda self, format: np.zeros((1920, 1920), np.uint8),
                })()
                consumer.consume(index * 2, frame)
        self.assertEqual(detect.call_args_list[0].kwargs['roi'], None)
        self.assertIsNotNone(detect.call_args_list[1].kwargs['roi'])
        self.assertEqual(detect.call_args_list[2].kwargs['roi'], None)


class FusionBeforeTagsTests(unittest.TestCase):
    def test_refactoring_preserves_saved_prechange_solution(self):
        with np.load(Path(__file__).parent / 'fixtures/scene_fusion_before_tags.npz', allow_pickle=False) as data:
            times, poses, imu, expected = (data[key] for key in ('times', 'poses', 'imu', 'expected'))
        observation = VisualObservations(times, poses, np.arange(len(times)), 'scene')
        result = fusion.fuse(times, imu, {'front': observation}, {'front': np.eye(4)},
                             config.load('trajectory').values.fusion, config.load('trajectory').values.zupt, load_scene_parameters(None).for_algorithm(FixedMapParameters))
        np.testing.assert_allclose(result.T_reference_imu, expected, rtol=0, atol=1e-10)
        self.assertTrue(result.tracked.all())
        self.assertEqual(result.evidence['actual_error'], 'not_measured')


# 海报融合数值回归：真实时钟、任意外参、几何导数和坏证据。


def pose(rotation=(0., 0., 0.), translation=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    result[:3, 3] = translation
    return result


def make_poster_inputs(arm, baseline, times, imu, observations, T_cam_imu,
                       trajectory_parameters, parameters, detections, board, measured, front_time_offset_ns=0):
    return poster.PosterInputs(
        arm, baseline, times, imu, observations, T_cam_imu,
        MapInertialParameters(trajectory_parameters.fusion, trajectory_parameters.zupt),
        parameters.for_algorithm(MappedPosterParameters), detections, board, measured, front_time_offset_ns)


def poster_fusion_fixture():
    times = np.linspace(.1, 1.9, 31)
    truth = np.tile(np.eye(4), (len(times), 1, 1))
    truth[:, 0, 3] = .12*times
    imu_t = np.linspace(0., 2., 1001)
    imu = np.column_stack((imu_t, np.zeros((len(imu_t), 5)), np.full(len(imu_t), 9.81)))
    visual = truth.copy()
    visual[:, 1, 3] += .018*np.sin(times*4)
    obs = {'front': VisualObservations(times, visual, np.arange(len(times)), 'scene')}
    extrinsics = {'front': np.eye(4), 'rear': pose((.12, -.08, .05), (.04, -.03, .02))}
    measured = dict(camera_type='kb4', resolution=[1440,1440], fx=520., fy=525., cx=720., cy=720.,
                    k=[.01, -.001, .0001, -.00001], cam_time_offset_ns=13000000)
    model = poster_pose.camera_model(measured)
    board = tag_board.load()
    board_pose = pose((.12, .07, -.04), (-.5, -.65, 2.1))
    rear_times = times[1:-1:2]+.013  # intentionally not on working-lens nodes
    ids = np.array([i for j in range(len(rear_times)) for i in ((12, 48, 88) if j%2 else (18, 42, 82))])
    samples = np.repeat(np.arange(len(rear_times)), 3)
    points = np.array([board.corners_m[i] for i in ids])
    cameras = resample_pose_series(times, truth, rear_times[samples]) @ np.linalg.inv(extrinsics['rear'])
    pixels, _ = poster_pose.project_observations(board_pose, cameras, points, model)
    sides = np.linalg.norm(pixels-np.roll(pixels, -1, axis=1), axis=2).min(axis=1)
    detections = poster_detection.TagObservations(np.arange(len(rear_times))*2, rear_times-.013, samples, ids, pixels+240.,
        sides, np.ones(len(ids), bool), np.ones(len(ids), bool), np.zeros(len(rear_times), int), np.zeros(len(rear_times), int))
    parameters = replace(load_scene_parameters(None), tag_fusion_enabled=True)
    trajectory = replace(config.load('trajectory').values, zupt=replace(config.load('trajectory').values.zupt, enabled=False))
    baseline = fusion.fuse(times, imu, obs, extrinsics, trajectory.fusion, trajectory.zupt, parameters.for_algorithm(FixedMapParameters))
    return baseline, times, imu, obs, extrinsics, trajectory, parameters, detections, board, measured, truth, board_pose


class PosterGeometryTests(unittest.TestCase):
    def test_kb4_jacobian_optical_axis_and_fisheye(self):
        model = poster_pose.camera_model(poster_fusion_fixture()[9])
        points = np.array([[0.,0.,2.], [1e-10,0.,2.], [.4,-.7,1.2], [1.,.2,-.1]])
        numeric = np.stack([(model.project(points+np.eye(3)[i]*1e-6)-model.project(points-np.eye(3)[i]*1e-6))/2e-6 for i in range(3)], axis=2)
        np.testing.assert_allclose(model.project_jacobian(points), numeric, atol=2e-6, rtol=1e-6)

    def test_corner_factor_jacobians_match_gtsam_retract(self):
        data = poster_fusion_fixture()
        body = pose((.1,-.2,.03), (.2,-.1,.04))
        board = data[-1]
        points = data[8].corners_m[48]
        model = poster_pose.camera_model(data[9])
        ext = data[4]['rear']
        pixels = np.full((4,2), 700.)
        _, jb, jp = poster_pose.residual_and_jacobians(body, board, ext, points, pixels, model)
        for variable, analytic in ((0,jb), (1,jp)):
            numerical = []
            for axis in np.eye(6)*1e-6:
                plus, minus = [body, board], [body, board]
                plus[variable] = gtsam.Pose3(plus[variable]).retract(axis).matrix()
                minus[variable] = gtsam.Pose3(minus[variable]).retract(-axis).matrix()
                a = poster_pose.residual_and_jacobians(*plus, ext, points, pixels, model)[0]
                b = poster_pose.residual_and_jacobians(*minus, ext, points, pixels, model)[0]
                numerical.append((a-b)/2e-6)
            np.testing.assert_allclose(analytic, np.array(numerical).T, atol=2e-5, rtol=2e-6)

    def test_rear_seat_jacobian_matches_gtsam_retract(self):
        data = poster_fusion_fixture()
        detections, board, measured = data[7:10]
        body = pose((.1, -.2, .03), (.2, -.1, .04))
        board_pose = data[-1]
        base_ext = data[4]['rear']
        model = poster_pose.camera_model(measured)
        tag_id = int(detections.tag_ids[0])
        points = board.corners_m[tag_id]
        pixels = detections.corners_px[0]
        pose_key, board_key, seat_key = (gtsam.symbol('x', 0), gtsam.symbol('b', 0),
                                         gtsam.symbol('s', 0))
        correction = gtsam.Rot3(Rotation.from_rotvec((.03, -.04, .02)).as_matrix())
        state = gtsam.Values()
        state.insert(pose_key, gtsam.Pose3(body))
        state.insert(board_key, gtsam.Pose3(board_pose))
        state.insert(seat_key, correction)
        analytic = poster_factors.rear_rotation_error(
            state, pose_key, board_key, seat_key, base_ext, points, pixels, model)[3]
        numerical = []
        for axis in np.eye(3):
            plus = gtsam.Values(state)
            minus = gtsam.Values(state)
            plus.update(seat_key, correction.retract(1e-6 * axis))
            minus.update(seat_key, correction.retract(-1e-6 * axis))
            plus_residual = poster_factors.rear_rotation_error(
                plus, pose_key, board_key, seat_key, base_ext, points, pixels, model)[0]
            minus_residual = poster_factors.rear_rotation_error(
                minus, pose_key, board_key, seat_key, base_ext, points, pixels, model)[0]
            numerical.append((plus_residual - minus_residual) / 2e-6)
        np.testing.assert_allclose(analytic, np.array(numerical).T, atol=2e-4, rtol=2e-6)

    def test_batched_tag_cost_preserves_corner_order_without_derivatives(self):
        data = poster_fusion_fixture()
        body, board = pose((.1, -.2, .03), (.2, -.1, .04)), data[-1]
        model, ext = poster_pose.camera_model(data[9]), data[4]['rear']
        corners = np.array([data[8].corners_m[tag] for tag in (12, 48, 88) * 3])
        observed = np.arange(corners.shape[0] * 8).reshape(-1, 4, 2) + 650.
        body_key, board_key, rear_key = (gtsam.symbol(char, 0) for char in 'xbs')
        correction = gtsam.Rot3(Rotation.from_rotvec((.03, -.04, .02)).as_matrix())
        state = gtsam.Values()
        state.insert(body_key, gtsam.Pose3(body))
        state.insert(board_key, gtsam.Pose3(board))
        state.insert(rear_key, correction)
        for enabled in (False, True):
            with self.subTest(refine_rotation=enabled):
                graph = gtsam.NonlinearFactorGraph()
                poster_factors.add_tag_factors(graph, pose_key=lambda i: body_key,
                    board_key=board_key, rear_key=rear_key, base_ext=ext, model=model,
                    points=corners, pixels=observed, tag_indices=np.zeros(len(corners), int),
                    parameters=replace(data[6], tag_rear_rotation_refinement_enabled=enabled))
                calibrated = ext.copy()
                if enabled:
                    calibrated[:3, :3] = correction.matrix() @ ext[:3, :3]
                expected = np.concatenate([poster_pose.residual_and_jacobians(
                    body, board, calibrated, pts, px, model)[0] for pts, px in zip(corners, observed)])
                self.assertEqual(graph.size(), 2)
                with patch.object(type(model), 'project_jacobian', side_effect=AssertionError):
                    actual = np.concatenate([graph.at(i).unwhitenedError(state) for i in range(graph.size())])
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

    def test_native_poster_factors_match_python_linearization(self):
        native = poster_factors.native_factors()
        if not hasattr(native, 'PosterCornerFactor'):
            self.skipTest('Optional native poster factors are unavailable')
        data = poster_fusion_fixture()
        model = poster_pose.camera_model(data[9])
        body = pose((.1, -.2, .03), (.2, -.1, .04))
        board = pose((-.2, .3, -.1), (-.3, .2, .1))
        ext = data[4]['rear']
        body_key, board_key, rear_key = (gtsam.symbol(char, 0) for char in 'xbs')
        state = gtsam.Values()
        state.insert(body_key, gtsam.Pose3(body))
        state.insert(board_key, gtsam.Pose3(board))
        correction = gtsam.Rot3.RzRyRx(.03, -.04, .02)
        state.insert(rear_key, correction)
        # Exercise off-axis negative depth, an active barrier on each side of
        # its minimum, and an inactive barrier. Nine sightings span two batches.
        camera_points = np.array([[.1, -.2, -.03], [.03, .02, .0008],
                                  [-.02, .04, .0013], [.05, -.14, .9]])
        camera_from_board = ext @ np.linalg.inv(body) @ board
        corners = poster_pose.transform(np.linalg.inv(camera_from_board), camera_points)
        points = np.tile(corners, (9, 1, 1))
        pixels = np.tile(model.project(camera_points), (9, 1, 1))
        pixels[1:8] += [10., -16.]  # Mixed inlier/outliers in one batch, then a partial batch.
        modes = [(False, False), (True, False)]
        if hasattr(native.PosterCornerFactor, 'per_tag_huber'):
            modes.extend([(False, True), (True, True)])
        for refine, per_tag in modes:
            graphs = []
            for extension in (None, native):
                graph = gtsam.NonlinearFactorGraph()
                common = dict(pose_key=lambda _: body_key, board_key=board_key, base_ext=ext,
                              points=points, tag_indices=np.zeros(9, int), native=extension)
                poster_factors.add_tag_factors(graph, **common, rear_key=rear_key, model=model,
                    pixels=pixels, parameters=replace(data[6], tag_rear_rotation_refinement_enabled=refine),
                    per_tag_huber=per_tag)
                poster_factors.add_tag_depth_barrier_factors(graph, **common)
                graphs.append(graph)
            self.assertEqual([g.size() for g in graphs], [3, 3])
            for i, cls in enumerate((native.PosterCornerFactor, native.PosterCornerFactor, native.PosterDepthFactor)):
                reference, optimized = (g.at(i) for g in graphs)
                self.assertIsInstance(optimized, cls)
                self.assertEqual(reference.keys(), optimized.keys())
                np.testing.assert_allclose(optimized.unwhitenedError(state), reference.unwhitenedError(state),
                                           atol=2e-10)
                np.testing.assert_allclose(optimized.error(state), reference.error(state), atol=1e-9)
                for actual, expected in zip(optimized.linearize(state).jacobian(), reference.linearize(state).jacobian()):
                    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8)

    def test_per_tag_batches_match_native_huber_cost_gradient_and_curvature(self):
        data = poster_fusion_fixture()
        board, model, ext = data[-1], poster_pose.camera_model(data[9]), data[4]['rear']
        body = data[-2][0]
        corners = np.array([data[8].corners_m[tag] for tag in (12, 48, 88) * 3])
        _, camera_points = poster_factors.tag_camera_points(body, board, ext, corners.reshape(-1, 3))
        observed = model.project(camera_points).reshape(-1, 4, 2)
        observed[1::3, :, 0] += .1
        observed[2::3, :, 1] += 25.
        x, h = gtsam.symbol('x', 0), gtsam.symbol('h', 0)
        state = gtsam.Values()
        state.insert(x, gtsam.Pose3(body))
        state.insert(h, gtsam.Pose3(board))

        def add(graph, indices, per_tag):
            poster_factors.add_tag_factors(graph, pose_key=lambda _: x, board_key=h, rear_key=None,
                base_ext=ext, model=model, points=corners[indices], pixels=observed[indices],
                tag_indices=np.zeros(len(indices), dtype=int),
                parameters=replace(data[6], tag_rear_rotation_refinement_enabled=False), per_tag_huber=per_tag)

        native = gtsam.NonlinearFactorGraph()
        for row in range(len(corners)):
            add(native, [row], False)
        expected_h, expected_g = native.linearize(state).hessian()
        for order in (np.arange(len(corners)), np.roll(np.arange(len(corners)), 2)):
            graph = gtsam.NonlinearFactorGraph()
            add(graph, order, True)
            self.assertEqual(graph.size(), 2)
            self.assertAlmostEqual(graph.error(state), native.error(state), places=9)
            hessian, gradient = graph.linearize(state).hessian()
            np.testing.assert_allclose(hessian, expected_h, atol=1e-7, rtol=1e-10)
            np.testing.assert_allclose(gradient, expected_g, atol=1e-7, rtol=1e-10)

    def test_seed_recovers_pose_with_changing_visible_ids(self):
        data = poster_fusion_fixture()
        detections, board, measured, truth, expected = data[7:]
        cameras = resample_pose_series(data[1], truth, detections.imu_times_s(measured['cam_time_offset_ns'])) @ np.linalg.inv(data[4]['rear'])
        points = np.array([board.corners_m[i] for i in detections.tag_ids])
        actual, rms = poster_pose.estimate_board(cameras, points, detections.corners_px, poster_pose.camera_model(measured), 5.)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertLess(rms, 1e-7)

    def test_seed_reports_sparse_frame_and_id_residuals(self):
        data = poster_fusion_fixture()
        detections, board, measured, truth, _ = data[7:]
        cameras = resample_pose_series(data[1], truth, detections.imu_times_s(measured['cam_time_offset_ns'])) @ np.linalg.inv(data[4]['rear'])
        points = np.array([board.corners_m[i] for i in detections.tag_ids])
        actual, rms, metrics = poster_pose.estimate_board(
            cameras, points, detections.corners_px, poster_pose.camera_model(measured), 5.,
            frame_ids=detections.sample_indices, tag_ids=detections.tag_ids, return_metrics=True)
        self.assertEqual(len(metrics['frame_rms_px']), len(np.unique(detections.sample_indices)))
        self.assertEqual(set(metrics['id_rms_px']), {str(i) for i in np.unique(detections.tag_ids)})
        self.assertAlmostEqual(metrics['rms_px'], rms)
        self.assertLess(metrics['median_px'], 1e-7)
        self.assertLess(metrics['p95_px'], 1e-7)


class PosterFusionTests(unittest.TestCase):
    def test_joint_method_solves_from_visual_imu_seeds(self):
        data = poster_fusion_fixture()
        baseline, times, imu, observations, extrinsics, trajectory, parameters, detections, board, measured = data[:10]
        parameters = replace(parameters, tag_fusion_enabled=True, tag_fusion_method='joint')
        result = poster.fuse_poster(make_poster_inputs(
            'left', baseline, times, imu, observations, extrinsics, trajectory, parameters,
            detections, board, measured))
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['status'], 'fused')
        self.assertEqual(evidence['method'], 'joint')
        self.assertLess(evidence['final_reprojection_rms_px'], 1.)
        # 报告从顶层读融合级指标，与 staged 形状一致。
        self.assertEqual(result.evidence['visual_consistency_rms_m'], evidence['visual_consistency_rms_m'])
        self.assertEqual(result.T_reference_imu.shape, baseline.T_reference_imu.shape)

    def test_joint_without_observations_keeps_baseline_with_evidence(self):
        data = list(poster_fusion_fixture()[:10])
        data[6] = replace(data[6], tag_fusion_enabled=True, tag_fusion_method='joint')
        data[7] = replace(data[7], usable=np.zeros(len(data[7].usable), bool))
        result = poster.fuse_poster(make_poster_inputs('left', *data))
        self.assertIs(result.T_reference_imu, data[0].T_reference_imu)
        evidence = result.evidence['environment_poster']
        self.assertEqual((evidence['status'], evidence['method']), ('insufficient_observations', 'joint'))

    def test_joint_post_solve_cull_reoptimizes_without_the_bad_sighting(self):
        data = list(poster_fusion_fixture()[:10])
        data[6] = replace(data[6], tag_fusion_enabled=True, tag_fusion_method='joint')
        corners = data[7].corners_half_px.copy()
        corners[0] += [20., 0.]
        data[7] = replace(data[7], corners_half_px=corners)
        # 跳过求解前剔除，让坏观测留到求解后的剔除分支。
        with patch.object(poster, '_initial_sighting_errors', return_value=np.zeros(len(corners))):
            result = poster.fuse_poster(make_poster_inputs('left', *data))
        evidence = result.evidence['environment_poster']
        self.assertEqual((evidence['status'], evidence['method']), ('fused', 'joint'))
        self.assertGreaterEqual(evidence['culled_observations'], 1)

    def test_joint_map_inconsistency_is_a_poster_error_not_a_localization_failure(self):
        data = list(poster_fusion_fixture()[:10])
        data[6] = replace(data[6], tag_fusion_enabled=True, tag_fusion_method='joint')
        with patch.object(poster.fusion, 'measure_map_consistency',
                          side_effect=poster.fusion.SceneFusionQualityError('不一致')), \
                self.assertRaises(poster_pose.PosterGeometryError) as caught:
            poster.fuse_poster(make_poster_inputs('left', *data))
        self.assertNotIsInstance(caught.exception, poster.fusion.SceneFusionQualityError)

    def test_final_cull_cannot_remove_over_budget_even_if_overall_rms_is_small(self):
        data = poster_fusion_fixture()
        project = poster.project_observations

        def biased_prediction(*args):
            pixels, points = project(*args)
            pixels[:int(np.ceil(.16*len(pixels)))] += [11., 0.]
            return pixels, points

        with patch.object(poster, '_initial_sighting_errors', return_value=np.zeros(len(data[7].tag_ids))), \
                patch.object(poster, 'project_observations', side_effect=biased_prediction), \
                self.assertRaises(poster_pose.PosterGeometryError) as caught:
            poster.fuse_poster(make_poster_inputs('left', *data[:10]))
        self.assertEqual(caught.exception.metrics['culling']['reason'], 'count_budget')
        self.assertLess(caught.exception.metrics['rms_px'], data[6].tag_maximum_reprojection_rms_px)

    def test_disabling_rear_refinement_fixes_the_exact_calibrated_rotation(self):
        data = list(poster_fusion_fixture()[:10])
        data[6] = replace(data[6], tag_rear_rotation_refinement_enabled=False)
        result = poster.fuse_poster(make_poster_inputs('left', *data))
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['rear_rotation_correction_deg'], 0.)
        self.assertFalse(evidence['rear_rotation_refinement']['enabled'])
        np.testing.assert_array_equal(evidence['T_rear_camera_imu_optimized'], data[4]['rear'])

    def test_large_rear_correction_is_rejected_by_the_actual_solver(self):
        data = list(poster_fusion_fixture()[:10])
        data[6] = replace(data[6], tag_rear_rotation_maximum_correction_deg=1e-8)
        with self.assertRaisesRegex(poster_pose.PosterGeometryError, '旋转修正超限'):
            poster.fuse_poster(make_poster_inputs('left', *data))

    def test_disabled_fusion_only_diagnoses_whatever_the_method(self):
        for method in poster.FUSION_METHODS:
            with self.subTest(method=method):
                data = list(poster_fusion_fixture()[:10])
                data[6] = replace(data[6], tag_fusion_enabled=False, tag_fusion_method=method)
                result = poster.fuse_poster(make_poster_inputs('left', *data))
                self.assertIs(result.T_reference_imu, data[0].T_reference_imu)
                self.assertEqual(result.evidence['environment_poster']['status'], 'diagnostics_only')

    def test_unknown_method_is_rejected_by_registry(self):
        data = list(poster_fusion_fixture()[:10])
        data[6] = replace(data[6], tag_fusion_enabled=True, tag_fusion_method='nope')
        with self.assertRaisesRegex(ValueError, 'staged'):
            poster.fuse_poster(make_poster_inputs('left', *data))

    def test_joint_method_matches_staged_relative_motion_on_fixture(self):
        data = poster_fusion_fixture()
        baseline, times, imu, observations, extrinsics, trajectory, parameters, detections, board, measured, truth = data[:11]
        staged, joint = (poster.fuse_poster(make_poster_inputs(
            'left', baseline, times, imu, observations, extrinsics, trajectory,
            replace(parameters, tag_fusion_enabled=True, tag_fusion_method=method),
            detections, board, measured)) for method in ('staged', 'joint'))

        def centered_position_error(result):
            error = result.T_reference_imu[:, :3, 3] - truth[:, :3, 3]
            error -= error.mean(axis=0)
            return float(np.linalg.norm(error))

        self.assertLessEqual(centered_position_error(joint), centered_position_error(staged) * 1.1)

    def test_joint_method_culls_a_bad_tag_sighting(self):
        data = list(poster_fusion_fixture())
        detections = data[7]
        corners = detections.corners_half_px.copy()
        corners[0] += [20., 0.]
        data[7] = replace(detections, corners_half_px=corners)
        parameters = replace(data[6], tag_fusion_enabled=True, tag_fusion_method='joint')
        result = poster.fuse_poster(make_poster_inputs('left', *data[:6], parameters, data[7], data[8], data[9]))
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['status'], 'fused')
        self.assertGreaterEqual(evidence['culled_observations'], 1)

    def test_joint_fusion_improves_relative_motion_and_uses_exact_rear_times(self):
        data = poster_fusion_fixture()
        baseline, times = data[:2]
        result = poster.fuse_poster(make_poster_inputs('left', *data[:10]))
        before = baseline.T_reference_imu[:, :3, 3]-data[10][:, :3, 3]
        after = result.T_reference_imu[:, :3, 3]-data[10][:, :3, 3]
        # An unknown board cannot establish absolute world translation; assess motion.
        before -= before.mean(axis=0); after -= after.mean(axis=0)
        self.assertLess(np.linalg.norm(after), np.linalg.norm(before)*.75)
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['status'], 'fused')
        self.assertEqual(evidence['added_nodes'], len(data[7].video_times_s))
        self.assertLess(evidence['final_reprojection_rms_px'], evidence['initial_reprojection_rms_px'])
        self.assertEqual(result.T_reference_imu.shape, baseline.T_reference_imu.shape)
        np.testing.assert_array_equal(result.tracked, baseline.tracked)

    def test_diagnostics_and_empty_data_leave_baseline_exact(self):
        data = list(poster_fusion_fixture()[:10])
        data[6] = replace(data[6], tag_fusion_enabled=False)
        diagnosed = poster.fuse_poster(make_poster_inputs('left', *data))
        self.assertEqual(diagnosed.evidence['environment_poster']['status'], 'diagnostics_only')
        self.assertIs(diagnosed.T_reference_imu, data[0].T_reference_imu)
        data[6] = replace(data[6], tag_fusion_enabled=True)
        data[7] = replace(data[7], usable=np.zeros(len(data[7].usable), bool))
        with patch.object(poster, 'estimate_board', side_effect=AssertionError('must skip empty evidence')):
            empty = poster.fuse_poster(make_poster_inputs('left', *data))
        self.assertEqual(empty.evidence['environment_poster']['status'], 'insufficient_observations')
        self.assertIs(empty.T_reference_imu, data[0].T_reference_imu)

    def test_diagnostics_culls_an_isolated_bad_tag_sighting(self):
        data = list(poster_fusion_fixture()[:10])
        detections = data[7]
        corners = detections.corners_half_px.copy()
        corners[0] += [20., 0.]
        data[7] = replace(detections, corners_half_px=corners)
        data[6] = replace(data[6], tag_fusion_enabled=False)
        result = poster.fuse_poster(make_poster_inputs('left', *data))
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['status'], 'diagnostics_only')
        self.assertGreaterEqual(evidence['culled_observations'], 1)
        self.assertGreater(evidence['pre_cull_reprojection_rms_px'],
                           evidence['initial_reprojection_rms_px'])
        self.assertIs(result.T_reference_imu, data[0].T_reference_imu)

    def test_moved_poster_is_not_silently_swallowed_as_localization_failure(self):
        data = list(poster_fusion_fixture()[:10])
        corners = data[7].corners_half_px.copy()
        corners[len(corners)//2:] += [50., 0.]
        data[7] = replace(data[7], corners_half_px=corners)
        with self.assertRaises(poster_pose.PosterGeometryError) as caught:
            poster.fuse_poster(make_poster_inputs('left', *data))
        self.assertNotIsInstance(caught.exception, fusion.SceneFusionQualityError)
        data[6] = replace(data[6], tag_fusion_enabled=False)
        diagnostic = poster.fuse_poster(make_poster_inputs('left', *data))
        self.assertEqual(diagnostic.evidence['environment_poster']['status'], 'rejected_diagnostics')
        self.assertIn('reprojection_metrics', diagnostic.evidence['environment_poster'])
        self.assertIn('median_px', diagnostic.evidence['environment_poster']['reprojection_metrics'])
        self.assertIs(diagnostic.T_reference_imu, data[0].T_reference_imu)

    def test_new_recording_reinitializes_poster_position(self):
        data = list(poster_fusion_fixture())
        detections, board, measured = data[7:10]
        moved = data[-1].copy(); moved[:3,3] += [.2, .1, .1]
        cameras = resample_pose_series(data[1], data[10], detections.imu_times_s(measured['cam_time_offset_ns'])) @ np.linalg.inv(data[4]['rear'])
        pixels, _ = poster_pose.project_observations(moved, cameras, np.array([board.corners_m[i] for i in detections.tag_ids]), poster_pose.camera_model(measured))
        data[7] = replace(detections, corners_half_px=pixels+240.)
        refined = poster.fuse_poster(make_poster_inputs('left', *data[:10]))
        self.assertEqual(refined.evidence['environment_poster']['status'], 'fused')
        np.testing.assert_allclose(np.array(refined.evidence['environment_poster']['T_scene_board'])[:3,3], moved[:3,3], atol=.03)

    def test_shared_fusion_accepts_different_ids_on_each_arm(self):
        data = poster_fusion_fixture()
        baseline, times, imu, observations, extrinsics, trajectory, parameters, detections, board, measured = data[:10]
        common = dict(times=times, imu=imu, observations=observations, T_cam_imu=extrinsics,
                      trajectory_parameters=trajectory, parameters=replace(parameters, tag_fusion_enabled=True),
                      board=board, measured=measured)
        keep = [i for i, tag_id in enumerate(detections.tag_ids) if int(tag_id) in {18, 42, 82}]
        partial = replace(detections,
            sample_indices=detections.sample_indices[keep], tag_ids=detections.tag_ids[keep],
            corners_half_px=detections.corners_half_px[keep], side_px=detections.side_px[keep],
            in_calibrated_crop=detections.in_calibrated_crop[keep], usable=detections.usable[keep])
        contexts = [make_poster_inputs(**common, arm='left', baseline=baseline, detections=detections),
                    make_poster_inputs(**common, arm='right', baseline=replace(baseline, T_reference_imu=baseline.T_reference_imu.copy()),
                         detections=partial)]
        result = poster.refine_shared_poster_pose(contexts)
        self.assertEqual(result['left'].evidence['environment_poster']['status'], 'fused')
        self.assertEqual(result['right'].evidence['environment_poster']['status'], 'fused')
        self.assertEqual(result['right'].evidence['environment_poster']['visible_ids'], [18, 42, 82])

    def test_shared_fusion_keeps_distinct_camera_models_and_partial_tag_sets(self):
        data = poster_fusion_fixture()
        left = make_poster_inputs('left', *data[:10])
        detections = left.detections
        selected = np.flatnonzero(np.isin(detections.tag_ids, [18, 42, 82]))
        measured = dict(left.measured, fx=561.6, fy=488.25, cam_time_offset_ns=18000000)
        samples, ids = detections.sample_indices[selected], detections.tag_ids[selected]
        times = detections.video_times_s[samples] + .018
        cameras = resample_pose_series(left.times, data[10], times) @ np.linalg.inv(left.T_cam_imu['rear'])
        points = np.array([left.board.corners_m[int(tag)] for tag in ids])
        pixels, _ = poster_pose.project_observations(data[11], cameras, points, poster_pose.camera_model(measured))
        partial = replace(detections, sample_indices=samples, tag_ids=ids, corners_half_px=pixels+240.,
            side_px=np.linalg.norm(pixels-np.roll(pixels, -1, axis=1), axis=2).min(axis=1),
            in_calibrated_crop=detections.in_calibrated_crop[selected], usable=detections.usable[selected])
        right = replace(left, arm='right', measured=measured, detections=partial)
        results = poster.refine_shared_poster_pose([left, right])
        for result in results.values():
            evidence = result.evidence['environment_poster']
            self.assertEqual(evidence['status'], 'fused')
            self.assertLess(evidence['final_reprojection_rms_px'], 1.)
        self.assertEqual(results['right'].evidence['environment_poster']['visible_ids'], [18, 42, 82])

    def test_shared_post_solve_cull_happens_before_the_overall_rms_gate(self):
        data = poster_fusion_fixture()
        context = make_poster_inputs('right', *data[:10])
        pixels = context.detections.corners_half_px.copy()
        pixels[0] += [60., 0.]
        left = replace(context, arm='left', detections=replace(context.detections, corners_half_px=pixels))
        with patch.object(poster, '_initial_sighting_errors', return_value=np.zeros(len(pixels))):
            results = poster.refine_shared_poster_pose([left, context])
        evidence = results['left'].evidence['environment_poster']
        self.assertEqual(evidence['culling']['stage'], 'optimization')
        self.assertEqual(evidence['culled_observations'], 1)
        self.assertGreater(evidence['pre_cull_reprojection_rms_px'], context.parameters.tag_maximum_reprojection_rms_px)
        self.assertLess(evidence['final_reprojection_rms_px'], context.parameters.tag_maximum_reprojection_rms_px)
        self.assertEqual(len(evidence['culling']['removed']), 1)
        self.assertNotIn('culling', results['right'].evidence['environment_poster'])

    def test_shared_two_stage_culling_uses_original_per_arm_budget(self):
        data = poster_fusion_fixture()
        right = make_poster_inputs('right', *data[:10])
        pixels = right.detections.corners_half_px.copy()
        pixels[:2] += [60., 0.]
        left = replace(right, arm='left', detections=replace(right.detections, corners_half_px=pixels))
        # Only the first bad sighting is detected by initialization.
        initial_errors = poster._initial_sighting_errors
        def mark_first(*args):
            errors = initial_errors(*args)
            errors[1:] = 0.
            return errors

        with patch.object(poster, '_initial_sighting_errors', side_effect=mark_first):
            result = poster.refine_shared_poster_pose([left, right])
            evidence = result['left'].evidence['environment_poster']
            self.assertEqual(evidence['culled_observations'], 2)
            audit = evidence['culling']
            self.assertEqual(audit['passes'], 2)
            self.assertEqual(audit['original_observations'], len(pixels))
            self.assertEqual([r['observation'] for r in audit['removed']], [0, 1])
            self.assertNotIn('culling', result['right'].evidence['environment_poster'])
            # Both single-stage proposals fit a one-observation budget. Their union does not.
            limited = replace(left, parameters=replace(left.parameters, tag_maximum_cull_fraction=.9/len(pixels)))
            with self.assertRaisesRegex(poster.PosterGeometryError, '累计剔除') as rejected:
                poster.refine_shared_poster_pose([limited, right])
            self.assertEqual(rejected.exception.metrics['culling']['reason'], 'count_budget')

    def test_shared_fusion_with_fixed_rear_extrinsics(self):
        data = list(poster_fusion_fixture()[:10])
        data[6] = replace(data[6], tag_rear_rotation_refinement_enabled=False)
        context = make_poster_inputs('left', *data)
        results = poster.refine_shared_poster_pose([context, replace(context, arm='right')])
        for result in results.values():
            evidence = result.evidence['environment_poster']
            self.assertEqual(evidence['rear_rotation_correction_deg'], 0.)
            np.testing.assert_array_equal(evidence['T_rear_camera_imu_optimized'], data[4]['rear'])

    def test_shared_poster_map_conflict_is_not_a_map_fallback_signal(self):
        data = poster_fusion_fixture()
        context = make_poster_inputs('left', *data[:10])
        with patch.object(poster.fusion, 'measure_map_consistency',
                          side_effect=poster.fusion.SceneFusionQualityError('不一致')), \
                self.assertRaises(poster_pose.PosterGeometryError) as caught:
            poster.refine_shared_poster_pose([context, replace(context, arm='right')])
        self.assertNotIsInstance(caught.exception, poster.fusion.SceneFusionQualityError)

    def test_shared_fusion_seed_follows_configured_method(self):
        data = poster_fusion_fixture()
        baseline, times, imu, observations, extrinsics, trajectory, parameters, detections, board, measured, truth = data[:11]
        common = dict(times=times, imu=imu, observations=observations, T_cam_imu=extrinsics,
                      trajectory_parameters=trajectory, parameters=replace(parameters, tag_fusion_enabled=True),
                      board=board, measured=measured, detections=detections)
        seeded = []
        for name, method in poster.FUSION_METHODS.items():
            def record(inputs, seed=method, name=name):
                seeded.append(name)
                return seed(inputs)
            configured = dict(common, parameters=replace(common['parameters'], tag_fusion_method=name))
            with patch.dict(poster.FUSION_METHODS, {name: record}):
                result = poster.refine_shared_poster_pose([
                    make_poster_inputs(**configured, arm='left', baseline=baseline),
                    make_poster_inputs(**configured, arm='right',
                                        baseline=replace(baseline, T_reference_imu=baseline.T_reference_imu.copy()))])
            for arm in ('left', 'right'):
                evidence = result[arm].evidence['environment_poster']
                self.assertEqual((evidence['status'], evidence['method']), ('fused', name))
            error = result['left'].T_reference_imu[:, :3, 3] - truth[:, :3, 3]
            self.assertLess(np.linalg.norm(error - error.mean(axis=0)), .05)
        self.assertEqual(seeded, ['staged', 'staged', 'joint', 'joint'])

    def test_shared_fusion_keeps_arm_without_poster_observations_on_baseline(self):
        data = poster_fusion_fixture()
        baseline, times, imu, observations, extrinsics, trajectory, parameters, detections, board, measured = data[:10]
        empty = replace(detections,
            sample_indices=np.empty(0, dtype=int), tag_ids=np.empty(0, dtype=int),
            corners_half_px=np.empty((0, 4, 2), dtype=float), side_px=np.empty(0),
            in_calibrated_crop=np.empty(0, dtype=bool), usable=np.empty(0, dtype=bool))
        common = dict(times=times, imu=imu, observations=observations, T_cam_imu=extrinsics,
                      trajectory_parameters=trajectory, parameters=replace(parameters, tag_fusion_enabled=True),
                      board=board, measured=measured)
        right_baseline = replace(baseline, T_reference_imu=baseline.T_reference_imu.copy())
        result = poster.refine_shared_poster_pose([
            make_poster_inputs(**common, arm='left', baseline=baseline, detections=detections),
            make_poster_inputs(**common, arm='right', baseline=right_baseline, detections=empty),
        ])
        self.assertEqual(result['left'].evidence['environment_poster']['status'], 'fused')
        self.assertEqual(result['right'].evidence['environment_poster']['status'], 'not_observed_shared_board')
        self.assertIs(result['right'].T_reference_imu, right_baseline.T_reference_imu)

    def test_fusion_culls_one_bad_environment_sighting_and_reoptimizes(self):
        data = list(poster_fusion_fixture())
        detections = data[7]
        corners = detections.corners_half_px.copy()
        corners[0] += [20., 0.]
        data[7] = replace(detections, corners_half_px=corners)
        result = poster.fuse_poster(make_poster_inputs('left', *data[:10]))
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['status'], 'fused')
        self.assertEqual(evidence['culled_observations'], 1)
        self.assertLess(evidence['final_reprojection_rms_px'], .2)


class PosterWorkflowTests(unittest.TestCase):
    def test_configuration_requires_layout_and_valid_thresholds(self):
        import tempfile

        from umi.infra.file_store import write_yaml
        from umi.scene.collection import configuration as scene_inputs
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'batch/collection.yaml'
            write_yaml(path, dict(kit='rig', arms=['left', 'right'],
                                  description='', poster=dict(detection=True),
                                  map=dict(layout='shared', reference_arm='left'),
                                  backend=dict(mode='scene_map')))
            with self.assertRaisesRegex(ValueError, 'poster.layout 是必需字段'):
                scene_inputs.read_scene(path)
            write_yaml(path, dict(kit='rig', arms=['left', 'right'],
                                  description='',
                                  map=dict(layout='shared', reference_arm='left'),
                                  poster=dict(layout=str(tag_board.DEFAULT_LAYOUT), detection=True),
                                  backend=dict(mode='scene_map', scene_map=dict(poster_constraint='staged'))))
            expert = Path(temporary)/'expert.yaml'
            write_yaml(expert, dict(poster=dict(standalone_fusion=True)))
            with self.assertRaisesRegex(ValueError, 'collection.yaml'):
                SceneOverrides.load(expert)
            self.assertTrue(scene_inputs.read_scene(path).scene_map_poster_constraint_enabled)
            for override in (dict(tag_minimum_frames=2), dict(tag_corner_sigma_px=0.), dict(tag_fusion_enabled=1)):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    load_scene_parameters(overrides=override)

    def test_workflow_freezes_layout_and_reports_missing_rear_without_localizing_it(self):
        import json
        import shutil

        import test_scene

        from umi.infra.file_store import read_yaml, write_yaml
        from umi.scene import run as workflow
        from umi.store.result_io import read_arm_summary
        helper = test_scene.SceneWorkflowTests('runTest')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        helper.build_scene()
        original = helper.root/'poster.json'
        shutil.copyfile(tag_board.DEFAULT_LAYOUT, original)
        declaration_path = helper.batch_root / 'collection.yaml'
        declaration = read_yaml(declaration_path)
        declaration['poster'] = dict(layout=str(original), detection=True)
        declaration['backend']['scene_map'] = dict(poster_constraint='off')
        write_yaml(declaration_path, declaration)
        identities = []

        def cached(video, output, settings, **kwargs):
            identities.append(kwargs['layout'].asset_hash)
            # Simulate the live asset being edited after the first arm starts.
            original.write_text('{}')
            report = dict(status='fixture', observations=0)
            kwargs['on_ready'](None, report)
            return report

        with patch.object(poster_detection, 'cached', side_effect=cached):
            workflow.run_episode(helper.episode_path)
        output = helper.task_root/'results'
        frozen = result_paths(output).run / 'devices/environment_poster.json'
        self.assertEqual(tag_board.load(frozen).asset_hash, tag_board.load().asset_hash)
        self.assertEqual(identities, [tag_board.load().asset_hash]*2)
        for arm in ('left', 'right'):
            summary = read_arm_summary(output / 'data' / arm)
            self.assertEqual(summary['fusion']['environment_poster']['status'], 'missing_rear_calibration')
        metadata = json.loads((output / 'data/episode.json').read_text())
        self.assertIn('devices/environment_poster.json', metadata['evidence_assets'])
        asset_identity.validate_assets(output.resolve(), metadata['assets'])

    def test_nanosecond_equivalent_times_merge_but_real_lens_offsets_do_not(self):
        front = np.array([.1, .2, .3, .4])
        rear = np.array([.1+1e-15, .2-1e-15, .313, .313])
        nodes, output, sightings = poster.observation_nodes(front, rear)
        np.testing.assert_array_equal(nodes[output], front)
        np.testing.assert_array_equal(sightings, [0, 1, 3, 3])
        self.assertEqual(len(nodes), 5)
        self.assertGreater(np.min(np.diff(nodes)), .01)

    def test_result_distinguishes_rear_localization_from_poster_fusion(self):
        import test_scene

        from umi.reconstruction.estimation.models import FusedTrajectory
        from umi.scene import publication as trajectory_writers
        helper = test_scene.SceneWorkflowTests('runTest')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        recording = helper.prepared_recordings[helper.recording_paths['left']]
        count = len(recording.video_times_s)
        for status in ('fused', 'diagnostics_only', 'insufficient_observations'):
            fused = FusedTrajectory(np.tile(np.eye(4), (count,1,1)), np.ones(count,bool), np.ones(count,bool),
                                    dict(environment_poster=dict(status=status)))
            track = trajectory_writers.write_arm_trajectory(recording, fused, 0., dict(frame_id='test'), helper.root/status)
            rear = track.metadata['frame_geometry']['rear_lens']
            self.assertFalse(rear['used_for_localization'])
            self.assertEqual(rear['used_for_poster_fusion'], status == 'fused')
            self.assertEqual(rear['used'], status == 'fused')


# 独立图海报：实际 GTSAM 图、偏移时间、分段尺度与空观测回归。


def independent_poster_fixture(tag_ids=(12, 48, 88)):
    from test_scene import scene_motion as motion
    times,truth,imu = motion()
    parameters = config.load('trajectory').values
    parameters = replace(parameters,fusion=replace(parameters.fusion,node_stride=3),
                         environment_tags=replace(parameters.environment_tags,tag_fusion_enabled=True))
    camera = dict(camera_type='kb4',resolution=[1440,1440],fx=520.,fy=525.,cx=720.,cy=720.,k=[.01,-.001,.0001,-.00001],
                  q_xyzw=[0.,0.,0.,1.],p=[0.,0.,0.],cam_time_offset_ns=17000000)
    T_imu_rear = pose((.12,-.08,.05),(.04,-.03,.02))
    rear = dict(camera,q_xyzw=Rotation.from_matrix(T_imu_rear[:3,:3]).as_quat().tolist(),p=T_imu_rear[:3,3].tolist(),cam_time_offset_ns=30000000)
    noisy = truth.copy(); noisy[:,1,3] += .008*np.sin(times*3)
    observations = {k:(p[:3,3]/2.3,p[:3,:3]) for k,p in enumerate(noisy)}
    board = tag_board.load()
    board_pose = pose((.1,.08,0.),(2.5,-2.7,3.5))
    rear_times = times[1:-1:3]+.013
    sample = np.repeat(np.arange(len(rear_times)), len(tag_ids))
    ids = np.tile(tag_ids, len(rear_times))
    points = np.array([board.corners_m[i] for i in ids])
    cameras = resample_pose_series(times,truth,rear_times[sample]) @ T_imu_rear
    pixels,_ = poster_pose.project_observations(board_pose,cameras,points,poster_pose.camera_model(rear))
    side = np.linalg.norm(pixels-np.roll(pixels,-1,axis=1),axis=2).min(axis=1)
    detection = poster_detection.TagObservations(np.arange(len(rear_times))*6,rear_times-.013,sample,ids,pixels+240.,side,
        np.ones(len(ids),bool),np.ones(len(ids),bool),np.zeros(len(rear_times),int),np.zeros(len(rear_times),int))
    inputs = FusionInputs(dict(front=observations,rear={}),dict(front={},rear={}),imu,times,camera,
        T_imu_rear[:3,:3],np.linalg.inv(T_imu_rear)[:3,3],fusion_poster.PosterInputs(board,detection,rear))
    return inputs,parameters


class IndependentPosterTests(unittest.TestCase):
    def solve(self, inputs, parameters):
        return fusion_solver.solve(inputs,parameters,log=lambda *a,**k:None)

    def test_clock_shift_requires_independent_rear_confirmation_and_imu_coverage(self):
        from umi.core.frame import FPS_SRC
        from umi.reconstruction.estimation.fusion.clock_check import correct_recording_clock

        inputs, params = independent_poster_fixture(tag_ids=(12, 18, 40, 48, 74, 88))
        env = params.environment_tags
        for sign in (-1, 1):
            imu = inputs.imu_video_clock.copy()
            imu[:, 0] += sign / FPS_SRC
            shifted = replace(inputs, imu_video_clock=imu)
            corrected, report = correct_recording_clock(shifted, env)
            self.assertEqual(report['decision']['applied_imu_timestamp_shift_ns'], round(-sign*1e9/FPS_SRC))
            np.testing.assert_allclose(corrected.imu_video_clock, inputs.imu_video_clock, rtol=0, atol=1e-9)
            np.testing.assert_array_equal(shifted.imu_video_clock[:, 1:], inputs.imu_video_clock[:, 1:])
            self.assertIs(corrected.video_times_s, inputs.video_times_s)
        # Rear observations now agree with the shifted gyro, while the front requests a correction.
        rear = shifted.poster
        conflict = replace(shifted, poster=replace(rear, measured=dict(rear.measured,
            cam_time_offset_ns=rear.measured['cam_time_offset_ns'] + round(1e9/FPS_SRC))))
        kept, report = correct_recording_clock(conflict, env)
        self.assertIs(kept, conflict)
        self.assertEqual(report['decision']['reason'], 'rear_does_not_confirm_shift')
        trimmed = replace(shifted, imu_video_clock=imu[imu[:, 0] <= inputs.video_times_s[-1]+.001])
        kept, report = correct_recording_clock(trimmed, env)
        self.assertIs(kept, trimmed)
        self.assertEqual(report['decision']['reason'], 'shift_exceeds_imu_coverage')
        kept, report = correct_recording_clock(inputs, env)
        self.assertIs(kept, inputs)
        self.assertEqual(report['decision']['applied_imu_timestamp_shift_ns'], 0)

    def test_clock_keeps_calibration_for_changing_lag_or_larger_than_one_frame(self):
        from umi.core.frame import FPS_SRC
        from umi.reconstruction.estimation.fusion.clock_check import correct_recording_clock

        inputs, params = independent_poster_fixture()
        for shifts, reason in [(np.where(inputs.imu_video_clock[:, 0] < 4., -1., 1.), 'front_does_not_confirm_shift'),
                               (1.7, 'front_lag_outside_one_frame_hypothesis')]:
            imu = inputs.imu_video_clock.copy()
            imu[:, 0] += shifts / FPS_SRC
            shifted = replace(inputs, imu_video_clock=imu)
            kept, report = correct_recording_clock(shifted, params.environment_tags)
            self.assertIs(kept, shifted)
            self.assertEqual(report['decision']['reason'], reason)

    def test_joint_native_graph_preserves_timeline_and_optimizes_scale(self):
        inputs,parameters = independent_poster_fixture()
        result = self.solve(inputs,parameters)
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['status'],'fused')
        self.assertGreater(evidence['added_nodes'],0)
        self.assertEqual(evidence['frames'],len(inputs.poster.observations.video_times_s))
        np.testing.assert_array_equal(result.video_times_s,inputs.video_times_s)
        self.assertTrue(np.isfinite(result.T_world_imu).all())
        self.assertLess(evidence['final_reprojection_rms_px'],evidence['initial_reprojection_rms_px'])
        self.assertGreater(len(result.segments),0)
        self.assertEqual(evidence['map_scale'],'joint_segment_variables')
        self.assertAlmostEqual(evidence['first_video_clock_time_s'],inputs.poster.observations.video_times_s[0]+.013)

    def test_empty_and_diagnostics_keep_baseline_poses_exact(self):
        inputs,parameters = independent_poster_fixture()
        disabled = replace(parameters,environment_tags=replace(parameters.environment_tags,tag_fusion_enabled=False))
        baseline = self.solve(replace(inputs,poster=None),disabled)
        diagnosed = self.solve(inputs,disabled)
        self.assertEqual(diagnosed.evidence['environment_poster']['status'],'diagnostics_only')
        np.testing.assert_array_equal(diagnosed.T_world_imu,baseline.T_world_imu)
        detections = replace(inputs.poster.observations,usable=np.zeros(len(inputs.poster.observations.usable),bool))
        empty = self.solve(replace(inputs,poster=replace(inputs.poster,observations=detections)),parameters)
        np.testing.assert_array_equal(empty.T_world_imu,baseline.T_world_imu)
        self.assertEqual(empty.evidence['environment_poster']['status'],'insufficient_observations')

    def test_fixed_rear_rotation_and_correction_limit_use_the_shared_factors(self):
        from umi.core.geometry import imu_from_camera
        inputs, parameters = independent_poster_fixture()
        fixed = replace(parameters, environment_tags=replace(parameters.environment_tags,
            tag_rear_rotation_refinement_enabled=False))
        evidence = self.solve(inputs, fixed).evidence['environment_poster']
        np.testing.assert_array_equal(evidence['T_rear_camera_imu_optimized'], np.linalg.inv(imu_from_camera(inputs.poster.measured)))
        self.assertEqual(evidence['rear_rotation_correction_deg'], 0.)
        limited = replace(parameters, environment_tags=replace(parameters.environment_tags,
            tag_rear_rotation_maximum_correction_deg=1e-8))
        with self.assertRaisesRegex(poster_pose.PosterGeometryError, '旋转修正超限'):
            self.solve(inputs, limited)

    def test_warm_start_tracks_renumbered_exposures_and_preserves_new_seeds(self):
        import gtsam
        from gtsam.symbol_shorthand import B, V, X

        def states(times):
            values = gtsam.Values()
            for k, t in times.items():
                xyz = np.full(3, t)
                values.insert(X(k), gtsam.Pose3(gtsam.Rot3(), xyz))
                values.insert(V(k), xyz)
                values.insert(B(k), gtsam.imuBias.ConstantBias(xyz, xyz))
            return values

        old_times = {0: 0., 8: .12, 9: .24}
        new_times = {0: 0., 8: .24, 9: .36}
        old = states(old_times)
        initial = states({k: -1. for k in new_times})
        old.insert(fusion_poster.BOARD_KEY, gtsam.Pose3(gtsam.Rot3(), np.ones(3)))
        initial.insert(fusion_poster.BOARD_KEY, gtsam.Pose3())
        fusion_poster.PosterWarmStart(old, old_times).apply(initial, new_times)
        for k, expected in [(0, 0.), (8, .24), (9, -1.)]:
            np.testing.assert_allclose(initial.atPose3(X(k)).translation(), expected)
            np.testing.assert_allclose(initial.atVector(V(k)), expected)
            np.testing.assert_allclose(initial.atConstantBias(B(k)).accelerometer(), expected)
        np.testing.assert_array_equal(initial.atPose3(fusion_poster.BOARD_KEY).translation(), np.ones(3))

    def test_post_solve_cull_reuses_the_initialization_and_keeps_one_budget(self):
        inputs, parameters = independent_poster_fixture()
        corners = inputs.poster.observations.corners_half_px.copy()
        corners[0] += [25., 0.]
        inputs = replace(inputs, poster=replace(inputs.poster,
            observations=replace(inputs.poster.observations, corners_half_px=corners)))
        original = fusion_poster.isolated_cull
        calls = []

        def skip_seed_once(*args, **kwargs):
            calls.append(None)
            return False if len(calls) == 1 else original(*args, **kwargs)

        with patch.object(fusion_poster, 'isolated_cull', side_effect=skip_seed_once), \
                patch.object(fusion_solver, 'align_recording', wraps=fusion_solver.align_recording) as alignment:
            result = self.solve(inputs, parameters)
        self.assertEqual(alignment.call_count, 1)
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['culled_observations'], 1)
        self.assertEqual(evidence['culling']['stage'], 'optimization')
        self.assertEqual(evidence['culling']['passes'], 1)

    def test_initial_and_post_solve_culls_use_the_original_budget_and_mask(self):
        inputs, parameters = independent_poster_fixture()
        tags = inputs.poster.observations
        corners = tags.corners_half_px.copy()
        corners[[0, -1]] += [25., 0.]
        inputs = replace(inputs, poster=replace(inputs.poster, observations=replace(tags, corners_half_px=corners)))
        original = fusion_poster.isolated_cull
        for maximum, should_pass in [(0.15, True), (0.9/len(corners), False)]:
            current = replace(parameters, environment_tags=replace(parameters.environment_tags,
                                                                   tag_maximum_cull_fraction=maximum))
            initial = True
            def first_sighting_at_initialization(mask, *args, **kwargs):
                nonlocal initial
                if initial:
                    mask[1:] = False
                    initial = False
                return original(mask, *args, **kwargs)

            with self.subTest(maximum=maximum), patch.object(fusion_poster, 'isolated_cull',
                                                              side_effect=first_sighting_at_initialization):
                if should_pass:
                    evidence = self.solve(inputs, current).evidence['environment_poster']
                    audit = evidence['culling']
                    self.assertEqual(audit['passes'], 2)
                    self.assertEqual([entry['observation'] for entry in audit['removed']], [0, len(corners)-1])
                    self.assertEqual(evidence['observations'], audit['original_observations'] - 2)
                    self.assertTrue(evidence['final_checks']['positive_depth'])
                else:
                    with self.assertRaisesRegex(poster_pose.PosterGeometryError, '累计剔除') as rejected:
                        self.solve(inputs, current)
                    self.assertEqual(rejected.exception.metrics['culling']['reason'], 'count_budget')

    def test_post_solve_culls_require_progress_and_stop_at_two(self):
        inputs, parameters = independent_poster_fixture()
        measure = fusion_poster.PosterContext.measure
        for proposals, accepted in [([2, 1], True), ([2, 2], False), ([3, 2, 1], False)]:
            pending = iter(proposals)
            def propose(context, *args):
                count = next(pending, 0)
                if not count:
                    return measure(context, *args)
                mask = np.arange(len(context.selected)) < count
                audit = fusion_poster.culling_record(mask, context.selected, context.frame_ids, context.tag_ids,
                                                     context.times, context.parameters, stage='optimization')
                audit.update(pre_cull_reprojection_rms_px=1., pre_cull_reprojection_metrics={},
                             pre_cull_invalid_depth_sightings=0)
                raise fusion_poster.PosterCullRequest(audit)

            with self.subTest(proposals=proposals), patch.object(fusion_poster.PosterContext, 'measure', propose):
                if accepted:
                    audit = self.solve(inputs, parameters).evidence['environment_poster']['culling']
                    self.assertEqual([row['observation'] for row in audit['removed']], [0, 1, 2])
                    self.assertEqual([step['observations'] for step in audit['stages']], proposals)
                else:
                    with self.assertRaisesRegex(poster_pose.PosterGeometryError, '有限轮次'):
                        self.solve(inputs, parameters)

    def test_conflicting_board_rejects_and_missing_assets_cannot_silently_disable(self):
        inputs,parameters = independent_poster_fixture()
        with self.assertRaisesRegex(ValueError,'布局'):
            self.solve(replace(inputs,poster=None),parameters)
        pixels = inputs.poster.observations.corners_half_px.copy(); pixels[len(pixels)//2:] += [80.,0.]
        detections = replace(inputs.poster.observations,corners_half_px=pixels)
        with self.assertRaises(poster_pose.PosterGeometryError):
            self.solve(replace(inputs,poster=replace(inputs.poster,observations=detections)),parameters)

    def test_dual_observations_retimed_without_crossing_map_switch_or_gaps(self):
        inputs,_ = independent_poster_fixture()
        times = np.array([0.,.1,.2,.3,.4])
        rear = {k:(np.array([t+.013,0.,0.]),np.eye(3)) for k,t in enumerate(times)}
        inputs = replace(inputs,video_times_s=times,observations=dict(front={},rear=rear),map_ids=dict(front={},rear={0:'a',1:'a',2:'b',3:'b',4:'b'}))
        from umi.reconstruction.estimation.fusion.data import retime_rear
        result = retime_rear(inputs, inputs.poster.measured)
        self.assertNotIn(0,result.observations['rear'])
        self.assertNotIn(2,result.observations['rear'])
        self.assertAlmostEqual(result.observations['rear'][1][0][0],.1)
        self.assertAlmostEqual(result.observations['rear'][3][0][0],.3)


def write_camera(path, measured, *, serial='test-camera', rear=False):
    import json
    document = dict(value0=dict(T_imu_cam=[dict(zip(('px','py','pz','qx','qy','qz','qw'),[*measured['p'],*measured['q_xyzw']]))],
        intrinsics=[dict(camera_type='kb4',intrinsics={**{k:measured[k] for k in ('fx','fy','cx','cy')},**dict(zip(('k1','k2','k3','k4'),measured['k']))})],
        resolution=[[1440,1440]],cam_time_offset_ns=measured['cam_time_offset_ns']))
    if rear: document.update(kind='umi_rear_calibration',camera_serial=serial,lens='rear',validation=dict(verdict='PASS'))
    path.write_text(json.dumps(document))


class IndependentPosterPipelineTests(unittest.TestCase):
    def test_frozen_input_reaches_real_fusion_and_detects_tampering(self):
        import csv
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout
        from dataclasses import asdict
        from pathlib import Path

        from umi.reconstruction.segment import environment_assets as calibration_assets
        from umi.reconstruction.segment import fusion_runner
        inputs,parameters = independent_poster_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video,front,rear = root/'raw.insv',root/'front.json',root/'rear.json'
            video.write_bytes(b'synthetic container; detector mocked')
            write_camera(front,inputs.camera)
            write_camera(rear,inputs.poster.measured,rear=True)
            with patch.object(poster_detection,'detect',return_value=inputs.poster.observations):
                manifest = calibration_assets.prepare_poster(video,root/'poster',tag_board.DEFAULT_LAYOUT,rear,parameters.environment_tags)
            loaded = calibration_assets.load_poster(manifest)
            self.assertEqual(loaded.measured['cam_time_offset_ns'],30000000)
            frames,imu,times = root/'frames.csv',root/'imu.csv',root/'times.csv'
            with frames.open('w') as stream:
                writer = csv.writer(stream); writer.writerow('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw,map'.split(','))
                for k,(p,r) in inputs.observations['front'].items():
                    writer.writerow([k,inputs.video_times_s[k],2,1,*p,*Rotation.from_matrix(r).as_quat(),'map0'])
            with imu.open('w') as stream:
                writer=csv.writer(stream);writer.writerow(['#t','wx','wy','wz','ax','ay','az'])
                for row in inputs.imu_video_clock:
                    writer.writerow([round((row[0]+1)*1e9)+17000000,*row[1:]])
            times.write_text('t_canonical_s\n'+''.join(f'{t+1:.12f}\n' for t in inputs.video_times_s))
            parameter_file=root/'parameters.json';parameter_file.write_text(json.dumps(asdict(parameters)))
            with redirect_stdout(io.StringIO()):
                fusion_runner.fuse(fusion_runner.FusionRequest(frames,frames,imu,17000000,(0,0,0,1),(0,0,0),root/'out',
                    front,parameter_file,times=times,poster=manifest))
            evidence=json.loads((root/'out/fuse_info.json').read_text())
            self.assertEqual(evidence['environment_poster']['status'],'fused')
            self.assertEqual(np.loadtxt(root/'out/traj_tum.txt').shape,(len(inputs.video_times_s),8))
            (manifest.parent/'rear.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'哈希'):
                calibration_assets.load_poster(manifest)

    def test_preflight_requires_registered_matching_rear(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from umi.reconstruction.estimation.models import TrajectoryRequest
        from umi.reconstruction.segment import environment_assets as calibration_assets
        inputs,_ = independent_poster_fixture()
        request=TrajectoryRequest(configuration=config.load('trajectory'),tag_layout=tag_board.DEFAULT_LAYOUT)
        selected=SimpleNamespace(camera_serial='test-camera',intrinsics_source='measured',
                                 assets=SimpleNamespace(camera_json=Path('front.json')))
        with patch('umi.devices.rear_fallback.derive', side_effect=ValueError('无法读取 INSV 工厂参数')):
            with self.assertRaisesRegex(ValueError,'工厂参数'):
                calibration_assets.select_environment_assets(Path('raw.insv'),selected,request)
        with tempfile.TemporaryDirectory() as temporary:
            rear=Path(temporary)/'rear.json';write_camera(rear,inputs.poster.measured,rear=True)
            request=replace(request,rear_calibration='002')
            with patch.object(registry,'rear_path',return_value=rear):
                valid=calibration_assets.select_environment_assets(Path('raw.insv'),selected,request)
                self.assertEqual(valid.rear.identity.version,'002')
                write_camera(rear,inputs.poster.measured,rear=True,serial='another-camera')
                with self.assertRaisesRegex(ValueError,'camera_serial'):
                    calibration_assets.select_environment_assets(Path('raw.insv'),selected,request)

    def test_single_arm_fallback_carries_scene_tags_to_native_engine(self):
        import json
        from types import SimpleNamespace

        from test_scene import SceneFallbackAdapterTests

        from umi.reconstruction.segment import environment_assets as calibration_assets
        helper = SceneFallbackAdapterTests('runTest')
        helper.setUp();self.addCleanup(helper.doCleanups)
        inputs,_=independent_poster_fixture()
        helper.episode.recordings['right'].write_bytes(b'synthetic container')
        rear=helper.root/'rear.json';write_camera(rear,inputs.poster.measured,rear=True)
        helper.episode.scene.tag_layout=tag_board.DEFAULT_LAYOUT
        helper.episode.scene.environment_tag_detection=True
        helper.episode.scene.parameters=replace(helper.episode.scene.parameters,tag_fusion_enabled=True)
        helper.recording=replace(helper.recording,camera=SimpleNamespace(
            files=dict(helper.recording.camera.files,rear=rear),measurements=dict(rear=inputs.poster.measured)))
        def engine(command,**kwargs):
            result=helper.engines(command,**kwargs)
            return result
        from umi.reconstruction.segment import fusion_runner
        def fuse(request,**kwargs):
            helper.engines.fuse(request,**kwargs)
            calibration_assets.load_poster(request.poster)
            parameters=json.loads(request.parameters.read_text())
            self.assertTrue(parameters['environment_tags']['tag_fusion_enabled'])
            evidence=request.output/'fuse_info.json'
            document=json.loads(evidence.read_text());document['environment_poster']=dict(status='fused',actual_error='not_measured')
            evidence.write_text(json.dumps(document))
        with patch.object(poster_detection,'detect',return_value=inputs.poster.observations),patch("subprocess.run",side_effect=engine),patch.object(fusion_runner,'execute',side_effect=fuse):
            result=helper.run_fallback()
        self.assertEqual(result.evidence['environment_poster']['status'],'fused')
        self.assertEqual(result.evidence['mode'],'independent')

    def test_dual_native_graph_uses_measured_rear_and_distinct_board_key(self):
        from umi.core.geometry import imu_from_camera
        inputs,parameters=independent_poster_fixture()
        # Supply a second independent visual-map scale, in the rear camera's own clock.
        poses=np.tile(np.eye(4),(len(inputs.video_times_s),1,1))
        for k,(position,rotation) in inputs.observations['front'].items():
            poses[k,:3,:3]=rotation;poses[k,:3,3]=position*2.3
        shifted=inputs.video_times_s[:-1]+.013
        rear_poses=resample_pose_series(inputs.video_times_s,poses,shifted) @ imu_from_camera(inputs.poster.measured)
        rear_obs={k:(pose[:3,3]/1.7,pose[:3,:3]) for k,pose in enumerate(rear_poses)}
        inputs=replace(inputs,observations=dict(inputs.observations,rear=rear_obs))
        parameters=replace(parameters,input=replace(parameters.input,mode='dual'))
        result=fusion_solver.solve(inputs,parameters,log=lambda *a,**k:None)
        self.assertEqual(result.evidence['environment_poster']['status'],'fused')
        self.assertGreater(result.evidence['obs']['rear'],0)
        self.assertIsNotNone(result.evidence['rear_lever_final'])
        self.assertTrue(np.isfinite(result.T_world_imu).all())


# Poster independent backend: board frame, whole-task routing and publication.


class MapSelectionTests(unittest.TestCase):
    def test_map_directory_and_manifest_form_one_state(self):
        directory, manifest = Path('/unread/map'), {'version': 'one'}
        selection = MapSelection.available(directory, manifest)
        self.assertEqual((selection.directory, selection.manifest), (directory, manifest))
        reason = {'category': 'map_unavailable'}
        self.assertEqual(MapSelection.without_map(reason).failure, reason)
        self.assertIsNone(MapSelection.without_map().manifest)
        with self.assertRaises(ValueError):
            MapSelection.available(None, None)
        for values in (dict(directory=directory), dict(manifest=manifest),
                       dict(directory=directory, manifest=manifest, failure=reason)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                MapSelection(**values)


class AutomaticBackendChainTests(unittest.TestCase):
    def test_auto_generated_poster_vio_preparation_quality_can_fall_through(self):
        episode = SimpleNamespace(
            scene=SimpleNamespace(environment_tag_detection=True,
                                  parameters=SimpleNamespace(tag_fusion_enabled=True),
                                  reconstruction_mode='auto'))
        inputs = SimpleNamespace(episode=episode)
        with patch.object(poster_vio, 'prepare_task_inputs', side_effect=PosterInitializationQualityError('too few tags')):
            with self.assertRaises(PosterVioQualityError):
                poster_vio.execute_task(inputs)

    def test_poster_vio_bad_input_remains_fatal_in_auto_and_explicit_modes(self):
        issue = ValueError('bad frozen input')
        for mode in ('auto', 'poster_vio'):
            episode = SimpleNamespace(scene=SimpleNamespace(environment_tag_detection=True,
                parameters=SimpleNamespace(tag_fusion_enabled=True), reconstruction_mode=mode))
            with self.subTest(mode=mode), patch.object(poster_vio, 'prepare_task_inputs', side_effect=issue):
                with self.assertRaises(ValueError) as caught:
                    poster_vio.execute_task(SimpleNamespace(episode=episode))
                self.assertIs(caught.exception, issue)

    def test_map_quality_then_poster_vio_quality_then_independent(self):
        from umi.scene import routing as estimation
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            calls = []
            @dataclass(frozen=True)
            class Inputs:
                directory: Path
                episode: object
                console: object
            console = SimpleNamespace(messages=[], write=lambda message, **_: console.messages.append(message))
            policy = SimpleNamespace(reconstruction_mode='auto', recording_trim={},
                                     auto_order=('scene_map', 'poster_vio', 'poster_independent'))
            inputs = Inputs(root, SimpleNamespace(scene=policy), console)
            expected = SimpleNamespace(coordinate_record={})

            def scene(inputs, selection):
                calls.append('scene_map')
                raise SceneFusionQualityError('map coverage too low')

            def vio(inputs, selection):
                calls.append('poster_vio')
                raise PosterVioQualityError('poster support too low')

            def independent(inputs, selection):
                calls.append(('poster_independent', selection.failure))
                return expected

            with patch.dict(estimation.ESTIMATORS, dict(scene_map=scene, poster_vio=vio,
                                                        poster_independent=independent)):
                result = estimation.execute(inputs, MapSelection.available(root / 'map', {'version': 'v'}))
            self.assertIs(result, expected)
            self.assertEqual([item if isinstance(item, str) else item[0] for item in calls],
                             ['scene_map', 'poster_vio', 'poster_independent'])
            failure = calls[-1][1]
            self.assertEqual(failure['category'], 'scene_quality')
            self.assertEqual([item['backend'] for item in failure['attempts']],
                             ['scene_map', 'poster_vio'])
            attempts = json.loads((root / 'reconstruction_attempts/attempts.json').read_text())
            self.assertEqual(attempts['selected_backend'], 'poster_independent')
            self.assertEqual([item['status'] for item in attempts['attempts']],
                             ['failed', 'failed', 'succeeded'])
            self.assertIn('自动后端路由', console.messages[0])
            self.assertIn('后端切换：scene_map → poster_vio', console.messages[2])
            self.assertIn('后端选定：poster_independent', console.messages[-1])

    def test_last_candidate_retains_its_quality_error(self):
        from umi.scene import routing as estimation
        scene = SimpleNamespace(reconstruction_mode='auto', recording_trim={}, auto_order=('poster_vio',))
        with tempfile.TemporaryDirectory() as root:
            inputs = TaskInputs(SimpleNamespace(scene=scene), {}, {}, Path(root), None, None, None)
            error = PosterVioQualityError('poster support too low')

            def reject(*args):
                raise error

            with patch.dict(estimation.ESTIMATORS, poster_vio=reject), self.assertRaises(PosterVioQualityError) as caught:
                estimation.execute(inputs, MapSelection.without_map())
            self.assertIs(caught.exception, error)


class PosterIndependentWorkflowTests(unittest.TestCase):
    def setUp(self):
        import test_scene
        self.helper = test_scene.SceneWorkflowTests('runTest')
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        # Workflow fixtures replace the solver and use placeholder INSV files.
        detector = patch.object(measurements, 'prepare_detections', return_value=None)
        detector.start()
        self.addCleanup(detector.stop)

    def configure(self, mode='poster_independent', *, confirmed=False):
        h = self.helper
        collection = h.batch_root / 'collection.yaml'
        doc = read_yaml(collection)
        doc['backend'] = dict(mode=mode, scene_map=dict(poster_constraint='joint'))
        doc['poster'] = dict(layout=str(tag_board.DEFAULT_LAYOUT), detection=True)
        doc['tasks'] = dict(take_001=dict(instruction='move both grippers', arms=['left', 'right'], sync=dict(
            mode='manual', offsets_s=dict(left=0., right=0.), confirmed=confirmed,
            note='fixture common clock' if confirmed else '')))
        write_yaml(collection, doc)
        for video, camera in list(h.cameras.items()):
            h.cameras[video] = replace(camera, files=dict(camera.files, rear=camera.files['front']),
                                      measurements=dict(camera.measurements, rear=camera.measurements['front']))

    def solved(self, episode, arm, recording, directory, *, require_poster=False):
        self.assertTrue(require_poster)
        raw = self.helper.raw[arm].T_reference_camera.copy()
        raw[:, :3, 3] *= 2.3
        return FusedTrajectory(raw, np.ones(len(raw), bool), np.ones(len(raw), bool), dict(
            mode='poster_independent', scene_map_priors_used=False,
            localization_fraction=1., maximum_visual_gap_s=0.,
            independent=dict(gravity_output_m_s2=[0., 0., -9.81]),
            environment_poster=dict(status='fused', layout=tag_board.load().record(),
                observation_video_times_s=recording.video_times_s.tolist(),
                position_support_video_times_s=recording.video_times_s.tolist())))

    def run_poster(self):
        with patch.object(independent_reconstruction, 'reconstruct', side_effect=self.solved), \
                patch.object(poster_vio, 'execute_task', side_effect=PosterVioQualityError('too few tags')):
            return workflow.run_episode(self.helper.episode_path)

    def test_no_map_bimanual_publication_reuse_and_viewer_contract(self):
        self.configure(confirmed=True)
        h = self.helper
        with patch.object(map_lifecycle_binding, 'bind_map', side_effect=AssertionError('must not bind map')):
            result = self.run_poster()
        self.assertEqual(result['status'], 'ok')
        manifest = json.loads((h.task_root / 'results/data/episode.json').read_text())
        self.assertEqual(manifest['kind'], 'umi_poster_episode')
        self.assertIsNone(manifest['map'])
        self.assertIsNone(manifest['inputs']['scene_map_sha256'])
        self.assertEqual(scene_reference(manifest)['scope'], 'task_poster')
        self.assertTrue((h.task_root / 'results/data/bimanual.csv').is_file())
        summary = read_arm_summary(h.task_root / 'results/data/left')
        self.assertEqual(summary['mode'], 'poster')
        from umi.apps.viewer import maps
        self.assertEqual(maps.load(h.task_root / 'results/data/left', summary)['maps'], [])
        from umi.apps.viewer.catalog import RecordingCatalog
        catalog = RecordingCatalog(h.root)
        entries = catalog.scan()['recordings']
        entry = next(item for item in entries if item.get('available') and item['serial'] == '任务海报轨迹')
        loaded = catalog.load(entry['id'])
        self.assertEqual(loaded['summary']['mode'], 'poster')
        self.assertIsNotNone(loaded['companion'])
        self.assertIn('imu', loaded['tracks'])
        self.assertEqual(workflow.run_episode(h.episode_path)['status'], 'skipped')

    def test_auto_missing_map_falls_back_without_building(self):
        self.configure('auto')
        with patch.object(map_lifecycle_binding, 'build_scene', side_effect=AssertionError('must not build')):
            result = self.run_poster()
        self.assertEqual(result['status'], 'ok')
        scene = json.loads((self.helper.task_root / 'results/data/episode.json').read_text())['scene']
        self.assertEqual(scene['reconstruction_backend'], 'poster_independent')
        self.assertIn('地图不可用', scene['fallback']['reason'])
        self.assertEqual(scene['fallback']['category'], 'map_unavailable')

    def test_auto_poster_success_does_not_prepare_map_and_survives_map_publication(self):
        self.configure('auto', confirmed=True)
        h = self.helper
        collection = h.batch_root / 'collection.yaml'
        doc = read_yaml(collection)
        doc['backend']['auto_order'] = ['poster_vio', 'poster_independent', 'scene_map']
        write_yaml(collection, doc)

        def vio(inputs, *, fallback):
            result = poster_independent.execute_task(inputs, fallback=fallback)
            result.coordinate_record['reconstruction_backend'] = 'poster_vio'
            return result

        with patch.object(map_repository_binding, 'bind_map') as bind, patch.object(map_repository, 'load_published') as load, \
                patch.object(poster_vio, 'execute_task', side_effect=vio) as solve, \
                patch.object(independent_reconstruction, 'reconstruct', side_effect=self.solved):
            self.assertEqual(workflow.run_episode(h.episode_path, force=True)['status'], 'ok')
        solve.assert_called_once()
        bind.assert_not_called()
        load.assert_not_called()
        manifest = json.loads((h.task_root / 'results/data/episode.json').read_text())
        self.assertEqual(manifest['scene']['reconstruction_backend'], 'poster_vio')
        self.assertNotIn('auto_map_attempted', manifest['inputs'])
        h.build_scene()
        with patch.object(map_repository, 'load_published') as load:
            self.assertEqual(workflow.run_episode(h.episode_path)['status'], 'skipped')
        load.assert_not_called()
        doc['backend']['auto_order'] = ['poster_independent', 'poster_vio', 'scene_map']
        write_yaml(collection, doc)
        with patch.object(map_repository, 'load_published') as load:
            self.assertEqual(self.run_poster()['status'], 'ok')
        load.assert_not_called()
        manifest = json.loads((h.task_root / 'results/data/episode.json').read_text())
        self.assertEqual([item['backend'] for item in manifest['scene']['attempts']], ['poster_independent'])

    def test_auto_poster_input_error_does_not_build_map(self):
        self.configure('auto')
        h = self.helper
        collection = h.batch_root / 'collection.yaml'
        doc = read_yaml(collection)
        doc['backend']['auto_order'] = ['poster_vio', 'poster_independent', 'scene_map']
        write_yaml(collection, doc)
        with patch.object(poster_vio, 'execute_task', side_effect=PosterVioQualityError('too few tags')), \
                patch.object(poster_independent, 'execute_task', side_effect=ValueError('bad IMU')), \
                patch.object(map_repository_binding, 'bind_map') as bind, patch.object(map_repository, 'load_published') as load:
            with self.assertRaisesRegex(ValueError, 'bad IMU'):
                workflow.run_episode(h.episode_path)
        bind.assert_not_called()
        load.assert_not_called()

    def test_unexpected_map_reader_errors_do_not_trigger_fallback(self):
        self.configure('auto')
        episode = episode_inputs.read_episode(self.helper.episode_path)
        from umi.scene.maps import binding
        for error in (ValueError('program error'), KeyError('program key'), RuntimeError('engine crash')):
            with self.subTest(error=error), patch.object(map_repository, 'load_published', side_effect=error):
                with self.assertRaises(type(error)):
                    binding.select_map(episode, None)

    def test_map_compatibility_failure_preserves_category_and_error_type(self):
        self.configure('auto')
        episode = episode_inputs.read_episode(self.helper.episode_path)
        from umi.scene.maps import binding
        with patch.object(map_repository, 'load_published', side_effect=MapCompatibilityError('engine changed')):
            _, selection = binding.select_map(episode, None)
        self.assertEqual(selection.failure['category'], 'map_incompatible')
        self.assertEqual(selection.failure['error_type'], 'MapCompatibilityError')

    def test_auto_corrupt_map_falls_back_without_using_any_scene_pose(self):
        h = self.helper
        h.build_scene()
        directory, _ = map_repository.load_published(h.scene)
        (directory / 'maps/shared_front/map.osa').write_bytes(b'corrupt map')
        self.configure('auto')
        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene', side_effect=AssertionError('no scene poses')):
            result = self.run_poster()
        self.assertEqual(result['status'], 'ok')
        manifest = json.loads((h.task_root / 'results/data/episode.json').read_text())
        self.assertIsNone(manifest['inputs']['scene_map_sha256'])
        self.assertIsNone(manifest['map'])

    def test_auto_missing_declared_map_does_not_block_layout_inference(self):
        self.configure('auto')
        h = self.helper
        path = h.batch_root / 'collection.yaml'
        doc = read_yaml(path)
        doc['map']['layout'] = 'auto'
        doc['map']['version'] = '20261117_000000-deadbeef'
        write_yaml(path, doc)
        batch = batches.read_inputs(h.batch_root, include_mapping=False)
        self.assertEqual(batch.scene.mode, 'shared_front')
        batches.save_record(batch)
        episode = episode_inputs.read_episode(h.episode_path)
        self.assertEqual(episode.map_version, doc['map']['version'])
        from umi.scene.maps import binding
        with patch.object(map_repository, 'load_published', side_effect=MapUnavailableError('missing')) as load:
            _, selected = binding.select_map(episode, None)
        self.assertEqual(load.call_args.kwargs['version'],
                         doc['map']['version'])
        self.assertIsNone(selected.manifest)

    def test_missing_poster_configuration_fails_before_any_engine_call(self):
        self.configure()
        path = self.helper.batch_root / 'collection.yaml'
        doc = read_yaml(path)
        doc['poster']['detection'] = False
        write_yaml(path, doc)
        with patch.object(independent_reconstruction, 'reconstruct') as solve:
            with self.assertRaisesRegex(ValueError, 'poster.detection'):
                workflow.run_episode(self.helper.episode_path)
        solve.assert_not_called()

    def test_auto_scene_quality_failure_recomputes_both_hands(self):
        h = self.helper
        h.build_scene()
        self.configure('auto')
        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene',
                          side_effect=SceneFusionQualityError('scene coverage too low')), \
                patch.object(poster_vio, 'execute_task', side_effect=PosterVioQualityError('too few tags')), \
                patch.object(independent_reconstruction, 'reconstruct', side_effect=self.solved) as solve:
            result = workflow.run_episode(h.episode_path)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual({call.args[1] for call in solve.call_args_list}, {'left', 'right'})
        manifest = json.loads((h.task_root / 'results/data/episode.json').read_text())
        self.assertIsNone(manifest['map'])
        self.assertIsNotNone(manifest['inputs']['scene_map_sha256'])
        self.assertEqual(manifest['scene']['fallback']['category'], 'scene_quality')

    def test_auto_input_error_does_not_trigger_poster_fallback(self):
        h = self.helper
        h.build_scene()
        self.configure('auto')
        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene', side_effect=ValueError('bad IMU')), \
                patch.object(independent_reconstruction, 'reconstruct') as solve:
            with self.assertRaisesRegex(ValueError, 'bad IMU'):
                workflow.run_episode(h.episode_path)
        solve.assert_not_called()

    def test_auto_success_publishes_scene_candidate_without_local_fallback(self):
        h = self.helper
        h.build_scene()
        self.configure('auto')

        def scene_solve(episode, arm, recording, *args, **kwargs):
            trajectory = self.solved(episode, arm, recording, h.root, require_poster=True)
            trajectory.evidence['environment_poster']['T_scene_board'] = np.eye(4).tolist()
            trajectory = replace(trajectory, evidence=dict(trajectory.evidence, scene_map_priors_used=True,
                                                          visual_consistency_rms_m=0.))
            return scene_reconstruction.ArmReconstruction(trajectory)

        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene', side_effect=scene_solve), \
                patch.object(independent_reconstruction, 'reconstruct') as solve:
            result = workflow.run_episode(h.episode_path)
        solve.assert_not_called()
        self.assertEqual(result['status'], 'ok')
        manifest = json.loads((h.task_root / 'results/data/episode.json').read_text())
        self.assertEqual(manifest['kind'], 'umi_scene_episode')
        self.assertIsNotNone(manifest['map'])
        self.assertEqual(scene_reference(manifest)['scope'], 'task_poster')
        self.assertEqual(manifest['scene']['reconstruction_backend'], 'scene_map')
        self.assertNotEqual(manifest['scene']['frame_id'], manifest['inputs']['scene_map_reference']['frame_id'])

    def test_missing_poster_arm_never_replaces_previous_success(self):
        self.configure()
        h = self.helper
        self.run_poster()
        previous = (h.task_root / 'results').resolve()

        def broken(*args, **kwargs):
            result = self.solved(*args, **kwargs)
            if args[1] == 'right':
                return replace(result, evidence=dict(result.evidence, environment_poster=dict(status='insufficient_observations')))
            return result

        with patch.object(independent_reconstruction, 'reconstruct', side_effect=broken):
            with self.assertRaisesRegex(ValueError, '未完成海报融合'):
                workflow.run_episode(h.episode_path, force=True)
        self.assertEqual((h.task_root / 'results').resolve(), previous)

    def test_disagreeing_gravity_rejects_bimanual_anchor(self):
        self.configure()

        def broken(*args, **kwargs):
            result = self.solved(*args, **kwargs)
            if args[1] == 'right':
                return replace(result, evidence=dict(result.evidence,
                    independent=dict(gravity_output_m_s2=[9.81, 0., 0.])))
            return result

        with patch.object(independent_reconstruction, 'reconstruct', side_effect=broken):
            with self.assertRaisesRegex(ValueError, '重力方向不一致'):
                workflow.run_episode(self.helper.episode_path)

    def test_backend_gravity_gate_uses_its_own_parameter(self):
        self.configure()
        h = self.helper
        episode = episode_inputs.read_episode(h.episode_path)
        parameters = replace(episode.scene.parameters, maximum_placement_rotation_deg=.1,
                             poster_maximum_gravity_difference_deg=3.)
        episode = replace(episode, session=replace(episode.session,
            scene=replace(episode.scene, parameters=parameters)))

        def tilted(*args, **kwargs):
            result = self.solved(*args, **kwargs)
            if args[1] == 'right':
                gravity = Rotation.from_euler('x', 2., degrees=True).apply([0., 0., -9.81])
                result = replace(result, evidence=dict(result.evidence,
                    independent=dict(gravity_output_m_s2=gravity.tolist())))
            return result

        prepared = {arm: h.prepared_recordings[path] for arm, path in h.recording_paths.items()}
        with patch.object(independent_reconstruction, 'reconstruct', side_effect=tilted):
            poster_independent.solve_task(episode, prepared, h.root / 'solve')
            episode = replace(episode, session=replace(episode.session, scene=replace(episode.scene,
                parameters=replace(parameters, poster_maximum_gravity_difference_deg=1.))))
            with self.assertRaisesRegex(ValueError, '重力方向不一致'):
                poster_independent.solve_task(episode, prepared, h.root / 'solve')

    def test_task_override_and_mapless_batch_input(self):
        self.configure()
        h = self.helper
        collection = h.batch_root / 'collection.yaml'
        doc = read_yaml(collection)
        doc['map']['layout'] = 'auto'
        write_yaml(collection, doc)
        inputs = batches.read_inputs(h.batch_root)
        self.assertEqual(inputs.mapping_records, {})
        self.assertEqual(inputs.scene.mode, 'shared_front')
        doc['backend'] = dict(mode='scene_map')
        doc['tasks']['take_001']['backend'] = dict(mode='poster_independent')
        write_yaml(collection, doc)
        episode = episode_inputs.read_episode(h.episode_path)
        self.assertEqual(episode.scene.reconstruction_mode, 'poster_independent')


class PosterIndependentAdapterTests(unittest.TestCase):
    def setUp(self):
        from test_scene import SceneFallbackAdapterTests

        self.helper = SceneFallbackAdapterTests('runTest')
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        inputs, _ = independent_poster_fixture()
        h = self.helper
        h.episode.root = h.root / 'take_001'
        h.episode.scene.tag_layout = tag_board.DEFAULT_LAYOUT
        h.episode.scene.environment_tag_detection = True
        h.episode.scene.parameters = replace(h.episode.scene.parameters, tag_fusion_enabled=True,
                                              tag_fusion_method='joint')
        h.episode.recordings['right'].write_bytes(b'fixture')
        rear = h.root / 'rear.json'
        write_camera(rear, inputs.poster.measured, rear=True)
        from types import SimpleNamespace
        h.recording = replace(h.recording, camera=SimpleNamespace(
            files=dict(h.recording.camera.files, rear=rear), measurements=dict(rear=inputs.poster.measured)))
        self.observations = inputs.poster.observations
        self.board_pose = np.eye(4)
        self.board_pose[:3, :3] = Rotation.from_euler('x', 30, degrees=True).as_matrix()
        self.board_pose[:3, 3] = [1., 2., 3.]

    def execute(self, *, status='fused', times=True, sparse_visual=False, sparse_tags=False, weak_tags=False):
        from umi.perception.poster import detection as environment_tags
        h = self.helper

        from umi.reconstruction.segment import fusion_runner

        def fuse(request, **options):
            self.assertTrue(request.correct_poster_clock)
            h.engines.fuse(request, **options)
            if True:
                path = request.output / 'fuse_info.json'
                evidence = json.loads(path.read_text())
                if sparse_visual:
                    evidence['chains']['front'] = [([0, 1], 'full')]
                evidence['environment_poster'] = dict(status=status,
                    T_world_board=self.board_pose.tolist(),
                    observation_video_times_s=(h.recording.video_times_s[:3].tolist() if sparse_tags else
                                              h.recording.video_times_s.tolist()) if times else [],
                    position_support_video_times_s=[] if weak_tags or not times else (
                        h.recording.video_times_s[:3].tolist() if sparse_tags else h.recording.video_times_s.tolist()))
                path.write_text(json.dumps(evidence))

        with patch.object(environment_tags, 'detect', return_value=self.observations), \
                patch.object(fusion_runner, 'execute', side_effect=fuse):
            return independent_reconstruction.reconstruct(h.episode, 'right', h.recording,
                                        h.root / 'poster-independent', require_poster=True)

    def test_backend_scoped_fusion_settings_reach_the_engine(self):
        h = self.helper
        scene = h.episode.scene
        base = scene.trajectory.values.fusion
        scene.parameters = replace(scene.parameters, poster_independent_node_stride=2,
                                   poster_independent_imu_white_noise_multiplier=3.)
        self.execute()
        settings = json.loads((h.root / 'poster-independent/debug/settings/parameters.json').read_text())
        self.assertEqual(settings['fusion']['node_stride'], 2)
        self.assertAlmostEqual(settings['fusion']['acc_noise'], 3. * base.acc_noise)
        self.assertAlmostEqual(settings['fusion']['gyro_noise'], 3. * base.gyro_noise)
        self.assertEqual(settings['fusion']['acc_bias_rw'], base.acc_bias_rw)
        self.assertEqual(settings['fusion']['gyro_bias_rw'], base.gyro_bias_rw)
        self.assertEqual(scene.trajectory.values.fusion, base)

    def test_local_world_to_board_rotation_translation_and_gravity(self):
        result = self.execute()
        local = np.tile(np.eye(4), (len(result.tracked), 1, 1))
        local[:, 0, 3] = np.arange(len(local)) * .01
        np.testing.assert_allclose(result.T_reference_imu, np.linalg.inv(self.board_pose) @ local, atol=1e-6)
        np.testing.assert_allclose(result.evidence['independent']['gravity_output_m_s2'],
            self.board_pose[:3, :3].T @ [0., 0., -9.81], atol=1e-6)
        self.assertFalse(result.evidence['scene_map_priors_used'])

    def test_diagnostics_only_does_not_count_as_poster_fusion(self):
        with self.assertRaisesRegex(ValueError, '未实际融合'):
            self.execute(status='diagnostics_only')

    def test_missing_observation_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '时间证据不足'):
            self.execute(times=False)

    def test_continuous_poster_constraints_cover_a_natural_visual_gap(self):
        result = self.execute(sparse_visual=True)
        self.assertGreater(result.evidence['maximum_visual_gap_s'], 2.)
        self.assertLess(result.evidence['independent']['maximum_position_support_gap_s'], .04)

    def test_many_weak_poster_sightings_are_not_position_support(self):
        with self.assertRaisesRegex(ValueError, '几何有效的位置支持不足'):
            self.execute(sparse_visual=True, weak_tags=True)

    def test_missing_natural_and_poster_support_rejects_long_gap(self):
        with self.assertRaisesRegex(ValueError, '共同约束缺口过长'):
            self.execute(sparse_visual=True, sparse_tags=True)
