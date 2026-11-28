"""夹爪宽度、标签距离、稀疏采样和补检。"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import av
import cv2
import numpy as np
import test_scene as scene_tests
from registry_fakes import write_batch, write_kit

from umi.apps import calib as calib_cli
from umi.apps.viewer import ViewerDataError
from umi.apps.viewer.trajectory_reader import read_tag_distance
from umi.devices import registry
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import read_yaml, write_json, write_yaml
from umi.perception.gripper import calibration, frame_detection, width_estimation
from umi.perception.gripper import calibration as gripper_calibration
from umi.perception.gripper import distance as tag_distance
from umi.perception.gripper import frame_detection as sampled_detection
from umi.perception.gripper import frame_detection as sampling
from umi.perception.gripper import gap_recovery as recovery
from umi.perception.gripper import marker_detection as detection
from umi.perception.gripper import video_tools as width_video
from umi.perception.gripper.distance import TagDistanceRequest, TagDistanceResult, TagDistanceSeries
from umi.perception.gripper.frame_detection import crop_gray
from umi.perception.gripper.marker_detection import FrameDetection, MarkerDetector
from umi.perception.gripper.tag_pose import TagPoseEstimator, TagPoseQuality
from umi.scene.collection import batch as batch_inputs
from umi.scene.collection import batch as batches
from umi.scene.collection import configuration as scene_inputs
from umi.scene.collection import kit as kit_inputs
from umi.scene.inputs import recordings as camera_assets
from umi.store.cache import camera_identity

"""真实 KB4 投影下的中心间距、拒绝条件、尺寸配置、缓存和场景输出。"""


CAMERA = dict(camera_type='kb4', resolution=[1440, 1440], fx=532., fy=531., cx=724., cy=725.,
              k=[.099, -.066, .032, -.0068])
SIZES = dict(tag_sizes_mm={'0': 6., '1': 6.})


def estimator(camera=CAMERA, sizes=SIZES, **kwargs):
    return TagPoseEstimator(camera, sizes, quality=TagPoseQuality(kwargs.get('error', 2.), kwargs.get('ambiguity', 2.)))


def projected(e=None, *, poses=None):
    e = e or estimator()
    poses = poses or [(np.array([2.8, .2, .1]), np.array([-.025, .015, .10])),
                      (np.array([2.6, -.3, .1]), np.array([.025, .017, .11]))]
    corners = np.array([cv2.fisheye.projectPoints(e.object_corners_m_by_tag[tag].reshape(4, 1, 3), r, t, e.camera_matrix, e.distortion_coefficients)[0][:, 0] + 240
                        for tag, (r, t) in enumerate(poses)])
    return corners, float(np.linalg.norm(poses[0][1] - poses[1][1]) * 1000)


class DistanceGeometryTests(unittest.TestCase):
    def test_unequal_sizes_and_depths_recover_euclidean_millimetres(self):
        e = estimator(sizes=dict(tag_sizes_mm={'0': 6., '1': 9.}))
        corners, expected = projected(e)
        actual, source = e.distance(corners)
        self.assertEqual(source, 'two_tags')
        self.assertAlmostEqual(actual, expected, places=6)

    def test_rigid_camera_motion_does_not_change_metric_distance(self):
        e = estimator()
        poses = [(np.array([2.8, .2, .1]), np.array([-.025, .015, .10])),
                 (np.array([2.6, -.3, .1]), np.array([.025, .017, .11]))]
        R = cv2.Rodrigues(np.array([.1, -.1, .2]))[0]
        moved = [(cv2.Rodrigues(R @ cv2.Rodrigues(r)[0])[0], R @ t + np.array([.005, .01, .02])) for r, t in poses]
        corners, expected = projected(e, poses=moved)
        self.assertAlmostEqual(e.distance(corners)[0], expected, places=6)

    def test_physical_size_controls_scale_without_pixel_anchors(self):
        corners, expected = projected()
        changed = estimator(sizes=dict(tag_sizes_mm={'0': 12., '1': 12.}))
        self.assertAlmostEqual(changed.distance(corners)[0], expected * 2, places=6)

    def test_missing_outside_calibration_and_degenerate_corners_remain_invalid(self):
        base, _ = projected()
        for change, expected in [('missing', 'missing'), ('outside', 'outside_calibration'), ('collapsed', 'invalid_geometry')]:
            corners = base.copy()
            if change == 'missing': corners[0] = np.nan
            elif change == 'outside': corners[0, 0, 0] = 200
            else: corners[0, 1:] = corners[0, 0]
            with self.subTest(change=change):
                value, source = estimator().distance(corners)
                self.assertTrue(np.isnan(value))
                self.assertEqual(source, expected)

    def test_bad_reprojection_is_rejected(self):
        corners, _ = projected()
        corners[0, 0] += [15., -15.]
        value, source = estimator(error=.1).distance(corners)
        self.assertTrue(np.isnan(value))
        self.assertEqual(source, 'high_reprojection_error')

    def test_similar_reprojection_with_disagreeing_centres_is_ambiguous(self):
        corners, _ = projected()
        rotations = [np.zeros((3, 1)), np.zeros((3, 1))]
        translations = [np.array([[0.], [0.], [.1]]), np.array([[.01], [0.], [.1]])]
        with patch.object(cv2, 'solvePnPGeneric', return_value=(2, rotations, translations, None)), \
             patch.object(cv2.fisheye, 'projectPoints', return_value=((corners[0]-240).reshape(4, 1, 2), None)):
            value, source = estimator().distance(corners)
        self.assertTrue(np.isnan(value))
        self.assertEqual(source, 'ambiguous_pose')

    def test_camera_model_resolution_and_quality_parameters_are_checked(self):
        for camera in (dict(CAMERA, camera_type='pinhole'), dict(CAMERA, resolution=[720,720]),
                       dict(CAMERA, fx=float('nan')), dict(CAMERA, k=[0,0])):
            with self.assertRaises(ValueError): estimator(camera=camera)
        for args in (dict(error=0), dict(ambiguity=-1)):
            with self.assertRaises(ValueError): estimator(**args)


class DistanceConfigurationTests(unittest.TestCase):
    def test_kit_selects_and_snapshots_sizes_independently_of_camera_identity(self):
        from registry_fakes import write_batch, write_kit
        f=WidthKitTests();f.setUp();self.addCleanup(f.doCleanups)
        path=f.width.parent/'tags.json';write_json(path,SIZES)
        f.arms['left']['gripper']['tag_distance']='tags.json'
        write_kit('rig',f.arms)
        batch=batch_inputs.read_batch(write_batch(f.scene,'tags',kit='rig'))
        scene=replace(batch.scene,devices=batch.devices)
        with patch.object(registry,'read_insv_serial',return_value='CAMERA_A'):
            camera=camera_assets.select_camera(f.f.root/'raw.insv',scene)
            frozen=camera_assets.snapshot_cameras({'left':camera},f.f.root/'snapshot')['left']
            before=camera_identity(camera)
            write_json(path,dict(tag_sizes_mm={'0':8.,'1':9.}))
            changed=camera_assets.select_camera(f.f.root/'raw.insv',scene)
        self.assertEqual(before,camera_identity(changed))
        self.assertEqual(frozen.tag_distance_measurement,SIZES)
        self.assertEqual(gripper_calibration.read_tag_sizes(frozen.files['tag_distance']),SIZES)
        self.assertEqual(changed.tag_distance_measurement['tag_sizes_mm'],{'0':8.,'1':9.})
        with self.assertRaisesRegex(ValueError,'尺寸.*变化'):
            camera_assets.snapshot_cameras({'left':camera},f.f.root/'changed_snapshot')

    def test_both_tag_sizes_are_explicit_positive_and_configurable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'tags.json'
            write_json(path, dict(tag_sizes_mm={'0': 6, '1': 8}))
            self.assertEqual(gripper_calibration.read_tag_sizes(path)['tag_sizes_mm'], {'0': 6., '1': 8.})
            for invalid in ({}, dict(tag_sizes_mm={'0': 6}), dict(tag_sizes_mm={'0': True,'1':6}),
                            dict(tag_sizes_mm={'0':0,'1':6}), dict(tag_sizes_mm={'0':float('nan'),'1':6}),
                            dict(tag_sizes_mm={'0':6,'1':6}, anchors=[])):
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError): gripper_calibration.read_tag_sizes(path)

    def test_kit_paths_allow_shared_root_and_reject_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            result=kit_inputs.gripper_paths(dict(tcp='tcp.json',tag_distance='tags.json'), root)
            self.assertEqual(result['tag_distance'],str(root/'tags.json'))
            shared = kit_inputs.gripper_paths(dict(tcp='tcp.json', tag_distance='../tags.json'), root)
            self.assertEqual(shared['tag_distance'], str(root.parent/'tags.json'))
            for name in ('../../tags.json','/etc/tags.json','tags.yaml'):
                with self.assertRaises(ValueError):
                    kit_inputs.gripper_paths(dict(tcp='tcp.json',tag_distance=name), root)


class DistancePipelineTests(unittest.TestCase):
    def test_bimanual_viewer_crops_primary_distance_on_same_clock_as_trajectory(self):
        from umi.apps.viewer import catalog as episodes
        scene=dict(version='v1',frame_id='scene/v1')
        paths={arm:Path('/fixture')/arm for arm in ('left','right')}
        def data(times,values):
            return dict(times=times,summary=dict(scene=scene),frames=[{} for _ in times],
                        tracks=dict(tcp=dict(position=[[0.,0.,0.] for _ in times],quaternion=[[0.,0.,0.,1.] for _ in times])),
                        tag_distance=dict(distance_mm=values,valid=[v is not None for v in values],
                                          source=['two_tags' if v is not None else 'missing' for v in values]))
        arms={episodes.recording_key(paths['left']):data([0.,.1,.2,.3],[10.,11.,None,13.]),
              episodes.recording_key(paths['right']):data([.1,.2,.3,.4],[20.,21.,22.,23.])}
        catalog=SimpleNamespace(load=lambda key:arms[key],contains=lambda path:True)
        from umi.store.result_io import scene_map_identity
        inputs=dict(map_mode='shared_front',scene_assets={},
                    scene_map_sha256=scene_map_identity(dict(scene,mode='shared_front',assets={})))
        metadata=dict(schema_version=5,scene=scene,inputs=inputs)
        with patch.object(episodes,'read_episode',return_value=(metadata,paths)):
            result=episodes.load_episode(catalog,'episode',Path('/fixture/results'))
        self.assertEqual(result['times'],[.1,.2,.3])
        self.assertEqual(result['tag_distance']['distance_mm'],[11.,None,13.])
        self.assertEqual(result['tag_distance']['valid'],[True,False,True])
        self.assertEqual(result['companion']['tag_distance']['distance_mm'],[20.,21.,22.,23.])

    def test_size_and_camera_changes_reuse_corners_but_recompute_3d_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            corners, expected=projected()
            def process(sizes=SIZES,camera=CAMERA):
                request=TagDistanceRequest(video=root/'raw.insv',recording_sha256='recording',measurement=sizes,
                    camera=camera,video_times_s=np.arange(2)/30,episode_offset_s=-.12,
                    measurement_sha256=artifact_identity.canonical_hash(sizes),camera_sha256=artifact_identity.canonical_hash(camera),
                    quality=TagPoseQuality(2.,2.))
                return tag_distance.process(request,root)
            frame = SimpleNamespace(to_ndarray=lambda **kwargs: np.zeros((1,1921),dtype=np.uint8))
            found = detection.FrameDetection(tuple(polygon[None] for polygon in corners),np.array([[0],[1]]),
                                             {tag:corners[tag].mean(axis=0) for tag in (0,1)})
            def frames(video,times,indices):
                return iter((int(index),float(times[index]),frame) for index in indices)
            with patch.object(sampled_detection,'sampled_frames',side_effect=frames) as decode, \
                    patch.object(sampled_detection.FrameDetector, 'detect', return_value=found):
                first=process()
                again=process()
                doubled=process(dict(tag_sizes_mm={'0':12.,'1':12.}))
                process(camera=dict(CAMERA,fx=530.))
                self.assertEqual(decode.call_count,1)
            self.assertEqual(len(list((root/'tag_distance').glob('*/ready.json'))),3)
            np.testing.assert_allclose(first.samples.distance_mm,expected,atol=1e-5)
            np.testing.assert_array_equal(first.samples.distance_mm,again.samples.distance_mm)
            np.testing.assert_allclose(doubled.samples.distance_mm,2*expected,atol=1e-5)
            np.testing.assert_allclose(first.samples.times_s,[-.12,1/30-.12])
            # 任一缓存篡改都不能静默当作有效距离。
            for file in (root/'tag_distance').glob('*/distances.npz'): file.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'哈希'): process()

    def test_viewer_uses_separate_distance_field_and_null_for_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trajectory.csv'
            path.write_text('t_episode_s,width_mm,tag_distance_mm,tag_distance_valid,tag_distance_source\n'
                            '0,,50.25,1,two_tags\n0.1,,,0,missing\n')
            result=read_tag_distance(path,[0.,.1])
            self.assertEqual(result['distance_mm'],[50.25,None])
            self.assertEqual(result['valid'],[True,False])
            path.write_text(path.read_text().replace('50.25','nan'))
            with self.assertRaises(ViewerDataError): read_tag_distance(path,[0.,.1])


class DistanceWorkflowTests(unittest.TestCase):
    def test_scene_publishes_distances_without_width(self):
        import test_scene as fixtures

        from umi.scene import run as workflow
        f=fixtures.SceneWorkflowTests();f.setUp();self.addCleanup(f.doCleanups)
        f.build_scene()
        for video,camera in list(f.cameras.items()):
            path=camera.files['front'].parent/'tags.json';write_json(path,SIZES)
            f.cameras[video]=replace(camera, files=dict(camera.files,tag_distance=path),
                                    tag_distance_measurement=gripper_calibration.read_tag_sizes(path))
        def measure(request,cache,**kwargs):
            times,offset,measurement=request.video_times_s,request.episode_offset_s,request.measurement
            values=np.full(len(times),50.);source=np.full(len(times),'two_tags',dtype='<U32')
            values[2]=np.nan;source[2]='missing'
            return TagDistanceResult(TagDistanceSeries(times+offset,values,source),
                dict(valid_fraction=float(np.isfinite(values).mean()),tag_sizes_mm=measurement['tag_sizes_mm']))
        with patch.object(tag_distance,'process',side_effect=measure) as process, \
                patch('umi.perception.decoding.VideoDecode.run', return_value=[]):
            workflow.run_episode(f.episode_path)
            self.assertEqual(process.call_count,2)
            self.assertEqual(workflow.run_episode(f.episode_path)['status'],'skipped')
        output=f.task_root/'results'
        data=json.loads((output / 'data/episode.json').read_text())
        self.assertEqual(set(data['tag_distance']),{'left','right'})
        self.assertNotIn('gripper_width',data)
        for arm in ('left','right'):
            path=output / 'data' / arm/'trajectory.csv'
            with path.open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['tag_distance_mm'],'50.000000')
            self.assertFalse({'width_mm','close_01','closed','valid','source'} & rows[0].keys())
            self.assertNotIn('tag_distance',data['arm_results'][arm])
            self.assertEqual(rows[2]['tag_distance_valid'],'0')
            self.assertEqual(rows[2]['tag_distance_mm'],'')
            times=[float(row['t_episode_s']) for row in rows]
            self.assertIsNone(read_tag_distance(path,times)['distance_mm'][2])


# Exposure evidence, geometric gates and cache isolation for gap recovery.


class GapRecoveryTests(unittest.TestCase):
    def test_native_code_check_compensates_for_a_brightness_gradient(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 1, 72)
        grid = marker[6::12, 6::12] > 0
        level = np.mgrid[0:6, 0:6][0] * 28. + 20. + grid * 65.
        patch = np.pad(np.repeat(np.repeat(level, 12, axis=0), 12, axis=1), 6, mode='edge').astype('uint8')
        other = cv2.aruco.generateImageMarker(dictionary, 0, 72)[6::12, 6::12] > 0
        self.assertFalse(recovery._reads_code(patch, grid))
        self.assertTrue(recovery._reads_code(patch, grid, local_contrast=True))
        self.assertFalse(recovery._reads_code(patch, other, local_contrast=True))

    def setUp(self):
        self.estimator = estimator()
        self.pair, self.expected = projected()
        self.origin = np.array([600, 860])
        size = (800, 480)
        self.gray = np.full(size[::-1], 155, dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        for tag, corners in enumerate(self.pair):
            marker = cv2.aruco.generateImageMarker(dictionary, tag, 72)
            source = np.float32([[0, 0], [71, 0], [71, 71], [0, 71]])
            transform = cv2.getPerspectiveTransform(source, np.float32(corners - self.origin))
            pixels = cv2.warpPerspective(marker, transform, size, borderValue=155)
            mask = cv2.warpPerspective(np.full_like(marker, 255), transform, size)
            self.gray[mask > 128] = pixels[mask > 128]
        times = np.arange(7) / 30.0
        self.valid = np.array([True] + [False] * 5 + [True])
        corners = np.repeat(self.pair[None], len(times), axis=0)
        corners[~self.valid] = np.nan
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cache = SimpleNamespace(
            directory=Path(folder.name),
            video=Path('video.insv'),
            roi_px=tuple(map(int, [*self.origin, *(self.origin + size)])),
            native_times_s=times,
            query_times_s=times,
            attempted=np.zeros(7, bool),
            blocked=np.zeros(7, bool),
            quick_corners=corners,
            corners=corners,
            recovery_detector=lambda: None,
        )

    def scan(self, video, times, indices):
        for i in indices:
            yield i, times[i], self.gray

    def run_recovery(self):
        valid = self.valid.copy()
        distances = np.where(valid, self.expected, np.nan)
        sources = np.where(valid, 'two_tags', 'missing').astype('<U32')
        reports = recovery.recover(self.cache, [[1, 6]], valid, self.estimator, distances, sources, {}, None)
        return valid, distances, reports

    def test_raw_border_refinement_brings_perturbed_corners_back_to_known_edges(self):
        true = np.float32(self.pair - self.origin)
        seed = np.float32(true + [[2., 0.], [1., -2.], [-2., 1.], [0., 1.]])
        raw = cv2.GaussianBlur(self.gray, (3, 3), .65).astype(np.float32)
        fitted = np.asarray([recovery._refine_border(raw, corners) for corners in seed])
        self.assertLess(np.linalg.norm(fitted - true, axis=-1).mean(), .6)
        distance, status = self.estimator.distance(fitted + self.origin)
        self.assertEqual(status, 'two_tags')
        self.assertAlmostEqual(distance, self.expected, delta=.7)

    def test_enhancement_methods_cannot_both_pass_with_disagreeing_corners(self):
        native = self.pair - self.origin
        choices = [sampling._frame_detection(native), sampling._frame_detection(native + [4., 0.])]
        detector = recovery._DirectDetector(self.cache.roi_px, self.estimator)
        with (patch.object(detector.detector, 'detect', side_effect=choices),
              patch.object(detector, '_valid', return_value=True),
              patch.object(recovery, '_refine_border', side_effect=lambda image, corners: corners)):
            _, method = detector.detect(self.gray, [])
        self.assertEqual(method, 'conflict')

    def test_short_gap_recovery_keeps_existing_rows_and_reuses_verified_candidates(self):
        rows = np.array([2, 4])

        def recover():
            valid = self.valid.copy()
            distances = np.where(valid, self.expected, np.nan)
            sources = np.where(valid, 'two_tags', 'missing').astype('<U32')
            report = recovery.recover_short(self.cache, rows, valid, self.estimator, distances, sources, {}, None)
            self.assertEqual(report['accepted_rows'], [2, 4])
            np.testing.assert_array_equal(valid, [True, False, True, False, True, False, True])
            np.testing.assert_array_equal(distances[[0, -1]], [self.expected] * 2)
            return distances

        with (patch.object(sampling, 'sampled_frames', side_effect=self.scan),
              patch.object(recovery.frames, 'crop_gray', side_effect=lambda frame, *_: frame)):
            initial = recover()
        with patch.object(sampling, 'sampled_frames', side_effect=AssertionError('unexpected decode')):
            np.testing.assert_array_equal(recover(), initial)

    def test_cached_native_measurements_preserve_anchors_and_reject_blocked_frames(self):
        self.cache.blocked[3] = True
        with (
            patch.object(sampling, 'sampled_frames', side_effect=self.scan),
            patch.object(recovery.frames, 'crop_gray', side_effect=lambda frame, *_: frame),
        ):
            valid, distances, reports = self.run_recovery()
        np.testing.assert_array_equal(valid, [True, True, True, False, True, True, True])
        self.assertEqual(reports[0]['accepted_rows'], [1, 2, 4, 5])
        np.testing.assert_allclose(distances[valid], self.expected, atol=2)
        np.testing.assert_array_equal(distances[[0, -1]], [self.expected] * 2)
        with patch.object(sampling, 'sampled_frames', side_effect=AssertionError('unexpected decode')):
            np.testing.assert_array_equal(self.run_recovery()[1], distances)

    def test_hidden_tag_cannot_be_filled_from_identical_anchors(self):
        centre = np.mean(self.pair[1], axis=0) - self.origin
        x, y = centre.astype(int)
        self.gray[y - 40 : y + 40, x - 40 : x + 40] = 110
        with (
            patch.object(sampling, 'sampled_frames', side_effect=self.scan),
            patch.object(recovery.frames, 'crop_gray', side_effect=lambda frame, *_: frame),
        ):
            valid, _, _ = self.run_recovery()
        np.testing.assert_array_equal(valid, self.valid)

    def test_enhanced_decode_still_requires_correct_cells_on_raw_pixels(self):
        self.gray.fill(100)
        decoded = sampling._frame_detection(self.pair - self.origin)
        with (
            patch.object(sampling, 'sampled_frames', side_effect=self.scan),
            patch.object(recovery.frames, 'crop_gray', side_effect=lambda frame, *_: frame),
            patch.object(recovery.MarkerDetector, 'detect', return_value=decoded),
        ):
            valid, _, _ = self.run_recovery()
        np.testing.assert_array_equal(valid, self.valid)

    def test_position_conflict_between_two_good_geometries_is_rejected(self):
        native = self.pair - self.origin
        shifted = sampling._frame_detection(native + [4.0, 0.0])
        with (
            patch.object(sampling, 'sampled_frames', side_effect=self.scan),
            patch.object(recovery.frames, 'crop_gray', side_effect=lambda frame, *_: frame),
            patch.object(recovery.MarkerDetector, 'detect', return_value=shifted),
            patch.object(recovery._Tracker, 'observe', return_value=native),
        ):
            valid, _, reports = self.run_recovery()
        np.testing.assert_array_equal(valid, self.valid)
        self.assertEqual(reports[0]['counts'], {'conflict': 5})

    def test_boundary_gaps_do_not_extrapolate_and_timestamp_mismatch_fails(self):
        result = recovery.recover(
            self.cache, [[0, 2], [4, 7]], self.valid, self.estimator, np.zeros(7), np.full(7, 'missing'), {}, None
        )
        self.assertEqual(result, [])
        self.cache.native_times_s = self.cache.query_times_s + 0.01
        with (
            patch.object(sampling, 'sampled_frames', side_effect=self.scan),
            self.assertRaisesRegex(ValueError, '曝光'),
        ):
            self.run_recovery()


"""main 宽度算法的场景集成：标定冻结、原片检测、时钟和缺失值。"""


def measurement():
    return dict(mapping=dict(status='test_fixture', anchors=[dict(x=200., mm=0.), dict(x=400., mm=100.)]))


class WidthNumericalTests(unittest.TestCase):
    def test_mapping_clamps_closed_and_flags_extrapolated_openings(self):
        anchors = [dict(x=277.35, mm=0.), dict(x=543.02, mm=112.)]
        widths_mm, exceeds_range = calibration.pixel_distances_to_widths(
            np.array([200., 277.35, 410.185, 600.]), anchors)
        np.testing.assert_allclose(widths_mm[:3], [0., 0., 56.])
        self.assertGreater(widths_mm[3], 112.)
        np.testing.assert_array_equal(exceeds_range, [False, False, False, True])

    def test_calibration_accepts_both_main_formats_and_checks_monotonicity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'calibration.json'
            data = measurement()
            write_json(path, data)
            self.assertNotIn('verified', calibration.read(path))
            write_json(path, data['mapping'])
            self.assertEqual(calibration.read(path)['open_width_mm'], 100.)
            data['mapping']['anchors'][1]['x'] = 100.
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, '递增'):
                calibration.read(path)

    def test_short_gaps_fill_but_long_gaps_and_out_of_range_stay_invalid(self):
        times = np.arange(100)/100.
        widths = np.full(100, 30.)
        widths[12:17] = np.nan
        widths[30:70] = np.nan
        above = np.zeros(100, bool)
        above[80] = True
        aligned = width_estimation.align(times, widths, above, times[::2], .2)
        self.assertEqual(aligned.source[7], width_estimation.INTERPOLATED)
        self.assertTrue(aligned.valid[7])
        self.assertFalse(aligned.valid[25])
        self.assertTrue(np.isnan(aligned.width_mm[25]))
        self.assertEqual(aligned.source[40], width_estimation.OUT_OF_RANGE)
        self.assertFalse(aligned.valid[40])

    def test_detection_cache_survives_measurement_policy_and_clock_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'width.json'
            write_json(path, measurement())
            times = np.arange(10)/60.
            detected = frame_detection.DetectionSeries(times, np.full(10, 240.), np.full((10, 2, 4, 2), np.nan))
            def process(offset=-.12, *, gap=.2, threshold=3., recording='a'*64):
                return width_estimation.process(root/'raw.insv', calibration.read(path), times[::2], offset, root/'cache', recording,
                                        calibration_sha256=artifact_identity.file_hash(path), max_gap_s=gap, closed_threshold_mm=threshold)
            with patch.object(frame_detection, 'scan_width', return_value=detected) as detect:
                process()
                result = process(-.24)
                measured = result.samples
                np.testing.assert_allclose(measured.times_s, times[::2]-.24, atol=1e-9)
                np.testing.assert_allclose(measured.width_mm, 20.)
                updated = measurement()
                updated['mapping']['anchors'][-1]['mm'] = 120.
                write_json(path, updated)
                result = process(gap=0., threshold=30.)
                measured = result.samples
                np.testing.assert_allclose(measured.width_mm, 24.)
                self.assertEqual(result.report['closed_fraction'], 1.)
                self.assertNotIn('calibration_source', result.report)
                self.assertEqual(detect.call_count, 1)
                process(recording='b'*64)
                self.assertEqual(detect.call_count, 2)
                with patch.object(frame_detection, 'identity', return_value=dict(detector='changed')):
                    process()
                self.assertEqual(detect.call_count, 3)

    def test_detection_cache_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            measured = frame_detection.DetectionSeries(np.array([0., .1]), np.array([240., np.nan]), np.full((2, 2, 4, 2), np.nan))
            with patch.object(frame_detection, 'scan_width', return_value=measured) as detect:
                frame_detection.cached_width(root / 'raw.insv', root, 'a'*64)
                next(root.rglob('detections.npz')).write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, '缓存身份或哈希'):
                    frame_detection.cached_width(root / 'raw.insv', root, 'a'*64)
                self.assertEqual(detect.call_count, 1)

    def test_measurement_rejects_processing_options(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'width.json'
            for field in ('max_gap_s', 'closed_threshold_mm'):
                write_json(path, dict(measurement(), **{field: .2}))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, '--expert-config'):
                    calibration.read(path)

    def test_detector_reads_native_front_half_and_preserves_pts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'markers.mp4'
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            image = np.full((1920, 3840, 3), 255, np.uint8)
            for tag, u in ((0, 200), (1, 500)):
                marker = cv2.aruco.generateImageMarker(dictionary, tag, 80)
                # 显式选择整幅原始镜头范围时，仍按原始坐标检测，不能套用轨迹裁切坐标。
                image[1750:1830, 1920+u:2000+u] = marker[:, :, None]
            with av.open(str(path), 'w') as container:
                stream = container.add_stream('mpeg4', rate=30)
                stream.width, stream.height, stream.pix_fmt = 3840, 1920, 'yuv420p'
                for k in range(3):
                    frame = av.VideoFrame.from_ndarray(image, format='rgb24')
                    frame.pts, frame.time_base = k, Fraction(1, 30)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            detected = frame_detection.scan_width(path, roi=(0, 0, 1920, 1920))
            np.testing.assert_allclose(detected.times_s, np.arange(3)/30., atol=1e-8)
            np.testing.assert_allclose(detected.distance_px, 300., atol=1.)
            self.assertEqual(detected.corners_px.shape, (3, 2, 4, 2))
            np.testing.assert_allclose(detected.corners_px[0].mean(axis=1), [[239.5,1789.5],[539.5,1789.5]], atol=1.)


class WidthKitTests(unittest.TestCase):
    def setUp(self):
        self.f = scene_tests.RegistryVersionTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        registry.register('CAMERA_B', self.f.bundle)
        self.scene = self.f.root / 'scene'
        self.scene.mkdir()
        self.width = kit_inputs.KITS / 'rig/left_width.json'
        write_json(self.width, measurement())
        self.arms = {
            'left': dict(serial='CAMERA_A', calibration=registry.pin('CAMERA_A')['calibration'],
                         gripper=dict(tcp='tcp_camera.json', width='left_width.json')),
            'right': dict(serial='CAMERA_B', calibration=registry.pin('CAMERA_B')['calibration'],
                          gripper=dict(tcp='tcp_camera.json')),
        }

    def test_kit_reads_relative_width_without_registry_registration(self):
        kit = write_kit('rig', self.arms)
        self.assertEqual(kit['arms']['left']['gripper']['width'], str(self.width))
        self.assertEqual(read_yaml(self.width.parent / 'kit.yaml')['arms']['left']['gripper']['width'],
                         'left_width.json')
        manifest = write_batch(self.scene, 'morning', kit='rig')
        batch = batch_inputs.read_batch(manifest)
        scene = replace(batch.scene, devices=batch.devices)
        with patch.object(registry, 'read_insv_serial', return_value='CAMERA_A'):
            selected = camera_assets.select_camera(self.f.root / 'recording.insv', scene)
        self.assertEqual(selected.files['width'], self.width)
        self.assertNotIn('width', camera_identity(selected)['assets'])
        self.assertFalse((registry.CAMERAS / 'CAMERA_A/gripper_width').exists())

    def test_missing_and_invalid_width_sources_are_rejected(self):
        write_kit('rig', self.arms)
        manifest = write_batch(self.scene, 'missing', kit='rig')
        batch = batch_inputs.read_batch(manifest)
        scene = replace(batch.scene, devices=batch.devices)
        with patch.object(registry, 'read_insv_serial', return_value='CAMERA_A'):
            self.width.unlink()
            with self.assertRaises(OSError):
                camera_assets.select_camera(self.f.root / 'recording.insv', scene)
            write_json(self.width, dict(anchors=[]))
            with self.assertRaisesRegex(ValueError, 'anchors'):
                camera_assets.select_camera(self.f.root / 'recording.insv', scene)

    def test_user_width_paths_must_stay_inside_the_kit_or_shared_root(self):
        for filename in ('../../escape.json', str(self.f.root / 'width.json'), '', 'left_width.yaml', None):
            with self.subTest(filename=filename):
                self.arms['left']['gripper']['width'] = filename
                write_yaml(self.width.parent / 'kit.yaml', dict(arms={
                    arm: dict(serial=binding['serial'], intrinsics=dict(front='measured', rear='factory'),
                              gripper=binding['gripper']) for arm, binding in self.arms.items()}))
                with self.assertRaisesRegex(ValueError, 'gripper.width'):
                    kit_inputs.read_kit('rig')
        outside = self.f.root / 'outside.json'
        write_json(outside, measurement())
        (self.width.parent / 'linked.json').symlink_to(outside)
        self.arms['left']['gripper']['width'] = 'linked.json'
        write_yaml(self.width.parent / 'kit.yaml', dict(arms={
                    arm: dict(serial=binding['serial'], intrinsics=dict(front='measured', rear='factory'),
                              gripper=binding['gripper']) for arm, binding in self.arms.items()}))
        with self.assertRaisesRegex(ValueError, '套件目录或共享 kits 目录内'):
            kit_inputs.read_kit('rig')

    def test_changed_width_file_is_read_without_editing_kit_yaml(self):
        first = write_kit('rig', self.arms)
        before = (self.width.parent / 'kit.yaml').read_bytes()
        write_batch(self.scene, 'morning', kit='rig')
        updated = measurement()
        updated['mapping']['anchors'][-1]['mm'] = 120.
        write_json(self.width, updated)
        batch_inputs = batches.read_inputs(self.scene / 'morning', require_mapping=False)
        self.assertEqual(batch_inputs.kit, first)
        self.assertEqual(calibration.read(Path(first['arms']['left']['gripper']['width']))['open_width_mm'], 120.)
        self.assertEqual((self.width.parent / 'kit.yaml').read_bytes(), before)

    def test_install_width_cli_is_removed(self):
        import io
        from contextlib import redirect_stderr

        from umi.apps import calib as cli
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            cli.build_parser().parse_args(['install-width', 'CAMERA_A', str(self.width),
                                          '--gripper', 'test', '--gripper-version', '001'])
        self.assertEqual(error.exception.code, 2)


"""全段快速扫描、按需补检、真实 PTS 与每手有效比例验收。"""


def found_pair():
    corners, _ = projected()
    return detection.FrameDetection(tuple(polygon[None] for polygon in corners), np.array([[0], [1]]),
                                    {tag: corners[tag].mean(axis=0) for tag in (0, 1)})


EMPTY = detection.FrameDetection((), None, {})


class SampledTagTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.request = TagDistanceRequest(self.root / 'raw.insv', 'recording', SIZES, 'sizes', CAMERA, 'camera',
                                          np.arange(4) / 30, -.12, TagPoseQuality(2., 2.))
        self.quick_results = {0: found_pair(), 3: found_pair()}
        self.complete_result = found_pair()
        self.quick_calls, self.recovery_calls = [], []

        def frames(video, times, indices):
            for index in np.sort(indices):
                frame = SimpleNamespace(index=int(index))
                yield int(index), float(times[index]), frame

        def detect(frame_detector, frame):
            index = frame.index
            if frame_detector.detector._fast:
                self.quick_calls.append(index)
                return self.quick_results.get(index, EMPTY)
            self.recovery_calls.append(index)
            return self.complete_result

        self.addCleanup(patch.stopall)
        patch.object(sampled_detection, 'sampled_frames', side_effect=frames).start()
        patch.object(sampled_detection.FrameDetector, 'detect', autospec=True, side_effect=detect).start()

    def process(self, fraction=.5):
        return tag_distance.process(replace(self.request, min_valid_fraction=fraction), self.root)

    def test_scans_every_output_frame_before_accepting_exactly_fifty_percent(self):
        result = self.process()
        self.assertEqual(self.quick_calls, [0, 1, 2, 3])
        self.assertEqual(self.recovery_calls, [])
        self.assertEqual(result.report['valid_fraction'], .5)
        self.assertTrue(result.report['acceptance']['passed'])
        np.testing.assert_array_equal(result.samples.valid, [True, False, False, True])
        np.testing.assert_allclose(result.samples.times_s, self.request.video_times_s - .12)

    def test_recovery_stops_when_required_count_is_reached(self):
        self.quick_results = {0: found_pair()}
        result = self.process()
        self.assertEqual(len(self.quick_calls), 4)
        self.assertEqual(len(self.recovery_calls), 1)
        self.assertEqual(result.report['acceptance']['valid_rows'], 2)
        self.assertEqual(result.report['detection']['recovery_attempted'], 1)

    def test_increasing_threshold_resumes_recovery_without_rescanning_or_repeating_frames(self):
        self.quick_results = {0: found_pair()}
        self.process()
        recovered_before = list(self.recovery_calls)
        result = self.process(.75)
        self.assertEqual(len(self.quick_calls), 4)
        self.assertEqual(len(self.recovery_calls), 2)
        self.assertNotEqual(self.recovery_calls[-1], recovered_before[0])
        self.assertEqual(result.report['acceptance']['valid_rows'], 3)
        self.assertEqual(result.report['acceptance']['minimum_valid_fraction'], .75)
        warmed = self.process(.75)
        np.testing.assert_array_equal(warmed.samples.distance_mm, result.samples.distance_mm)
        self.assertEqual(len(self.recovery_calls), 2)

    def test_below_threshold_fails_and_reuses_exhausted_detection_cache(self):
        self.quick_results = {}
        self.complete_result = EMPTY
        for _ in range(2):
            with self.assertRaises(tag_distance.TagDistanceAcceptanceError) as raised:
                self.process()
            self.assertEqual(raised.exception.report['valid_fraction'], 0.)
            self.assertIn('0/4', str(raised.exception))
            self.assertFalse(raised.exception.report['acceptance']['passed'])
        self.assertEqual(len(self.quick_calls), 4)
        self.assertEqual(len(self.recovery_calls), 4)

    def test_duplicate_ids_cannot_be_rescued_by_another_pass_missing_one_duplicate(self):
        pair = found_pair()
        duplicated = detection.FrameDetection((pair.corners[0], pair.corners[0] + 50, pair.corners[1]),
                                              np.array([[0], [0], [1]]), {1: pair.centres[1]})
        self.quick_results = {index: duplicated for index in range(4)}
        with self.assertRaises(tag_distance.TagDistanceAcceptanceError) as raised:
            self.process()
        self.assertEqual(self.recovery_calls, [])
        self.assertEqual(raised.exception.report['sources'], {'ambiguous_detection': 4})

    def test_detected_pair_with_invalid_geometry_does_not_count_toward_threshold(self):
        corners, _ = projected()
        corners[0, 1:] = corners[0, 0]
        invalid = detection.FrameDetection(tuple(polygon[None] for polygon in corners), np.array([[0], [1]]),
                                           {tag: corners[tag].mean(axis=0) for tag in (0, 1)})
        self.quick_results = {index: invalid for index in range(4)}
        self.complete_result = invalid
        with self.assertRaises(tag_distance.TagDistanceAcceptanceError) as raised:
            self.process()
        self.assertEqual(raised.exception.report['acceptance']['valid_rows'], 0)
        self.assertNotIn('two_tags', raised.exception.report['sources'])

    def test_changed_output_times_have_a_separate_detection_cache(self):
        self.process()
        self.request = replace(self.request, video_times_s=np.arange(4) / 40)
        self.process()
        self.assertEqual(len(self.quick_calls), 8)

    def test_clipped_cache_only_reads_full_quick_scan_on_a_miss(self):
        full_times = self.request.video_times_s
        self.quick_results = {index: found_pair() for index in range(len(full_times))}
        self.request = replace(self.request, video_times_s=full_times[1:], detection_times_s=full_times)
        first = self.process()
        self.assertEqual(self.quick_calls, list(range(len(full_times))))

        read = sampled_detection.SampledDetectionCache._read

        def read_clipped(cache, stage):
            self.assertEqual(len(cache.query_times_s), len(full_times) - 1)
            return read(cache, stage)

        with patch.object(sampled_detection.SampledDetectionCache, '_read', read_clipped):
            reopened = self.process()
        np.testing.assert_array_equal(first.samples.distance_mm, reopened.samples.distance_mm)
        self.assertEqual(first.report['acceptance'], reopened.report['acceptance'])

    def test_changed_roi_has_a_separate_corner_cache_and_json_safe_identity(self):
        self.process()
        self.request = replace(self.request, roi_px=(550, 950, 1400, 1300))
        result = self.process()
        self.assertEqual(len(self.quick_calls), 8)
        self.assertEqual(result.report['inputs']['roi_px'], [550, 950, 1400, 1300])

    def test_tampered_quick_cache_is_rejected(self):
        self.process()
        next((self.root / 'sampled_tags').glob('*/quick.npz')).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, '哈希'):
            self.process()

    def test_threshold_uses_ceiling_for_odd_output_count(self):
        self.request = replace(self.request, video_times_s=np.arange(5) / 30)
        result = self.process()
        self.assertEqual(result.report['acceptance']['valid_rows'], 3)
        self.assertEqual(result.report['valid_fraction'], .6)

    def test_decimal_threshold_at_exact_ratio_does_not_require_an_extra_frame(self):
        self.request = replace(self.request, video_times_s=np.arange(50) / 30, max_missing_frames=50)
        self.quick_results = {index: found_pair() for index in range(7)}
        result = self.process(.14)
        self.assertEqual(result.report['valid_fraction'], .14)
        self.assertEqual(self.recovery_calls, [])

    def test_coverage_pass_still_recovers_an_overlong_gap_and_stops_after_bridging_it(self):
        self.request = replace(self.request, video_times_s=np.arange(8) / 30, max_missing_frames=2)
        self.quick_results = {index: found_pair() for index in (0, 1, 6, 7)}
        result = self.process()
        self.assertEqual(len(self.quick_calls), 8)
        self.assertGreater(len(self.recovery_calls), 0)
        self.assertLess(len(self.recovery_calls), 4)
        self.assertLessEqual(result.report['acceptance']['longest_missing_frames'], 2)

    def test_unrecoverable_tail_fails_even_with_more_than_fifty_percent_valid(self):
        self.request = replace(self.request, video_times_s=np.arange(8) / 30, max_missing_frames=2)
        self.quick_results = {index: found_pair() for index in range(5)}
        self.complete_result = EMPTY
        with self.assertRaisesRegex(tag_distance.TagDistanceAcceptanceError, '连续缺失 3 帧.*上限 2 帧') as raised:
            self.process()
        self.assertEqual(raised.exception.report['valid_fraction'], .625)
        self.assertEqual(raised.exception.report['acceptance']['overlong_gaps'][0]['start_row'], 5)

    def test_stricter_gap_limit_resumes_cached_recovery_without_fast_rescan(self):
        self.request = replace(self.request, video_times_s=np.arange(8) / 30, max_missing_frames=4)
        self.quick_results = {index: found_pair() for index in (0, 1, 6, 7)}
        self.process()
        self.assertEqual(self.recovery_calls, [])
        self.request = replace(self.request, max_missing_frames=2)
        result = self.process()
        self.assertEqual(len(self.quick_calls), 8)
        self.assertGreater(len(self.recovery_calls), 0)
        self.assertTrue(result.report['acceptance']['passed'])

    def test_failed_contiguous_window_stops_without_trying_unrelated_missing_frames(self):
        self.request = replace(self.request, video_times_s=np.arange(64) / 30, max_missing_frames=5)
        self.quick_results = {index: found_pair() for index in range(40)}
        self.complete_result = EMPTY
        with self.assertRaises(tag_distance.TagDistanceAcceptanceError) as raised:
            self.process()
        self.assertEqual(len(self.quick_calls), 64)
        self.assertEqual(len(self.recovery_calls), 6)
        self.assertEqual(raised.exception.report['detection']['stop_reason'], 'unrecoverable_gap')
        self.assertEqual(raised.exception.report['acceptance']['longest_missing_frames'], 24)
        # 已有补检证据已足以证明超限，重跑不再做无关补检。
        with self.assertRaises(tag_distance.TagDistanceAcceptanceError):
            self.process()
        self.assertEqual(len(self.recovery_calls), 6)


class SourceFrameSelectionTests(unittest.TestCase):
    def source(self, pts):
        frames = [SimpleNamespace(pts=value, time_base=Fraction(1, 60000)) for value in pts]
        stream = SimpleNamespace(codec_context=SimpleNamespace(width=3840, height=1920),
                                 time_base=Fraction(1, 60000))
        container = MagicMock()
        container.__enter__.return_value = container
        container.streams.video = [stream]
        container.decode.return_value = iter(frames)
        return container, frames

    def test_nearest_pts_ties_use_earlier_frame_and_never_extrapolate(self):
        container, frames = self.source([0, 1001, 2002])
        period = 1001 / 60000
        queries = np.array([-.001, period / 2, period, 2 * period + .0001])
        with patch('av.open', return_value=container):
            selected = list(sampled_detection.sampled_frames(Path('raw.insv'), queries, np.arange(4)))
        self.assertIsNone(selected[0][2])
        self.assertIs(selected[1][2], frames[0])
        self.assertIs(selected[2][2], frames[1])
        self.assertIsNone(selected[3][2])

    def test_video_gap_does_not_assign_a_distant_source_frame(self):
        container, _ = self.source([0, 1001, 6000])
        queries = np.array([.05])
        with patch('av.open', return_value=container):
            selected = list(sampled_detection.sampled_frames(Path('raw.insv'), queries, np.array([0])))
        self.assertIsNone(selected[0][2])

    def test_duplicate_source_pts_are_rejected(self):
        container, _ = self.source([0, 1001, 1001])
        with patch('av.open', return_value=container), self.assertRaisesRegex(ValueError, '非递增'):
            list(sampled_detection.sampled_frames(Path('raw.insv'), np.array([.1]), np.array([0])))

    def test_late_recovery_seeks_before_target_and_preserves_nearest_source_pts(self):
        container, frames = self.source([600000, 601001, 602002])
        query = np.array([601001 / 60000])
        with patch('av.open', return_value=container):
            selected = list(sampled_detection.sampled_frames(Path('raw.insv'), query, np.array([0])))
        requested = container.seek.call_args.args[0]
        self.assertLess(requested, frames[0].pts)
        self.assertTrue(container.seek.call_args.kwargs['backward'])
        self.assertIs(selected[0][2], frames[1])
        self.assertEqual(selected[0][1], query[0])

    def test_recovery_prioritizes_known_tags_and_spreads_queries(self):
        corners = np.full((64, 2, 4, 2), np.nan)
        corners[32:, 0] = 1.
        order = sampled_detection.recovery_order(np.arange(64), corners)
        self.assertEqual(set(order), set(range(64)))
        self.assertTrue((order[:32] >= 32).all())
        self.assertGreater(int(np.ptp(order[:8])), 20)


class TagDistanceAcceptanceTests(unittest.TestCase):
    def test_default_threshold_and_configuration_bounds(self):
        self.assertEqual(scene_inputs.load_scene_parameters().tag_distance_min_valid_fraction, .5)
        self.assertEqual(scene_inputs.load_scene_parameters().tag_distance_max_missing_frames, 15)
        self.assertEqual(scene_inputs.load_scene_parameters().tag_distance_roi_px, [600, 980, 1350, 1250])
        for value in (0., -1., 1.01, float('nan'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scene_inputs.load_scene_parameters(overrides={'tag_distance_min_valid_fraction': value})
        for value in (-1, 1.5, True):
            with self.subTest(gap=value), self.assertRaises(ValueError):
                scene_inputs.load_scene_parameters(overrides={'tag_distance_max_missing_frames': value})
        self.assertEqual(scene_inputs.load_scene_parameters(overrides={
            'tag_distance_max_missing_frames': 0}).tag_distance_max_missing_frames, 0)
        for roi in ([1, 2, 3], [600., 980, 1350, 1250], [1350, 980, 600, 1250], [0, 0, 2000, 1920]):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                scene_inputs.load_scene_parameters(overrides={'tag_distance_roi_px': roi})

    def test_cli_parser_defers_an_unspecified_roi_to_the_shared_default(self):
        from umi.apps.calib import build_parser
        from umi.perception.gripper import default_tag_roi
        for command in ('preview-width', 'detect-width'):
            parsed = build_parser().parse_args([command, 'raw.insv'])
            self.assertIsNone(parsed.roi)
        self.assertEqual(tuple(scene_inputs.load_scene_parameters().tag_distance_roi_px), default_tag_roi())

    def test_actual_quick_and_recovery_search_do_not_escape_configured_roi(self):
        from test_perception import bright_markers
        for positions, succeeds in [(((0, 700, 1050), (1, 1000, 1050), (0, 100, 100)), True),
                                     (((0, 700, 1050), (1, 1600, 1050)), False)]:
            with self.subTest(positions=positions), tempfile.TemporaryDirectory() as folder:
                half = bright_markers(positions, shape=(1920, 1920))
                pixels = np.column_stack((np.zeros_like(half), half))
                frame = av.VideoFrame.from_ndarray(pixels, format='gray')
                request = TagDistanceRequest(Path('raw.insv'), 'recording', SIZES, 'sizes', CAMERA, 'camera',
                                             np.arange(2) / 30, 0., TagPoseQuality(2., 2.))

                def frames(video, times, indices):
                    return iter((int(index), float(times[index]), frame) for index in np.sort(indices))

                with patch.object(sampled_detection, 'sampled_frames', side_effect=frames):
                    if succeeds:
                        result = tag_distance.process(request, Path(folder))
                        self.assertEqual(result.report['valid_fraction'], 1.)
                        self.assertTrue((result.samples.distance_mm > 20).all())
                    else:
                        with self.assertRaises(tag_distance.TagDistanceAcceptanceError) as raised:
                            tag_distance.process(request, Path(folder))
                        self.assertEqual(raised.exception.report['acceptance']['valid_rows'], 0)

    def test_gap_boundaries_include_leading_trailing_and_all_missing_frames(self):
        for valid, expected in [([False, False, True, False], [[0, 2], [3, 4]]),
                                ([False, False], [[0, 2]]), ([True, True], [])]:
            with self.subTest(valid=valid):
                np.testing.assert_array_equal(tag_distance.missing_runs(np.array(valid)),
                                              np.array(expected, dtype=int).reshape(-1, 2))

    def test_exactly_fifteen_missing_frames_pass_and_sixteen_fail_at_high_coverage(self):
        request = TagDistanceRequest(Path('raw.insv'), 'recording', SIZES, 'sizes', CAMERA, 'camera',
                                     np.arange(40) / 30, 0., TagPoseQuality(2., 2.))
        for count, expected in [(15, True), (16, False)]:
            valid = np.ones(40, dtype=bool)
            valid[:count] = False
            report = tag_distance.acceptance_report(valid, request)
            self.assertEqual(report['passed'], expected)
            self.assertEqual(report['longest_missing_frames'], count)

    def test_each_arm_reports_measurement_quality_without_blocking_accepted_tracks(self):
        import test_scene

        from umi.scene import run as workflow
        fixture = test_scene.SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build_scene()
        workflow.run_episode(fixture.episode_path)
        output = fixture.task_root / 'results'
        previous = output.resolve()
        for video, camera in list(fixture.cameras.items()):
            path = camera.files['front'].parent / 'tags.json'
            write_json(path, SIZES)
            fixture.cameras[video] = replace(camera, files=dict(camera.files, tag_distance=path),
                                            tag_distance_measurement=SIZES)
        report = dict(valid_fraction=.4, rows=10, sources={'two_tags': 4, 'missing': 6},
                      acceptance=dict(minimum_valid_fraction=.5, valid_rows=4, passed=False,
                                      maximum_missing_frames=15, longest_missing_frames=6,
                                      longest_missing_interval_s=[0., .2]))
        calls = []

        def measure(request, cache, **kwargs):
            calls.append(request)
            self.assertFalse(kwargs['strict'])
            times = request.video_times_s + request.episode_offset_s
            if request.video == fixture.recording_paths['right']:
                return TagDistanceResult(TagDistanceSeries(times, np.full(len(times), np.nan),
                                                           np.full(len(times), 'missing')), report)
            return TagDistanceResult(TagDistanceSeries(times, np.full(len(times), 50.),
                np.full(len(times), 'two_tags')), dict(valid_fraction=1., acceptance=dict(
                    passed=True, longest_missing_frames=0, maximum_missing_frames=15)))

        with patch.object(tag_distance, 'process', side_effect=measure), \
                patch('umi.perception.decoding.VideoDecode.run', return_value=[]):
            result = workflow.run_episode(fixture.episode_path)
        self.assertEqual(result['status'], 'ok')
        self.assertNotEqual(output.resolve(), previous)
        self.assertEqual(len(calls), 2)
        episode = json.loads((output / 'data/episode.json').read_text())
        self.assertTrue(episode['tag_distance']['left']['acceptance']['passed'])
        self.assertFalse(episode['tag_distance']['right']['acceptance']['passed'])


# 真实编码/解码验证宽度预览：标记叠加、原始像素单位、缺失与时间轴。


def write_preview_recording(path: Path, *, size=(3840, 1920)) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    with av.open(str(path), 'w', format='mp4') as container:
        stream = container.add_stream('mpeg4', rate=30)
        stream.width, stream.height, stream.pix_fmt = *size, 'yuv420p'
        for index, pts in enumerate((0, 1, 3)):
            pixels = np.full((size[1], size[0], 3), 255, np.uint8)
            if size == (3840, 1920):
                tags = [(0, 200)] if index == 1 else [(0, 200), (1, 500)]
                if index == 2:
                    tags.append((0, 700))
                for tag, x in tags:
                    marker = cv2.aruco.generateImageMarker(dictionary, tag, 80)
                    pixels[800:880, 1920+x:2000+x] = marker[:, :, None]
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            frame.pts, frame.time_base = pts, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class WidthPreviewTests(unittest.TestCase):
    def test_contrast_only_changes_display_roi_and_keeps_source_image(self):
        front = np.full((1920, 1920, 3), 20, np.uint8)
        original = front.copy()
        missing = FrameDetection((), None, {})
        with patch.object(cv2, 'putText', wraps=cv2.putText) as labels:
            rendered = width_video.annotate(front, missing, timestamp_s=0., frame_index=0,
                                        roi=(400, 800, 1000, 1200), contrast=2)
        np.testing.assert_array_equal(rendered[450, 300], [40, 40, 40])
        np.testing.assert_array_equal(rendered[750, 750], [20, 20, 20])
        np.testing.assert_array_equal(front, original)
        self.assertTrue(any('contrast x2' in call.args[1] for call in labels.call_args_list))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'width.insv'

    def test_cli_creates_annotated_video_without_measurement_or_registry(self):
        write_preview_recording(self.source)
        original = self.source.read_bytes()
        with patch.object(cv2, 'putText', wraps=cv2.putText) as labels, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(calib_cli.main(['preview-width', str(self.source), '--roi', '0', '0', '1920', '1920']), 0)
        summary = json.loads(output.getvalue())
        path = self.source.with_name('width_width_preview.mp4')
        self.assertEqual(summary['output'], str(path))
        self.assertEqual(summary['frames'], 3)
        self.assertEqual(summary['paired_frames'], 1)
        texts = [call.args[1] for call in labels.call_args_list]
        self.assertIn('ID 0', texts)
        self.assertIn('ID 1', texts)
        self.assertTrue(any('300.' in label and 'native 1920x1920' in label for label in texts))
        self.assertEqual(sum('Distance: N/A' in label for label in texts), 2)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            self.assertEqual((stream.width, stream.height), (960, 960))
            self.assertEqual(stream.codec_context.name, 'h264')
            frames = list(container.decode(stream))
            self.assertEqual(len(frames), 3)
            times = [float(frame.pts * frame.time_base) for frame in frames]
            np.testing.assert_allclose(times, [0., 1/30., .1], atol=1e-5)
            pixels = frames[0].to_ndarray(format='bgr24')
            self.assertTrue(((pixels[390:445, 95:290, 1] > 160)
                             & (pixels[390:445, 95:290, 0] < 100)).any())
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual({path.name for path in self.root.iterdir()}, {'width.insv', 'width_width_preview.mp4'})

    def test_preview_and_extraction_agree_on_bright_unambiguous_tags(self):
        write_preview_recording(self.source)
        measured = frame_detection.scan_width(self.source, roi=(0, 0, 1920, 1920))
        with patch.object(width_video, 'annotate', wraps=width_video.annotate) as annotation:
            width_video.preview(self.source, roi=(0, 0, 1920, 1920))
        displayed = [call.args[1].distance_px for call in annotation.call_args_list]
        np.testing.assert_allclose(displayed, measured.distance_px, equal_nan=True)

    def test_cli_fixed_roi_crops_before_detection_and_reports_native_units(self):
        write_preview_recording(self.source)
        with redirect_stdout(io.StringIO()) as output:
            status = calib_cli.main(['preview-width', str(self.source), '--roi', '100', '700', '650', '1000'])
        self.assertEqual(status, 0)
        summary = json.loads(output.getvalue())
        self.assertEqual(summary['roi'], [100, 700, 650, 1000])
        self.assertEqual(summary['paired_frames'], 2)
        self.assertEqual(summary['distance_units'], 'native_pixels')

    def test_cli_rejects_invalid_roi_without_output(self):
        write_preview_recording(self.source)
        self.assertEqual(calib_cli.main(['preview-width', str(self.source), '--roi', '600', '100', '500', '1000']), 2)
        self.assertFalse(self.source.with_name('width_width_preview.mp4').exists())

    def test_custom_output_and_existing_file_protection(self):
        write_preview_recording(self.source)
        target = self.root / 'nested/preview.mp4'
        width_video.preview(self.source, target)
        before = target.read_bytes()
        with self.assertRaisesRegex(FileExistsError, '已存在'):
            width_video.preview(self.source, target)
        self.assertEqual(target.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'MP4'):
            width_video.preview(self.source, self.root / 'preview.json')

    def test_failed_detection_does_not_publish_partial_video(self):
        write_preview_recording(self.source)
        target = self.root / 'preview.mp4'
        with patch.object(MarkerDetector, 'detect', side_effect=RuntimeError('detector failed')):
            with self.assertRaisesRegex(RuntimeError, 'detector failed'):
                width_video.preview(self.source, target)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.iterdir()), [self.source])

    def test_invalid_recording_geometry_is_rejected(self):
        write_preview_recording(self.source, size=(320, 240))
        with self.assertRaisesRegex(ValueError, '3840'):
            width_video.preview(self.source)
        self.assertFalse(self.source.with_name('width_width_preview.mp4').exists())

    def test_cli_reports_decoder_errors_without_publishing_output(self):
        self.source.write_bytes(b'not a video')
        self.assertEqual(calib_cli.main(['preview-width', str(self.source)]), 2)
        self.assertFalse(self.source.with_name('width_width_preview.mp4').exists())


# 无界面采样、固定范围、原始坐标、原速节拍与 CSV 原子输出。


ROI = (100, 700, 900, 1000)


class WidthMonitorTests(unittest.TestCase):
    def test_cli_contrast_reaches_detection_and_saved_video(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as diagnostics, \
             patch.object(cv2, 'putText', wraps=cv2.putText) as labels:
            status = calib_cli.main(['detect-width', str(self.video), '--roi', *map(str, ROI),
                               '--contrast', '2', '--benchmark'])
        self.assertEqual(status, 0)
        summary = json.loads(diagnostics.getvalue())
        self.assertEqual(summary['contrast'], 2.)
        self.assertEqual(summary['paired_samples'], 1)
        self.assertTrue(Path(summary['video_output']).is_file())
        self.assertTrue(any('contrast x2' in call.args[1] for call in labels.call_args_list))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.video = self.root / 'width.insv'
        write_preview_recording(self.video)

    def test_sampling_native_centres_and_missing_values_without_video_output(self):
        original = self.video.read_bytes()
        original_threads = cv2.getNumThreads()
        output = self.root / 'measurements.csv'
        summary = width_video.monitor(self.video, ROI, output=output, benchmark=True)
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual((summary['source_frames'], summary['samples'], summary['paired_samples']), (3, 3, 1))
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(float(rows[0]['distance_px']), 300., delta=.3)
        self.assertAlmostEqual(float(rows[0]['id0_x_px']), 239.5, delta=.3)
        self.assertAlmostEqual(float(rows[0]['id0_y_px']), 839.5, delta=.3)
        self.assertEqual(rows[1]['id1_x_px'], '')
        self.assertEqual(rows[1]['distance_px'], '')
        self.assertEqual(rows[2]['id0_x_px'], '')
        self.assertEqual(rows[2]['distance_px'], '')
        self.assertEqual(self.video.read_bytes(), original)
        self.assertEqual(cv2.getNumThreads(), original_threads)
        self.assertEqual({path.name for path in self.root.iterdir()}, {'width.insv', 'measurements.csv'})

    def test_lower_sampling_frequency_uses_video_timestamps(self):
        with redirect_stdout(io.StringIO()) as output:
            summary = width_video.monitor(self.video, ROI, hz=15, benchmark=True)
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(summary['samples'], 2)
        np.testing.assert_allclose([float(row['t_video_s']) for row in rows], [0., .1], atol=1e-6)

    def test_realtime_waits_for_video_clock_without_sleeping_in_test(self):
        now = [0.]
        def advance(seconds):
            self.assertGreater(seconds, 0.)
            now[0] += seconds
        with patch.object(width_video.time, 'perf_counter', side_effect=lambda: now[0]), \
             patch.object(width_video.time, 'sleep', side_effect=advance) as sleep, redirect_stdout(io.StringIO()):
            summary = width_video.monitor(self.video, ROI)
        self.assertEqual((summary['samples'], summary['late_dropped']), (3, 0))
        self.assertEqual(summary['mode'], 'realtime')
        self.assertGreaterEqual(sleep.call_count, 1)
        self.assertAlmostEqual(now[0], .1, delta=1e-6)

    def test_late_samples_are_dropped_instead_of_accumulating(self):
        now = [0.]
        detect = MarkerDetector.detect
        def slow_detect(detector, gray):
            now[0] += .1
            return detect(detector, gray)
        with patch.object(width_video.time, 'perf_counter', side_effect=lambda: now[0]), \
             patch.object(MarkerDetector, 'detect', slow_detect), redirect_stdout(io.StringIO()):
            summary = width_video.monitor(self.video, ROI, video_output=self.root / 'late.mp4')
        self.assertEqual(summary['late_dropped'], 1)
        self.assertEqual(summary['samples'], 2)
        with av.open(str(self.root / 'late.mp4')) as video:
            timestamps = [float(frame.pts * frame.time_base) for frame in video.decode(video=0)]
        np.testing.assert_allclose(timestamps, [0., .1], atol=1e-5)

    def test_yuv_and_rgb_crops_match_full_gray_for_odd_bounds(self):
        converter = av.video.reformatter.VideoReformatter()
        with av.open(str(self.video)) as source:
            frame = next(source.decode(video=0))
            rgb = av.VideoFrame.from_ndarray(frame.to_ndarray(format='rgb24'), format='rgb24')
            for current in (frame, rgb):
                roi = (101, 701, 649, 999)
                expected = current.to_ndarray(format='gray')[701:999, 2021:2569]
                actual = crop_gray(current, roi, converter)
                np.testing.assert_array_equal(actual, expected)

    def test_invalid_requests_existing_output_and_failed_detection_are_safe(self):
        existing = self.root / 'existing.csv'
        existing.write_text('keep')
        with self.assertRaises(FileExistsError):
            width_video.monitor(self.video, ROI, output=existing)
        self.assertEqual(existing.read_text(), 'keep')
        for hz in (0., -1., float('nan'), float('inf')):
            with self.subTest(hz=hz), self.assertRaises(ValueError):
                width_video.monitor(self.video, ROI, hz=hz)
        with self.assertRaises(ValueError):
            width_video.monitor(self.video, (-1, 0, 600, 1000))
        target = self.root / 'failed.csv'
        original_threads = cv2.getNumThreads()
        with patch.object(MarkerDetector, 'detect', side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                width_video.monitor(self.video, ROI, output=target, benchmark=True)
        self.assertFalse(target.exists())
        self.assertEqual(cv2.getNumThreads(), original_threads)
        self.assertEqual({path.name for path in self.root.iterdir()}, {'width.insv', 'existing.csv'})

    def test_cli_streams_csv_and_sends_summary_to_stderr(self):
        with redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as diagnostics, \
             patch.object(cv2, 'putText', wraps=cv2.putText) as labels:
            status = calib_cli.main(['detect-width', str(self.video), '--roi', *map(str, ROI), '--benchmark'])
        self.assertEqual(status, 0)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(output.getvalue())))), 3)
        summary = json.loads(diagnostics.getvalue())
        self.assertEqual(summary['samples'], 3)
        self.assertEqual(summary['output'], 'stdout')
        path = self.root / 'width_width_detect.mp4'
        self.assertEqual(summary['video_output'], str(path))
        self.assertEqual(summary['video_frames'], 3)
        texts = [call.args[1] for call in labels.call_args_list]
        self.assertEqual(texts.count('ROI'), 3)
        self.assertTrue(any('300.' in label and 'native 1920x1920' in label for label in texts))
        self.assertEqual(sum('Distance: N/A' in label for label in texts), 2)
        with av.open(str(path)) as video:
            stream = video.streams.video[0]
            self.assertEqual(stream.codec_context.name, 'h264')
            self.assertEqual((stream.width, stream.height), (960, 960))
            frames = list(video.decode(stream))
            np.testing.assert_allclose([float(frame.pts * frame.time_base) for frame in frames], [0., 1/30, .1], atol=1e-5)
            pixels = frames[0].to_ndarray(format='bgr24')
            roi_border = pixels[348:353, 100:200]
            self.assertTrue(((roi_border[:, :, 0] > 150) & (roi_border[:, :, 2] < 70)).any())

    def test_no_video_keeps_csv_only_behavior(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as diagnostics:
            status = calib_cli.main(['detect-width', str(self.video), '--roi', *map(str, ROI), '--benchmark', '--no-video'])
        self.assertEqual(status, 0)
        self.assertIsNone(json.loads(diagnostics.getvalue())['video_output'])
        self.assertEqual(list(self.root.iterdir()), [self.video])

    def test_encoding_failure_cleans_csv_and_video_and_preserves_existing_video(self):
        csv_output, video_output = self.root / 'result.csv', self.root / 'result.mp4'
        original_threads = cv2.getNumThreads()
        with patch.object(width_video.PreviewWriter, 'write', side_effect=RuntimeError('encode failed')):
            with self.assertRaisesRegex(RuntimeError, 'encode failed'):
                width_video.monitor(self.video, ROI, output=csv_output, video_output=video_output, benchmark=True)
        self.assertEqual(list(self.root.iterdir()), [self.video])
        self.assertEqual(cv2.getNumThreads(), original_threads)
        video_output.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            width_video.monitor(self.video, ROI, output=csv_output, video_output=video_output, benchmark=True)
        self.assertEqual(video_output.read_bytes(), b'keep')
        self.assertFalse(csv_output.exists())
