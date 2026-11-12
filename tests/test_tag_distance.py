from software import provenance as artifact_identity
from software.gripper_width import calibration as gripper_calibration
"""真实 KB4 投影下的中心间距、拒绝条件、尺寸配置、缓存和场景输出。"""
from software.scene.management import camera_assets
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from software import registry
from software.gripper_width import detection, tag_distance
from software.gripper_width import sampled_detection
from software.gripper_width.tag_geometry import TagPoseEstimator
from software.gripper_width.tag_models import TagDistanceRequest, TagDistanceResult, TagDistanceSeries, TagPoseQuality
from software.scene.management import inputs
from software.storage import write_json
from software.viewer.readers import read_tag_distance
from software.viewer.errors import ViewerDataError


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
        import test_gripper_width as fixtures
        from registry_fakes import write_kit, write_batch
        f=fixtures.WidthKitTests();f.setUp();self.addCleanup(f.doCleanups)
        path=f.width.parent/'tags.json';write_json(path,SIZES)
        f.arms['left']['gripper']['tag_distance']='tags.json'
        write_kit('rig',f.arms)
        batch=inputs.read_batch(write_batch(f.scene,'tags',kit='rig'))
        scene=replace(batch.scene,devices=batch.devices)
        with patch.object(registry,'read_insv_serial',return_value='CAMERA_A'):
            camera=camera_assets.select_camera(f.f.root/'raw.insv',scene)
            frozen=camera_assets.snapshot_cameras({'left':camera},f.f.root/'snapshot')['left']
            before=camera_assets.camera_identity(camera)
            write_json(path,dict(tag_sizes_mm={'0':8.,'1':9.}))
            changed=camera_assets.select_camera(f.f.root/'raw.insv',scene)
        self.assertEqual(before,camera_assets.camera_identity(changed))
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

    def test_kit_paths_stay_inside_kit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            result=inputs.gripper_paths(dict(tcp='tcp.json',tag_distance='tags.json'), root)
            self.assertEqual(result['tag_distance'],str(root/'tags.json'))
            for name in ('../tags.json','/tmp/tags.json','tags.yaml'):
                with self.assertRaises(ValueError):
                    inputs.gripper_paths(dict(tcp='tcp.json',tag_distance=name), root)


class DistancePipelineTests(unittest.TestCase):
    def test_bimanual_viewer_crops_primary_distance_on_same_clock_as_trajectory(self):
        from software.viewer import episodes
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
        with patch.object(episodes,'read_episode',return_value=(dict(frame_id='scene/v1',scene=scene),paths)):
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
                    patch.object(sampled_detection,'detect_frame',return_value=found):
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
    def test_scene_saves_distances_without_width_and_retains_geometry_on_export(self):
        import test_scene as fixtures
        from software.scene import workflow
        from software.scene.management import episodes
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
        with patch.object(tag_distance,'process',side_effect=measure) as process:
            workflow.run_episode(f.episode_path)
            self.assertEqual(process.call_count,2)
            self.assertEqual(workflow.run_episode(f.episode_path)['status'],'skipped')
        output=f.episode_path.parent/'results'
        data=json.loads((output/'episode.json').read_text())
        self.assertEqual(set(data['tag_distance']),{'left','right'})
        self.assertNotIn('gripper_width',data)
        for arm in ('left','right'):
            path=output/arm/'trajectory.csv'
            with path.open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['tag_distance_mm'],'50.000000')
            self.assertFalse({'width_mm','close_01','closed','valid','source'} & rows[0].keys())
            self.assertNotIn('tag_distance',data['arm_results'][arm])
            self.assertEqual(rows[2]['tag_distance_valid'],'0')
            self.assertEqual(rows[2]['tag_distance_mm'],'')
            times=[float(row['t_episode_s']) for row in rows]
            self.assertIsNone(read_tag_distance(path,times)['distance_mm'][2])
        episodes.review(f.episode_path,accept=True,reviewer='test')
        export=f.root/'export';episodes.export([f.episode_path],export)
        copies=list(export.rglob('trajectory.csv'))
        self.assertEqual(len(copies),2)
        self.assertTrue(all('tag_distance_mm' in path.read_text().splitlines()[0] for path in copies))
