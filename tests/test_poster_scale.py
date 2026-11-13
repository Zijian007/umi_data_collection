"""建图海报尺度：独立尺度恢复、精确固定尺度惯性复核和发布身份。"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from software import config
from software.scene.management import inputs
from software.scene.spatial import geometry, mapping, poster_scale
from software.vision import tag_board
from software.storage import write_yaml


def scale_fixture():
    from test_poster_fusion import fixture
    data = fixture()
    raw = data[10].copy()
    raw[:,:3,3] /= 2.3
    obs = geometry.VisualObservations(data[1], raw, np.arange(len(raw)), 'visual_map')
    rear_pose = np.linalg.inv(data[4]['rear'])
    rear = dict(data[9], p=rear_pose[:3,3].tolist(), q_xyzw=Rotation.from_matrix(rear_pose[:3,:3]).as_quat().tolist())
    front = dict(rear, p=[0.,0.,0.], q_xyzw=[0.,0.,0.,1.], cam_time_offset_ns=0)
    parameters = replace(data[6], tag_mapping_enabled=True)
    return obs, front, rear, data[7], data[8], 2.1, parameters


class PosterScaleNumericalTests(unittest.TestCase):
    def test_scale_recovery_partial_ids_offset_and_metric_lever(self):
        args = scale_fixture()
        result = poster_scale.estimate_scale(*args)
        self.assertEqual(result['status'], 'estimated')
        self.assertAlmostEqual(result['scale'], 2.3, delta=1e-5)
        self.assertGreater(result['relative_scale_std'], 0.)
        self.assertLess(result['reprojection_rms_px'], 1e-5)
        # Changing arbitrary visual-map origin cannot change scale or uncertainty.
        moved = args[0].T_reference_camera.copy(); moved[:,:3,3] += [300.,-400.,200.]
        again = poster_scale.estimate_scale(replace(args[0], T_reference_camera=moved), *args[1:])
        self.assertAlmostEqual(again['scale'], result['scale'], delta=1e-5)

    def test_no_tags_no_motion_and_map_gaps_are_unknown(self):
        args = list(scale_fixture())
        args[3] = replace(args[3], usable=np.zeros(len(args[3].usable), bool))
        self.assertEqual(poster_scale.estimate_scale(*args)['status'], 'insufficient_observations')
        args = list(scale_fixture())
        static = args[0].T_reference_camera.copy(); static[:,:3,3] = 0.
        args[0] = replace(args[0], T_reference_camera=static)
        self.assertEqual(poster_scale.estimate_scale(*args)['status'], 'insufficient_geometry')
        args = list(scale_fixture())
        # Every rear timestamp is between map frames; nonadjacent IDs prohibit interpolation.
        args[0] = replace(args[0], indices=args[0].indices*2)
        self.assertEqual(poster_scale.estimate_scale(*args)['status'], 'insufficient_observations')

    def test_moving_poster_rejects_static_scale_model(self):
        args = list(scale_fixture())
        corners = args[3].corners_half_px.copy()
        corners[len(corners)//2:] += [0., 70.]
        args[3] = replace(args[3], corners_half_px=corners)
        with self.assertRaises(poster_scale.PosterScaleConflict):
            poster_scale.estimate_scale(*args)

    def test_log_weighted_combination_floors_and_both_conflict_limits(self):
        p = inputs.load_scene_parameters()
        report = poster_scale.combine_scales(2., .01, 2.05, .02, p)
        self.assertTrue(report['compatible'])
        variance = np.array([.01**2+.03**2, .02**2+.02**2])
        weights = 1/variance; weights /= weights.sum()
        self.assertAlmostEqual(report['scale'], np.exp(weights@np.log([2.,2.05])))
        self.assertGreater(report['relative_scale_std'], 0.)
        self.assertFalse(poster_scale.combine_scales(2., 1., 3., 1., p)['compatible'])
        strict = replace(p, tag_mapping_imu_std_floor=.001, tag_mapping_tag_std_floor=.001)
        self.assertFalse(poster_scale.combine_scales(2., 0., 2.05, 0., strict)['compatible'])

    def test_fixed_scale_refits_gravity_and_bias_and_rejects_inertial_conflict(self):
        from test_scene import motion
        times, poses, imu = motion()
        raw = poses.copy(); raw[:,:3,3] /= 2.3
        obs = geometry.VisualObservations(times, raw, np.arange(len(times)), 'visual_map')
        p = inputs.load_scene_parameters()
        baseline = geometry.metric_alignment(obs, np.eye(4), imu, np.zeros(3), 9.81, p)
        refit = geometry.metric_alignment(obs, np.eye(4), imu, np.zeros(3), 9.81, p, fixed_scale=2.3)
        self.assertEqual(refit['scale'], 2.3)
        self.assertLess(refit['evidence']['rms_m'], .001)
        np.testing.assert_allclose(refit['T_scene_metric_map'], baseline['T_scene_metric_map'], atol=.002)
        with self.assertRaises(ValueError):
            geometry.metric_alignment(obs, np.eye(4), imu, np.zeros(3), 9.81, p, fixed_scale=4.6)

    def test_diagnostics_preserve_placement_and_conflicts_block_correction(self):
        args = scale_fixture()
        observations, front, rear, detections, board, _, parameters = args
        scene = SimpleNamespace(parameters=parameters, trajectory=config.load('trajectory'))
        recording = SimpleNamespace(camera=SimpleNamespace(measurements=dict(front=front,rear=rear)))
        placement = dict(scale=2.3, T_scene_metric_map=np.eye(4).tolist(), evidence=dict(relative_scale_std=.02))
        actual = poster_scale.correct_placement(placement, observations, 'front', recording, scene, (board,detections))
        self.assertEqual(actual['scale'], placement['scale'])
        self.assertEqual(actual['evidence']['environment_poster']['status'], 'diagnostics_only')
        self.assertNotIn('environment_poster', placement['evidence'])
        conflict = copy.deepcopy(placement); conflict['scale'] = 3.5
        diagnosed = poster_scale.correct_placement(conflict, observations, 'front', recording, scene, (board,detections))
        self.assertEqual(diagnosed['evidence']['environment_poster']['status'], 'conflict')
        scene.parameters = replace(parameters, tag_mapping_scale_enabled=True)
        with self.assertRaises(geometry.MapAlignmentError):
            poster_scale.correct_placement(conflict, observations, 'front', recording, scene, (board,detections))


class PosterScaleIdentityTests(unittest.TestCase):
    def test_mapping_identity_tracks_geometry_and_scale_policy_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'scene.yaml'
            board_path = Path(temporary)/'board.json'
            board_path.write_bytes(tag_board.DEFAULT_LAYOUT.read_bytes())
            write_yaml(path, dict(schema_version=3,tags=dict(layout=str(board_path))))
            scene = inputs.read_scene(path)
            before = mapping.definition(scene)
            scene = replace(scene, parameters=replace(scene.parameters, tag_fusion_enabled=True))
            self.assertEqual(mapping.definition(scene), before)
            scene = replace(scene, parameters=replace(scene.parameters, tag_mapping_enabled=True))
            enabled = mapping.definition(scene)
            self.assertNotEqual(enabled, before)
            changed = replace(scene, parameters=replace(scene.parameters, tag_mapping_scale_enabled=True))
            self.assertNotEqual(mapping.definition(changed), enabled)
            doc = json.loads(board_path.read_text())
            for marker in doc['markers']:
                marker['corners_m'] = (np.array(marker['corners_m'])*.99).tolist()
            board_path.write_text(json.dumps(doc))
            self.assertNotEqual(mapping.definition(scene), enabled)

    def test_scale_requires_diagnostics_and_layout(self):
        with self.assertRaisesRegex(ValueError, 'tag_mapping_enabled'):
            inputs.load_scene_parameters(overrides=dict(tag_mapping_scale_enabled=True))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'scene.yaml'
            write_yaml(path, dict(schema_version=3, parameters=dict(tag_mapping_enabled=True)))
            with self.assertRaisesRegex(ValueError, 'tags.layout'):
                inputs.read_scene(path)


class PosterMappingWorkflowTests(unittest.TestCase):
    def helper(self):
        import test_scene
        helper = test_scene.SceneWorkflowTests('runTest')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        return helper

    def configure(self, helper, *, correction):
        declaration = inputs.read_yaml(helper.scene_path)
        declaration.update(tags=dict(layout=str(tag_board.DEFAULT_LAYOUT)), parameters=dict(tag_mapping_enabled=True, tag_mapping_scale_enabled=correction))
        write_yaml(helper.scene_path, declaration)
        helper.scene = inputs.read_batch(helper.session_path).scene
        for video, camera in list(helper.cameras.items()):
            helper.cameras[video] = replace(camera, measurements=dict(camera.measurements, rear=camera.measurements['front']),
                                            files=dict(camera.files, rear=camera.files['front']))

    def detections(self, helper, *, empty=False, shifted=False):
        from software.vision import poster_pose, tags
        from test_scene import motion
        times, truth, _ = motion()
        frame_indices = np.arange(0,len(times),3)
        samples = np.repeat(np.arange(len(frame_indices)),3)
        ids = np.tile([12,48,88],len(frame_indices))
        board = tag_board.load()
        board_pose = np.eye(4); board_pose[:3,3] = [2.5,-2.7,3.5]
        points = np.array([board.corners_m[i] for i in ids])
        model = poster_pose.camera_model(helper.cameras[helper.recording_paths['left']].measurements['rear'])
        pixels, _ = poster_pose.project_observations(board_pose, truth[frame_indices[samples]], points, model)
        if shifted:
            pixels[len(pixels)//2:] += [80., 0.]
        sides = np.linalg.norm(pixels-np.roll(pixels,-1,axis=1),axis=2).min(axis=1)
        return tags.TagObservations(frame_indices, times[frame_indices], samples, ids, pixels+240., sides,
            np.ones(len(ids),bool), np.full(len(ids),not empty,bool), np.zeros(len(frame_indices),int), np.zeros(len(frame_indices),int))

    def fake_cache(self, detections):
        from software.storage import write_json
        def cached(video, output, settings, **kwargs):
            output.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output/'observations.npz', **{name:getattr(detections,name) for name in detections.__dataclass_fields__})
            write_json(output/'detection.json',dict(fixture=True))
            write_json(output/'report.json',dict(fixture=True,layout=kwargs['layout'].record()))
        return cached

    def test_enabled_correction_publishes_refit_and_frozen_observations(self):
        from software.vision import tags
        helper = self.helper()
        self.configure(helper, correction=True)
        with patch.object(tags,'cached',side_effect=self.fake_cache(self.detections(helper))):
            built = helper.build_scene()
        root = helper.scene.map_root/'versions'/built['version']
        metadata = json.loads((root/'manifest.json').read_text())
        placement = metadata['maps']['shared_front']['placement']
        report = placement['evidence']['environment_poster']
        self.assertEqual(report['status'],'corrected')
        self.assertTrue(report['applied'])
        self.assertAlmostEqual(placement['scale'],2.3,delta=.005)
        self.assertEqual(placement['scale'],report['combination']['scale'])
        self.assertEqual(placement['evidence']['relative_scale_std'],report['combination']['relative_scale_std'])
        self.assertIn('devices/environment_poster.json',metadata['assets'])
        self.assertIn('environment_tags/left/observations.npz',metadata['assets'])
        inputs.validate_assets(root,metadata['assets'])
        with patch.object(tags,'cached',side_effect=AssertionError('must reuse unchanged map')):
            self.assertEqual(helper.build_scene()['status'],'skipped')

    def test_empty_observations_preserve_imu_scale_and_conflict_does_not_replace_map(self):
        from software.vision import tags
        helper = self.helper()
        baseline = helper.build_scene()
        root = helper.scene.map_root
        old = json.loads((root/'versions'/baseline['version']/'manifest.json').read_text())['maps']['shared_front']['placement']
        self.configure(helper, correction=True)
        with patch.object(tags,'cached',side_effect=self.fake_cache(self.detections(helper,empty=True))):
            empty = helper.build_scene()
        saved = json.loads((root/'versions'/empty['version']/'manifest.json').read_text())['maps']['shared_front']['placement']
        self.assertEqual(saved['scale'],old['scale'])
        self.assertEqual(saved['T_scene_metric_map'],old['T_scene_metric_map'])
        self.assertEqual(saved['evidence']['environment_poster']['status'],'insufficient_observations')
        pointer = helper.session_path.read_bytes()
        with patch.object(tags,'cached',side_effect=self.fake_cache(self.detections(helper,shifted=True))):
            with self.assertRaisesRegex(ValueError,'海报'):
                helper.build_scene(force=True)
        self.assertEqual(helper.session_path.read_bytes(),pointer)
