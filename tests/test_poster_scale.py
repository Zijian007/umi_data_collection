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

from software.trajectory.algorithms.fixed_map.parameters import MapPlacementParameters, PosterScaleParameters
from software.configuration.loader import load
from software.configuration import loader as config
from software.scene.parameters import SceneOverrides
from software.infrastructure.file_store import read_yaml
from software.scene.management.inputs import batch as batch_inputs, scene as scene_inputs
from software.scene import spatial as mapping
from software.scene.spatial import correct_poster_placement
from software.trajectory.algorithms.fixed_map import poster
from software.trajectory.algorithms.fixed_map import geometry
from software.vision.poster import layout as tag_board, detection as environment_tags
tags = environment_tags
from software.infrastructure.file_store import write_yaml
from software.scene import result_io as asset_identity


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
        result = poster.estimate_scale(*args[:-1], args[-1].for_algorithm(PosterScaleParameters))
        self.assertEqual(result['status'], 'estimated')
        self.assertAlmostEqual(result['scale'], 2.3, delta=1e-5)
        self.assertGreater(result['relative_scale_std'], 0.)
        self.assertLess(result['reprojection_rms_px'], 1e-5)
        # Changing arbitrary visual-map origin cannot change scale or uncertainty.
        moved = args[0].T_reference_camera.copy(); moved[:,:3,3] += [300.,-400.,200.]
        again = poster.estimate_scale(replace(args[0], T_reference_camera=moved), *args[1:-1],
                                          args[-1].for_algorithm(PosterScaleParameters))
        self.assertAlmostEqual(again['scale'], result['scale'], delta=1e-5)

    def test_no_tags_no_motion_and_map_gaps_are_unknown(self):
        args = list(scale_fixture())
        args[3] = replace(args[3], usable=np.zeros(len(args[3].usable), bool))
        self.assertEqual(poster.estimate_scale(*args[:-1], args[-1].for_algorithm(PosterScaleParameters))['status'], 'insufficient_observations')
        args = list(scale_fixture())
        static = args[0].T_reference_camera.copy(); static[:,:3,3] = 0.
        args[0] = replace(args[0], T_reference_camera=static)
        self.assertEqual(poster.estimate_scale(*args[:-1], args[-1].for_algorithm(PosterScaleParameters))['status'], 'insufficient_geometry')
        args = list(scale_fixture())
        # Every rear timestamp is between map frames; nonadjacent IDs prohibit interpolation.
        args[0] = replace(args[0], indices=args[0].indices*2)
        self.assertEqual(poster.estimate_scale(*args[:-1], args[-1].for_algorithm(PosterScaleParameters))['status'], 'insufficient_observations')

    def test_moving_poster_rejects_static_scale_model(self):
        args = list(scale_fixture())
        corners = args[3].corners_half_px.copy()
        corners[len(corners)//2:] += [0., 70.]
        args[3] = replace(args[3], corners_half_px=corners)
        with self.assertRaises(poster.PosterScaleConflict):
            poster.estimate_scale(*args[:-1], args[-1].for_algorithm(PosterScaleParameters))

    def test_scale_estimation_culls_an_isolated_bad_tag_sighting(self):
        args = list(scale_fixture())
        detections = args[3]
        corners = detections.corners_half_px.copy()
        corners[0] += [20., 0.]
        args[3] = replace(detections, corners_half_px=corners)
        result = poster.estimate_scale(*args[:-1], args[-1].for_algorithm(PosterScaleParameters))
        self.assertEqual(result['status'], 'estimated')
        self.assertGreaterEqual(result['culled_observations'], 1)
        self.assertGreater(result['pre_cull_reprojection_rms_px'],
                           result['reprojection_rms_px'])

    def test_log_weighted_combination_floors_and_both_conflict_limits(self):
        p = scene_inputs.load_scene_parameters()
        report = poster.combine_scales(2., .01, 2.05, .02, p.for_algorithm(PosterScaleParameters))
        self.assertTrue(report['compatible'])
        variance = np.array([.01**2+.03**2, .02**2+.02**2])
        weights = 1/variance; weights /= weights.sum()
        self.assertAlmostEqual(report['scale'], np.exp(weights@np.log([2.,2.05])))
        self.assertGreater(report['relative_scale_std'], 0.)
        self.assertFalse(poster.combine_scales(2., 1., 3., 1., p.for_algorithm(PosterScaleParameters))['compatible'])
        strict = replace(p, tag_mapping_imu_std_floor=.001, tag_mapping_tag_std_floor=.001)
        self.assertFalse(poster.combine_scales(2., 0., 2.05, 0., strict.for_algorithm(PosterScaleParameters))['compatible'])

    def test_fixed_scale_refits_gravity_and_bias_and_rejects_inertial_conflict(self):
        from test_scene import motion
        times, poses, imu = motion()
        raw = poses.copy(); raw[:,:3,3] /= 2.3
        obs = geometry.VisualObservations(times, raw, np.arange(len(times)), 'visual_map')
        p = scene_inputs.load_scene_parameters()
        baseline = geometry.metric_alignment(obs, np.eye(4), imu, np.zeros(3), 9.81, p.for_algorithm(MapPlacementParameters), fusion_parameters=load('trajectory').values.fusion)
        refit = geometry.metric_alignment(obs, np.eye(4), imu, np.zeros(3), 9.81, p.for_algorithm(MapPlacementParameters), fusion_parameters=load('trajectory').values.fusion, fixed_scale=2.3)
        self.assertEqual(refit['scale'], 2.3)
        self.assertLess(refit['evidence']['rms_m'], .001)
        np.testing.assert_allclose(refit['T_scene_metric_map'], baseline['T_scene_metric_map'], atol=.002)
        with self.assertRaises(ValueError):
            geometry.metric_alignment(obs, np.eye(4), imu, np.zeros(3), 9.81, p.for_algorithm(MapPlacementParameters), fusion_parameters=load('trajectory').values.fusion, fixed_scale=4.6)

    def test_diagnostics_preserve_placement_and_conflicts_block_correction(self):
        args = scale_fixture()
        observations, front, rear, detections, board, _, parameters = args
        scene = SimpleNamespace(parameters=parameters, trajectory=config.load('trajectory'))
        recording = SimpleNamespace(camera=SimpleNamespace(measurements=dict(front=front,rear=rear)))
        placement = dict(scale=2.3, T_scene_metric_map=np.eye(4).tolist(), evidence=dict(relative_scale_std=.02))
        actual = correct_poster_placement(placement, observations, 'front', recording, scene, (board,detections))
        self.assertEqual(actual['scale'], placement['scale'])
        self.assertEqual(actual['evidence']['environment_poster']['status'], 'diagnostics_only')
        self.assertNotIn('environment_poster', placement['evidence'])
        conflict = copy.deepcopy(placement); conflict['scale'] = 3.5
        diagnosed = correct_poster_placement(conflict, observations, 'front', recording, scene, (board,detections))
        self.assertEqual(diagnosed['evidence']['environment_poster']['status'], 'conflict')
        scene.parameters = replace(parameters, tag_mapping_scale_enabled=True)
        with self.assertRaises(geometry.MapAlignmentError):
            correct_poster_placement(conflict, observations, 'front', recording, scene, (board,detections))


class PosterScaleIdentityTests(unittest.TestCase):
    def test_mapping_identity_tracks_geometry_and_scale_policy_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'batch/collection.yaml'
            board_path = Path(temporary)/'board.json'
            board_path.write_bytes(tag_board.DEFAULT_LAYOUT.read_bytes())
            write_yaml(path, dict(kit='rig', recording=dict(arms=['left', 'right']),
                                 scene=dict(description=''),
                                 map=dict(layout='shared', reference_arm='left', selection=dict(policy='latest_prior')),
                                 environment_poster=dict(layout_path=str(board_path), detection=dict(enabled=True)),
                                 reconstruction=dict(backend='scene_map')))
            scene = scene_inputs.read_scene(path)
            before = mapping.definition(scene)
            scene = replace(scene, parameters=replace(scene.parameters, tag_fusion_enabled=True))
            self.assertNotEqual(mapping.definition(scene), before)
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
            scene_inputs.load_scene_parameters(overrides=dict(tag_mapping_scale_enabled=True))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'batch/collection.yaml'
            write_yaml(path, dict(kit='rig', recording=dict(arms=['left']),
                                 scene=dict(description='', parameters=dict(tag_mapping_enabled=True)),
                                 map=dict(layout='shared', reference_arm='left', selection=dict(policy='latest_prior')),
                                 reconstruction=dict(backend='scene_map')))
            with self.assertRaisesRegex(ValueError, '不承载算法参数'):
                scene_inputs.read_scene(path)


class PosterMappingWorkflowTests(unittest.TestCase):
    def helper(self):
        import test_scene
        helper = test_scene.SceneWorkflowTests('runTest')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        return helper

    def configure(self, helper, *, correction):
        declaration_path = helper.batch_root / 'collection.yaml'
        declaration = read_yaml(declaration_path)
        declaration['environment_poster'] = dict(
            layout_path=str(tag_board.DEFAULT_LAYOUT), detection=dict(enabled=True),
            task_constraint=dict(enabled=False, method='staged'),
            mapping=dict(enabled=True, scale_correction=dict(enabled=correction)))
        write_yaml(declaration_path, declaration)
        expert = helper.root / 'expert.yaml'
        write_yaml(expert, dict(parameters=dict(mapping=dict(environment_tags=dict(minimum_frames=6)))))
        helper.scene_overrides = SceneOverrides.load(expert)
        helper.scene = batch_inputs.read_batch(helper.session_path, overrides=helper.scene_overrides).scene
        for video, camera in list(helper.cameras.items()):
            helper.cameras[video] = replace(camera, measurements=dict(camera.measurements, rear=camera.measurements['front']),
                                            files=dict(camera.files, rear=camera.files['front']))

    def detections(self, helper, *, empty=False, shifted=False):
        from software.vision.poster import geometry as poster_pose, detection as environment_tags
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
        return environment_tags.TagObservations(frame_indices, times[frame_indices], samples, ids, pixels+240., sides,
            np.ones(len(ids),bool), np.full(len(ids),not empty,bool), np.zeros(len(frame_indices),int), np.zeros(len(frame_indices),int))

    def fake_cache(self, detections):
        from software.infrastructure.file_store import write_json
        def cached(video, output, settings, **kwargs):
            output.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output/'observations.npz', **{name:getattr(detections,name) for name in detections.__dataclass_fields__})
            write_json(output/'detection.json',dict(fixture=True))
            write_json(output/'report.json',dict(fixture=True,layout=kwargs['layout'].record()))
        return cached

    def test_enabled_correction_publishes_refit_and_frozen_observations(self):
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
        asset_identity.validate_assets(root,metadata['assets'])
        with patch.object(tags,'cached',side_effect=AssertionError('must reuse unchanged map')):
            self.assertEqual(helper.build_scene()['status'],'skipped')

    def test_empty_observations_preserve_imu_scale_and_conflict_does_not_replace_map(self):
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
