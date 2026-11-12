"""Factory rear geometry, immutable run assets, and both selection paths."""
from dataclasses import replace
import json
from unittest.mock import patch

import gtsam
import numpy as np

from software import config, provenance, registry
from software.calibration.algorithms import factory_kb4
from software.camera import calibration_source, rear_fallback
from software.camera.geometry import imu_from_camera
from software.camera.x3 import factory
from software.scene.management import camera_assets
from software.scene.models import Scene
from software.storage import write_json
from software.trajectory.calibration import assets
from software.trajectory.models import TrajectoryRequest
from software.trajectory.pipeline import calibration_assets, reuse
from software.vision import tag_board
from test_intrinsics_source import native
from test_trajectory_workflow import RecordingFixture
import test_trajectory_workflow as workflow_tests


class RearFallbackTests(RecordingFixture):
    @classmethod
    def setUpClass(cls):
        cls.forward = factory_kb4.fit_forward(factory.lens_model(native()['lenses'][0], 1440),
                                             config.load('calibration').values.factory)

    def setUp(self):
        super().setUp()
        self.native = native()
        # Nonzero translation on both sides catches rotating index0's translation.
        self.native['lenses'][0].update(yaw=.581, pitch=.714, roll=90.712, tx=.002, ty=-.001, tz=.003)
        self.native['lenses'][1].update(yaw=-.518, pitch=.410, roll=90.417, tx=-.001542, ty=-.00001, tz=-.026248)
        for override in (patch.object(factory, 'read_factory', return_value=self.native),
                         patch.object(factory_kb4, 'fit_forward', return_value=self.forward)):
            override.start(); self.addCleanup(override.stop)
        self.request = config.resolve_trajectory_request(TrajectoryRequest(tag_layout=tag_board.DEFAULT_LAYOUT))

    def derive(self):
        return rear_fallback.derive(self.video, front_path=self.selected.assets.camera_json,
                                   serial='camera', front_source='measured')

    def test_geometry_matches_specialist_and_composes_front_b_and_clock(self):
        # Independent GTSAM equations from specialist.record.offset_v3.
        body_from_lens = []
        for lens in self.native['lenses']:
            rotation = gtsam.Rot3.Ypr(*np.deg2rad([lens['yaw'], lens['pitch'], lens['roll']]))
            translation = -rotation.rotate(np.array([lens['tx'], lens['ty'], lens['tz']]))
            if lens['index'] == 0:
                rotation = gtsam.Rot3.Rz(np.pi).compose(rotation)
            body_from_lens.append(gtsam.Pose3(rotation, translation))
        expected_relative = body_from_lens[1].inverse().compose(body_from_lens[0]).matrix()
        np.testing.assert_allclose(factory.working_from_environment(self.native), expected_relative, atol=1e-14)
        front = assets.read_basalt(self.selected.assets.camera_json)
        derived = self.derive()
        np.testing.assert_allclose(imu_from_camera(derived.measurement),
                                   imu_from_camera(front) @ expected_relative, atol=1e-14)
        self.assertEqual(derived.measurement['cam_time_offset_ns'], front['cam_time_offset_ns'])
        self.assertNotEqual(derived.measurement['fx'], front['fx'])
        self.assertEqual(derived.measurement['provenance']['actual_error'], 'not_measured')
        self.assertEqual(json.loads(derived.serialized)['validation']['verdict'], 'NOT_EVALUATED')

    def test_independent_preflight_is_read_only_freezes_and_checks_hashes(self):
        before = set(self.root.rglob('*'))
        selected = calibration_assets.select_environment_assets(self.video, self.selected, self.request)
        self.assertEqual(set(self.root.rglob('*')), before)
        self.assertIsNone(selected.rear.path)
        self.assertTrue(selected.rear.identity.version.startswith('factory_fallback_'))
        manifest = calibration_assets.freeze_rear_calibration(selected.rear, self.output/'devices/rear')
        measured = calibration_assets.load_frozen_rear(manifest)
        self.assertEqual(measured, selected.rear.factory.measurement)
        self.assertEqual(provenance.file_hash(manifest.parent/'calibration.json'), selected.rear.identity.sha256)
        with self.assertRaisesRegex(ValueError, '通过验收'):
            calibration_source.read_rear(manifest.parent/'calibration.json', serial='camera')
        (manifest.parent/'calibration.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, '哈希'):
            calibration_assets.load_frozen_rear(manifest)

    def test_formal_binding_is_authoritative_even_when_broken(self):
        broken = self.root/'broken.json'; write_json(broken, {})
        with patch.object(registry, 'rear_path', return_value=broken), \
             patch.object(rear_fallback, 'derive', side_effect=AssertionError('must not fall back')):
            with self.assertRaisesRegex(ValueError, 'camera_serial'):
                calibration_assets.select_environment_assets(self.video, self.selected,
                    replace(self.request, rear_calibration='explicit'))

    def test_fallback_result_reuse_verifies_the_frozen_snapshot(self):
        request = config.resolve_trajectory_request(TrajectoryRequest(mode='dual'))
        with patch('software.vision.capabilities.require_forward_kb4'):
            selected = calibration_assets.select_environment_assets(self.video, self.selected, request)
        fingerprint = reuse.run_fingerprint(self.video, self.selected, request, environment_assets=selected)
        self.successful_result()
        manifest = calibration_assets.freeze_rear_calibration(selected.rear, self.output/'devices/rear')
        write_json(self.output/'summary.json', dict(status='ok', run_fingerprint=fingerprint))
        self.assertIsNotNone(reuse.reuse_decision(self.output, fingerprint, notes=request.configuration.values.notes).summary)
        (manifest.parent/'calibration.json').write_text('{}')
        self.assertIsNone(reuse.reuse_decision(self.output, fingerprint, notes=request.configuration.values.notes).summary)

    def test_source_identity_and_front_changes_affect_fallback_hash(self):
        first = self.derive()
        document = json.loads(self.selected.assets.camera_json.read_text())
        document['value0']['cam_time_offset_ns'] += 1000000
        write_json(self.selected.assets.camera_json, document)
        self.assertNotEqual(first.sha256, self.derive().sha256)
        self.native['serial'] = 'another-camera'
        with self.assertRaisesRegex(ValueError, '序列号'):
            self.derive()
        with self.assertRaisesRegex(ValueError, '样例'):
            rear_fallback.derive(self.video, front_path=self.selected.assets.camera_json,
                                 serial='camera', front_source='sample')

    def test_invalid_factory_metadata_does_not_remove_previous_result(self):
        from software.trajectory import workflow
        self.successful_result()
        before = (self.output/'summary.json').read_bytes()
        with patch.object(factory, 'read_factory', side_effect=ValueError('unsupported v6')), \
             patch('software.trajectory.calibration.selection.select_calibration', return_value=self.selected):
            result = workflow.run_recording(self.video, self.output.parent, replace(self.request, force=True))
        self.assertEqual(result['status'], 'failed')
        self.assertIn('unsupported v6', result['reason'])
        self.assertEqual(before, (self.output/'summary.json').read_bytes())

    def test_shared_scene_and_dual_share_fallback_and_frozen_asset_identity(self):
        from software.scene.management import inputs
        front = assets.read_basalt(self.selected.assets.camera_json)
        record = dict(serial='camera', status='active', json=str(self.selected.assets.camera_json),
            dir=str(self.selected.directory), source='registry', cam_time_offset_ns=front['cam_time_offset_ns'],
            measurement=front, intrinsics_source='measured')
        scene = Scene(self.root/'scene.yaml', 'fallback', 'shared_front', 'left',
            {'left': self.video}, {}, config.load('trajectory'), inputs.load_scene_parameters(None),
            {'camera': dict(calibration='front-version')}, tag_layout=tag_board.DEFAULT_LAYOUT)
        with patch.object(registry, 'read_insv_serial', return_value='camera'), \
             patch.object(registry, 'get', return_value=record), \
             patch('software.vision.capabilities.require_forward_kb4') as capability:
            shared = camera_assets.select_camera(self.video, scene)
            capability.assert_not_called()
            dual = camera_assets.select_camera(self.video, replace(scene, mode='dual'))
            capability.assert_called_once()
            self.assertEqual(shared.asset_hashes['rear'], dual.asset_hashes['rear'])
            frozen = camera_assets.snapshot_cameras({'left': shared}, self.output)['left']
            self.assertIsNone(frozen.factory_rear)
            self.assertEqual(shared.asset_hashes, frozen.asset_hashes)
            self.assertEqual(json.loads(frozen.files['rear'].read_text())['kind'], rear_fallback.KIND)
            broken = self.root/'broken.json'; write_json(broken, {})
            with self.assertRaisesRegex(ValueError, 'camera_serial'):
                camera_assets.select_camera(self.video, replace(scene, rear_calibrations={'camera': broken}))

    def test_dual_workflow_passes_factory_snapshot_to_fusion(self):
        helper = workflow_tests.WorkflowTests('test_success_skip_and_reprocess_after_calibration_change')
        helper.setUp(); self.addCleanup(helper.doCleanups)
        with patch('software.vision.capabilities.require_forward_kb4'), \
             patch.object(factory, 'read_factory', return_value=self.native):
            result = helper.run_recording(TrajectoryRequest(mode='dual'))
        self.assertEqual(result['status'], 'ok', result)
        self.assertTrue(result['rear_calibration']['version'].startswith('factory_fallback_'))
        measured = calibration_assets.load_frozen_rear(helper.output/'devices/rear/manifest.json')
        self.assertEqual(measured['calibration_kind'], rear_fallback.KIND)
        from software.trajectory.reporting.metadata import collect_run_settings
        settings = collect_run_settings(helper.output, {}, 'dual', helper.video.name,
            calib=dict(json=str(helper.selected.assets.camera_json),
                       rear_calibration_manifest=str(helper.output/'devices/rear/manifest.json')))
        self.assertEqual(settings['frame_geometry']['rear_lens']['calibration_source'], 'insv_factory_fallback')
