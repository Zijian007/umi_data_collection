"""场景批次、设备套件和任务配置的合成与传递。"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import test_scene as scene_tests
import test_scene_runs as scene_run_tests
from collection_defaults_fakes import defaults_document, patched_defaults
from registry_fakes import write_batch, write_kit

from umi.apps import calib as calibration_cli
from umi.apps import scene as scene_cli
from umi.apps.viewer import server
from umi.config import loader as config
from umi.config import reconstruction as reconstruction_config
from umi.devices import registry
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import read_yaml, write_json, write_yaml
from umi.reconstruction.backends import scene_map as scene_reconstruction
from umi.reconstruction import mapping
from umi.reconstruction.estimation.fixed_map.fusion import SceneFusionQualityError
from umi.reconstruction.estimation.models import BackendResult
from umi.scene import routing as estimation
from umi.scene import run as processing
from umi.scene.collection import batch as batch_inputs
from umi.scene.collection import batch as batches
from umi.scene.collection import batch as episode_inputs
from umi.scene.collection import batch as processing_batch
from umi.scene.collection import configuration as kit_inputs_configuration
from umi.scene.collection import configuration as scene_inputs
from umi.scene.collection import kit as kit_inputs
from umi.scene.collection import kit as scene_inputs_kit
from umi.scene.collection.batch import read_episode
from umi.scene.collection.config_report import effective_report
from umi.scene.collection.configuration import SceneOverrides, apply_overrides, read_synchronization
from umi.scene.maps import binding as estimation_binding
from umi.scene.maps import binding as map_lifecycle_binding
from umi.scene.maps.binding import requested_map_version
from umi.scene.maps.epochs import MapEpochError
from umi.store import layout
from umi.store.models import TaskInputs
from umi.store.result_io import result_paths

# 场景元数据与专家算法配置的边界。


class SceneConfigurationTests(unittest.TestCase):
    def test_old_shared_orb_switch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '未知字段'):
            reconstruction_config.split(dict(poster={'front_map_pixel_fusion': True}), label='expert')

    def test_frozen_gripper_paths_follow_hardware_migration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kits = root / 'hardware/kits'
            frozen = dict(tcp=str(root / 'kits/tcp_camera.json'),
                          tag_distance=str(root / 'kits/kit_001/tag_distance.json'),
                          mask=str(root / 'kits/rig/hand.png'),
                          width=str(root / 'custom/width.json'))
            with patch.object(scene_inputs_kit, 'PROJECT_ROOT', root), patch.object(scene_inputs_kit, 'KITS', kits):
                current = scene_inputs_kit.gripper_paths(frozen)
            self.assertEqual(current, dict(tcp=str(kits / 'shared/tcp_camera.json'),
                                           tag_distance=str(kits / 'shared/tag_distance.json'),
                                           mask=str(kits / 'rig/hand.png'), width=frozen['width']))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'shelf'
        self.path = self.root / 'seed/collection.yaml'
        self.path.parent.mkdir(parents=True)

    def write_collection(self, **fields):
        write_yaml(self.path, dict(kit='rig', arms=['left', 'right'], map=dict(layout='shared'),
                                   backend=dict(mode='scene_map'), **fields))

    def test_collection_rejects_inline_algorithm_settings_and_file_pointers(self):
        for fields, message in (
            (dict(trajectory=dict(video=dict(size_px=960))), '未知字段.*trajectory'),
            (dict(parameters=dict(minimum_localization_fraction=.7)), '未知字段.*parameters'),
            (dict(devices={}), '未知字段.*devices'),
            (dict(trajectory_config='other.yaml'), '未知字段.*trajectory_config'),
        ):
            with self.subTest(fields=fields):
                self.write_collection(**fields)
                with self.assertRaisesRegex(ValueError, message):
                    scene_inputs.read_scene(self.path)

    def test_collection_schema_version_is_rejected_without_rewriting(self):
        self.write_collection(schema_version=3)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, '未知字段.*schema_version'):
            scene_inputs.read_scene(self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_build_requires_a_batch_instead_of_a_standalone_scene(self):
        self.write_collection()
        with self.assertRaisesRegex(ValueError, '建图需要采集批次'):
            map_lifecycle_binding.build_scene(self.path)

    def test_poster_position_support_thresholds_are_configurable_and_positive(self):
        parameters = scene_inputs.load_scene_parameters(overrides=dict(shared=dict(poster=dict(acceptance=dict(
            maximum_position_std_m=.02, maximum_rotation_std_deg=2.)))))
        self.assertEqual(parameters.tag_maximum_position_std_m, .02)
        self.assertEqual(parameters.tag_maximum_rotation_std_deg, 2.)
        for value in (0., -1., float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scene_inputs.load_scene_parameters(overrides=dict(shared=dict(poster=dict(acceptance=dict(
                    maximum_position_std_m=value)))))

    def test_poster_culling_and_rear_refinement_share_validated_configuration(self):
        parameters = scene_inputs.load_scene_parameters(overrides=dict(shared=dict(poster=dict(
            culling=dict(maximum_fraction=.1, minimum_retained_time_fraction=.9),
            rear_rotation_refinement=dict(enabled=False, prior_sigma_rad=.05, maximum_correction_deg=2.)))))
        self.assertEqual(parameters.tag_maximum_cull_fraction, .1)
        self.assertEqual(parameters.tag_minimum_retained_time_fraction, .9)
        self.assertFalse(parameters.tag_rear_rotation_refinement_enabled)
        self.assertEqual(parameters.tag_rear_rotation_prior_sigma_rad, .05)
        self.assertEqual(parameters.tag_rear_rotation_maximum_correction_deg, 2.)
        self.assertEqual(scene_inputs.load_scene_parameters(overrides=dict(tag_maximum_cull_fraction=0.)).
                         tag_maximum_cull_fraction, 0.)
        from umi.config import loader as config
        invalid = [dict(tag_maximum_cull_fraction=1.), dict(tag_maximum_cull_fraction=-.1),
            dict(tag_minimum_retained_time_fraction=1.1), dict(tag_minimum_retained_time_fraction=0.),
            dict(tag_rear_rotation_refinement_enabled=1), dict(tag_rear_rotation_prior_sigma_rad=0.),
            dict(tag_rear_rotation_prior_sigma_rad=float('nan')), dict(tag_rear_rotation_maximum_correction_deg=181.)]
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    scene_inputs.load_scene_parameters(overrides=values)
                with self.assertRaises(ValueError):
                    config.load('trajectory', inline=dict(environment_tags=values))

    def test_width_policy_defaults_overrides_and_validation(self):
        defaults = scene_inputs.load_scene_parameters()
        self.assertEqual(defaults.gripper_width_max_gap_s, .2)
        self.assertEqual(defaults.gripper_width_closed_threshold_mm, 3.)
        policy = dict(gripper_width_max_gap_s=0., gripper_width_closed_threshold_mm=0.)
        self.assertEqual(scene_inputs.load_scene_parameters(overrides=policy).gripper_width_max_gap_s, 0.)
        for invalid in (dict(gripper_width_max_gap_s=-1), dict(gripper_width_closed_threshold_mm=True),
                        dict(gripper_width_max_gap_s=float('nan'))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                scene_inputs.load_scene_parameters(overrides=invalid)

    def test_poster_backend_quality_parameters_are_independent_and_round_trip(self):
        from dataclasses import asdict

        from umi.config import reconstruction
        parameters = scene_inputs.load_scene_parameters(overrides=dict(backends=dict(scene_map=dict(
            mapping=dict(placement=dict(maximum_rotation_deg=1.)), localization=dict(maximum_visual_gap_s=.1)))))
        self.assertEqual(parameters.poster_maximum_gravity_difference_deg, 5.)
        self.assertEqual(parameters.poster_maximum_position_support_gap_s, 2.)
        self.assertEqual(parameters.poster_independent_imu_white_noise_multiplier, 1.)
        self.assertEqual(parameters.poster_independent_node_stride, 0)
        self.assertEqual(parameters.poster_independent_keyframe_maximum_dt_s, 0.)
        self.assertFalse(parameters.poster_independent_guidance)
        fusion = dict(node_stride=2, imu_white_noise_multiplier=3., poster_keyframe_maximum_dt_s=.1)
        configured = scene_inputs.load_scene_parameters(overrides=dict(backends=dict(poster_independent=dict(
            maximum_gravity_difference_deg=2., maximum_position_support_gap_s=.5, poster_guidance=True, fusion=fusion))))
        self.assertEqual(configured.poster_maximum_gravity_difference_deg, 2.)
        self.assertEqual(configured.poster_vio_maximum_position_support_gap_s, 2.)
        self.assertEqual(configured.maximum_visual_gap_s, 2.)
        tree = reconstruction.public_tree(scene=asdict(configured))
        self.assertEqual(tree['backends']['poster_independent'], dict(
            maximum_gravity_difference_deg=2., maximum_position_support_gap_s=.5, poster_guidance=True,
            front_map=dict(pixel_weights=False), fusion=fusion))
        decoded = reconstruction.split(tree, label='snapshot').scene
        self.assertEqual(decoded['poster_maximum_gravity_difference_deg'], 2.)
        self.assertEqual(decoded['poster_maximum_position_support_gap_s'], .5)
        self.assertEqual(decoded['poster_independent_node_stride'], 2)
        self.assertEqual(decoded['poster_independent_imu_white_noise_multiplier'], 3.)
        self.assertEqual(decoded['poster_independent_keyframe_maximum_dt_s'], .1)
        self.assertIs(decoded['poster_independent_guidance'], True)
        for invalid in (dict(node_stride=-1), dict(node_stride=1.5), dict(imu_white_noise_multiplier=0.),
                        dict(poster_keyframe_maximum_dt_s=-.1)):
            with self.subTest(fusion=invalid), self.assertRaises(ValueError):
                scene_inputs.load_scene_parameters(overrides=dict(backends=dict(poster_independent=dict(fusion=invalid))))
        for value in (0., -1., float('nan'), True, 181.):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scene_inputs.load_scene_parameters(overrides=dict(backends=dict(poster_independent=dict(
                    maximum_gravity_difference_deg=value))))

    def test_collection_yaml_is_single_user_scene_and_task_declaration(self):
        batch = self.root / '20261113'
        batch.mkdir()
        collection = batch / 'collection.yaml'
        write_yaml(collection, dict(kit='rig', arms=['left'], description='demo',
                                    map=dict(layout='shared'), backend=dict(mode='scene_map'),
                                    tasks=dict(take_001=dict(instruction='pick', arms=['left']))))
        scene = scene_inputs.read_scene(collection)
        self.assertEqual(scene.name, self.root.name)
        self.assertEqual(scene.mode, 'shared_front')
        self.assertEqual(layout.task_configuration(batch / '.pipeline/tasks/take_001'),
                         dict(instruction='pick', expected_arms=['left']))

    def test_collection_yaml_rejects_split_batch_and_task_overrides(self):
        batch = self.root / '20261113'
        batch.mkdir()
        write_yaml(batch / 'collection.yaml', dict(kit='rig', arms=['left'], backend=dict(mode='scene_map')))
        write_yaml(batch / 'batch.yaml', dict(kit='rig'))
        with self.assertRaisesRegex(ValueError, 'batch.yaml 已移除'):
            from umi.scene.collection import configuration as batches_configuration
            batches_configuration.read_collection(batch)
        (batch / 'batch.yaml').unlink()
        task = batch / 'take_001'
        task.mkdir()
        write_yaml(task / 'task.yaml', dict(instruction='old'))
        with self.assertRaisesRegex(ValueError, 'task.yaml 已移除'):
            layout.task_configuration(task)

    def test_synchronization_accepts_user_facing_shorthand(self):
        audio = read_synchronization('audio_auto', ('left', 'right'))
        self.assertEqual(audio['method'], 'audio_auto')
        self.assertEqual(audio['offsets_s'], {'left': 0., 'right': 0.})
        simultaneous = read_synchronization('assumed_simultaneous', ('left', 'right'))
        self.assertEqual(simultaneous['method'], 'assumed_simultaneous')

    def test_collection_inherits_map_defaults_and_resolves_explicit_layouts(self):
        defaults = defaults_document()
        defaults['map'] = dict(layout='shared', reference_arm='left')
        declaration = dict(kit='rig', arms=['left', 'right'], backend=dict(mode='scene_map'))
        with patched_defaults(self.root, defaults):
            write_yaml(self.path, declaration)
            inherited = scene_inputs.read_scene(self.path)
            self.assertEqual((inherited.name, inherited.mode, inherited.reference_arm), ('shelf', 'shared_front', 'left'))
            write_yaml(self.path, dict(declaration, description='', map=defaults['map']))
            self.assertEqual(scene_inputs.read_scene(self.path), inherited)
            for layout_name, mode in (('auto', 'auto'), ('per_arm', 'dual')):
                with self.subTest(layout=layout_name):
                    write_yaml(self.path, dict(declaration, map=dict(layout=layout_name, reference_arm='right')))
                    scene = scene_inputs.read_scene(self.path)
                    self.assertEqual((scene.mode, scene.reference_arm), (mode, 'right'))

    def test_auto_layout_handles_epoch_directory_shapes(self):
        from umi.scene.collection import batch as batches
        batch = self.root / '20261115'
        (batch / 'mapping/epoch_001').mkdir(parents=True)
        (batch / 'mapping/epoch_001/map.insv').write_bytes(b'')
        (batch / 'mapping/epoch_002').mkdir()
        (batch / 'mapping/epoch_002/map.insv').write_bytes(b'')
        self.assertEqual(batches._infer_mode(batch, {}), 'shared_front')
        dual_batch = self.root / '20261116'
        for epoch in ('epoch_001', 'epoch_002'):
            for arm in ('left', 'right'):
                path = dual_batch / 'mapping' / epoch / arm
                path.mkdir(parents=True, exist_ok=True)
                (path / 'map.insv').write_bytes(b'')
        self.assertEqual(batches._infer_mode(dual_batch, {}), 'dual')

    def test_environment_tag_policy_uses_canonical_enabled_fields(self):
        board = scene_inputs.ROOT / 'hardware/environment_poster/environment_poster.json'
        if not board.is_file():
            self.skipTest('environment poster fixture unavailable')
        collection = self.root / '20261118' / 'collection.yaml'
        collection.parent.mkdir()
        write_yaml(collection, dict(kit='rig', arms=['left', 'right'],
                                    map=dict(layout='shared', poster_scale='diagnose'),
                                    poster=dict(layout=str(board), detection=True),
                                    backend=dict(mode='scene_map', scene_map=dict(poster_constraint='joint'))))
        from umi.config.collection_schema import apply_task_backend
        scene = apply_task_backend(scene_inputs.read_scene(collection))
        self.assertTrue(scene.environment_tag_detection)
        self.assertEqual(scene.parameters.tag_fusion_method, 'joint')
        self.assertEqual(scene.environment_tag_policy,
                         dict(enabled=True,
                              mapping=dict(enabled=True, scale=dict(enabled=False)),
                              fusion=dict(enabled=True, method='joint')))

    def test_environment_tag_enabled_fields_reject_inconsistent_combinations(self):
        board = scene_inputs.ROOT / 'hardware/environment_poster/environment_poster.json'
        if not board.is_file():
            self.skipTest('environment poster fixture unavailable')
        from umi.config.collection_schema import apply_task_backend, normalize_collection

        document = dict(kit='rig', arms=['left', 'right'], poster=dict(layout=str(board), detection=False),
                        backend=dict(mode='scene_map', scene_map=dict(poster_constraint='staged')))
        write_yaml(self.path, document)
        declaration = normalize_collection(document)
        scene = scene_inputs.read_scene(self.path, collection=declaration)
        with self.assertRaisesRegex(ValueError, '冲突'):
            apply_task_backend(scene, declaration['disabled_poster_switches'])


# 套件选择、批次声明和设备快照的契约测试。


class KitTests(unittest.TestCase):
    def setUp(self):
        self.fixture = scene_tests.RegistryVersionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        registry.register('CAMERA_B', self.fixture.bundle)
        self.scene_path = self.scene('lab')

    def scene(self, name):
        root = self.root / 'scenes' / name
        root.mkdir(parents=True, exist_ok=True)
        return root

    def test_left_right_and_bimanual_kits_use_one_current_config(self):
        left = write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A'}).items()})
        right = write_kit('right_rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'right': 'CAMERA_B'}).items()})
        both = write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A', 'right': 'CAMERA_B'}).items()})
        self.assertEqual(set(right['arms']), {'right'})
        self.assertNotEqual(left['arms'], both['arms'])
        self.assertEqual(kit_inputs.read_kit('rig'), both)
        self.assertFalse((kit_inputs.KITS / 'rig/versions').exists())
        self.assertFalse((kit_inputs.KITS / 'rig/current.json').exists())
        self.assertEqual(set(both), {'id', 'arms'})
        with self.assertRaisesRegex(ValueError, '不同相机'):
            write_kit('bad', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A', 'right': 'CAMERA_A'}).items()})

    def test_global_kit_is_reusable_and_batches_keep_their_snapshots(self):
        shelf, counter = self.scene('shelf'), self.scene('counter')
        left = write_kit('rig', {'left': dict(serial='CAMERA_A', gripper=dict(tcp='tcp_camera.json'))},
                         description='设备备注')
        snapshot_kit = dict(id='rig', arms=left['arms'])
        first = write_batch(shelf, 'morning', kit='rig')
        second = write_batch(counter, 'morning', kit='rig')
        self.assertEqual(batch_inputs.read_batch(first).kit, snapshot_kit)
        self.assertEqual(batch_inputs.read_batch(second).kit, snapshot_kit)
        snapshot = read_yaml(first)
        self.assertEqual(set(snapshot), {'schema_version', 'scene', 'kit', 'map_policy', 'resolved_mode', 'expected_arms'})
        self.assertEqual(snapshot['kit'], snapshot_kit)
        stored = read_yaml(kit_inputs.KITS / 'rig/kit.yaml')
        self.assertEqual(stored['description'], '设备备注')
        self.assertNotIn('id', stored)
        right = write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'right': 'CAMERA_B'}).items()})
        later = write_batch(counter, 'afternoon', kit='rig')
        self.assertEqual(batch_inputs.read_batch(later).kit, right)
        self.assertEqual(batch_inputs.read_batch(later).expected_arms, ('right',))
        for path in (first, second):
            frozen = batch_inputs.read_batch(path)
            self.assertEqual(frozen.kit, snapshot_kit)
            self.assertEqual(frozen.expected_arms, ('left',))
        self.assertIsNone(batch_inputs.read_batch(first).map_version)
        self.assertEqual(kit_inputs.read_kit('rig'), right)

    def test_batch_names_are_unique_within_scene_and_manifests_cannot_be_nested(self):
        write_kit('left_rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A'}).items()})
        write_kit('right_rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'right': 'CAMERA_B'}).items()})
        left = write_batch(self.scene_path, 'morning', kit='left_rig')
        self.assertEqual(batch_inputs.read_batch(left).kit['id'], 'left_rig')
        right = write_batch(self.scene_path, 'afternoon', kit='right_rig')
        self.assertNotEqual(left, right)
        self.assertEqual(left, self.scene_path / 'morning/.pipeline/manifest.yaml')
        self.assertEqual(right, self.scene_path / 'afternoon/.pipeline/manifest.yaml')
        self.assertEqual(batch_inputs.read_batch(right).expected_arms, ('right',))
        data = read_yaml(right)
        data['scene'] = '../../afternoon/collection.yaml'
        misplaced = self.scene_path / 'nested/afternoon/.pipeline/manifest.yaml'
        write_yaml(misplaced, data)
        with self.assertRaisesRegex(ValueError, '采集批次必须位于'):
            batch_inputs.read_batch(misplaced)
        for name in sorted(layout.INTERNAL_DIRECTORIES | {'kits', 'sessions'}):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, '批次名'):
                kit_inputs_configuration.validate_batch_name(name)

    def test_manual_batch_can_select_one_hand_from_a_bimanual_kit(self):
        write_kit('rig', {
            'left': dict(serial='CAMERA_A', gripper=dict(tcp='tcp_camera.json')),
            'right': dict(serial='CAMERA_B', gripper=dict(tcp='tcp_camera.json')),
        })
        batch = self.scene_path / 'morning'
        declaration = batch / 'collection.yaml'
        write_yaml(declaration, dict(kit='rig',
            arms=['right'], description='',
            map=dict(layout='shared', reference_arm='right'),
            backend=dict(mode='scene_map')))
        before = declaration.read_bytes()
        self.assertFalse((batch / '.pipeline').exists())
        resolved = batch_inputs.read_batch(batches.save_record(batches.read_inputs(batch, require_mapping=False)))
        self.assertEqual(declaration.read_bytes(), before)
        data = read_yaml(resolved.path)
        self.assertIn('scene', data)
        self.assertIn('resolved_mode', data)
        self.assertEqual(data['expected_arms'], ['right'])
        self.assertEqual(set(data['kit']), {'id', 'arms'})
        self.assertEqual(resolved.expected_arms, ('right',))
        self.assertTrue(declaration.is_file())

    def test_existing_session_does_not_read_current_kit(self):
        original = write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A'}).items()})
        session = write_batch(self.scene_path, 'morning', kit='rig')
        write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'right': 'CAMERA_B'}).items()})
        kit_file = kit_inputs.KITS / 'rig/kit.yaml'
        kit_file.write_text('tampered')
        self.assertEqual(batch_inputs.read_batch(session).kit, original)
        with self.assertRaises(ValueError):
            write_batch(self.scene_path, 'broken', kit='rig')
        kit_file.unlink()
        self.assertEqual(batch_inputs.read_batch(session).kit, original)
        with self.assertRaises(OSError):
            write_batch(self.scene_path, 'missing', kit='rig')

    def test_session_requires_its_own_snapshot(self):
        write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A'}).items()})
        session = write_batch(self.scene_path, 'morning', kit='rig')
        data = read_yaml(session)
        data['kit'] = dict(id='rig')
        write_yaml(session, data)
        with self.assertRaisesRegex(ValueError, '套件'):
            batch_inputs.read_batch(session)

    def test_camera_selection_rejects_invalid_measurement_before_processing(self):
        write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A'}).items()})
        kit_file = kit_inputs.KITS / 'rig/kit.yaml'
        data = read_yaml(kit_file)
        data['arms']['left']['calibration'] = 'missing'
        write_yaml(kit_file, data)
        with self.assertRaisesRegex(ValueError, '当前 kit'):
            write_batch(self.scene_path, 'invalid', kit='rig')

    def test_current_kit_identity_matches_its_directory(self):
        write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A'}).items()})
        kit_file = kit_inputs.KITS / 'rig/kit.yaml'
        data = read_yaml(kit_file)
        data['id'] = 'other'
        write_yaml(kit_file, data)
        with self.assertRaisesRegex(ValueError, '未知项'):
            kit_inputs.read_kit('rig')


# Execution and configuration preview share task decisions without video IO.


class TaskConfigurationTests(unittest.TestCase):
    def setUp(self):
        from test_scene import SceneWorkflowTests
        self.helper = SceneWorkflowTests('runTest')
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.collection = self.helper.batch_root / 'collection.yaml'
        self.plan = layout.metadata(self.helper.batch_root, 'map_epochs.yaml')
        record = read_yaml(self.helper.session_path)
        record['map']['version'] = 'map_default'
        write_yaml(self.helper.session_path, record)

    def configure(self, *, backend='scene_map', task=None, poster=False):
        document = read_yaml(self.collection)
        document['backend'] = dict(mode=backend)
        document['tasks'] = dict(take_001=task or {})
        if poster:
            document['poster'] = dict(layout=str(
                Path(__file__).resolve().parents[1] / 'hardware/environment_poster/environment_poster.json'))
        write_yaml(self.collection, document)
        return document

    def epoch_plan(self, *, pending=False):
        write_yaml(self.plan, dict(schema_version=1,
            maps=[dict(epoch='epoch_a', map_version='map_a', sequence=0),
                  dict(epoch='epoch_b', map_version=None if pending else 'map_b', sequence=3,
                       **(dict(recording='mapping/b/video.insv') if pending else {}))],
            tasks=[dict(task='take_001', sequence=4)]))

    def preview(self):
        with patch('umi.scene.collection.batch.read_recordings',
                   side_effect=AssertionError('preview must not inspect recordings')):
            return effective_report([self.helper.task_root])[0]

    def test_epoch_default_and_explicit_choices_match_execution(self):
        self.epoch_plan()
        for task, version, reason in (({}, 'map_b', 'collection_sequence'),
                ({'map_epoch': 'epoch_a'}, 'map_a', 'task_map_epoch'),
                ({'map_version': 'map_a'}, 'map_a', 'task_map_version')):
            declared = {'map': {key.removeprefix('map_'): value for key, value in task.items()}}
            # Public map uses epoch/version, while the resolver carries normalized requests.
            self.configure(task=declared)
            episode = read_episode(self.helper.episode_path, verify=False)
            report = self.preview()
            self.assertEqual(report['selection']['map_version'], version)
            self.assertEqual(report['effective']['map_selection'], episode.map_selection)
            self.assertEqual(episode.map_selection['reason'], reason)
            self.assertEqual(requested_map_version(episode, None), version)

    def test_frozen_batch_default_matches_execution(self):
        self.configure(task={'instruction': 'pick'})
        episode = read_episode(self.helper.episode_path, verify=False)
        report = self.preview()
        self.assertEqual(report['selection']['map_version'], 'map_default')
        self.assertEqual(report['effective']['map_selection'], episode.map_selection)
        self.assertEqual(report['effective']['instruction'], episode.instruction)

    def test_auto_order_inherits_and_task_override_replaces_the_list(self):
        order = ['poster_vio', 'poster_independent', 'scene_map']
        document = self.configure(backend='auto', poster=True)
        document['backend']['auto_order'] = order
        write_yaml(self.collection, document)
        self.assertEqual(self.preview()['effective']['backend']['auto_order'], order)
        task = document['tasks']['take_001']
        task['backend'] = dict(auto_order=['poster_independent'])
        write_yaml(self.collection, document)
        self.assertEqual(read_episode(self.helper.episode_path).scene.auto_order, ('poster_independent',))
        self.assertEqual(self.preview()['effective']['backend']['auto_order'], ['poster_independent'])
        self.assertIsNone(self.preview()['effective']['map_selection'])
        for invalid in ([], 'poster_vio', ['auto'], ['poster_vio', 'poster_vio'], ['scene_map', {}]):
            with self.subTest(order=invalid), self.assertRaisesRegex(ValueError, r'tasks\.take_001\.backend\.auto_order'):
                task['backend']['auto_order'] = invalid
                write_yaml(self.collection, document)
                self.preview()

    def test_instruction_edit_preserves_explicit_single_hand_selection(self):
        document = self.configure(task=dict(arms=['right']))
        path = self.helper.episode_path
        before = path.read_bytes()
        document['tasks']['take_001']['instruction'] = '把盒子放入托盘'
        write_yaml(self.collection, document)

        episode = read_episode(path)
        self.assertEqual(episode.instruction, '把盒子放入托盘')
        self.assertEqual(episode.expected_arms, ('right',))
        self.assertEqual(set(episode.recordings), {'right'})
        self.assertTrue(episode.synchronization['confirmed'])
        self.assertEqual(path.read_bytes(), before)

    def test_instruction_edit_preserves_confirmed_manual_sync(self):
        sync = dict(mode='manual', offsets_s=dict(left=0., right=-.12), confirmed=True, note='首尾已核验')
        document = self.configure(task=dict(sync=sync))
        path = self.helper.episode_path
        document['tasks']['take_001']['instruction'] = '双手移动盒子'
        write_yaml(self.collection, document)
        episode = read_episode(path)
        self.assertEqual(episode.offsets(), sync['offsets_s'])
        self.assertEqual(episode.synchronization['note'], sync['note'])
        self.assertTrue(episode.synchronization['confirmed'])

        sync['note'] = ''
        write_yaml(self.collection, document)
        with self.assertRaisesRegex(ValueError, '核验依据'):
            read_episode(path)

    def test_audio_sync_uses_task_then_batch_then_global_settings(self):
        user_values = defaults_document()
        manual = dict(mode='manual', offsets_s=dict(left=0., right=.12), confirmed=False)
        for global_enabled, batch_enabled, task, expected in (
            (True, None, {}, 'audio_auto'),
            (False, None, {}, 'assumed_simultaneous'),
            (False, True, {}, 'audio_auto'),
            (True, False, {}, 'assumed_simultaneous'),
            (False, True, {'sync': manual}, 'manual'),
            (True, False, {'sync': 'audio_auto'}, 'audio_auto'),
            (False, True, {'arms': ['right']}, 'single_camera'),
        ):
            with self.subTest(global_enabled=global_enabled, batch_enabled=batch_enabled, task=task):
                user_values['sync']['mode'] = 'audio_auto' if global_enabled else 'assumed_simultaneous'
                with patched_defaults(self.helper.root, user_values):
                    document = self.configure(task=task)
                    if batch_enabled is None:
                        document.pop('sync', None)
                    else:
                        document['sync'] = dict(mode='audio_auto' if batch_enabled else 'assumed_simultaneous')
                    write_yaml(self.collection, document)
                    episode = read_episode(self.helper.episode_path, verify=False)
                    self.assertEqual(episode.synchronization['method'], expected)
                    report = self.preview()['effective']
                    self.assertIs(report['batch_audio_sync_enabled'],
                                  global_enabled if batch_enabled is None else batch_enabled)
                    self.assertEqual(report['synchronization'], episode.synchronization)

    def test_pending_epoch_is_visible_but_blocks_scene_map_execution(self):
        self.configure()
        self.epoch_plan(pending=True)
        report = self.preview()
        self.assertEqual(report['effective']['map_selection']['status'], 'pending')
        self.assertEqual(report['effective']['map_selection']['epoch'], 'epoch_b')
        self.assertIn('尚未构建', report['effective']['map_selection']['reason'])
        with self.assertRaisesRegex(MapEpochError, '尚未构建'):
            read_episode(self.helper.episode_path, verify=False)

    def test_auto_retains_pending_decision_for_fallback(self):
        document = self.configure(backend='auto', poster=True)
        document['backend']['auto_order'] = ['poster_vio', 'scene_map']
        write_yaml(self.collection, document)
        self.epoch_plan(pending=True)
        episode = read_episode(self.helper.episode_path, verify=False)
        self.assertEqual(episode.map_selection, self.preview()['effective']['map_selection'])
        self.assertEqual(episode.map_selection['status'], 'pending')
        self.assertTrue(episode.scene.parameters.tag_fusion_enabled)

    def test_poster_backends_derive_capabilities_and_ignore_map_plan(self):
        write_yaml(self.plan, dict(schema_version='invalid'))
        for backend in ('poster_vio', 'poster_independent'):
            with self.subTest(backend=backend):
                self.configure(backend=backend, poster=True)
                episode = read_episode(self.helper.episode_path, verify=False)
                report = self.preview()
                self.assertTrue(episode.scene.environment_tag_detection)
                self.assertTrue(episode.scene.parameters.tag_fusion_enabled)
                self.assertEqual(report['effective']['environment_tag_policy'], episode.scene.environment_tag_policy)
                self.assertIsNone(episode.map_selection)
                self.assertIsNone(report['selection']['map_version'])
                batch_report = effective_report([self.helper.batch_root])[0]
                self.assertEqual(batch_report['effective']['environment_tag_policy'],
                                 episode.scene.environment_tag_policy)
                self.assertIsNone(batch_report['selection']['map_version'])

    def test_explicit_disabled_capabilities_conflict_with_poster_backend(self):
        document = self.configure(backend='poster_vio', poster=True)
        document['poster']['detection'] = False
        write_yaml(self.collection, document)
        with self.assertRaisesRegex(ValueError, '冲突'):
            read_episode(self.helper.episode_path, verify=False)
        with self.assertRaisesRegex(ValueError, '冲突'):
            self.preview()

    def test_map_uses_declared_layout_by_default_and_respects_explicit_opt_out(self):
        self.configure(poster=True, task=dict(backend=dict(mode='poster_vio')))
        self.assertTrue(read_episode(self.helper.episode_path, verify=False).scene.parameters.tag_fusion_enabled)
        document = self.configure(poster=True)
        self.assertTrue(read_episode(self.helper.episode_path, verify=False).scene.parameters.tag_fusion_enabled)
        document['backend']['scene_map'] = dict(poster_constraint='off')
        write_yaml(self.collection, document)
        self.assertFalse(read_episode(self.helper.episode_path, verify=False).scene.parameters.tag_fusion_enabled)

    def test_audio_algorithm_and_base_tag_use_global_defaults_and_explicit_overrides(self):
        user_values = defaults_document()
        user_values['sync'].update(mode='audio_auto', algorithm='kiwi')
        user_values['backend']['poster_vio']['base_tag'] = False

        with patched_defaults(self.helper.root, user_values):
            document = self.configure(backend='auto', poster=True)
            document.pop('sync', None)
            document['backend']['poster_vio'] = dict(maximum_observation_gap_s=4.)
            write_yaml(self.collection, document)
            for scope, algorithm, base_tag in (('global', 'kiwi', False), ('batch', 'specialist', True),
                                               ('task', 'specialist', False)):
                with self.subTest(scope=scope):
                    if scope == 'batch':
                        document['sync'] = dict(algorithm='specialist')
                        document['backend']['poster_vio']['base_tag'] = True
                    if scope == 'task':
                        document['tasks']['take_001']['backend'] = dict(poster_vio=dict(base_tag=False))
                    write_yaml(self.collection, document)
                    episode = read_episode(self.helper.episode_path, verify=False)
                    report = self.preview()['effective']
                    self.assertEqual(episode.synchronization['algorithm'], algorithm)
                    self.assertIs(episode.scene.poster_vio_base_tag_enabled, base_tag)
                    self.assertEqual(report['batch_audio_sync_algorithm'], algorithm)
                    self.assertIs(report['poster_vio_base_tag_enabled'], base_tag)

    def test_invalid_epoch_selection_is_reported_and_execution_rejects_it(self):
        self.configure(task={'map': {'epoch': 'missing'}})
        self.epoch_plan()
        selection = self.preview()['effective']['map_selection']
        self.assertEqual(selection['status'], 'unavailable')
        with self.assertRaises(MapEpochError):
            read_episode(self.helper.episode_path, verify=False)

    def test_backend_options_apply_through_batch_task_and_expert_configuration(self):
        document = self.configure(poster=True)
        document['backend'] = dict(
            mode='auto',
            scene_map=dict(poster_constraint='joint'),
            poster_vio=dict(base_tag=False, maximum_observation_gap_s=4.),
            poster_independent=dict(maximum_position_support_gap_s=2.5, maximum_gravity_difference_deg=6.),
            edge_trim=dict(enabled=True, minimum_missing_duration_s=1.4))
        document['tasks']['take_001']['backend'] = dict(
            mode='poster_independent',
            scene_map=dict(poster_constraint='off'),
            poster_vio=dict(maximum_observation_gap_s=4.5),
            poster_independent=dict(maximum_gravity_difference_deg=4.),
            edge_trim=dict(minimum_missing_duration_s=1.8))
        write_yaml(self.collection, document)

        batch = effective_report([self.helper.batch_root])[0]['effective']['backend']
        self.assertEqual(batch['mode'], 'auto')
        self.assertTrue(batch['edge_trim']['scene_map_applies'])
        episode = read_episode(self.helper.episode_path, verify=False)
        task = self.preview()['effective']['backend']
        self.assertEqual(task['mode'], episode.scene.reconstruction_mode)
        self.assertEqual(task['scene_map']['poster_constraint'], 'off')
        self.assertFalse(task['edge_trim']['scene_map_applies'])
        self.assertTrue(task['edge_trim']['enabled'])
        self.assertEqual(task['edge_trim']['minimum_missing_duration_s'], 1.8)
        self.assertTrue(episode.scene.parameters.tag_fusion_enabled)
        self.assertFalse(episode.scene.poster_vio_base_tag_enabled)
        self.assertEqual(episode.scene.parameters.poster_vio_maximum_observation_gap_s, 4.5)
        self.assertEqual(episode.scene.parameters.poster_maximum_position_support_gap_s, 2.5)
        self.assertEqual(episode.scene.parameters.poster_maximum_gravity_difference_deg, 4.)
        self.assertEqual(episode.scene.recording_trim, dict(enabled=True, minimum_missing_duration_s=1.8))
        self.assertTrue(episode.scene.scene_map_edge_trim)

        expert_path = self.helper.batch_root / 'expert.yaml'
        write_yaml(expert_path, dict(backends=dict(poster_vio=dict(maximum_observation_gap_s=7.))))
        expert = SceneOverrides.load(expert_path)
        episode = read_episode(self.helper.episode_path, verify=False, overrides=expert)
        self.assertEqual(episode.scene.parameters.poster_vio_maximum_observation_gap_s, 7.)
        self.assertEqual(episode.scene.parameters.poster_maximum_gravity_difference_deg, 4.)

    def test_show_config_reports_the_effective_retry_target_and_its_origin(self):
        document = self.configure(poster=True)
        document['backend'] = dict(mode='poster_independent',
                                   poster_independent=dict(maximum_position_support_gap_s=3.))
        task_backend = dict(poster_independent=dict(maximum_position_support_gap_s=4.))
        document['tasks']['take_001']['backend'] = task_backend
        write_yaml(self.collection, document)
        expert_path = self.helper.batch_root / 'expert.yaml'
        write_yaml(expert_path, dict(shared=dict(poster=dict(acceptance=dict(maximum_position_support_gap_s=5.)))))

        project_source = 'common.yaml:shared.poster.acceptance.maximum_position_support_gap_s'
        batch_source = ':backend.poster_independent.maximum_position_support_gap_s'
        task_source = ':tasks.take_001.backend.poster_independent.maximum_position_support_gap_s'
        for use_task, use_batch, expert, expected, source in (
                (False, False, None, 2., project_source), (False, True, None, 3., batch_source),
                (True, True, None, 4., task_source),
                (True, True, SceneOverrides.load(expert_path), 5., 'expert.yaml:shared.poster.acceptance.maximum_position_support_gap_s')):
            with self.subTest(source=source):
                batch_options = dict(poster_independent=dict(maximum_position_support_gap_s=3.)) if use_batch else {}
                document['backend'] = dict(mode='poster_independent', **batch_options)
                document['tasks']['take_001'] = {'backend': task_backend} if use_task else {}
                write_yaml(self.collection, document)
                effective = effective_report([self.helper.task_root], overrides=expert)[0]['effective']
                support = effective['position_support']['backends']['poster_independent']
                self.assertEqual((support['retry_target_s'], support['acceptance_maximum_s']), (expected, expected))
                self.assertTrue(support['source'].endswith(source), support['source'])
                episode = read_episode(self.helper.episode_path, verify=False, overrides=expert)
                self.assertEqual(episode.scene.parameters.poster_maximum_position_support_gap_s, expected)
                self.assertNotIn('standalone', effective['reconstruction'])

    def test_auto_map_constraint_is_independent_of_poster_fallback(self):
        document = self.configure(poster=True)
        document['backend'] = dict(mode='auto', scene_map=dict(poster_constraint='off'))
        write_yaml(self.collection, document)
        episode = read_episode(self.helper.episode_path, verify=False)
        inputs = TaskInputs(episode, {}, {}, self.helper.task_root, None, None, None)
        result = BackendResult({}, {}, {})

        def run_map(current, **kwargs):
            self.assertFalse(current.episode.scene.parameters.tag_fusion_enabled)
            self.assertFalse(current.episode.scene.environment_tag_detection)
            raise SceneFusionQualityError('map coverage')

        def run_vio(current, selection):
            self.assertTrue(current.episode.scene.parameters.tag_fusion_enabled)
            self.assertTrue(current.episode.scene.environment_tag_detection)
            return result

        selection = estimation_binding.MapSelection.available(self.helper.batch_root, dict(version='v1', frame_id='map:v1'))
        with patch.object(estimation.scene_map, 'execute_task', side_effect=run_map), \
             patch.dict(estimation.ESTIMATORS, poster_vio=run_vio):
            self.assertIs(estimation.execute(inputs, selection).coordinate_record, result.coordinate_record)


# 配置优先级与只读展示边界，不启动算法或网络服务。


class EffectiveConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_yaml_auto_order_inherits_and_each_declared_layer_replaces_it(self):
        from umi.scene.collection.config_report import effective_report
        collection = self.root / 'batch/collection.yaml'
        doc = dict(kit='kit_001', arms=['left'], tasks={'take_001': {'backend': {'auto_order': ['poster_vio']}}})
        take = collection.parent / 'take_001'
        take.mkdir(parents=True)
        defaults = defaults_document()
        defaults['backend']['auto_order'] = ['poster_independent', 'scene_map']
        write_yaml(collection, doc)
        with patched_defaults(self.root, defaults):
            batch = effective_report([collection])[0]['effective']
            self.assertEqual(batch['backend']['auto_order'], ['poster_independent', 'scene_map'])
            self.assertIn('backend.auto_order', batch['project_defaults_used'])
            doc['backend'] = {'auto_order': ['scene_map', 'poster_vio']}
            write_yaml(collection, doc)
            batch = effective_report([collection])[0]['effective']
            task = effective_report([take])[0]['effective']
        self.assertEqual(batch['backend']['auto_order'], ['scene_map', 'poster_vio'])
        self.assertEqual(task['backend']['auto_order'], ['poster_vio'])
        self.assertNotIn('backend.auto_order', task['project_defaults_used'])

    def test_scene_expert_and_snapshot_share_the_same_input_scope(self):
        expert = self.root / 'expert.yaml'
        settings = {'standalone': {'input': {'mode': 'dual', 'allow_irregular_frames': True}}}
        write_yaml(expert, settings)
        standalone = config.load('trajectory', expert).values.input
        self.assertTrue(standalone.allow_irregular)
        self.assertEqual(standalone.mode, 'dual')
        with self.assertRaisesRegex(ValueError, '只用于独立逐段引擎'):
            SceneOverrides.load(expert)
        with self.assertRaisesRegex(ValueError, '只用于独立逐段引擎'):
            SceneOverrides.from_snapshot(dict(path=str(expert), sha256='previous', values=settings))

    def test_scene_retry_gap_has_clear_scope_for_overrides_and_snapshots(self):
        expert = self.root / 'expert.yaml'
        settings = dict(standalone=dict(retry=dict(target_maximum_position_support_gap_s=9.)),
                        segment_engine=dict(retry=dict(maximum_tries=2)))
        write_yaml(expert, settings)
        standalone = config.load('trajectory', expert).values.retry
        self.assertEqual(standalone.maximum_position_support_gap_s, 9.)
        with self.assertRaisesRegex(ValueError, r'retry\.target_maximum_position_support_gap_s.*backends\.poster_independent'):
            SceneOverrides.load(expert)

        snapshot = dict(path=str(expert), sha256='previous', values=settings)
        with self.assertRaisesRegex(ValueError, '只用于独立逐段引擎'):
            SceneOverrides.from_snapshot(snapshot)

    def test_processing_options_report_the_winning_layer_per_switch(self):
        from umi.scene.collection.config_report import effective_report
        collection = self.root / 'batch/collection.yaml'
        doc = dict(kit='kit_001', arms=['left'])
        write_yaml(collection, doc)
        def options(expert=None):
            return effective_report([collection], overrides=expert)[0]['effective']['processing_options']

        defaults = read_yaml(reconstruction_config.COMMON_PATH)
        vio = read_yaml(reconstruction_config.BACKEND_PATHS['poster_vio'])
        project = dict(
            robust_inertial_seed=vio['initialization']['robust_inertial_seed'],
            verified_gap_recovery=defaults['measurements']['gripper_tags']['verified_gap_recovery'])
        self.assertEqual(options(), {name: dict(value=value, source='project') for name, value in project.items()})
        doc['measurements'] = dict(gripper_tags=dict(verified_gap_recovery=not project['verified_gap_recovery']))
        write_yaml(collection, doc)
        self.assertEqual(options(), dict(
            robust_inertial_seed=dict(value=project['robust_inertial_seed'], source='project'),
            verified_gap_recovery=dict(value=not project['verified_gap_recovery'], source='collection')))
        expert_path = self.root / 'expert.yaml'
        write_yaml(expert_path, dict(
            backends=dict(poster_vio=dict(initialization=dict(robust_inertial_seed=True))),
            measurements=dict(gripper_tags=dict(verified_gap_recovery=True))))
        self.assertEqual(options(SceneOverrides.load(expert_path)),
                         dict(robust_inertial_seed=dict(value=True, source='expert'),
                              verified_gap_recovery=dict(value=True, source='expert')))

    def test_scene_precedence_and_immutable_loaded_overlay(self):
        scene_path = self.root/'batch/collection.yaml'
        write_yaml(scene_path, dict(description='test', kit='rig',
                                    arms=['left', 'right'],
                                    map=dict(layout='shared', reference_arm='left'),
                                    backend=dict(mode='scene_map')))
        override_path = self.root/'override.yaml'
        write_yaml(override_path, dict(recording=dict(video=dict(frame_step=4, crf=15)),
                                      measurements=dict(gripper_width=dict(maximum_interpolation_gap_s=.8))))
        overrides = SceneOverrides.load(override_path)
        override_path.write_text('{}')
        scene = scene_inputs.read_scene(scene_path, overrides=overrides)
        self.assertEqual(scene.trajectory.values.video.frame_step, 4)
        self.assertEqual(scene.trajectory.values.video.crf, 15)
        self.assertEqual(scene.parameters.gripper_width_max_gap_s, .8)
        self.assertEqual(scene.trajectory.sources[-1], str(override_path))
        self.assertNotEqual(scene_inputs.read_scene(scene_path).trajectory.values.video.frame_step, 4)

    def test_expert_files_reject_scene_policy_and_unsupported_sections(self):
        path = self.root / 'override.yaml'
        for values, message in (
            (dict(standalone=dict(poster_fusion=True)), 'collection.yaml'),
            (dict(parameters=dict(localization=dict(minimum_fraction=.7))), r'parameters 是未知字段'),
            (dict(trajectory=dict(fusion=dict(node_stride=3))), r'trajectory 是未知字段'),
        ):
            with self.subTest(values=values):
                write_yaml(path, values)
                with self.assertRaisesRegex(ValueError, message):
                    SceneOverrides.load(path)

    def test_frozen_expert_uses_recorded_values_without_reopening_the_file(self):
        path = self.root / 'missing.yaml'
        values = dict(shared=dict(solver=dict(node_stride=3)),
                      backends=dict(poster_vio=dict(minimum_training_observation_times=9),
                                    scene_map=dict(mapping=dict(poster_scale=dict(minimum_observation_times=7)),
                                                   localization=dict(minimum_localized_fraction=.7))))
        frozen = SceneOverrides.from_snapshot(dict(path=str(path), sha256='frozen', values=values))
        parameters = scene_inputs.load_scene_parameters(overrides=dict(tag_mapping_enabled=True,
                                                                      tag_mapping_scale_enabled=True))
        trajectory, effective = apply_overrides(config.load('trajectory'), parameters, frozen)
        self.assertEqual(trajectory.values.fusion.node_stride, 3)
        self.assertTrue(effective.tag_mapping_scale_enabled)
        self.assertEqual(effective.tag_mapping_minimum_frames, 7)
        self.assertEqual(effective.poster_vio_minimum_tag_frames, 9)
        self.assertEqual(effective.minimum_localization_fraction, .7)

    def test_scene_overlay_rejects_asset_fields_and_unknown_parameters(self):
        override_path = self.root/'override.yaml'
        for declaration in (dict(kit='kit_002'), dict(video=[]), dict(fusion=dict(unknown=1)), {'video': {}, 1: 2},
                            dict(video=dict(unknown=1))):
            with self.subTest(declaration=declaration):
                write_yaml(override_path, declaration)
                with self.assertRaises(ValueError):
                    SceneOverrides.load(override_path)

    def test_backend_gap_does_not_change_the_reported_shared_default(self):
        from umi.scene.collection.config_report import effective_report

        collection = self.root / 'batch/collection.yaml'
        write_yaml(collection, dict(kit='kit_001', arms=['left']))
        expert = self.root / 'expert.yaml'
        threshold = dict(maximum_position_support_gap_s=1.5)
        write_yaml(expert, dict(backends=dict(poster_independent=threshold)))
        effective = effective_report([collection], overrides=SceneOverrides.load(expert))[0]['effective']
        self.assertEqual(effective['changed_from_defaults'], dict(backends=dict(poster_independent=threshold)))
        self.assertEqual(effective['reconstruction']['shared']['poster']['acceptance']['maximum_position_support_gap_s'],
                         read_yaml(reconstruction_config.COMMON_PATH)['shared']['poster']['acceptance']['maximum_position_support_gap_s'])

    def test_scene_show_config_never_processes_inputs(self):
        collection = self.root/'batch/collection.yaml'
        write_yaml(collection, dict(description='test', kit='kit_001',
                                    arms=['left', 'right'],
                                    map=dict(layout='shared', reference_arm='left'),
                                    backend=dict(mode='scene_map')))
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        open_path = Path.open
        collection_reads = []

        def count_reads(path, *args, **kwargs):
            if path == collection:
                collection_reads.append(path)
            return open_path(path, *args, **kwargs)

        with patch.object(scene_cli.scene_run, 'run') as run, \
             patch.object(Path, 'open', count_reads), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(scene_cli.main(['run', str(collection.parent), '--show-config', '--json', '--force']), 0)
        run.assert_not_called()
        self.assertEqual(collection_reads, [collection])
        report = json.loads(output.getvalue())
        self.assertEqual(report['scope'], 'configuration_only')
        self.assertEqual(report['configuration'][0]['effective']['reconstruction'],
                         reconstruction_config.public_tree(trajectory=asdict(config.load('trajectory').values),
                                                    scene=asdict(scene_inputs.load_scene_parameters()), for_scene=True))
        self.assertEqual(report['configuration'][0]['effective']['changed_from_defaults'], {})
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}, before)

    def test_scene_show_config_prints_a_summary_and_full_json_on_request(self):
        defaults = defaults_document()
        defaults['backend']['poster_vio']['base_tag'] = True
        self.enterContext(patched_defaults(self.root, defaults))
        collection = self.root/'batch/collection.yaml'
        write_yaml(collection, dict(kit='kit_001', arms=['left', 'right'], sync=dict(algorithm='kiwi'),
                                    backend=dict(mode='scene_map'), tasks=dict(take_002=dict(backend=dict(mode='poster_vio'))),
                                    poster=dict(layout=str(Path(__file__).resolve().parents[1] /
                                        'hardware/environment_poster/environment_poster.json'))))
        expert = self.root/'expert.yaml'
        write_yaml(expert, dict(shared=dict(solver=dict(node_stride=3)),
                                backends=dict(scene_map=dict(localization=dict(minimum_localized_fraction=0.6)))))
        arguments = ['run', str(collection.parent), '--show-config', '--expert-config', str(expert)]
        with patch.object(scene_cli.scene_run, 'run') as run, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(scene_cli.main(arguments), 0)
        run.assert_not_called()
        summary = output.getvalue()
        with self.assertRaises(json.JSONDecodeError):
            json.loads(summary)
        for expected in ('  后端      scene_map\n', '  任务覆盖  take_002 → poster_vio\n',
                         '布局 auto（项目默认）', '参考手 left（项目默认）',
                         '  音频同步  开启（项目默认） · kiwi\n', '  基座标签  开启（项目默认）\n',
                         '鲁棒惯性初始化 开启（项目默认）', f'  专家覆盖  {expert}\n',
                         '    shared.solver.node_stride = 3\n',
                         '    backends.scene_map.localization.minimum_localized_fraction = 0.6\n', '加 --json'):
            self.assertIn(expected, summary)
        with patch.object(scene_cli.scene_run, 'run'), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(scene_cli.main([*arguments, '--json']), 0)
        effective = json.loads(output.getvalue())['configuration'][0]['effective']
        self.assertEqual(effective['changed_from_defaults'], dict(
            shared=dict(solver=dict(node_stride=3)),
            backends=dict(scene_map=dict(localization=dict(minimum_localized_fraction=0.6)))))
        self.assertEqual(set(effective['project_defaults_used']),
                         {'map.layout', 'map.reference_arm', 'sync.mode', 'backend.poster_vio.base_tag'})

    def test_task_show_config_preserves_overrides_and_never_reads_recordings(self):
        from umi.scene.collection import config_report as effective_config
        from umi.scene.collection import kit as effective_config_kit
        batch = self.root / 'batch'
        task = batch / 'take_001'
        task.mkdir(parents=True)
        write_yaml(batch / 'collection.yaml', dict(kit='rig', arms=['left', 'right'],
            description='', map=dict(layout='per_arm'), backend=dict(mode='scene_map'),
            poster=dict(layout=str(Path(__file__).resolve().parents[1] /
                'hardware/environment_poster/environment_poster.json')),
            tasks=dict(take_001=dict(arms=['right'], backend=dict(mode='poster_vio')))))
        episode = batch / '.pipeline/tasks/take_001/episode.yaml'
        write_yaml(episode, dict(schema_version=5, recordings={}))
        before = {p: p.read_bytes() for p in batch.rglob('*') if p.is_file()}
        with patch.object(effective_config_kit, 'read_kit',
                          return_value=dict(id='rig', arms=dict(left={}, right={}))), \
                patch.object(episode_inputs, 'read_recordings', side_effect=AssertionError('recording IO')):
            for target in (task, episode, episode.parent):
                with self.subTest(target=target):
                    report = effective_config.effective_report([target])[0]
                    self.assertEqual(report['scope'], 'task')
                    self.assertEqual(report['target'], str(task))
                    self.assertEqual(report['selection']['arms'], ['right'])
                    self.assertEqual(report['effective']['reconstruction_mode'], 'poster_vio')
                    self.assertEqual(report['effective']['mode'], 'shared_front')
                    self.assertEqual(report['effective']['synchronization']['method'], 'single_camera')
            batch_report = effective_config.effective_report([batch])[0]
            self.assertEqual(batch_report['selection']['arms'], ['left', 'right'])
            self.assertEqual(batch_report['effective']['reconstruction_mode'], 'scene_map')
        self.assertEqual(before, {p: p.read_bytes() for p in batch.rglob('*') if p.is_file()})

    def test_calibration_show_config_without_recordings_or_environment(self):
        overlay = self.root/'calibration.yaml'
        write_yaml(overlay, dict(trajectory_validation=dict(config='trajectory.yaml')))
        write_yaml(self.root/'trajectory.yaml', dict(standalone=dict(input=dict(mode='dual'), poster_fusion=True),
            recording=dict(video=dict(frame_step=3))))
        with patch.object(calibration_cli.inputs, 'prepare_inputs') as parse_recordings, \
             patch.object(calibration_cli.runtime, 'use_calibration_environment') as environment:
            for command in (['run', 'missing/front'], ['verify', 'missing.insv'], ['run', 'missing/rear']):
                with self.subTest(command=command), redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(calibration_cli.main(command + ['--expert-config', str(overlay),
                        '--show-config', '--tag-size-mm', '36'], prepare_environment=True), 0)
                effective = json.loads(output.getvalue())['effective']
                trajectory = effective['trajectory_validation']['values']
                self.assertEqual(trajectory['orb']['initial_tag_size_m'], .036)
                self.assertEqual(trajectory['video']['frame_step'], 3)
                self.assertEqual(trajectory['input']['mode'], 'handheld')
                self.assertFalse(trajectory['environment_tags']['tag_fusion_enabled'])
                if command[-1] == 'missing/rear':
                    self.assertFalse(effective['scene_parameters']['tag_mapping_enabled'])
            parse_recordings.assert_not_called()
            environment.assert_not_called()

    def test_viewer_show_config_does_not_scan_or_listen(self):
        with patch.object(server, 'RecordingCatalog') as catalog, \
             patch.object(server, 'ThreadingHTTPServer') as http_server, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(server.main([str(self.root/'missing'), '--port', '0', '--show-config']), 0)
        catalog.assert_not_called()
        http_server.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['effective']['port'], 0)


"""当前配置是运行输入；内部记录不覆盖配置，历史成果保持原内容。"""


class CurrentConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.f = scene_run_tests.ProcessingTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.batch = self.f.batch
        self.kit = kit_inputs.KITS / 'rig/kit.yaml'

    def update_calibration(self, value='updated'):
        self.f.f.device_pins['SERIAL_left']['calibration'] = value

    def test_single_mapping_has_no_left_label_and_retired_schema_is_rejected(self):
        processing_batch.prepare_batch(self.batch)
        path = self.batch / '.pipeline/manifest.yaml'
        data = read_yaml(path)
        self.assertEqual(data['schema_version'], 5)
        self.assertEqual(set(data['mapping']), {'serial', 'path', 'sha256'})
        self.assertEqual(batch_inputs.read_batch(path).mapping, {'left': data['mapping']})
        write_yaml(path, dict(data, schema_version=4, mapping={'left': data['mapping']}))
        with self.assertRaisesRegex(ValueError, 'schema_version: 5'):
            batch_inputs.read_batch(path)

    def test_changed_calibration_rebuilds_map_and_keeps_historical_result(self):
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        old_session = batch_inputs.read_batch(self.batch)
        output = (self.batch / 'take_001/results').resolve()
        self.update_calibration()
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        current = batch_inputs.read_batch(self.batch)
        self.assertEqual(current.kit['arms']['left']['calibration'], 'updated')
        self.assertNotEqual(current.map_version, old_session.map_version)
        self.assertNotEqual((self.batch / 'take_001/results').resolve(), output)
        self.assertFalse(output.exists())  # 被取代的运行由自动清理删除
        self.assertEqual(processing.run([self.batch])[0]['status'], 'skipped')

    def test_mapping_replacement_is_discovered_without_new_batch(self):
        processing_batch.prepare_batch(self.batch)
        source = self.batch / 'mapping/map.INSV'
        before = batch_inputs.read_batch(self.batch).mapping['left']['sha256']
        source.write_text('left_new_mapping')
        processing_batch.prepare_batch(self.batch)
        latest = batch_inputs.read_batch(self.batch)
        self.assertNotEqual(latest.mapping['left']['sha256'], before)
        self.assertEqual(latest.mapping['left']['sha256'], artifact_identity.file_hash(source))

    def test_changed_config_preview_is_readonly_and_task_entry_refreshes_it(self):
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        self.update_calibration()
        before = {p: p.read_bytes() for p in self.batch.rglob('*') if p.is_file()}
        self.assertEqual(processing.run([self.batch], dry_run=True)[0]['status'], 'planned')
        self.assertEqual({p: p.read_bytes() for p in self.batch.rglob('*') if p.is_file()}, before)
        self.assertEqual(processing.run([self.batch / 'take_001'])[0]['status'], 'ok')
        self.assertEqual(batch_inputs.read_batch(self.batch).kit['arms']['left']['calibration'], 'updated')

    def test_organizer_batch_with_optional_scene_and_instruction_can_run(self):
        path = self.batch / 'collection.yaml'
        document = read_yaml(path)
        document.pop('description', None)
        document.pop('instruction', None)
        write_yaml(path, document)
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        document['instruction'] = 'Pick up the cup'
        write_yaml(path, document)
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')

    def test_width_only_change_reuses_map(self):
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        version = batch_inputs.read_batch(self.batch).map_version
        data = read_yaml(self.kit)
        data['arms']['left']['gripper']['width'] = 'left_width.json'
        write_json(self.kit.parent / 'left_width.json', dict(anchors=[dict(x=100., mm=0.), dict(x=200., mm=50.)]))
        write_yaml(self.kit, data)
        with patch.object(mapping, 'build_maps',
                side_effect=AssertionError('width-only change must not rebuild map')):
            self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        self.assertEqual(batch_inputs.read_batch(self.batch).map_version, version)

    def test_configuration_change_during_fusion_keeps_the_selected_snapshot(self):
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        previous = (self.batch / 'take_001/results').resolve()
        source = self.f.f.source / 'left/tcp.json'
        original_geometry = source.read_bytes()
        original_trajectory = (previous / 'data/left/trajectory.csv').read_bytes()
        original = scene_reconstruction.reconstruct_arm_in_scene
        def solve(*args, **kwargs):
            self.update_calibration()
            changed = json.loads(original_geometry)
            changed['T_camera_tcp']['t_m'] = [10., 0., 0.]
            write_json(source, changed)
            return original(*args, **kwargs)
        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene', side_effect=solve):
            result = processing.run([self.batch], force=True)
        self.assertEqual(result[0]['status'], 'ok')
        current = (self.batch / 'take_001/results').resolve()
        self.assertNotEqual(current, previous)
        record = json.loads((current / 'data/episode.json').read_text())
        self.assertEqual(record['kit']['arms']['left']['calibration'], 'test')
        self.assertEqual((result_paths(current).run / 'devices/SERIAL_left/tcp.json').read_bytes(), original_geometry)
        self.assertNotEqual(source.read_bytes(), original_geometry)
        self.assertEqual((current / 'data/left/trajectory.csv').read_bytes(), original_trajectory)
        self.assertFalse(previous.exists())  # 被取代的运行由自动清理删除

    def test_width_policy_changes_do_not_rebuild_the_map(self):
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        version = batch_inputs.read_batch(self.batch).map_version
        expert = self.f.f.root / 'width-expert.yaml'
        write_yaml(expert, dict(measurements=dict(gripper_width=dict(
            maximum_interpolation_gap_s=0., closed_threshold_mm=5.))))
        with patch.object(mapping, 'build_maps', side_effect=AssertionError('width policy rebuilt map')):
            self.assertEqual(processing.run([self.batch], overrides=SceneOverrides.load(expert))[0]['status'], 'ok')
        self.assertEqual(batch_inputs.read_batch(self.batch).map_version, version)
