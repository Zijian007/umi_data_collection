from umi.workflows.scene.task_configuration import read_synchronization
"""场景元数据与专家算法配置的边界。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from umi.workflows.scene.inputs import scene as scene_inputs
from umi.store import layout
from umi.workflows.scene.parameters import SceneOverrides
from umi.workflows.scene.maps import repository as map_lifecycle
from umi.infra.file_store import write_yaml as _write_yaml


def write_yaml(path, data):
    """Write current collection fixtures while keeping test cases compact."""
    if path.name == 'collection.yaml' and data.get('schema_version') == 1:
        scene = dict(data.get('scene', {}))
        tags = scene.pop('tags', None)
        normalized = dict(
            kit=data.get('kit', 'rig'),
            recording=dict(arms=data.get('arms', ['left', 'right'])),
            scene=dict(description=scene.pop('description', '')),
            map=dict(layout=scene.pop('layout', 'auto'),
                     reference_arm=scene.pop('map_reference_arm', 'left'),
                     selection=dict(policy='latest_prior')),
            reconstruction=dict(backend='scene_map'))
        normalized['scene'].update(scene)
        if tags is not None:
            normalized['environment_poster'] = dict(
                layout_path=tags.get('layout'),
                detection=dict(enabled=tags.get('enabled', False)),
                mapping=dict(enabled=tags.get('mapping', {}).get('enabled', False),
                             scale_correction=dict(enabled=tags.get('mapping', {}).get('scale', {}).get('enabled', False))),
                task_constraint=dict(enabled=tags.get('fusion', {}).get('enabled', False),
                                     method=tags.get('fusion', {}).get('method', 'staged')))
        if data.get('tasks'):
            normalized['tasks'] = {}
            for name, task in data['tasks'].items():
                item = dict(task)
                if 'expected_arms' in item:
                    item['arms'] = item.pop('expected_arms')
                if 'reconstruction' in item and 'mode' in item['reconstruction']:
                    item['reconstruction'] = dict(backend=item['reconstruction']['mode'])
                normalized['tasks'][name] = item
        data = normalized
    _write_yaml(path, data)


class SceneConfigurationTests(unittest.TestCase):
    def test_frozen_gripper_paths_follow_hardware_migration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kits = root / 'hardware/kits'
            frozen = dict(tcp=str(root / 'kits/tcp_camera.json'),
                          tag_distance=str(root / 'kits/kit_001/tag_distance.json'),
                          mask=str(root / 'kits/rig/hand.png'),
                          width=str(root / 'custom/width.json'))
            with patch.object(scene_inputs, 'PROJECT_ROOT', root), patch.object(scene_inputs, 'KITS', kits):
                current = scene_inputs.gripper_paths(frozen)
            self.assertEqual(current, dict(tcp=str(kits / 'shared/tcp_camera.json'),
                                           tag_distance=str(kits / 'shared/tag_distance.json'),
                                           mask=str(kits / 'rig/hand.png'), width=frozen['width']))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'shelf'
        self.path = self.root / 'seed/collection.yaml'
        self.path.parent.mkdir(parents=True)

    def write_scene(self, scene):
        write_yaml(self.path, dict(schema_version=1, scene=scene, kit='rig'))

    def test_inline_algorithm_settings_are_rejected(self):
        self.write_scene(dict(layout='shared', trajectory=dict(video=dict(size_px=960)),
                              parameters=dict(minimum_localization_fraction=.7)))
        with self.assertRaisesRegex(ValueError, '不承载算法参数'):
            scene_inputs.read_scene(self.path)

    def test_scene_rejects_input_sources_file_pointers_and_invalid_inline_values(self):
        bad = [dict(mapping={}), dict(devices={}), dict(trajectory_config='other.yaml'),
               dict(parameters='other.yaml'), dict(trajectory=[]), dict(schema_version=True),
               dict(trajectory=dict(video=dict(size_px=719))), dict(parameters=dict(unknown=1)),
               dict(parameters=dict(minimum_localization_fraction=2.)),
               dict(parameters=dict(maximum_visual_gap_s=float('nan')))]
        for data in bad:
            with self.subTest(data=data):
                self.write_scene(dict(layout='shared', **data))
                with self.assertRaises(ValueError):
                    scene_inputs.read_scene(self.path)

    def test_legacy_schema_is_rejected_without_rewriting(self):
        self.write_scene(dict(layout='shared', schema_version=3))
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'scene 只允许 description'):
            scene_inputs.read_scene(self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_build_requires_a_batch_instead_of_a_standalone_scene(self):
        self.write_scene(dict(layout='shared'))
        with self.assertRaisesRegex(ValueError, '建图需要采集批次'):
            map_lifecycle.build_scene(self.path)

    def test_inline_algorithm_settings_rejected_regardless_of_comments_and_order(self):
        self.path.write_text('kit: rig\nrecording:\n  arms: [left, right]\nscene:\n  description: test\n  trajectory:\n    video:\n      size_px: 960\nreconstruction:\n  backend: scene_map\n')
        with self.assertRaisesRegex(ValueError, '不承载算法参数'):
            scene_inputs.read_scene(self.path)
        self.path.write_text('kit: rig\nrecording:\n  arms: [left, right]\nscene:\n  description: test\n  trajectory:\n    video:\n      size_px: 960\nreconstruction:\n  backend: scene_map\n')
        with self.assertRaisesRegex(ValueError, '不承载算法参数'):
            scene_inputs.read_scene(self.path)

    def test_poster_position_support_thresholds_are_configurable_and_positive(self):
        parameters = scene_inputs.load_scene_parameters(overrides=dict(fusion=dict(environment_tags=dict(
            maximum_position_std_m=.02, maximum_rotation_std_deg=2.))))
        self.assertEqual(parameters.tag_maximum_position_std_m, .02)
        self.assertEqual(parameters.tag_maximum_rotation_std_deg, 2.)
        for value in (0., -1., float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scene_inputs.load_scene_parameters(overrides=dict(fusion=dict(environment_tags=dict(
                    maximum_position_std_m=value))))

    def test_poster_culling_and_rear_refinement_share_validated_configuration(self):
        parameters = scene_inputs.load_scene_parameters(overrides=dict(fusion=dict(environment_tags=dict(
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
        self.assertFalse(defaults.gripper_width_diagnostics)
        policy = dict(gripper_width_max_gap_s=0., gripper_width_closed_threshold_mm=0.,
                      gripper_width_diagnostics=True)
        self.assertTrue(scene_inputs.load_scene_parameters(overrides=policy).gripper_width_diagnostics)
        for invalid in (dict(gripper_width_max_gap_s=-1), dict(gripper_width_closed_threshold_mm=True),
                        dict(gripper_width_diagnostics=1), dict(gripper_width_max_gap_s=float('nan'))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                scene_inputs.load_scene_parameters(overrides=invalid)

    def test_poster_backend_quality_parameters_are_independent_and_round_trip(self):
        from dataclasses import asdict
        from umi.config.scene_parameters import nested_snapshot, flatten
        parameters = scene_inputs.load_scene_parameters(overrides=dict(
            placement=dict(maximum_rotation_deg=1.), localization=dict(maximum_visual_gap_s=.1)))
        self.assertEqual(parameters.poster_maximum_gravity_difference_deg, 5.)
        self.assertEqual(parameters.poster_maximum_position_support_gap_s, 2.)
        configured = scene_inputs.load_scene_parameters(overrides=dict(poster_independent=dict(
            maximum_gravity_difference_deg=2., maximum_position_support_gap_s=.5)))
        self.assertEqual(configured.poster_maximum_gravity_difference_deg, 2.)
        self.assertEqual(configured.maximum_visual_gap_s, 2.)
        snapshot = nested_snapshot(asdict(configured))
        # 报告快照中的共享环境参数是独立信息，不属于场景覆盖的 YAML 根分组。
        snapshot.pop('environment_tags')
        decoded = flatten(snapshot)
        self.assertEqual(decoded['poster_maximum_gravity_difference_deg'], 2.)
        self.assertEqual(decoded['poster_maximum_position_support_gap_s'], .5)
        for value in (0., -1., float('nan'), True, 181.):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scene_inputs.load_scene_parameters(overrides=dict(poster_independent=dict(
                    maximum_gravity_difference_deg=value)))

    def test_collection_yaml_is_single_user_scene_and_task_declaration(self):
        batch = self.root / '20261113'
        batch.mkdir()
        collection = batch / 'collection.yaml'
        write_yaml(collection, dict(schema_version=1, kit='rig', arms=['left'],
            scene=dict(layout='shared', map_reference_arm='left', description='demo'),
            tasks=dict(take_001=dict(instruction='pick', expected_arms=['left']))))
        scene = scene_inputs.read_scene(collection)
        self.assertEqual(scene.name, self.root.name)
        self.assertEqual(scene.mode, 'shared_front')
        self.assertEqual(layout.task_configuration(batch / '.pipeline/tasks/take_001'),
                         dict(instruction='pick', expected_arms=['left']))

    def test_collection_yaml_rejects_split_batch_and_task_overrides(self):
        batch = self.root / '20261113'
        batch.mkdir()
        write_yaml(batch / 'collection.yaml', dict(schema_version=1, kit='rig', scene={}))
        write_yaml(batch / 'batch.yaml', dict(kit='rig'))
        with self.assertRaisesRegex(ValueError, 'batch.yaml 已移除'):
            from umi.workflows.scene import batch_snapshot as batches
            batches.read_collection(batch)
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

    def test_collection_layout_is_user_semantic_and_resolves_to_internal_mode(self):
        collection = self.root / '20261114' / 'collection.yaml'
        collection.parent.mkdir()
        write_yaml(collection, dict(schema_version=1, kit='rig', scene=dict(layout='auto')))
        self.assertEqual(scene_inputs.read_scene(collection).mode, 'auto')
        write_yaml(collection, dict(schema_version=1, kit='rig', scene=dict(layout='per_arm', map_reference_arm='left')))
        self.assertEqual(scene_inputs.read_scene(collection).mode, 'dual')

    def test_auto_layout_handles_epoch_directory_shapes(self):
        from umi.workflows.scene import batch_snapshot as batches
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

    def test_retired_flat_and_layout_only_tag_forms_are_rejected(self):
        board = scene_inputs.ROOT / 'hardware/environment_poster/environment_poster.json'
        if not board.is_file():
            self.skipTest('environment poster fixture unavailable')
        for index, poster in enumerate((dict(layout_path=str(board), detection=True),
                                        dict(layout_path=str(board), task_constraint=True))):
            with self.subTest(poster=poster):
                collection = self.root / f'legacy_{index}' / 'collection.yaml'
                collection.parent.mkdir()
                write_yaml(collection, dict(kit='rig',
                    recording=dict(arms=['left', 'right']), scene=dict(description='test'),
                    map=dict(layout='shared', reference_arm='left', selection=dict(policy='latest_prior')),
                    environment_poster=poster, reconstruction=dict(backend='scene_map')))
                with self.assertRaisesRegex(ValueError, 'environment_poster'):
                    scene_inputs.read_scene(collection)

    def test_expert_config_cannot_override_collection_tag_switches(self):
        expert = self.root / 'expert.yaml'
        for values in (dict(parameters=dict(fusion=dict(environment_tags=dict(enabled=True)))),
                       dict(parameters=dict(mapping=dict(environment_tags=dict(enabled=True)))),
                       dict(parameters=dict(fusion=dict(environment_tags=dict(method='joint')))),
                       dict(trajectory=dict(environment_tags=dict(tag_fusion_enabled=True)))):
            write_yaml(expert, values)
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'collection.yaml'):
                SceneOverrides.load(expert)

    def test_environment_tag_policy_uses_canonical_enabled_fields(self):
        board = scene_inputs.ROOT / 'hardware/environment_poster/environment_poster.json'
        if not board.is_file():
            self.skipTest('environment poster fixture unavailable')
        collection = self.root / '20261118' / 'collection.yaml'
        collection.parent.mkdir()
        write_yaml(collection, dict(schema_version=1, kit='rig', scene=dict(
            layout='shared', map_reference_arm='left', tags=dict(
                enabled=True, layout=str(board),
                mapping=dict(enabled=True, scale=dict(enabled=False)),
                              fusion=dict(enabled=True, method='joint')))))
        scene = scene_inputs.read_scene(collection)
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
        cases = [
            dict(enabled=False, mapping=dict(enabled=False), fusion=dict(enabled=True)),
            dict(enabled=True, mapping=dict(enabled=False, scale=dict(enabled=True)), fusion=dict(enabled=False)),
        ]
        for policy in cases:
            with self.subTest(policy=policy):
                collection = self.root / 'invalid' / 'collection.yaml'
                collection.parent.mkdir(exist_ok=True)
                tags = dict(policy, layout=str(board))
                write_yaml(collection, dict(schema_version=1, kit='rig', scene=dict(
                    layout='shared', map_reference_arm='left', tags=tags)))
                with self.assertRaises(ValueError):
                    scene_inputs.read_scene(collection)

    def test_environment_tag_fusion_method_is_validated(self):
        board = scene_inputs.ROOT / 'hardware/environment_poster/environment_poster.json'
        if not board.is_file():
            self.skipTest('environment poster fixture unavailable')
        collection = self.root / 'invalid_method' / 'collection.yaml'
        collection.parent.mkdir()
        write_yaml(collection, dict(schema_version=1, kit='rig', scene=dict(
            layout='shared', map_reference_arm='left', tags=dict(
                enabled=True, layout=str(board),
                fusion=dict(enabled=True, method='unknown')))))
        with self.assertRaisesRegex(ValueError, 'method|staged|joint'):
            scene_inputs.read_scene(collection)
