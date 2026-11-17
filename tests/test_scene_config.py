"""场景元数据与专家算法配置的边界。"""
from pathlib import Path
import tempfile
import unittest

from software.infrastructure.file_store import read_yaml
from software.scene.management import batch_inputs, episode_inputs, kit_inputs, scene_inputs
from software.scene import layout
from software.scene.configuration import SceneOverrides
from software.scene.management import scene_inputs
from software.scene.pipeline import map_lifecycle
from software.infrastructure.file_store import write_yaml


class SceneConfigurationTests(unittest.TestCase):
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
        self.write_scene(dict(layout='shared', schema_version=2))
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'schema_version: 3'):
            scene_inputs.read_scene(self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_build_requires_a_batch_instead_of_a_standalone_scene(self):
        self.write_scene(dict(layout='shared'))
        with self.assertRaisesRegex(ValueError, '建图需要采集批次'):
            map_lifecycle.build_scene(self.path)

    def test_inline_algorithm_settings_rejected_regardless_of_comments_and_order(self):
        self.path.write_text('schema_version: 1\nscene:\n  layout: shared\n  trajectory:\n    video:\n      size_px: 960\n')
        with self.assertRaisesRegex(ValueError, '不承载算法参数'):
            scene_inputs.read_scene(self.path)
        self.path.write_text('schema_version: 1\nscene:\n  layout: shared\n  trajectory:\n    video:\n      size_px: 960\n')
        with self.assertRaisesRegex(ValueError, '不承载算法参数'):
            scene_inputs.read_scene(self.path)

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
            from software.scene.management import batches
            batches._declaration(batch)
        (batch / 'batch.yaml').unlink()
        task = batch / 'take_001'
        task.mkdir()
        write_yaml(task / 'task.yaml', dict(instruction='old'))
        with self.assertRaisesRegex(ValueError, 'task.yaml 已移除'):
            layout.task_configuration(task)

    def test_synchronization_accepts_user_facing_shorthand(self):
        audio = episode_inputs.read_synchronization('audio_auto', ('left', 'right'))
        self.assertEqual(audio['method'], 'audio_auto')
        self.assertEqual(audio['offsets_s'], {'left': 0., 'right': 0.})
        simultaneous = episode_inputs.read_synchronization('assumed_simultaneous', ('left', 'right'))
        self.assertEqual(simultaneous['method'], 'assumed_simultaneous')

    def test_collection_layout_is_user_semantic_and_resolves_to_internal_mode(self):
        collection = self.root / '20261114' / 'collection.yaml'
        collection.parent.mkdir()
        write_yaml(collection, dict(schema_version=1, kit='rig', scene=dict(layout='auto')))
        self.assertEqual(scene_inputs.read_scene(collection).mode, 'auto')
        write_yaml(collection, dict(schema_version=1, kit='rig', scene=dict(layout='per_arm', map_reference_arm='left')))
        self.assertEqual(scene_inputs.read_scene(collection).mode, 'dual')

    def test_auto_layout_handles_epoch_directory_shapes(self):
        from software.scene.management import batches
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
        board = scene_inputs.ROOT / 'hardware/calibration_targets/environment_poster.json'
        if not board.is_file():
            self.skipTest('environment poster fixture unavailable')
        for index, tags in enumerate((dict(layout=str(board), detection=True, scale=False, fusion=True),
                                      dict(layout=str(board)))):
            with self.subTest(tags=tags):
                collection = self.root / f'legacy_{index}' / 'collection.yaml'
                collection.parent.mkdir()
                write_yaml(collection, dict(schema_version=1, kit='rig', scene=dict(
                    layout='shared', map_reference_arm='left', tags=tags)))
                with self.assertRaisesRegex(ValueError, 'scene.tags'):
                    scene_inputs.read_scene(collection)

    def test_expert_config_overrides_collection_tag_switches(self):
        board = scene_inputs.ROOT / 'hardware/calibration_targets/environment_poster.json'
        if not board.is_file():
            self.skipTest('environment poster fixture unavailable')
        collection = self.root / 'expert' / 'collection.yaml'
        collection.parent.mkdir()
        write_yaml(collection, dict(schema_version=1, kit='rig', scene=dict(
            layout='shared', map_reference_arm='left', tags=dict(
                enabled=True, layout=str(board), fusion=dict(enabled=False)))))
        expert = self.root / 'expert.yaml'
        write_yaml(expert, dict(parameters=dict(fusion=dict(environment_tags=dict(enabled=True)))))
        scene = scene_inputs.read_scene(collection, overrides=SceneOverrides.load(expert))
        self.assertTrue(scene.parameters.tag_fusion_enabled)
        # 覆盖之后仍要求总开关：tags.enabled=false 时专家配置不能单独打开融合。
        write_yaml(collection, dict(schema_version=1, kit='rig', scene=dict(
            layout='shared', map_reference_arm='left', tags=dict(enabled=False, layout=str(board)))))
        with self.assertRaisesRegex(ValueError, 'scene.tags.enabled'):
            scene_inputs.read_scene(collection, overrides=SceneOverrides.load(expert))

    def test_environment_tag_policy_uses_canonical_enabled_fields(self):
        board = scene_inputs.ROOT / 'hardware/calibration_targets/environment_poster.json'
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
        board = scene_inputs.ROOT / 'hardware/calibration_targets/environment_poster.json'
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
        board = scene_inputs.ROOT / 'hardware/calibration_targets/environment_poster.json'
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
