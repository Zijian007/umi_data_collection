"""场景参数用户分组与内部规范化字段的契约测试。"""

import unittest

from umi.config.scene_parameters import flatten, nested_snapshot


class SceneParameterSchemaTests(unittest.TestCase):
    def test_nested_defaults_flatten_to_internal_fields(self):
        values = flatten({
            'schema_version': 2,
            'localization': {
                'bidirectional': False,
                'minimum_fraction': 0.6,
            },
            'fusion': {
                'environment_tags': {'method': 'joint'},
            },
            'measurements': {
                'gripper_tags': {'roi_px': [1, 2, 3, 4]},
            },
        }, require_schema=True)
        self.assertEqual(values['schema_version'], 1)
        self.assertFalse(values['localization_bidirectional'])
        self.assertEqual(values['minimum_localization_fraction'], 0.6)
        self.assertEqual(values['tag_distance_roi_px'], [1, 2, 3, 4])
        self.assertEqual(values['tag_fusion_method'], 'joint')

    def test_nested_snapshot_uses_user_paths(self):
        snapshot = nested_snapshot({
            'localization_bidirectional': False,
            'tag_mapping_scale_enabled': True,
            'tag_fusion_method': 'joint',
            'gripper_width_diagnostics': True,
        })
        self.assertEqual(snapshot['schema_version'], 2)
        self.assertFalse(snapshot['localization']['bidirectional'])
        self.assertTrue(snapshot['mapping']['environment_tags']['scale']['enabled'])
        self.assertEqual(snapshot['fusion']['environment_tags']['method'], 'joint')
        self.assertTrue(snapshot['diagnostics']['gripper_width'])

    def test_unknown_nested_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown|未知'):
            flatten({'schema_version': 2, 'localization': {'unknown': True}}, require_schema=True)


if __name__ == '__main__':
    unittest.main()
