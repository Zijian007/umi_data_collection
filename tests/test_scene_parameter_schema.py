"""场景参数用户分组与内部规范化字段的契约测试。"""

import unittest

from umi.config.scene_parameters import flatten, nested_snapshot


class SceneParameterSchemaTests(unittest.TestCase):
    def test_project_defaults_require_explicit_version_but_overlays_can_omit_it(self):
        document = {'localization': {'bidirectional': False}}
        self.assertFalse(flatten(document)['localization_bidirectional'])
        with self.assertRaisesRegex(ValueError, 'schema_version'):
            flatten(document, require_schema=True)

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

    def test_renamed_observation_thresholds_keep_the_normalized_values(self):
        legacy = dict(mapping=dict(environment_tags=dict(minimum_frames=7, minimum_motion_m=.2)),
                      poster_vio=dict(minimum_tag_frames=9))
        current = dict(mapping=dict(poster_scale=dict(minimum_scale_observation_times=7, minimum_motion_m=.2)),
                       poster_vio=dict(minimum_training_observation_times=9))
        with self.assertRaisesRegex(ValueError, '未知字段'):
            flatten(legacy)
        self.assertEqual(flatten(current), flatten(legacy, historical=True))
        self.assertEqual(nested_snapshot(flatten(legacy, historical=True))['mapping'], current['mapping'])

    def test_backend_gap_overrides_the_shared_default_regardless_of_yaml_order(self):
        document = dict(poster_independent=dict(maximum_position_support_gap_s=4.),
                        poster_acceptance=dict(maximum_position_support_gap_s=3.))
        origins = {}
        values = flatten(document, gap_sources=origins)
        self.assertEqual(values['poster_maximum_position_support_gap_s'], 4.)
        self.assertEqual(values['poster_vio_maximum_position_support_gap_s'], 3.)
        self.assertIn(':poster_independent.', origins['poster_maximum_position_support_gap_s'])
        self.assertIn(':poster_acceptance.', origins['poster_vio_maximum_position_support_gap_s'])


if __name__ == '__main__':
    unittest.main()
