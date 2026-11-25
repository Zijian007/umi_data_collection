import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from umi.config import user_settings
from umi.config.collection_schema import normalize_collection
from umi.infra.file_store import read_yaml, write_yaml


class CollectionSchemaTests(unittest.TestCase):
    def base(self):
        return dict(
            kit='kit_001',
            recording=dict(arms=['left', 'right']),
            scene=dict(description='shelf'),
            environment_poster=dict(
                layout_path='poster.json', detection=dict(enabled=True),
                mapping=dict(enabled=False, scale_correction=dict(enabled=False))),
            backend=dict(mode='auto', scene_map=dict(poster_constraint=dict(enabled=True, method='staged'))))

    def test_reads_reconstruction_defaults_from_user_yaml_and_honors_explicit_overrides(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'user.yaml'
            options = read_yaml(user_settings.DEFAULT_PATH)
            options['reconstruction']['backend']['mode'] = 'poster_vio'
            options['reconstruction']['map'] = dict(layout='per_arm', reference_arm='right')
            write_yaml(path, options)
            with patch.object(user_settings, 'current', return_value=user_settings.load(path)):
                document = dict(kit='kit', recording=dict(arms=['left']), mapping={})
                inherited = normalize_collection(document)
                overridden = normalize_collection({**document, 'backend': {'mode': 'auto'},
                                                   'map': {'layout': 'shared'}})
        self.assertEqual(inherited['reconstruction']['backend'], 'poster_vio')
        self.assertEqual(inherited['scene']['layout'], 'per_arm')
        self.assertEqual(inherited['scene']['map_reference_arm'], 'right')
        self.assertEqual(overridden['reconstruction']['backend'], 'auto')
        self.assertEqual(overridden['scene']['layout'], 'shared')
        self.assertEqual(overridden['scene']['map_reference_arm'], 'right')
    def test_normalizes_map_poster_reconstruction_and_task_override(self):
        document = self.base()
        document['backend']['poster_vio'] = dict(base_tag=dict(enabled=False))
        document.update(map=dict(layout='per_arm', reference_arm='right',
                                 selection=dict(policy='latest_prior', version='map-v1')),
                        tasks=dict(take_001=dict(
                            arms=['right'], synchronization='assumed_simultaneous',
                            map=dict(epoch='epoch_001'),
                            backend=dict(mode='poster_vio', poster_vio=dict(base_tag=dict(enabled=True))))))
        normalized = normalize_collection(document)
        self.assertNotIn('schema_version', normalized)
        self.assertEqual(normalized['scene']['layout'], 'per_arm')
        self.assertEqual(normalized['scene']['map_reference_arm'], 'right')
        self.assertEqual(normalized['map_version'], 'map-v1')
        self.assertEqual(normalized['tasks']['take_001']['expected_arms'], ['right'])
        self.assertEqual(normalized['tasks']['take_001']['map_epoch'], 'epoch_001')
        self.assertEqual(normalized['tasks']['take_001']['reconstruction']['backend'], 'poster_vio')
        self.assertFalse(normalized['reconstruction']['base_tag']['enabled'])
        self.assertTrue(normalized['tasks']['take_001']['reconstruction']['base_tag']['enabled'])

    def test_task_base_tag_switch_can_inherit_batch_setting(self):
        document = self.base()
        document['backend']['poster_vio'] = dict(base_tag=dict(enabled=False))
        document['tasks'] = dict(take_001=dict(backend=dict(mode='poster_vio')))
        normalized = normalize_collection(document)
        self.assertNotIn('base_tag', normalized['tasks']['take_001']['reconstruction'])

    def test_recording_arms_are_required_and_tasks_cannot_expand_them(self):
        document = self.base()
        document['recording'] = {}
        with self.assertRaisesRegex(ValueError, 'recording.arms'):
            normalize_collection(document)
        document['recording'] = dict(arms=['right'])
        document['tasks'] = dict(take_001=dict(arms=['left']))
        with self.assertRaisesRegex(ValueError, 'tasks.take_001.arms'):
            normalize_collection(document)
        for value in (None, [], ['right', 'right'], [['right']]):
            document['tasks']['take_001']['arms'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_collection(document)

    def test_batch_audio_sync_requires_a_boolean_when_specified(self):
        document = self.base()
        self.assertNotIn('audio_sync_enabled', normalize_collection(document))
        for enabled in (True, False):
            document['recording']['audio_sync'] = dict(enabled=enabled)
            self.assertIs(normalize_collection(document)['audio_sync_enabled'], enabled)
        for enabled in (None, 1, 'false'):
            document['recording']['audio_sync'] = dict(enabled=enabled)
            with self.subTest(enabled=enabled), self.assertRaisesRegex(ValueError, r'recording\.audio_sync\.enabled'):
                normalize_collection(document)

    def test_rejects_old_schema_and_conflicting_task_map_selectors(self):
        with self.assertRaisesRegex(ValueError, '当前字段布局'):
            normalize_collection(dict(schema_version=1))
        document = self.base()
        document['tasks'] = dict(take_001=dict(map=dict(epoch='e', version='v')))
        with self.assertRaisesRegex(ValueError, '不能同时'):
            normalize_collection(document)

    def test_rejects_retired_backend_and_poster_declarations(self):
        document = self.base()
        document['reconstruction'] = dict(backend='auto')
        with self.assertRaisesRegex(ValueError, '未知字段.*reconstruction'):
            normalize_collection(document)
        document.pop('reconstruction')
        document['environment_poster']['task_constraint'] = dict(enabled=True)
        with self.assertRaisesRegex(ValueError, 'environment_poster 含未知字段'):
            normalize_collection(document)
        document['environment_poster'].pop('task_constraint')
        document['tasks'] = dict(take_001=dict(reconstruction={}))
        with self.assertRaisesRegex(ValueError, r'tasks\.take_001 含未知字段'):
            normalize_collection(document)

    def test_backend_reports_invalid_options_at_the_public_paths(self):
        document = dict(kit='kit_001', recording=dict(arms=['right']), scene={})
        for section, options, field in (
            ('scene_map', dict(poster_constraint=dict(method='other')), 'poster_constraint.method'),
            ('poster_vio', dict(base_tag=dict(enabled=None)), 'base_tag.enabled'),
            ('poster_vio', dict(initialization=dict(max_observation_gap_s=float('inf'))), 'max_observation_gap_s'),
            ('poster_independent', dict(max_position_support_gap_s=False), 'max_position_support_gap_s'),
            ('poster_independent', dict(max_gravity_difference_deg=181), 'max_gravity_difference_deg'),
            ('poster_edge_trim', dict(min_missing_duration_s=0), 'min_missing_duration_s'),
        ):
            document['backend'] = {'mode': 'auto', section: options}
            with self.subTest(section=section, field=field), self.assertRaisesRegex(ValueError, field):
                normalize_collection(document)

    def test_rejects_invalid_backend_declarations(self):
        for field in ('backend', 'reconstruction'):
            for record in (dict(mode=True), dict(mode='auto', extra=True), 'auto'):
                document = dict(kit='kit_001', recording=dict(arms=['right']), scene={})
                document[field] = record
                with self.subTest(field=field, record=record), self.assertRaisesRegex(ValueError, field):
                    normalize_collection(document)


if __name__ == '__main__':
    unittest.main()
