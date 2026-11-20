import unittest

from software.scene.management.collection_schema import normalize_collection


class CollectionSchemaTests(unittest.TestCase):
    def base(self):
        return dict(
            kit='kit_001',
            recording=dict(arms=['left', 'right']),
            scene=dict(description='shelf'),
            environment_poster=dict(
                layout_path='poster.json', detection=dict(enabled=True),
                mapping=dict(enabled=False, scale_correction=dict(enabled=False)),
                task_constraint=dict(enabled=True, method='staged')),
            reconstruction=dict(backend='auto'))

    def test_normalizes_map_poster_reconstruction_and_task_override(self):
        document = self.base()
        document['reconstruction']['base_tag'] = dict(enabled=False)
        document.update(map=dict(layout='per_arm', reference_arm='right',
                                 selection=dict(policy='latest_prior', version='map-v1')),
                        tasks=dict(take_001=dict(
                            arms=['right'], synchronization='assumed_simultaneous',
                            map=dict(epoch='epoch_001'),
                            reconstruction=dict(backend='poster_vio',
                                                base_tag=dict(enabled=True)))))
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
        document['reconstruction']['base_tag'] = dict(enabled=False)
        document['tasks'] = dict(take_001=dict(reconstruction=dict(backend='poster_vio')))
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

    def test_rejects_old_schema_and_conflicting_task_map_selectors(self):
        with self.assertRaisesRegex(ValueError, '当前字段布局'):
            normalize_collection(dict(schema_version=1))
        document = self.base()
        document['tasks'] = dict(take_001=dict(map=dict(epoch='e', version='v')))
        with self.assertRaisesRegex(ValueError, '不能同时'):
            normalize_collection(document)


if __name__ == '__main__':
    unittest.main()
