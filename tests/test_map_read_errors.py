"""地图读取失败分类；资产损坏允许回退，程序异常不得变成地图不可用。"""

import json
from copy import deepcopy
import unittest
from unittest.mock import patch

from umi.workflows.scene.errors import MapUnavailableError, MapCompatibilityError, fallback_record
from umi.workflows.scene.maps import repository as lifecycle
from umi.workflows.scene.maps import repository
from umi.workflows.scene.maps import repository as map_repository


class MapManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(name='lab', mode='shared_front', version='test-v1', frame_id='lab/test-v1',
            inputs=dict(definition={}, engines={}), assets={'maps/front.osa': 'a'*64},
            maps=dict(shared_front=dict(archive='maps/front.osa', index=0,
                placement=dict(scale=1., T_scene_metric_map=[
                    [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]))))

    def test_validation_needs_no_files_or_engine_and_does_not_modify_input(self):
        original = deepcopy(self.manifest)
        with patch.object(repository.artifact_identity, 'engine_identity', side_effect=AssertionError('engine accessed')), \
                patch.object(repository, 'validate_assets', side_effect=AssertionError('files accessed')):
            repository.validate_manifest(self.manifest, version='test-v1')
        self.assertEqual(self.manifest, original)

    def test_invalid_structure_and_geometry_raise_known_map_error(self):
        cases = [None, [], {}, dict(self.manifest, inputs=[]), dict(self.manifest, assets=[]),
                 dict(self.manifest, frame_id=''), dict(self.manifest, version='other'),
                 dict(self.manifest, maps=dict(shared_front={})),
                 dict(self.manifest, maps=dict(shared_front=dict(archive='maps/front.osa', index=True,
                                                                placement=dict(scale=1.))))]
        for scale in (0., -1., float('nan'), True):
            invalid = deepcopy(self.manifest)
            invalid['maps']['shared_front']['placement']['scale'] = scale
            cases.append(invalid)
        invalid = deepcopy(self.manifest)
        invalid['maps']['shared_front']['placement']['T_scene_metric_map'][0][0] = -1.
        cases.append(invalid)
        for document in cases:
            with self.subTest(document=document), self.assertRaises(MapUnavailableError):
                repository.validate_manifest(document, version='test-v1')


class MapReadErrorTests(unittest.TestCase):
    def setUp(self):
        import test_scene
        self.fixture = test_scene.SceneWorkflowTests('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.build_scene()
        self.directory, self.manifest = map_repository.load_published(self.fixture.scene)

    def load(self):
        return repository.load_published(self.fixture.scene, version=self.manifest['version'])

    def test_missing_file_preserves_original_os_error(self):
        (self.directory / 'manifest.json').unlink()
        with self.assertRaises(MapUnavailableError) as caught:
            self.load()
        self.assertIsInstance(caught.exception.__cause__, FileNotFoundError)
        self.assertEqual(fallback_record(caught.exception)['error_type'], 'FileNotFoundError')

    def test_malformed_json_and_incomplete_manifest_are_known_asset_errors(self):
        for text in ('{broken', '[]', '{}', json.dumps(dict(self.manifest, assets={'bad': None}))):
            with self.subTest(text=text[:30]):
                (self.directory / 'manifest.json').write_text(text)
                with self.assertRaises(MapUnavailableError):
                    self.load()

    def test_engine_incompatibility_has_a_distinct_category(self):
        with patch.object(repository.artifact_identity, 'engine_identity', return_value={'orb_library': 'changed'}):
            with self.assertRaises(MapCompatibilityError) as caught:
                self.load()
        self.assertEqual(fallback_record(caught.exception)['category'], 'map_incompatible')

    def test_corrupt_nested_map_entries_are_data_errors_rather_than_program_key_errors(self):
        role = next(iter(self.manifest['maps']))
        original = self.manifest['maps'][role]
        for item in ({}, dict(original, archive='unregistered.osa'),
                     dict(original, placement=dict(scale=-1)),
                     dict(original, placement=dict(scale=1, T_scene_metric_map=None))):
            broken = dict(self.manifest, maps={role: item})
            (self.directory / 'manifest.json').write_text(json.dumps(broken))
            with self.subTest(item=item), self.assertRaises(MapUnavailableError):
                self.load()

    def test_unexpected_validation_errors_propagate(self):
        for error in (KeyError('bug'), RuntimeError('bug'), OSError('engine unavailable')):
            with self.subTest(error=error), patch.object(repository.artifact_identity, 'engine_identity', side_effect=error):
                with self.assertRaises(type(error)):
                    self.load()
