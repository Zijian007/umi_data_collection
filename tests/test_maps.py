"""场景地图、地图轮次、尺度、重定位与缓存。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_scene as scene_tests
from scipy.spatial.transform import Rotation

from umi.apps.viewer import maps
from umi.config import loader as config
from umi.config.loader import load
from umi.infra import provenance
from umi.infra.file_store import read_yaml, write_yaml
from umi.perception import atlas
from umi.perception import slam as visual_slam
from umi.perception.models import LensVideo, MapAtlas, ProcessingError
from umi.perception.poster import detection as poster_detection
from umi.perception.poster import layout as tag_board
from umi.reconstruction import mapping
from umi.reconstruction import mapping as slam
from umi.reconstruction.estimation.fixed_map import geometry, poster
from umi.reconstruction.estimation.fixed_map import geometry as relocalization
from umi.reconstruction.estimation.fixed_map.parameters import MapPlacementParameters, PosterScaleParameters
from umi.reconstruction.mapping import correct_poster_placement
from umi.reconstruction.segment import map_data
from umi.scene.collection import batch as batch_inputs
from umi.scene.collection import configuration as scene_inputs
from umi.scene.collection.configuration import SceneOverrides, load_scene_parameters
from umi.scene.errors import MapUnavailableError, fallback_record
from umi.scene.maps import build as map_build
from umi.scene.maps import versions as map_repository
from umi.scene.maps import versions as repository
from umi.scene.maps.epochs import (
    MapEpoch,
    MapEpochError,
    TaskCollection,
    draft_plan,
    read_plan,
    resolve_task_map,
    select_many,
    select_map_epoch,
)
from umi.store import result_io as asset_identity
from umi.store.cache import StageCache, preparation_inputs

# 地图导出、相似变换和目录边界；不运行 SLAM。


class SharedMapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene = self.root / 'lab'
        self.directory = self.scene / 'tasks/take/left'
        self.directory.mkdir(parents=True)
        self.path = self.scene / '.pipeline/maps/versions/abcdef-12345678/map.json'
        self.path.parent.mkdir(parents=True)
        self.data = dict(kind='umi_map_view', schema_version=1, maps=[])
        self.path.write_text(json.dumps(self.data))
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.summary = dict(mode='scene', scene=dict(name='lab', version=self.path.parent.name,
            frame_id='lab/'+self.path.parent.name), map=dict(path=os.path.relpath(self.path, self.directory), sha256=digest))
        self.manifest = dict(kind='umi_scene', name='lab', version=self.path.parent.name,
            frame_id=self.summary['scene']['frame_id'], assets={'map.json': digest})
        (self.path.parent / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_shared_map_loads_without_a_copy_or_exporter(self):
        with patch.object(maps, '_convert') as convert:
            self.assertEqual(maps.load(self.directory, self.summary), self.data)
        convert.assert_not_called()
        self.assertFalse((self.directory / 'details/map.json').exists())

    def test_shared_map_hash_change_is_rejected(self):
        self.path.write_text(json.dumps(self.data)+'\n')
        with self.assertRaisesRegex(ValueError, '内容已变化'):
            maps.load(self.directory, self.summary)

    def test_shared_map_frame_mismatch_is_rejected(self):
        self.summary['scene']['frame_id'] = 'other/frame'
        with self.assertRaisesRegex(ValueError, '坐标系'):
            maps.load(self.directory, self.summary)

    def test_shared_map_outside_current_scene_is_rejected(self):
        self.summary['map']['path'] = os.path.relpath(self.root / 'other/.pipeline/maps/versions/abcdef-12345678/map.json', self.directory)
        with self.assertRaisesRegex(ValueError, '越界'):
            maps.load(self.directory, self.summary)

    def test_shared_map_symlink_cannot_leave_scene(self):
        outside = self.root / 'outside.json'
        outside.write_text(json.dumps(self.data))
        self.path.unlink()
        self.path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, '越界'):
            maps.load(self.directory, self.summary)


class SimilarityTests(unittest.TestCase):
    def test_recovers_metric_transform_despite_outliers(self):
        source = np.random.default_rng(4).normal(size=(80, 3))
        rotation = Rotation.from_euler('xyz', [.4, -.3, 1.2]).as_matrix()
        target = 1.7*source @ rotation.T+[.1, -.2, 1.]
        target[-4:] += [8., 9., 10.]
        fit = map_data.fit_similarity(source, target)
        self.assertEqual(fit['status'], 'aligned')
        self.assertAlmostEqual(fit['scale'], 1.7, places=9)
        np.testing.assert_allclose(fit['rotation'], rotation, atol=1e-9)
        np.testing.assert_allclose(fit['translation'], [.1, -.2, 1.], atol=1e-9)
        self.assertLess(fit['rmse_m'], 1e-9)
        self.assertEqual(fit['pairs'], 76)

    def test_degenerate_and_nonfinite_motion_is_not_overlaid(self):
        for points in (np.zeros((10, 3)), np.tile(np.arange(10)[:, None], (1, 3)),
                       np.full((10, 3), np.nan), np.zeros((3, 3))):
            with self.subTest(points=points.shape):
                self.assertEqual(map_data.fit_similarity(points, points)['status'], 'unaligned')

    def test_large_inconsistent_motion_is_not_overlaid(self):
        random = np.random.default_rng(9)
        fit = map_data.fit_similarity(random.normal(size=(100, 3)), random.normal(size=(100, 3)))
        self.assertEqual(fit['status'], 'unaligned')


class MapExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root/'result'
        self.attempt = self.output/'debug/reconstruction/try_2'
        self.archive = self.attempt/'front/map/map.osa'
        self.archive.parent.mkdir(parents=True)
        self.archive.write_bytes(b'original atlas')
        self.executable, self.vocabulary = self.root/'exporter', self.root/'vocabulary'
        for path in (self.executable, self.vocabulary):
            path.write_text('dependency')
        angles = np.linspace(0, 3, 30)
        self.points = np.c_[np.sin(angles), np.cos(angles), angles*.1]
        self.raw = dict(schema_version=1, maps=[dict(index=0, id=3, points=self.points.tolist(),
            keyframes=[dict(id=7, time=1., position=self.points[10].tolist(), quaternion=[0, 0, 0, 1])])])
        locations = self.attempt/'front/loc_m0/frames.csv'
        locations.parent.mkdir()
        locations.write_text('frame_idx,t,ok,tx,ty,tz\n'+''.join(
            f'{i},{i*.1:.9f},1,{p[0]},{p[1]},{p[2]}\n' for i,p in enumerate(self.points)))
        (self.output/'details').mkdir()
        (self.output/'details/traj_camera.txt').write_text(''.join(
            f'{i*.1:.9f} {2*p[0]+1} {2*p[1]} {2*p[2]} 0 0 0 1\n' for i,p in enumerate(self.points)))

    def export(self):
        def run(command, **kwargs):
            Path(command[-1]).write_text(json.dumps(self.raw))
            return SimpleNamespace(returncode=0)
        with patch.object(atlas, 'exporter_paths', return_value=(self.executable, self.vocabulary)), \
                patch.object(atlas.subprocess, 'run', side_effect=run):
            return map_data.export_attempt(self.attempt, self.output)

    def test_export_aligns_points_and_keeps_original_archive_unchanged(self):
        data = maps.validate(self.export())
        item = data['maps'][0]
        self.assertEqual(item['coordinates'], 'trajectory')
        np.testing.assert_allclose(item['points'], self.points*2+[1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(item['keyframes'][0]['position'], self.points[10]*2+[1, 0, 0], atol=1e-8)
        self.assertEqual(self.archive.read_bytes(), b'original atlas')

    def test_missing_localization_retains_raw_map_instead_of_inventing_alignment(self):
        (self.attempt/'front/loc_m0/frames.csv').unlink()
        item = self.export()['maps'][0]
        self.assertEqual(item['coordinates'], 'map')
        np.testing.assert_allclose(item['points'], self.points, atol=1e-7)

    def test_scene_export_removes_empty_log_and_preserves_diagnostics(self):
        placement = dict(scale=2., T_scene_metric_map=np.eye(4).tolist(), evidence=dict(rms_m=0.))
        for message in ('', 'export diagnostic\n'):
            with self.subTest(message=message):
                directory = self.root / ('empty_log' if not message else 'diagnostic_log')
                directory.mkdir()
                def run(command, log, **kwargs):
                    Path(command[-1]).write_text(json.dumps(self.raw))
                    log.write_text(message)
                with patch.object(atlas, 'exporter_paths', return_value=(self.executable, self.vocabulary)), \
                        patch.object(atlas.runtime, 'run_logged', side_effect=run):
                    exported = slam.export_map(self.archive, 0, placement, directory)
                np.testing.assert_allclose(exported['points'], self.points*2, atol=1e-7)
                self.assertFalse((directory / 'raw_atlas.json').exists())
                self.assertEqual((directory / 'export.log').exists(), bool(message))
                if message:
                    self.assertEqual((directory / 'export.log').read_text(), message)
        self.assertEqual(self.archive.read_bytes(), b'original atlas')

    def test_failed_scene_export_keeps_log(self):
        def run(command, log, **kwargs):
            log.write_text('export failed\n')
            raise RuntimeError('export failed')
        with patch.object(atlas, 'exporter_paths', return_value=(self.executable, self.vocabulary)), \
                patch.object(atlas.runtime, 'run_logged', side_effect=run), self.assertRaisesRegex(RuntimeError, 'export failed'):
            slam.export_map(self.archive, 0, {}, self.root)
        self.assertEqual((self.root / 'export.log').read_text(), 'export failed\n')

    def test_only_selected_attempt_is_read(self):
        self.assertEqual(map_data.attempt_directory(self.output, {'chosen_try': 2}), self.attempt)
        self.assertIsNone(map_data.attempt_directory(self.output, {'chosen_try': 1}))

    def test_symbolic_links_cannot_read_maps_outside_result(self):
        self.archive.unlink()
        outside = self.root/'outside.osa'
        outside.write_bytes(b'outside')
        self.archive.symlink_to(outside)
        with patch.object(atlas.subprocess, 'run') as execute, self.assertRaises(map_data.MapDataError):
            map_data.export_attempt(self.attempt, self.output)
        execute.assert_not_called()

    def test_prepared_map_loads_without_exporter_and_rejects_outside_symlink(self):
        data = self.export()
        prepared = self.output/'details/map.json'
        prepared.parent.mkdir(exist_ok=True)
        prepared.write_text(json.dumps(data))
        with patch.object(maps, '_convert') as convert:
            self.assertEqual(maps.load(self.output, {}), data)
        convert.assert_not_called()
        prepared.unlink()
        outside = self.root/'outside.json'
        outside.write_text(json.dumps(data))
        prepared.symlink_to(outside)
        with self.assertRaises(map_data.MapDataError):
            maps.load(self.output, {})

    def test_merged_localizations_are_filtered_by_map_id(self):
        locations = self.attempt/'front/loc_m0/frames.csv'
        # 用明确的两张图构造合并表，另一张图的坐标不可参与拟合。
        locations.write_text('t,ok,tx,ty,tz,map\n0,1,1,2,3,m0\n0.1,1,4,5,6,m1\n')
        source, target = map_data._correspondences(locations, self.output/'details/traj_camera.txt', 0)
        self.assertEqual(source, [[1., 2., 3.]])
        self.assertEqual(len(target), 1)

    def test_invalid_geometry_is_rejected_before_serving_browser(self):
        data = self.export()
        data['maps'][0]['points'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            maps.validate(data)

    def test_invalid_map_transform_is_rejected_before_drawing_axes(self):
        for key, value in [('scale', 0), ('rotation', [[1, 0, 0]]), ('translation', [0, 0])]:
            with self.subTest(key=key):
                data = self.export()
                data['maps'][0]['alignment'][key] = value
                with self.assertRaisesRegex(ValueError, '地图坐标变换无效'):
                    maps.validate(data)


class MapEpochTests(unittest.TestCase):
    def setUp(self):
        self.epochs = [
            MapEpoch('epoch_001', 'map_a', 0),
            MapEpoch('epoch_002', 'map_b', 3),
        ]

    def test_latest_map_at_or_before_task_sequence(self):
        self.assertEqual(select_map_epoch(TaskCollection('take_001', 1), self.epochs)['map_version'], 'map_a')
        self.assertEqual(select_map_epoch(TaskCollection('take_002', 3), self.epochs)['map_version'], 'map_b')
        self.assertEqual(select_map_epoch(TaskCollection('take_003', 8), self.epochs)['epoch'], 'epoch_002')

    def test_explicit_override_is_provenanced(self):
        result = select_map_epoch(TaskCollection('take_001', 1), self.epochs, override='map_b')
        self.assertEqual(result['selection'], 'explicit_override')
        self.assertEqual(result['reason'], 'task_map_version')
        self.assertEqual(result['status'], 'ready')

    def test_batch_default_is_resolved_by_shared_resolver(self):
        result = resolve_task_map(None, 'take_001', batch_map_version='map_a')
        self.assertEqual(result['map_version'], 'map_a')
        self.assertEqual(result['selection'], 'batch_default')

    def test_explicit_epoch_override_is_provenanced(self):
        result = select_map_epoch(TaskCollection('take_001', 1), self.epochs,
                                  epoch_override='epoch_002')
        self.assertEqual(result['epoch'], 'epoch_002')
        self.assertEqual(result['map_version'], 'map_b')
        self.assertEqual(result['selection'], 'explicit_epoch')

    def test_rejects_task_before_first_map_and_unknown_override(self):
        with self.assertRaises(MapEpochError):
            select_map_epoch(TaskCollection('take_000', 0), [MapEpoch('epoch_002', 'map_b', 3)])
        with self.assertRaises(MapEpochError):
            select_map_epoch(TaskCollection('take_001', 1), self.epochs, override='map_missing')

    def test_rejects_duplicate_sequences_and_tasks(self):
        with self.assertRaises(MapEpochError):
            select_many([TaskCollection('take_001', 1)], [MapEpoch('a', 'map_a', 0), MapEpoch('b', 'map_b', 0)])
        with self.assertRaises(MapEpochError):
            select_many([TaskCollection('take_001', 1)], [MapEpoch('a', 'map_a', 0), MapEpoch('b', 'map_a', 2)])
        with self.assertRaises(MapEpochError):
            select_many([TaskCollection('take_001', 1), TaskCollection('take_001', 2)], self.epochs)

    def test_reads_optional_batch_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map_epochs.yaml'
            write_yaml(path, dict(schema_version=1,
                maps=[dict(epoch='epoch_001', map_version='map_a', sequence=0),
                      dict(epoch='epoch_002', map_version='map_b', sequence=3)],
                tasks=[dict(task='take_001', sequence=1), dict(task='take_002', sequence=4)]))
            plan = read_plan(path)
            self.assertEqual(plan.select('take_002')['map_version'], 'map_b')

    def test_unbuilt_epoch_requires_recording_and_cannot_be_selected(self):
        pending = MapEpoch('epoch_pending', None, 0, 'mapping/epoch_pending/video.insv')
        result = select_map_epoch(TaskCollection('take_001', 1), [pending])
        self.assertEqual(result['status'], 'pending')

    def test_plan_accepts_pending_map_with_source_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map_epochs.yaml'
            write_yaml(path, dict(schema_version=1,
                maps=[dict(epoch='epoch_001', map_version=None, sequence=0,
                           recording='mapping/epoch_001/video.insv')], tasks=[]))
            plan = read_plan(path)
            self.assertEqual(plan.epochs[0].recording, 'mapping/epoch_001/video.insv')

    def test_plan_accepts_dual_epoch_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map_epochs.yaml'
            write_yaml(path, dict(schema_version=1,
                maps=[dict(epoch='epoch_001', map_version=None, sequence=0,
                           recording={'left': 'mapping/left/video.insv',
                                      'right': 'mapping/right/video.insv'})], tasks=[]))
            plan = read_plan(path)
            self.assertEqual(set(plan.epochs[0].recording), {'left', 'right'})

    def test_draft_plan_uses_embedded_time_and_latest_prior_order(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            mapping = dict(path=batch / 'mapping' / 'map_001' / 'video.insv',
                           serial='camera-front', started_at_unix_s=100.)
            tasks = [dict(task='take_001', recordings={
                'left': [dict(path=batch / 'take_001' / 'left' / 'video.insv',
                              serial='camera-left', started_at_unix_s=110.)]}),
                     dict(task='take_002', recordings={
                'left': [dict(path=batch / 'take_002' / 'left' / 'video.insv',
                              serial='camera-left', started_at_unix_s=130.)]})]
            result = draft_plan(batch=batch, mode='shared_front', mapping_records=[mapping],
                                task_records=tasks)
            self.assertEqual(result['status'], 'ready')
            self.assertEqual(result['plan']['maps'][0]['sequence'], 0)
            self.assertEqual([item['sequence'] for item in result['plan']['tasks']], [1, 2])
            self.assertEqual(result['plan']['maps'][0]['recording'], 'mapping/map_001/video.insv')

    def test_draft_plan_refuses_missing_or_tied_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            result = draft_plan(
                batch=batch, mode='shared_front',
                mapping_records=[dict(path=batch / 'mapping' / 'map_001.insv', started_at_unix_s=None)],
                task_records=[dict(task='take_001', recordings={
                    'left': [dict(path=batch / 'take_001' / 'left' / 'video.insv', started_at_unix_s=100.)]})])
            self.assertEqual(result['status'], 'needs_review')
            self.assertTrue(any('开始时间' in reason for reason in result['reasons']))

            tied = draft_plan(
                batch=batch, mode='shared_front',
                mapping_records=[dict(path=batch / 'mapping' / 'map_a.insv', started_at_unix_s=100.)],
                task_records=[dict(task='take_001', recordings={
                    'left': [dict(path=batch / 'take_001' / 'left' / 'video.insv', started_at_unix_s=100.)]})])
            self.assertEqual(tied['status'], 'needs_review')
            self.assertTrue(any('时间相同' in reason for reason in tied['reasons']))

    def test_draft_plan_pairs_dual_mapping_by_arm_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            records = [
                dict(path=batch / 'mapping' / 'epoch_a' / 'left' / 'v.insv',
                     serial='L', started_at_unix_s=100.),
                dict(path=batch / 'mapping' / 'epoch_a' / 'right' / 'v.insv',
                     serial='R', started_at_unix_s=100.),
            ]
            result = draft_plan(batch=batch, mode='dual', mapping_records=records,
                                task_records=[], arm_serials={'left': 'L', 'right': 'R'})
            self.assertEqual(result['status'], 'ready')
            self.assertEqual(set(result['plan']['maps'][0]['recording']), {'left', 'right'})

    def test_draft_plan_rejects_take_before_first_map(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            result = draft_plan(
                batch=batch, mode='shared_front',
                mapping_records=[dict(path=batch / 'mapping' / 'map.insv', started_at_unix_s=200.)],
                task_records=[dict(task='take_001', recordings={
                    'left': [dict(path=batch / 'take_001' / 'left' / 'video.insv', started_at_unix_s=100.)]})])
            self.assertEqual(result['status'], 'needs_review')
            self.assertTrue(any('早于第一张地图' in reason for reason in result['reasons']))


# 地图读取失败分类；资产损坏允许回退，程序异常不得变成地图不可用。


class MapManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(name='lab', mode='shared_front', version='test-v1', frame_id='lab/test-v1',
            inputs=dict(definition={}), assets={'maps/front.osa': 'a'*64},
            maps=dict(shared_front=dict(archive='maps/front.osa', index=0,
                placement=dict(scale=1., T_scene_metric_map=[
                    [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]))))

    def test_validation_needs_no_files_or_engine_and_does_not_modify_input(self):
        original = deepcopy(self.manifest)
        with patch.object(repository, 'validate_assets', side_effect=AssertionError('files accessed')):
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


# 相同验收失败停止重建；变化的轨迹和暂时性引擎错误仍保留重试。


class MapRetryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = scene_tests.SceneWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def failure(self):
        files = list((self.fixture.scene.map_root / 'versions').glob('*/error.json'))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text())

    def test_identical_rejection_stops_after_two_attempts_and_preserves_published_map(self):
        fixture = self.fixture
        fixture.build_scene()
        pointer = fixture.session_path  # 当前地图只绑定在批次清单
        previous = pointer.read_bytes()
        with patch.object(slam, 'build', side_effect=fixture.build) as build, \
                patch.object(mapping, 'metric_alignment', side_effect=ValueError('固定尺度不一致')):
            with self.assertRaisesRegex(ValueError, '停止重复重建'):
                fixture.build_scene(force=True)
        self.assertEqual(build.call_count, 2)
        self.assertEqual([call.kwargs['refresh'] for call in build.call_args_list], [False, True])
        report = self.failure()['retry']
        self.assertEqual(report['stop_reason'], 'repeated_visual_failure')
        first, second = report['attempts']
        self.assertEqual(first['signature'], second['signature'])
        self.assertEqual(second['repeat_of_attempt'], 1)
        self.assertEqual(first['category'], 'map_acceptance')
        self.assertEqual(pointer.read_bytes(), previous)
        directory, manifest = map_repository.load_published(fixture.scene)
        asset_identity.validate_assets(directory, manifest['assets'])

    def test_changed_trajectory_can_recover_on_third_attempt(self):
        fixture = self.fixture
        visits = []
        original_alignment = mapping.metric_alignment

        def candidates(recording, lens, archive, directory, **kwargs):
            observed = fixture.observations(recording, lens)
            visits.append(directory)
            poses = observed.T_reference_camera.copy()
            if len(visits) < 3:
                poses[0, 0, 3] += .01 * len(visits)
            yield 0, replace(observed, T_reference_camera=poses)

        def alignment(*args, **kwargs):
            if len(visits) < 3:
                raise ValueError('固定尺度不一致')
            return original_alignment(*args, **kwargs)

        with patch.object(slam, 'atlas_candidates', side_effect=candidates), \
                patch.object(mapping, 'metric_alignment', side_effect=alignment):
            fixture.build_scene()
        self.assertEqual(len(visits), 3)
        _, manifest = map_repository.load_published(fixture.scene)
        report = manifest['retry']
        self.assertEqual(report['stop_reason'], 'accepted')
        self.assertEqual([attempt['status'] for attempt in report['attempts']], ['failed', 'failed', 'accepted'])
        self.assertNotEqual(report['attempts'][0]['signature'], report['attempts'][1]['signature'])

    def test_changed_rejection_reason_keeps_retry_available(self):
        fixture = self.fixture
        original_alignment = mapping.metric_alignment
        visits = []

        def alignment(*args, **kwargs):
            visits.append(None)
            if len(visits) == 1:
                raise ValueError('运动退化')
            if len(visits) == 2:
                raise ValueError('重力不一致')
            return original_alignment(*args, **kwargs)

        with patch.object(mapping, 'metric_alignment', side_effect=alignment):
            fixture.build_scene()
        _, manifest = map_repository.load_published(fixture.scene)
        first, second, accepted = manifest['retry']['attempts']
        self.assertEqual(first['maps'][0]['observations_sha256'], second['maps'][0]['observations_sha256'])
        self.assertNotEqual(first['signature'], second['signature'])
        self.assertEqual(accepted['status'], 'accepted')

    def test_engine_errors_with_same_message_do_not_count_as_identical_trajectories(self):
        with patch.object(slam, 'build', side_effect=ProcessingError('建图', '引擎暂时退出', return_code=1)) as build:
            with self.assertRaisesRegex(ValueError, '场景地图未通过验收'):
                self.fixture.build_scene()
        self.assertEqual(build.call_count, 3)
        report = self.failure()['retry']
        self.assertEqual(report['stop_reason'], 'attempt_limit')
        self.assertTrue(all(attempt['category'] == 'engine' and 'signature' not in attempt
                            for attempt in report['attempts']))

    def test_maximum_one_attempt_does_not_claim_duplicate_evidence(self):
        fixture = self.fixture
        parameters = replace(fixture.scene.trajectory.values,
                             retry=replace(fixture.scene.trajectory.values.retry, max_tries=1))
        scene = replace(fixture.scene, trajectory=replace(fixture.scene.trajectory, values=parameters))
        with patch.object(mapping, 'metric_alignment', side_effect=ValueError('固定尺度不一致')):
            with self.assertRaisesRegex(ValueError, '场景地图未通过验收'):
                map_build.build_version(scene)
        report = self.failure()['retry']
        self.assertEqual(report['stop_reason'], 'attempt_limit')
        self.assertEqual(len(report['attempts']), 1)

    def test_all_rejected_submaps_participate_in_comparison(self):
        fixture = self.fixture
        recording = fixture.prepared_recordings[fixture.recording_paths['left']]
        observed = fixture.raw['left']
        errors = []
        for change in (0., .001):
            poses = observed.T_reference_camera.copy()
            poses[0, 1, 3] += change
            candidates = [(0, observed), (1, replace(observed, T_reference_camera=poses))]
            with patch.object(slam, 'atlas_candidates', return_value=iter(candidates)), \
                    patch.object(mapping, 'metric_alignment', side_effect=ValueError('固定尺度不一致')):
                with self.assertRaises(mapping.MapAcceptanceError) as caught:
                    mapping.choose_map(recording, 'front', fixture.root / 'map.osa',
                                       fixture.root / f'candidates-{len(errors)}', fixture.scene)
            errors.append(caught.exception)
        self.assertEqual(errors[0].rejected_maps[0], errors[1].rejected_maps[0])
        self.assertNotEqual(errors[0].retry_signature, errors[1].retry_signature)

    def test_no_candidate_data_cannot_prove_duplicate_trajectory(self):
        with patch.object(slam, 'atlas_candidates', return_value=iter(())):
            with self.assertRaises(ValueError):
                self.fixture.build_scene()
        report = self.failure()['retry']
        self.assertEqual(report['stop_reason'], 'attempt_limit')
        self.assertEqual(len(report['attempts']), 3)
        self.assertTrue(all(attempt['signature'] is None for attempt in report['attempts']))

    def test_fixed_scale_failure_reports_actual_video_interval(self):
        fixture = self.fixture
        recording = fixture.prepared_recordings[fixture.recording_paths['left']]
        diagnostics = dict(fixed_map_validation=[dict(status='inconsistent', rms_m=.933, k=[3, 10])])
        error = mapping.MapAlignmentError('固定尺度不一致', diagnostics)
        with patch.object(mapping, 'metric_alignment', side_effect=error):
            with self.assertRaises(mapping.MapAcceptanceError) as caught:
                mapping.choose_map(recording, 'front', fixture.root / 'map.osa', fixture.root / 'failure', fixture.scene)
        expected = [float(recording.video_times_s[index]) for index in (3, 10)]
        self.assertEqual(caught.exception.rejected_maps[0]['video_interval_s'], expected)
        self.assertIn(f'录像 {expected[0]:.2f}–{expected[1]:.2f} s', str(caught.exception))


# 建图内容复用、阶段缓存失效、失败保护和最终地图定位回归。


class CacheStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = StageCache(self.root / 'cache')

    def run_stage(self, name, *, identity=None, fail=False, cache=None, refresh=False):
        output = self.root / name
        selected = cache or self.cache
        with selected.stage('build', '视觉建图', identity or {'video': 'one'}, output,
                            roots=('result',), refresh=refresh) as reused:
            if not reused:
                (output / 'result').mkdir(parents=True)
                (output / 'result/map.osa').write_bytes(b'complete atlas')
                if fail:
                    raise RuntimeError('interrupted')
        return output, reused

    def test_reuse_validates_and_copies_outputs_without_sharing_writable_files(self):
        original, reused = self.run_stage('first')
        self.assertFalse(reused)
        shutil.rmtree(original)
        restored, reused = self.run_stage('second')
        self.assertTrue(reused)
        (restored / 'result/map.osa').write_bytes(b'work changed')
        final, reused = self.run_stage('third')
        self.assertTrue(reused)
        self.assertEqual((final / 'result/map.osa').read_bytes(), b'complete atlas')
        self.assertEqual([entry['status'] for entry in self.cache.timings], ['computed', 'cached', 'cached'])

    def test_linked_stage_shares_read_only_files_and_rejects_writes(self):
        def run_linked(name):
            output = self.root / name
            with self.cache.stage('prepare', '录像准备', {'video': 'one'}, output,
                                  roots=('result',), link=True) as reused:
                if not reused:
                    (output / 'result').mkdir(parents=True)
                    (output / 'result/front.mp4').write_bytes(b'decoded')
            return output / 'result/front.mp4', reused
        first, reused = run_linked('first')
        self.assertFalse(reused)
        second, reused = run_linked('second')
        self.assertTrue(reused)
        cached = next((self.root / 'cache/prepare').rglob('front.mp4'))
        self.assertEqual({first.stat().st_ino, second.stat().st_ino}, {cached.stat().st_ino})
        with self.assertRaises(PermissionError):
            second.write_bytes(b'changed')
        self.assertEqual(cached.read_bytes(), b'decoded')

    def test_corrupted_cache_recomputes_instead_of_returning_damaged_atlas(self):
        self.run_stage('first')
        next((self.root / 'cache').rglob('map.osa')).write_bytes(b'broken')
        _, reused = self.run_stage('second')
        self.assertFalse(reused)
        self.assertTrue(self.run_stage('third')[1])

    def test_interrupted_stage_never_becomes_ready_and_can_be_retried(self):
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.run_stage('failed', fail=True)
        self.assertEqual(list((self.root / 'cache').rglob('ready.json')), [])
        self.assertEqual(self.cache.timings[-1]['status'], 'failed')
        self.assertFalse(self.run_stage('retry')[1])

    def test_force_and_retry_refresh_recompute_but_failed_refresh_preserves_cache(self):
        self.run_stage('first')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.run_stage('failed', fail=True, refresh=True)
        self.assertTrue(self.run_stage('after_failure')[1])
        forced = StageCache(self.root / 'cache', force=True)
        self.assertFalse(self.run_stage('forced', cache=forced)[1])
        self.assertFalse(self.run_stage('retry', refresh=True)[1])

    def test_input_change_and_incomplete_manifest_cannot_hit_cache(self):
        self.run_stage('first')
        self.assertFalse(self.run_stage('changed', identity={'video': 'two'})[1])
        for ready in (self.root / 'cache').rglob('ready.json'):
            ready.unlink()
        self.assertFalse(self.run_stage('partial')[1])


class MapReuseTests(unittest.TestCase):
    def setUp(self):
        self.fixture = scene_tests.SceneWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_same_recording_in_another_batch_reuses_map_and_keeps_original_provenance(self):
        first = self.fixture.build_scene()
        source = self.fixture.recording_paths['left']
        copied = self.fixture.scene.root / 'another_batch/mapping/copy.insv'
        copied.parent.mkdir(parents=True)
        shutil.copyfile(source, copied)
        self.fixture.cameras[copied] = self.fixture.cameras[source]
        moved_scene = replace(self.fixture.scene, recordings={'left': copied})
        with patch.object(slam, 'build', side_effect=AssertionError('same content rebuilt')):
            second = map_build.build_version(moved_scene)
        self.assertEqual(second['status'], 'skipped')
        self.assertEqual(second['version'], first['version'])
        _, manifest = map_repository.load_published(moved_scene)
        self.assertEqual(manifest['inputs']['recordings']['left']['path'], str(source))
        self.assertEqual(manifest['inputs']['recordings']['left']['sha256'], provenance.file_hash(copied))

    def test_changed_content_with_same_size_and_mtime_requires_new_map(self):
        import os
        first = self.fixture.build_scene()
        source = self.fixture.recording_paths['left']
        before = source.stat()
        source.write_bytes(b'LEFT')
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        second = map_build.build_version(self.fixture.scene)
        self.assertNotEqual(first['version'], second['version'])

    def test_tcp_change_keeps_map_but_calibration_and_threshold_changes_invalidate(self):
        first = self.fixture.build_scene()
        tcp = self.fixture.cameras[self.fixture.recording_paths['left']].files['tcp']
        measurement = json.loads(tcp.read_text())
        measurement['T_camera_tcp']['t_m'] = [0.01, 0, 0]
        tcp.write_text(json.dumps(measurement))
        self.assertEqual(self.fixture.build_scene()['version'], first['version'])
        tuned = replace(self.fixture.scene, parameters=replace(self.fixture.scene.parameters,
            maximum_alignment_condition=self.fixture.scene.parameters.maximum_alignment_condition * 2))
        self.assertNotEqual(map_build.build_version(tuned)['version'], first['version'])
        camera = self.fixture.cameras[self.fixture.recording_paths['left']]
        self.fixture.cameras[self.fixture.recording_paths['left']] = replace(camera,
            selected=replace(camera.selected, calibration_id='new-calibration'))
        self.assertNotEqual(self.fixture.build_scene()['version'], first['version'])

    def test_preparation_ignores_alignment_and_orb_parameters_but_not_video_changes(self):
        scene = self.fixture.scene
        source = self.fixture.recording_paths['left']
        camera = self.fixture.cameras[source]
        original = preparation_inputs(source, camera, scene)
        tuned = replace(scene, parameters=replace(scene.parameters, maximum_scale_relative_std=.4),
                        trajectory=replace(scene.trajectory, values=replace(scene.trajectory.values,
                            orb=replace(scene.trajectory.values.orb, features=999))))
        self.assertEqual(preparation_inputs(source, camera, tuned), original)
        resized = replace(scene, trajectory=replace(scene.trajectory, values=replace(scene.trajectory.values,
                          video=replace(scene.trajectory.values.video, size_px=640))))
        self.assertNotEqual(preparation_inputs(source, camera, resized), original)

    def test_preparation_ignores_calibration_and_code(self):
        source = self.fixture.recording_paths['left']
        camera = self.fixture.cameras[source]
        original = preparation_inputs(source, camera, self.fixture.scene)
        recalibrated = replace(camera, selected=replace(camera.selected, calibration_id='new-calibration'))
        self.assertEqual(preparation_inputs(source, recalibrated, self.fixture.scene), original)
        self.assertFalse({'code', 'engines'} & original.keys())


class VisualStageCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, content in [('video.mp4', 'pixels'), ('front.yaml', 'intrinsics'), ('mask.png', 'mask')]:
            (self.root / name).write_text(content)
        self.video = LensVideo('front', self.root / 'video.mp4', self.root / 'front.yaml', self.root / 'mask.png')
        self.recording = SimpleNamespace(
            lens_video=lambda lens: self.video, video_times_s=np.array([0., 1.]),
            context=SimpleNamespace(frame_count=2, parameters=config.load('trajectory').values),
            camera=SimpleNamespace(measurements={'front': {'cam_time_offset_ns': 0}}))
        self.cache = StageCache(self.root / 'cache')
        self.build_patch = patch.object(slam.visual_slam, 'build_maps', side_effect=self.build_engine)
        self.localize_patch = patch.object(slam.visual_slam, 'localize_map', side_effect=self.localize_engine)
        self.build = self.build_patch.start()
        self.localize = self.localize_patch.start()
        self.addCleanup(self.build_patch.stop)
        self.addCleanup(self.localize_patch.stop)

    def build_engine(self, videos, context, directory):
        archive = directory / 'front/map/map.osa'
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b'atlas')
        return (MapAtlas(videos[0], archive),)

    def localize_engine(self, atlas, index, output, *args, **kwargs):
        output.mkdir(parents=True)
        (output / 'frames.csv').write_text('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw\n'
                                           '0,0,2,1,0,0,0,0,0,0,1\n1,1,2,1,1,0,0,0,0,0,1\n')
        (output / 'log.txt').write_text('atlas_map index 0 id 0 keyframes 100\n'
                                       'atlas_map index 1 id 1 keyframes 20\n')

    def run_visual(self, name, *, cache=None):
        cache = cache or self.cache
        directory = self.root / name
        atlas = slam.build(self.recording, 'front', directory / 'build', cache=cache)
        return list(slam.atlas_candidates(self.recording, 'front', atlas.archive,
                                         directory / 'localize', cache=cache))

    def test_reuses_atlas_and_all_submap_localizations_with_identical_observations(self):
        original = self.run_visual('first')
        restored = self.run_visual('second')
        self.assertEqual(self.build.call_count, 1)
        self.assertEqual(self.localize.call_count, 2)
        self.assertEqual([index for index, _ in restored], [0, 1])
        for (_, before), (_, after) in zip(original, restored):
            np.testing.assert_array_equal(before.indices, after.indices)
            np.testing.assert_array_equal(before.T_reference_camera, after.T_reference_camera)
        # 同一固定地图观测按本次标定时间偏移读取，不缓存已偏移的 IMU 时间。
        self.recording.camera.measurements['front']['cam_time_offset_ns'] = 1000000
        shifted = self.run_visual('shifted')
        np.testing.assert_allclose(shifted[0][1].imu_times_s, original[0][1].imu_times_s + .001)
        self.assertEqual(self.localize.call_count, 2)

    def test_camera_settings_and_pixels_invalidate_visual_stages(self):
        self.run_visual('first')
        self.video.settings.write_text('new intrinsics')
        self.run_visual('new_intrinsics')
        self.video.path.write_text('new pixels')
        self.run_visual('new_pixels')
        self.assertEqual(self.build.call_count, 3)
        self.assertEqual(self.localize.call_count, 6)

    def test_recompute_ignores_earlier_caches_but_reuses_ones_made_inside(self):
        from umi.infra.file_store import recompute
        self.run_visual('first')
        self.run_visual('cached')
        self.assertEqual(self.build.call_count, 1)
        with recompute():
            self.run_visual('forced')
            self.assertEqual(self.build.call_count, 2)
            self.run_visual('reused_inside')
            self.assertEqual(self.build.call_count, 2)
        self.run_visual('after')
        self.assertEqual(self.build.call_count, 2)

    def test_mask_affects_build_and_force_recomputes_all_visual_stages(self):
        self.run_visual('first')
        self.video.mask.write_text('new mask')
        self.run_visual('new_mask')
        self.assertEqual(self.build.call_count, 2)
        # 测试引擎产生完全相同的地图字节，因此定位可以复用；真实地图改变则哈希改变。
        self.assertEqual(self.localize.call_count, 2)
        forced = StageCache(self.root / 'cache', force=True)
        self.run_visual('forced', cache=forced)
        self.assertEqual(self.build.call_count, 3)
        self.assertEqual(self.localize.call_count, 4)


# Reverse localization must restore original clocks and reject unanchored or inconsistent runs.


def observations(indices, *, shift=0., rotation=0.):
    indices = np.asarray(indices, int)
    poses = np.tile(np.eye(4), (len(indices), 1, 1))
    poses[:, 0, 3] = indices * .01 + shift
    poses[:, :3, :3] = Rotation.from_euler('z', rotation, degrees=True).as_matrix()
    return geometry.VisualObservations(indices / 30 + .007, poses, indices, 'visual_map')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.parameters = load_scene_parameters()

    def merge(self, forward, reverse, scale=1.):
        return relocalization.merge(forward, reverse, metric_scale=scale, parameters=self.parameters)

    def test_prefix_and_internal_gap_recovered_without_changing_forward_poses_or_time(self):
        forward = observations([5, 8, 9, 10, 11])
        reverse = observations(range(10), shift=.001)
        merged, report = self.merge(forward, reverse)
        np.testing.assert_array_equal(merged.indices, np.arange(12))
        np.testing.assert_array_equal(merged.imu_times_s, np.arange(12)/30+.007)
        np.testing.assert_array_equal(merged.T_reference_camera[forward.indices], forward.T_reference_camera)
        np.testing.assert_array_equal(merged.T_reference_camera[:5], reverse.T_reference_camera[:5])
        self.assertEqual(report['added_frames'], 7)
        self.assertEqual(report['segments'][0]['overlap_frames'], 3)

    def test_reverse_segments_require_their_own_overlap_and_consistency(self):
        forward = observations([0, 1, 2, 10, 11, 12, 20, 21])
        reverse = observations([*range(6), *range(8, 15), *range(18, 23)])
        reverse.T_reference_camera[6:13, 0, 3] += .1
        merged, report = self.merge(forward, reverse)
        np.testing.assert_array_equal(merged.indices, [0, 1, 2, 3, 4, 5, 10, 11, 12, 20, 21])
        self.assertEqual([s['status'] for s in report['segments']],
                         ['accepted', 'inconsistent', 'insufficient_overlap'])

    def test_no_overlap_and_no_forward_observations_are_not_used(self):
        for forward in (observations([]), observations([10, 11, 12])):
            merged, report = self.merge(forward, observations(range(5)))
            np.testing.assert_array_equal(merged.indices, forward.indices)
            self.assertEqual(report['added_frames'], 0)

    def test_translation_uses_map_scale_and_rotation_is_checked_independently(self):
        forward = observations([3, 4, 5])
        reverse = observations(range(6), shift=.02)
        self.assertEqual(self.merge(forward, reverse, scale=.5)[1]['added_frames'], 3)
        self.assertEqual(self.merge(forward, reverse, scale=2.)[1]['added_frames'], 0)
        self.assertEqual(self.merge(forward, observations(range(6), rotation=4.))[1]['added_frames'], 0)

    def test_one_conflicting_overlap_rejects_segment_instead_of_averaging_it_away(self):
        forward = observations(range(3, 20))
        reverse = observations(range(20))
        reverse.T_reference_camera[10, 0, 3] += .04
        self.assertEqual(self.merge(forward, reverse)[1]['added_frames'], 0)

    def test_mismatched_times_unsorted_indices_and_bad_scales_are_rejected(self):
        forward, reverse = observations([3, 4, 5]), observations(range(6))
        with self.assertRaisesRegex(ValueError, '时间戳'):
            self.merge(forward, replace(reverse, imu_times_s=reverse.imu_times_s + .1))
        with self.assertRaises(ValueError):
            self.merge(forward, observations([2, 1, 0]))
        for scale in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                self.merge(forward, reverse, scale)

    def test_complete_and_disabled_paths_skip_reverse(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            recording = SimpleNamespace(video_times_s=np.arange(6)/30)
            for parameters, result, status in (
                (self.parameters, observations(range(6)), 'forward_complete'),
                (replace(self.parameters, localization_bidirectional=False), observations([3,4,5]), 'disabled')):
                with patch.object(slam, 'localize', return_value=result) as localize:
                    output = slam.localize_with_recovery(recording, 'front', directory/'map.osa', 0,
                                                         directory, 1., parameters)
                self.assertIs(output, result)
                self.assertEqual(localize.call_count, 1)
                self.assertEqual(json.loads((directory/'recovery.json').read_text())['status'], status)

    def test_reverse_read_restores_original_frame_order_and_camera_imu_offset(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'frames.csv'
            path.write_text('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw\n'
                            '2,0.066666667,2,1,2,0,0,0,0,0,1\n'
                            '1,0.033333333,4,0,,,,,,,\n'
                            '0,0,2,1,0,0,0,0,0,0,1\n')
            result = slam.read_visual_observations(path, np.array([0., .033, .067]), 7000000)
            np.testing.assert_array_equal(result.indices, [0, 2])
            np.testing.assert_allclose(result.imu_times_s, [.007, .074])

    def test_policy_rejects_insufficient_overlap_and_invalid_boolean(self):
        for data in ({'localization_minimum_overlap_frames':2}, {'localization_bidirectional':1},
                     {'localization_max_position_difference_m':0}, {'localization_max_rotation_difference_deg':181},
                     {'localization_reverse_context_s':0}, {'localization_reverse_context_s':float('nan')}):
            with self.assertRaises(ValueError):
                load_scene_parameters(overrides=data)

    def test_pipeline_runs_reverse_only_after_gap_and_records_accepted_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            recording = SimpleNamespace(video_times_s=np.arange(6)/30)
            with patch.object(slam, 'localize', side_effect=[observations([3, 4, 5]), observations(range(6))]) as localize:
                output = slam.localize_with_recovery(recording, 'front', directory/'map.osa', 0,
                                                     directory, 1., self.parameters)
            np.testing.assert_array_equal(output.indices, np.arange(6))
            self.assertEqual(localize.call_args.kwargs, {'reverse':True})
            self.assertEqual(json.loads((directory/'recovery.json').read_text())['added_frames'], 3)

    def test_window_uses_original_times_and_spans_all_gaps(self):
        times = np.arange(300) / 30.
        times[150:] += .5  # 非均匀时间轴不能按固定 FPS 推算上下文。
        forward = observations(np.delete(np.arange(300), [100, 160]))
        window = relocalization.reverse_window(forward, times, self.parameters)
        self.assertEqual(window, (70, 191))
        self.assertIsNone(relocalization.reverse_window(observations([]), times, self.parameters))
        self.assertIsNone(relocalization.reverse_window(observations(range(300)), times, self.parameters))

    def test_complete_window_skips_full_reverse_and_retains_forward(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            recording = SimpleNamespace(video_times_s=np.arange(300)/30)
            forward = observations(range(10, 300))
            with patch.object(slam, 'localize', side_effect=[forward, observations(range(40), shift=.001)]) as localize:
                output = slam.localize_with_recovery(recording, 'front', directory/'map.osa', 0,
                                                     directory, 1., self.parameters)
            self.assertEqual(localize.call_count, 2)
            self.assertEqual(localize.call_args.kwargs, {'reverse': True, 'frame_range': (0, 40)})
            np.testing.assert_array_equal(output.indices, np.arange(300))
            np.testing.assert_array_equal(output.T_reference_camera[10:], forward.T_reference_camera)
            report = json.loads((directory/'recovery.json').read_text())
            self.assertEqual(report['reverse_processed_frames'], 40)
            self.assertFalse(report['full_reverse_fallback'])

    def test_incomplete_window_retries_full_reverse_against_original_forward_anchors(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            recording = SimpleNamespace(video_times_s=np.arange(300)/30)
            forward = observations(range(10, 300))
            window = observations(range(5, 40), shift=.001)
            whole = observations(range(300), shift=.002)
            with patch.object(slam, 'localize', side_effect=[forward, window, observations(range(40), shift=.1), whole]) as localize:
                output = slam.localize_with_recovery(recording, 'front', directory/'map.osa', 0,
                                                     directory, 1., self.parameters)
            self.assertEqual(localize.call_count, 4)
            self.assertEqual(localize.call_args.kwargs, {'reverse': True})
            np.testing.assert_array_equal(output.T_reference_camera[:10], whole.T_reference_camera[:10])
            report = json.loads((directory/'recovery.json').read_text())
            self.assertEqual(report['reverse_processed_frames'], 380)
            self.assertEqual(report['reverse_attempts'][1]['segments'][0]['status'], 'inconsistent')
            self.assertEqual(report['reverse_attempts'][0]['added_frames'], 5)
            self.assertTrue(report['full_reverse_fallback'])

    def test_warm_prefix_uses_original_anchors_and_keeps_original_timestamps(self):
        forward = observations(range(279, 702))
        recording = SimpleNamespace(video_times_s=np.arange(702) * 1001/30000)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(slam, 'localize', side_effect=[
                    forward, observations([281]), observations(range(308), shift=.001)]) as localize:
                output = slam.localize_with_recovery(recording, 'front', directory/'map.osa', 1,
                                                     directory, 1., self.parameters)
            self.assertEqual(localize.call_args.kwargs, {'reverse': True, 'frame_range': (0, 308), 'warm_reverse': True})
            np.testing.assert_array_equal(output.indices, np.arange(702))
            np.testing.assert_array_equal(output.imu_times_s, observations(range(702)).imu_times_s)
            np.testing.assert_array_equal(output.T_reference_camera[279:], forward.T_reference_camera)
            report = json.loads((directory/'recovery.json').read_text())
            self.assertEqual(report['selected_pass'], 'reverse_warm')
            self.assertEqual(report['warmup_frames'], 309)
            self.assertEqual(report['segments'][0]['overlap_frames'], 29)
            self.assertFalse(report['full_reverse_fallback'])

    def test_older_engine_falls_back_and_keeps_best_independently_verified_run(self):
        recording = SimpleNamespace(video_times_s=np.arange(300)/30)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(slam, 'localize', side_effect=[
                    observations(range(10, 300)), observations(range(5, 40)),
                    visual_slam.WarmReverseUnavailable('older driver'), observations([10])]):
                output = slam.localize_with_recovery(recording, 'front', directory/'map.osa', 0,
                                                     directory, 1., self.parameters)
            np.testing.assert_array_equal(output.indices, np.arange(5,300))
            report = json.loads((directory/'recovery.json').read_text())
            self.assertEqual(report['selected_pass'], 'reverse_window')
            self.assertEqual(report['reverse_attempts'][1]['status'], 'unsupported')
            self.assertEqual(report['warmup_frames'], 0)


class ReverseRuntimeTests(unittest.TestCase):
    def test_partial_csv_keeps_absolute_indices_and_checks_requested_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'logs').mkdir()
            settings = SimpleNamespace(environment={}, orb_directory=root, vocabulary=root/'vocab')
            context = SimpleNamespace(settings=settings, parameters=config.load('trajectory').values,
                                      frame_count=10, cores='')
            video = SimpleNamespace(lens='front', settings=root/'settings', path=root/'input.mp4')
            for case, reverse, indices in [('warm', True, [5,4,3]), ('reverse', True, [5,4,3]), ('forward', False, [3,4,5]),
                                           ('renumbered', True, [2,1,0]), ('missing', True, [5,3])]:
                output = root/case
                def run(command, log, **kwargs):
                    self.assertEqual('--warm_reverse' in command, case == 'warm')
                    self.assertEqual(command[command.index('--frame_start') + 1], '3')
                    self.assertEqual(command[command.index('--frame_end') + 1], '6')
                    log.write_text('TRACKED 3 / 3\n')
                    (output/'frames.csv').write_text('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw\n' +
                        ''.join(f'{i},{i/30},2,1,0,0,0,0,0,0,1\n' for i in indices))
                    return 0
                with patch.object(visual_slam.runtime, 'run_logged', side_effect=run):
                    if case in ('warm', 'reverse', 'forward'):
                        visual_slam.run_orb(video, output, context, root, stage=case,
                                           extra=['--localization_only'], reverse=reverse, frame_range=(3,6),
                                           warm_reverse=case == 'warm')
                        record = json.loads((output/'execution.json').read_text())
                        self.assertEqual(record['rows'], 3)
                        self.assertEqual(record['warmup_frames'], 7 if case == 'warm' else 0)
                    else:
                        with self.assertRaises(visual_slam.ProcessingError):
                            visual_slam.run_orb(video, output, context, root, stage=case,
                                               extra=['--localization_only'], reverse=reverse, frame_range=(3,6))

    def test_reverse_csv_must_have_all_original_frame_indices_in_reverse_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'logs').mkdir()
            settings = SimpleNamespace(environment={}, orb_directory=root, vocabulary=root/'vocab')
            context = SimpleNamespace(settings=settings, parameters=config.load('trajectory').values,
                                      frame_count=3, cores='')
            video = SimpleNamespace(lens='front', settings=root/'settings', path=root/'input.mp4')
            for case, indices in [('valid', [2,1,0]), ('wrong_order', [0,1,2]), ('missing', [2,0])]:
                output = root/case
                def run(command, log, **kwargs):
                    self.assertIn('--reverse', command)
                    log.write_text('TRACKED 3 / 3\n')
                    (output/'frames.csv').write_text('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw\n' +
                        ''.join(f'{i},{i/30},2,1,0,0,0,0,0,0,1\n' for i in indices))
                    return 0
                with patch.object(visual_slam.runtime, 'run_logged', side_effect=run):
                    if case == 'valid':
                        visual_slam.run_orb(video, output, context, root, stage='reverse',
                                            extra=['--localization_only'], reverse=True)
                        self.assertTrue(json.loads((output/'execution.json').read_text())['accepted'])
                    else:
                        with self.assertRaises(visual_slam.ProcessingError):
                            visual_slam.run_orb(video, output, context, root, stage='reverse',
                                                extra=['--localization_only'], reverse=True)


# 建图海报尺度：独立尺度恢复、精确固定尺度惯性复核和发布身份。


tags = poster_detection


def scale_fixture():
    from test_poster import poster_fusion_fixture as fixture
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
        from test_scene import scene_motion as motion
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
            write_yaml(path, dict(kit='rig', arms=['left', 'right'],
                                 description='',
                                 map=dict(layout='shared', reference_arm='left'),
                                 poster=dict(layout=str(board_path), detection=True),
                                 backend=dict(mode='scene_map')))
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
            write_yaml(path, dict(kit='rig', arms=['left'],
                                 description='', poster=None,
                                 map=dict(layout='shared', reference_arm='left', poster_scale='correct'),
                                 backend=dict(mode='scene_map')))
            with self.assertRaisesRegex(ValueError, 'map.poster_scale 需要 poster.layout'):
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
        declaration['poster'] = dict(layout=str(tag_board.DEFAULT_LAYOUT), detection=True)
        declaration.setdefault('map', {})['poster_scale'] = 'correct' if correction else 'diagnose'
        write_yaml(declaration_path, declaration)
        expert = helper.root / 'expert.yaml'
        write_yaml(expert, dict(backends=dict(scene_map=dict(mapping=dict(poster_scale=dict(minimum_observation_times=6))))))
        helper.scene_overrides = SceneOverrides.load(expert)
        helper.scene = batch_inputs.read_batch(helper.session_path, overrides=helper.scene_overrides).scene
        for video, camera in list(helper.cameras.items()):
            helper.cameras[video] = replace(camera, measurements=dict(camera.measurements, rear=camera.measurements['front']),
                                            files=dict(camera.files, rear=camera.files['front']))

    def detections(self, helper, *, empty=False, shifted=False):
        from test_scene import scene_motion as motion

        from umi.perception.poster import detection as environment_tags
        from umi.perception.poster import geometry as poster_pose
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
        from umi.infra.file_store import write_json
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
