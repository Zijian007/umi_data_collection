import unittest
from pathlib import Path
import tempfile

from software.scene.pipeline.maps.epoch_planning import draft_plan
from software.scene.pipeline.maps.epoch_selection import (
    MapEpoch,
    MapEpochError,
    TaskCollection,
    read_plan,
    resolve_task_map,
    select_many,
    select_map_epoch,
)
from software.infrastructure.file_store import write_yaml


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
