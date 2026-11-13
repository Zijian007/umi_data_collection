import unittest
from pathlib import Path
import tempfile

from software.scene.pipeline.map_epochs import MapEpoch, MapEpochError, TaskCollection, read_plan, select_many, select_map_epoch
from software.storage import write_yaml


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
