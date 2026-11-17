from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from software.infrastructure import provenance
from software.infrastructure.file_store import read_yaml, write_yaml
from software.scene.management.import_manifest import ensure_epoch_plan, read_manifest


class ImportManifestTests(unittest.TestCase):
    def make_batch(self, root: Path) -> Path:
        batch = root / 'scene' / 'batch'
        mapping = batch / 'mapping' / 'epoch_001' / 'map.insv'
        task = batch / 'take_001' / 'left' / 'task.insv'
        mapping.parent.mkdir(parents=True)
        task.parent.mkdir(parents=True)
        mapping.write_bytes(b'mapping')
        task.write_bytes(b'task')
        payload = {
            'schema_version': 1,
            'kind': 'umi_insv_import',
            'batch': 'batch',
            'maps': [{'epoch': 'epoch_001', 'sequence': 0,
                      'recording': 'mapping/epoch_001/map.insv',
                      'sha256': {'mapping/epoch_001/map.insv': provenance.file_hash(mapping)}}],
            'tasks': [{'task': 'take_001', 'sequence': 1, 'map_epoch': 'epoch_001',
                       'recordings': {'left': {'path': 'take_001/left/task.insv',
                                               'sha256': provenance.file_hash(task)}}}],
        }
        (batch / 'import_manifest.json').write_text(json.dumps(payload), encoding='utf-8')
        return batch

    def test_handoff_materializes_internal_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = self.make_batch(Path(directory))
            self.assertEqual(read_manifest(batch)['maps'][0]['epoch'], 'epoch_001')
            plan = ensure_epoch_plan(batch)
            self.assertEqual(read_yaml(plan)['tasks'], [{'task': 'take_001', 'sequence': 1}])

    def append_commit(self, batch: Path, *, mutate=None) -> None:
        """模拟 Organizer 的第二次提交：新增一张地图和一个任务。"""
        mapping = batch / 'mapping' / 'epoch_002' / 'map.insv'
        task = batch / 'take_002' / 'left' / 'task.insv'
        mapping.parent.mkdir(parents=True)
        task.parent.mkdir(parents=True)
        mapping.write_bytes(b'mapping 2')
        task.write_bytes(b'task 2')
        payload = json.loads((batch / 'import_manifest.json').read_text(encoding='utf-8'))
        payload['maps'].append({'epoch': 'epoch_002', 'sequence': 2, 'recording': 'mapping/epoch_002/map.insv',
                                'sha256': {'mapping/epoch_002/map.insv': provenance.file_hash(mapping)}})
        payload['tasks'].append({'task': 'take_002', 'sequence': 3,
                                 'recordings': {'left': {'path': 'take_002/left/task.insv',
                                                         'sha256': provenance.file_hash(task)}}})
        if mutate is not None:
            mutate(payload)
        (batch / 'import_manifest.json').write_text(json.dumps(payload), encoding='utf-8')

    def test_later_commit_appends_and_keeps_built_map_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = self.make_batch(Path(directory))
            plan_path = ensure_epoch_plan(batch)
            plan = read_yaml(plan_path)
            plan['maps'][0]['map_version'] = '20261117_100000-a1b2c3d4'  # build --epochs 已构建
            write_yaml(plan_path, plan)
            self.append_commit(batch)
            plan = read_yaml(ensure_epoch_plan(batch))
            self.assertEqual([(item['epoch'], item['map_version']) for item in plan['maps']],
                             [('epoch_001', '20261117_100000-a1b2c3d4'), ('epoch_002', None)])
            self.assertEqual([item['task'] for item in plan['tasks']], ['take_001', 'take_002'])

    def test_later_commit_cannot_change_or_drop_registered_entries(self):
        def change_sequence(payload):
            payload['maps'][0]['sequence'] = 5

        def drop_task(payload):
            payload['tasks'] = payload['tasks'][1:]

        for mutate, message in ((change_sequence, '修改了已登记的地图 epoch epoch_001'),
                                (drop_task, '删除了已登记的任务 take_001')):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                batch = self.make_batch(Path(directory))
                plan_path = ensure_epoch_plan(batch)
                before = plan_path.read_text()
                self.append_commit(batch, mutate=mutate)
                with self.assertRaisesRegex(ValueError, message):
                    ensure_epoch_plan(batch)
                self.assertEqual(plan_path.read_text(), before)

    def test_changed_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = self.make_batch(Path(directory))
            (batch / 'mapping/epoch_001/map.insv').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                read_manifest(batch)


if __name__ == '__main__':
    unittest.main()
