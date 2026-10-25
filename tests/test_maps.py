"""地图导出、相似变换和目录边界；不运行 SLAM。"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from software.trajectory import map_data
from software.viewer import maps


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
        (self.output/'traj_camera.txt').write_text(''.join(
            f'{i*.1:.9f} {2*p[0]+1} {2*p[1]} {2*p[2]} 0 0 0 1\n' for i,p in enumerate(self.points)))

    def export(self):
        def run(command, **kwargs):
            Path(command[-1]).write_text(json.dumps(self.raw))
            return SimpleNamespace(returncode=0)
        with patch.object(map_data, 'exporter_paths', return_value=(self.executable, self.vocabulary)), \
                patch.object(map_data.subprocess, 'run', side_effect=run):
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

    def test_only_selected_attempt_is_read(self):
        self.assertEqual(map_data.attempt_directory(self.output, {'chosen_try': 2}), self.attempt)
        self.assertIsNone(map_data.attempt_directory(self.output, {'chosen_try': 1}))

    def test_symbolic_links_cannot_read_maps_outside_result(self):
        self.archive.unlink()
        outside = self.root/'outside.osa'
        outside.write_bytes(b'outside')
        self.archive.symlink_to(outside)
        with patch.object(map_data.subprocess, 'run') as execute, self.assertRaises(map_data.MapDataError):
            map_data.export_attempt(self.attempt, self.output)
        execute.assert_not_called()

    def test_prepared_map_loads_without_exporter_and_rejects_outside_symlink(self):
        data = self.export()
        prepared = self.output/'details/map.json'
        prepared.parent.mkdir()
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
        source, target = map_data._correspondences(locations, self.output/'traj_camera.txt', 0)
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
