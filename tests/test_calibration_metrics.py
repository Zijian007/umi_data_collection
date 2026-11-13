"""A/B measurement definitions: solver log parsing, coverage metrics, KB4 type and board layout check."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from software import config
from software.calibration import acceptance
from software.calibration.pipeline import inputs
from software.calibration.solve import basalt
from software.calibration.algorithms import board_pnp
from software.calibration.algorithms.aprilgrid_detect import GridDetector
from software.calibration.models import IntrinsicsResult, Kb4

# Excerpt of a real basalt_calibrate log (IAQEB2609JPAR3, 2026-11-02); first and last optimisation steps.
BASALT_LOG = """\
Total number of messages: 98350
Image size: 2836
Cam 0: 519.642 519.642   719.5   719.5       0       0       0       0
RMS_REPROJ_EVIDENCE stage=cam eval=accepted n=385850 sum_norm=894513.302148 sum_sq=3873874.77583 mean=2.31829286549 rms=3.16857168293
Current error: 3.25224e+06 num_points 385850 mean_error 8.42877 reprojection_error 894513 mean reprojection 2.31829 opt_time 31ms.
RMS_REPROJ_EVIDENCE stage=cam eval=accepted n=385850 sum_norm=327449.075126 sum_sq=364346.704505 mean=0.848643449853 rms=0.971735727435
RMS_REPROJ_EVIDENCE stage=cam eval=linearize n=385850 sum_norm=327449.075126 sum_sq=364346.704505 mean=0.848643449853 rms=0.971735727435
Current error: 364345 num_points 385850 mean_error 0.944267 reprojection_error 327449 mean reprojection 0.848643 opt_time 21ms.
Optimization Converged !!
"""


class SolverLogTests(unittest.TestCase):
    def parse(self, text: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'camera_intrinsics.log'
            log.write_text(text)
            return basalt.last_reprojection(log)

    def test_final_step_and_image_count(self):
        metrics = self.parse(BASALT_LOG)
        self.assertEqual(metrics['images'], 2836)
        self.assertEqual(metrics['num_points'], 385850)
        self.assertEqual(metrics['corners_per_frame'], 136.1)
        self.assertAlmostEqual(metrics['mean_reprojection_px'], 0.848643)
        self.assertAlmostEqual(metrics['rms_reprojection_px'], 0.971735727435)
        self.assertGreaterEqual(metrics['rms_reprojection_px'], metrics['mean_reprojection_px'])
        self.assertEqual(metrics['converged'], 1)

    def test_unpatched_or_missing_log_gives_no_values(self):
        unpatched = '\n'.join(line for line in BASALT_LOG.splitlines() if not line.startswith('RMS_REPROJ_EVIDENCE'))
        self.assertIsNone(self.parse(unpatched)['rms_reprojection_px'])
        self.assertEqual(basalt.last_reprojection(Path('/nonexistent/log')), dict(
            converged=0, images=None, num_points=None, mean_reprojection_px=None, rms_reprojection_px=None,
            corners_per_frame=None))

    def test_append_only_log_uses_the_latest_solver_invocation(self):
        old = BASALT_LOG.replace('Image size: 2836', 'Image size: 10').replace('num_points 385850', 'num_points 100')
        current = BASALT_LOG.replace('Image size: 2836', 'Image size: 20').replace('num_points 385850', 'num_points 200')
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'camera_imu.log'
            log.write_text(old + '\n=== UMI_SOLVER_START camera_imu===\n' + current)
            metrics = basalt.last_reprojection(log)
        self.assertEqual(metrics['images'], 20)
        self.assertEqual(metrics['num_points'], 200)
        self.assertEqual(metrics['corners_per_frame'], 10.0)

    def test_rms_is_informational_and_mean_drives_the_threshold(self):
        parameters = config.load('calibration').values.checks
        metrics = dict(corners_per_frame=100., mean_reprojection_px=0.9, rms_reprojection_px=5.0)
        rows = []
        acceptance.solver_fit_items(rows, 'A', metrics, parameters.intrinsics_minimum_corners, parameters.intrinsics_reprojection_px)
        results = {row['check']: row['result'] for row in rows}
        self.assertEqual(results['内参：重投影误差均值 (px)'], 'OK')
        self.assertEqual(results['内参：重投影误差 RMS (px)'], 'INFO')
        self.assertEqual(acceptance.decision(rows).label, 'PASS')


class CoverageMetricTests(unittest.TestCase):
    grid = dict(tagCols=6, tagRows=6, tagSize=.024, tagSpacing=.3)
    kb4 = Kb4(500., 500., 720., 720., 0., 0., 0., 0.)

    def found(self):
        # Three frames, farthest corner at r = 0.3, 0.6, 0.9 (radius 960 px), all to the right of the principal point.
        def frame(r):
            x = 720 + r * 960
            return {0: np.array([[x, 720.], [x - 1, 720.], [x - 1, 721.], [x, 721.]])}
        return dict(n_frames=3, video_time_s=np.arange(3)/30, shape=(1440, 1440), dets={0: frame(.3), 1: frame(.6), 2: frame(.9)})

    def test_metric_names_do_not_encode_parameter_values(self):
        board = replace(config.load('calibration').values.board, coverage_percentile=50.0, coverage_reach=0.7)
        metrics = board_pnp.measure_board_coverage(self.found(), self.kb4, self.grid, parameters=board)
        self.assertAlmostEqual(metrics['reach_radius_percentile'], 0.6, places=2)
        self.assertEqual(metrics['coverage_percentile'], 50.0)
        self.assertEqual(metrics['coverage_radius_px'], 960.0)
        self.assertEqual(metrics['coverage_reach_threshold'], 0.7)
        self.assertEqual(metrics['frames_reaching_by_direction'], dict(right=1, left=0, down=0, up=0))
        rows = []
        acceptance.coverage_items(rows, metrics, 0.5)
        self.assertIn('第 50 百分位', rows[0]['check'])
        self.assertIn('达到 0.7 的帧数', rows[1]['check'])
        self.assertEqual(rows[1]['value'], '1 / 0 / 0 / 0')


class Kb4Tests(unittest.TestCase):
    def test_named_access_and_json_list(self):
        kb4 = board_pnp.read_kb4(Path('software/trajectory/assets/calib/calibration_r1440.json'))
        self.assertIsInstance(kb4, Kb4)
        self.assertEqual(kb4.principal_point, (kb4.cx, kb4.cy))
        self.assertEqual(json.loads(json.dumps(kb4)), list(kb4))
        stored = IntrinsicsResult(Path('unused'), json.loads(json.dumps(kb4)), {}, {})
        self.assertEqual(stored.kb4, kb4)
        self.assertEqual(stored.kb4.fx, kb4.fx)


class BoardLayoutTests(unittest.TestCase):
    def test_repository_prints_match_layout(self):
        self.assertEqual(inputs.board_layout_problems(), [])

    def test_measured_sizes_produce_the_existing_basalt_geometry(self):
        for size_mm in (24.0, 36.0, 23.9):
            with self.subTest(size_mm=size_mm):
                self.assertEqual(inputs.load_board(size_mm), dict(
                    tagCols=6, tagRows=6, tagSpacing=.3, tagSize=size_mm / 1000.0))
        for size_mm in (0, -1, float('nan'), float('inf')):
            with self.subTest(size_mm=size_mm), self.assertRaises(ValueError):
                inputs.load_board(size_mm)

    def test_shipped_print_images_decode_with_the_actual_detector(self):
        import cv2
        detector = GridDetector()
        for path in sorted((inputs.BOARD_JSON.parent / 'prints').glob('*.png')):
            with self.subTest(path=path.name):
                found = detector.detect(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
                self.assertEqual(set(found), set(range(36)))
                # Grid IDs increase rightwards, then upwards from the lower left.
                self.assertGreater(found[1][:, 0].mean(), found[0][:, 0].mean())
                self.assertLess(found[6][:, 1].mean(), found[0][:, 1].mean())

    def test_print_with_other_geometry_or_missing_asset_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            reference = directory / 'aprilgrid.json'
            reference.write_bytes(inputs.BOARD_JSON.read_bytes())
            prints = directory / 'prints'
            prints.mkdir()
            metadata = json.loads((inputs.BOARD_JSON.parent / 'prints/A3_36mm_meta.json').read_text())
            record = prints / 'A3_36mm_meta.json'
            record.write_text(json.dumps(metadata))
            for name in metadata['files']:
                (prints / name).touch()
            with patch.object(inputs, 'BOARD_JSON', reference):
                self.assertEqual(inputs.board_layout_problems(), [])
                metadata['layout']['rows'] = 7
                record.write_text(json.dumps(metadata))
                problems = inputs.board_layout_problems()
                self.assertEqual(len(problems), 1)
                self.assertIn(record.name, problems[0])
                self.assertIn('行列或间距不同', problems[0])
                metadata['layout']['rows'] = 6
                record.write_text(json.dumps(metadata))
                (prints / metadata['files'][0]).unlink()
                self.assertIn('打印资产不存在', inputs.board_layout_problems()[0])

    def test_incompatible_layout_is_rejected_before_solving(self):
        original = json.loads(inputs.BOARD_JSON.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / 'aprilgrid.json'
            for change in (dict(family='tag25h9'), dict(black_border_bits=1), dict(tagSize=.024),
                           dict(tagRows=0), dict(tagSpacing=float('nan'))):
                with self.subTest(change=change), patch.object(inputs, 'BOARD_JSON', reference):
                    reference.write_text(json.dumps(original | change))
                    with self.assertRaises(ValueError):
                        inputs.load_board(24.)
                    self.assertTrue(inputs.board_layout_problems())


if __name__ == '__main__':
    unittest.main()
