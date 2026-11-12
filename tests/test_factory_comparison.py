from software import provenance as artifact_identity
"""Factory optical diagnostics: geometry, held-out evidence and workflow isolation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from software.camera.x3 import factory as optical
from software.calibration.algorithms import board_pnp
from software.calibration.models import Kb4, Step
from software.calibration.validation import factory
from software.calibration import reports, result_io, workflow
from software.calibration.pipeline import inputs


def offset():
    first = [1.94817, 4625., 4624., 2970., 2990., 1., 2., 90., 0., 0., 0.,
             .39, 1.24, -3.86, -.0001, -.002, 11904., 5952., 71.]
    second = first.copy()
    second[3] += 5952
    second[8:11] = [.001, -.0001, -.026]
    return '_'.join(map(str, [2., *first, *second, 199424.]))


class FactoryGeometryTests(unittest.TestCase):
    def test_metadata_blocks_preserve_identity_and_unsupported_formats_are_refused(self):
        parsed = optical.parse_offset(offset())
        self.assertEqual(parsed['mapping_status'], 'fixed')
        self.assertEqual(parsed['lens_assignment'], {'working': 1, 'environment': 0})
        self.assertEqual([lens['role'] for lens in parsed['lenses']], ['environment', 'working'])
        self.assertEqual(parsed['lenses'][1]['lens_type'], 71)
        self.assertEqual(parsed['lenses'][1]['tz'], -.026)
        self.assertEqual(parsed['lenses'][0]['roll'], 90.)
        for raw in (None, '2_bad', offset()+'_0', offset().replace('4625.0', 'nan'),
                    offset().replace('11904.0', '11903.0'), '1_'+offset().split('_', 1)[1]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                optical.parse_offset(raw)

    def test_half_offset_and_crop_preserve_rays(self):
        native = optical.parse_offset(offset())
        a = optical.lens_model(native['lenses'][0], 1920)
        b = optical.lens_model(native['lenses'][1], 1920)
        crop = optical.lens_model(native['lenses'][0], 1440)
        np.testing.assert_allclose([a.cx, a.cy], [b.cx, b.cy])
        rays = np.array([[0., 0., 1.], [.4, -.2, 1.], [-.6, .7, 1.]])
        np.testing.assert_allclose(a.project(rays)-240, crop.project(rays))
        np.testing.assert_allclose(a.unproject(a.project(rays)), rays/np.linalg.norm(rays, axis=1)[:, None], atol=1e-9)
        self.assertEqual(a.fx, 4625*1920/5952)

    def test_kb4_matches_opencv_and_invalid_rays_are_masked(self):
        model = optical.Kb4Model(530., 531., 720., 721., .1, -.05, .02, -.003)
        rays = np.array([[0., 0., 1.], [.6, .2, 1.], [-1., .5, 1.]])
        K = np.array([[model.fx, 0, model.cx], [0, model.fy, model.cy], [0, 0, 1.]])
        expected = cv2.fisheye.projectPoints(rays[:, None], np.zeros(3), np.zeros(3), K,
            np.array([model.k1, model.k2, model.k3, model.k4]))[0][:, 0]
        np.testing.assert_allclose(model.project(rays), expected)
        np.testing.assert_allclose(model.unproject(expected), rays/np.linalg.norm(rays, axis=1)[:, None], atol=1e-10)
        mei = optical.lens_model(optical.parse_offset(offset())['lenses'][0], 1920)
        self.assertTrue(np.isnan(mei.project([[0., 0., -1.]])).all())
        self.assertTrue(np.isnan(mei.unproject([[1e9, 1e9]])).all())

    def test_full_view_separates_extrapolation_and_exports_heatmap(self):
        native = optical.parse_offset(offset())
        mei = optical.lens_model(native['lenses'][0], 1920)
        kb = optical.Kb4Model(530., 530., 720., 720., .1, -.05, .02, -.003)
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'rays.png'
            result = factory.ray_comparison(kb.full_view(), mei, 1920, image)
            self.assertGreater(result['regions']['outside_measured_crop']['sampled_points'], 0)
            self.assertGreater(result['regions']['measured_crop']['angular_degrees']['count'], 0)
            self.assertEqual(cv2.imread(str(image)).shape, (520, 520, 3))
            json.dumps(result, allow_nan=False)


class HeldoutComparisonTests(unittest.TestCase):
    def setUp(self):
        self.model = optical.Kb4Model(530., 531., 720., 721., .1, -.05, .02, -.003)
        self.grid = board_pnp.grid_points_3d(dict(tagCols=6, tagRows=6, tagSpacing=.3, tagSize=.024))
        R = cv2.Rodrigues(np.array([.12, -.25, .07]))[0]
        self.detections = {i: self.model.project(p@R.T+[-.09, -.08, .45]) for i, p in self.grid.items()}

    def test_heldout_tags_are_not_used_to_fit_pose(self):
        perfect = factory.fit_board(self.model, self.detections, self.grid)
        self.assertLess(perfect['errors'].max(), 1e-5)
        changed = copy.deepcopy(self.detections)
        for tag in sorted(changed)[::3]:
            changed[tag] += [5., 0.]
        result = factory.fit_board(self.model, changed, self.grid)
        np.testing.assert_allclose(result['errors'], 5., atol=1e-5)
        np.testing.assert_allclose(result['fit_errors'], perfect['fit_errors'], atol=1e-5)

    def test_statistics_use_common_frames_and_report_failures(self):
        class Invalid:
            def unproject(self, pixels):
                return np.full((len(pixels), 3), np.nan)
        summary, _, common = factory.board_comparison({0: self.detections},
            dict(kb4=self.model, bad=Invalid()), self.grid, 1440)
        self.assertEqual(common, [])
        self.assertEqual(summary['models']['kb4']['successful_frames'], 1)
        self.assertEqual(summary['models']['kb4']['heldout_pixels']['count'], 0)
        self.assertTrue(summary['models']['bad']['failures'])

    def test_comparison_artifacts_survive_work_cleanup_and_corruption_invalidates_reuse(self):
        native = optical.parse_offset(offset())
        native['lenses'].reverse()  # Select metadata identity, not the list position.
        native.update(serial='camera', firmware='test', camera_model='Insta360 X3')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root/'calibration.json'
            calibration.write_text('{}')
            (root/'factory').mkdir()
            for size in (1440, 1920):
                (root/f'factory/{size}_mei_lens0_rays.png').write_bytes(b'old v1 plot')
            obs = {1440: {0: self.detections}, 1920: {0: {k: p+240 for k, p in self.detections.items()}}}
            with patch.object(board_pnp, 'read_kb4', return_value=Kb4(*vars(self.model).values())), \
                 patch.object(factory, '_observations', return_value=(obs, {0: np.full((1920, 1920), 80, np.uint8)}, {0: 1.}, [])):
                result = factory.compare(Path('C.insv'), dict(consistent=True, recordings=dict(C=native)),
                    calibration, {}, dict(tagCols=6, tagRows=6, tagSpacing=.3, tagSize=.024), None, root)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['fits']['lens0']['status'], 'ok')
            self.assertEqual(result['fits']['lens1']['status'], 'ok')
            self.assertTrue((root/'factory/kb4_fit.json').is_file())
            self.assertEqual(calibration.read_text(), '{}')
            self.assertTrue(factory.reusable(root, result))
            self.assertEqual(result['views']['1920']['role'], 'exploratory_with_kb4_extrapolation')
            self.assertEqual(result['mapping_status'], 'fixed')
            self.assertEqual(result['lens_assignment'], {'working': 1, 'environment': 0})
            self.assertEqual(len(result['native']['recordings']['C']['lenses']), 2)
            for size, view in result['views'].items():
                self.assertEqual(set(view['candidates']), {'mei_lens1'})
                self.assertEqual(set(view['board']['models']), {'kb4', 'mei_lens1', 'factory_fitted_kb4'})
                self.assertAlmostEqual(view['candidates']['mei_lens1']['parameters']['fx'], 4625*1920/5952)
                self.assertFalse((root/f'factory/{size}_mei_lens0_rays.png').exists())
            self.assertFalse(factory.reusable(root, dict(result, method_version=1)))
            text = '\n'.join(reports.factory_comparison_lines(result))
            self.assertIn('不参与标定验收', text)
            self.assertIn('外推', text)
            self.assertIn('lens1 = 工作镜头', text)
            self.assertIn('未替代实测 A', text)
            self.assertIn('历史报告', '\n'.join(reports.factory_comparison_lines(
                dict(result, mapping_status='unverified_candidates'))))
            fit_path = root/'factory/kb4_fit.json'
            original_fit = fit_path.read_bytes()
            fit_path.write_bytes(b'broken')
            self.assertFalse(factory.reusable(root, result))
            fit_path.write_bytes(original_fit)
            self.assertTrue(factory.reusable(root, result))
            (root/'factory/1440_mei_lens1_rays.png').write_bytes(b'broken')
            self.assertFalse(factory.reusable(root, result))

    def test_fitted_parameters_survive_unavailable_board_video(self):
        native = optical.parse_offset(offset())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root/'calibration.json'
            calibration.write_text('{}')
            with patch.object(board_pnp, 'read_kb4', return_value=Kb4(*vars(self.model).values())), \
                 patch.object(factory, '_observations', side_effect=ValueError('bad video')):
                result = factory.compare(Path('C.insv'), dict(consistent=True, recordings=dict(C=native)),
                    calibration, {}, dict(tagCols=6, tagRows=6, tagSpacing=.3, tagSize=.024), None, root)
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['fits']['lens0']['status'], 'ok')
            self.assertEqual(result['fits']['lens1']['status'], 'ok')
            self.assertEqual(result['views']['1440']['board']['common_frames'], 0)
            self.assertIn('bad video', '\n'.join(reports.factory_comparison_lines(result)))
            self.assertFalse(factory.reusable(root, result))

    def test_environment_fit_failure_does_not_remove_working_lens_board_evidence(self):
        native = optical.parse_offset(offset())
        native['lenses'][0]['fx'] = 1.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root/'calibration.json'
            calibration.write_text('{}')
            (root/'factory').mkdir()
            stale = root/'factory/1440_lens0_kb4_fit_pixels.png'
            stale.write_bytes(b'old successful diagnostic')
            obs = {1440: {0: self.detections}, 1920: {0: {k: p+240 for k, p in self.detections.items()}}}
            with patch.object(board_pnp, 'read_kb4', return_value=Kb4(*vars(self.model).values())), \
                 patch.object(factory, '_observations', return_value=(obs, {}, {0: 1.}, [])):
                result = factory.compare(Path('C.insv'), dict(consistent=True, recordings=dict(C=native)),
                    calibration, {}, dict(tagCols=6, tagRows=6, tagSpacing=.3, tagSize=.024), None, root)
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['fits']['lens0']['status'], 'failed')
            self.assertEqual(result['fits']['lens1']['status'], 'ok')
            self.assertEqual(result['views']['1440']['board']['common_frames'], 1)
            self.assertFalse(stale.exists())
            self.assertFalse(factory.reusable(root, result))


class AutomaticWorkflowTests(unittest.TestCase):
    def test_factory_failure_after_metadata_keeps_raw_parameters(self):
        from test_calibration_workflow import FullWorkflowTests
        native = dict(consistent=True, recordings={'C': optical.parse_offset(offset())})
        with tempfile.TemporaryDirectory() as tmp, FullWorkflowTests().simulation(Path(tmp)) as (args, _, _):
            with patch.object(factory, 'sources', return_value=native), \
                 patch.object(factory, 'compare', side_effect=ValueError('video decode failed')):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            doc = result_io.load_result(Path(args.out))
            self.assertEqual(result_io.measurement(doc, Step.FACTORY)['native'], native)
            self.assertIn('video decode failed', (Path(args.out)/'report.txt').read_text())

    def test_default_run_records_nonfatal_factory_failure_and_preserves_acceptance(self):
        from test_calibration_workflow import FullWorkflowTests
        with tempfile.TemporaryDirectory() as tmp, FullWorkflowTests().simulation(Path(tmp)) as (args, _, _):
            with patch.object(factory, 'sources', side_effect=ValueError('missing offset_v3')) as read:
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            read.assert_called_once()
            doc = result_io.load_result(Path(args.out))
            self.assertEqual(doc['execution']['status'], 'complete')
            self.assertEqual(result_io.measurement(doc, Step.FACTORY)['status'], 'comparison_failed')
            self.assertIn('missing offset_v3', (Path(args.out)/'report.txt').read_text())
            self.assertTrue((Path(args.out)/'run/logs/factory_error.log').is_file())

    def test_factory_success_is_automatic_and_reused_after_cleanup(self):
        from test_calibration_workflow import FullWorkflowTests
        with tempfile.TemporaryDirectory() as tmp, FullWorkflowTests().simulation(Path(tmp)) as (args, _, _):
            root = Path(args.out)
            def compare(*arguments, **options):
                self.assertEqual(options['lens'], 'front')
                self.assertEqual(options['intrinsics_source'], 'measured')
                path = root/'factory/comparison.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
                return dict(status='ok', method_version=factory.METHOD_VERSION,
                    mapping_status='fixed', native=dict(consistent=True), views={},
                    artifacts={'factory/comparison.json': artifact_identity.file_hash(path)})
            with patch.object(factory, 'sources', return_value={}), patch.object(factory, 'compare', side_effect=compare) as call:
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
                self.assertFalse((root/'run/work').exists())
                args.resume = True
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
                self.assertEqual(call.call_count, 1)
                (root/'factory/comparison.json').write_text('changed')
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
                self.assertEqual(call.call_count, 2)


if __name__ == '__main__':
    unittest.main()
