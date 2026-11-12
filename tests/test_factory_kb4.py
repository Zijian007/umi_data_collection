"""Independent geometry and >180 degree field-of-view regression checks."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from software.calibration.algorithms import board_pnp, factory_kb4 as fit
from software.calibration.models import Kb4
from software.camera.x3.factory import Mei, Kb4Model


class FactoryKb4Tests(unittest.TestCase):
    def test_forward_conversion_rejects_nonmonotonic_or_inaccurate_fit(self):
        mei = Mei(3000., 2995., 718., 723., 1., 0., 0., 0., .002, -.003)
        fitted = dict(kb4=dict(fx=20., fy=20., cx=720., cy=720., k1=0., k2=0., k3=0., k4=0.),
                      views={'1440': dict(minimum_radial_derivative=-1.)})
        limits = SimpleNamespace(maximum_fit_rms_px=2., maximum_fit_p99_px=3.)
        with patch.object(fit, 'fit', return_value=fitted):
            with self.assertRaisesRegex(ValueError, '单调'):
                fit.fit_forward(mei, limits)
            fitted['views']['1440']['minimum_radial_derivative'] = 1.
            with self.assertRaisesRegex(ValueError, '误差超限'):
                fit.fit_forward(mei, limits)

    def test_fit_uses_heldout_domain_and_records_tangential_approximation_error(self):
        mei = Mei(3000., 2995., 718., 723., 1., 0., 0., 0., .002, -.003)
        result = fit.fit(mei)
        self.assertEqual(result['status'], 'ok')
        self.assertTrue(result['diagnostic_only'])
        self.assertEqual(result['acceptance'], 'not_evaluated')
        self.assertEqual(len(result['attempts']), 2)
        validation = result['views']['1440']
        self.assertGreater(validation['regions']['all']['pixels']['rms'], .01)
        self.assertGreater(validation['minimum_radial_derivative'], 0.)
        self.assertEqual(validation['regions']['all']['pixels']['invalid'], 0)
        self.assertEqual(validation['rectangle_samples'], validation['disc_samples']+validation['outside_nominal_disc'])
        self.assertGreater(validation['outside_nominal_disc'], 0)
        self.assertGreater(validation['boundary']['pixels']['count'], 100)
        training = set(map(tuple, fit.grid(1440, fit.FIT_SIDE, .5)))
        heldout = set(map(tuple, fit.grid(1440, fit.VALIDATION_SIDE, .37)))
        self.assertFalse(training & heldout)
        self.assertLess(validation['opencv']['projection_agreement_pixels']['maximum'], 1e-9)
        json.dumps(result, allow_nan=False)

    def test_unproject_preserves_backward_rays_and_board_pnp_does_not_fold_them(self):
        kb = Kb4Model(450., 451., 720., 719., .04, -.005, .001, 0.)
        theta = np.radians([0., 35., 89., 90., 95., 110., 140.])
        rays = np.column_stack((np.sin(theta), np.zeros(len(theta)), np.cos(theta)))
        pixels = kb.project(rays)
        np.testing.assert_allclose(kb.unproject(pixels), rays, atol=1e-10)
        actual = board_pnp.kb4_unproject(pixels, Kb4(*vars(kb).values()))
        np.testing.assert_allclose(actual, rays, atol=1e-10)
        self.assertTrue((actual[4:, 2] < 0).all())
        # A tag seen entirely behind the lens cannot seed the forward PnP stage.
        grid = board_pnp.grid_points_3d(dict(tagCols=6, tagRows=6, tagSpacing=.3, tagSize=.024))
        detections = {i: kb.project(points+[.2, .1, -.2]) for i, points in grid.items()}
        self.assertIsNone(board_pnp.pnp_from_detections(detections, Kb4(*vars(kb).values()), grid))

    def test_opencv_is_diagnostic_not_spherical_ground_truth(self):
        # The same model at both ends makes conversion error exactly zero,
        # but a 960 px disc with f=500 crosses the 90 degree ray angle.
        kb = Kb4Model(500., 500., 719.5, 719.5, 0., 0., 0., 0.)
        validation = fit.validate(kb, kb, 1440)
        self.assertLess(validation['regions']['all']['pixels']['maximum'], 1e-9)
        self.assertGreater(validation['theta_max_degrees'], 90.)
        self.assertGreater(validation['opencv']['inverse_mismatch_samples'], 0)
        self.assertGreater(validation['orb_slam3_inverse_limits']['clipped_radius_samples'], 0)
        self.assertGreater(validation['orb_slam3_inverse_limits']['backward_or_horizon_reference_samples'], 0)

    def test_forward_board_pose_survives_distorted_radius_above_opencv_clip(self):
        kb = Kb4Model(150., 151., 720., 719., .6, 0., 0., 0.)
        grid = board_pnp.grid_points_3d(dict(tagCols=6, tagRows=6, tagSpacing=.3, tagSize=.024))
        translation = np.array([.5, -.1, .3])
        detections = {i: kb.project(points+translation) for i, points in grid.items()}
        radius = np.linalg.norm((np.concatenate(list(detections.values()))-[kb.cx, kb.cy])/[kb.fx, kb.fy], axis=1)
        self.assertGreater(radius.max(), np.pi/2)
        pose = board_pnp.pnp_from_detections(detections, Kb4(*vars(kb).values()), grid)
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose['T_bc'][:3, 3], -translation, atol=1e-7)

    def test_narrow_rays_still_match_opencv_and_crop_resize_conventions(self):
        kb = Kb4Model(1000., 1001., 719.5, 719.5, .02, -.001, 0., 0.)
        validation = fit.validate(kb, kb, 1440)
        self.assertEqual(validation['opencv']['inverse_mismatch_samples'], 0)
        self.assertLess(validation['roundtrip_pixels']['maximum'], 1e-9)
        pixels = np.array([[0., 0.], [720., 600.], [1439., 1439.]])
        np.testing.assert_allclose(kb.unproject(pixels), kb.full_view().unproject(pixels+240), atol=1e-12)
        from software.camera.x3.frame import scale_intrinsics
        scaled = scale_intrinsics(vars(kb), 1440, 720)
        np.testing.assert_allclose(Kb4Model(**scaled).unproject((pixels+.5)*.5-.5), kb.unproject(pixels), atol=1e-12)

    def test_folds_and_invalid_domains_cannot_be_silently_accepted(self):
        kb = Kb4Model(500., 500., 720., 720., -1., 0., 0., 0.)
        self.assertLess(fit.minimum_radial_derivative(kb, 1.), 0.)
        self.assertTrue(np.isnan(kb.project([[0., 0., -1.]])).all())
        self.assertEqual(fit.stats([1., np.nan])['invalid'], 1)
        mei = Mei(.001, .001, -10000., -10000., 2., 0., 0., 0., 0., 0.)
        with self.assertRaisesRegex(ValueError, '有效射线不足'):
            fit.fit(mei)


if __name__ == '__main__':
    unittest.main()
