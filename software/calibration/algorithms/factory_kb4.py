"""Fit optical KB4 surrogates to factory MEI rays; no measured calibration or poses.

The nominal sensor disc is an explicit hypothesis, not a calibrated validity mask.
Validation points and boundary probes never enter fitting or initial-value selection.
The comparison uses this as a diagnostic; solve.factory reuses the fit and adds
forward-domain, recording-consistency, and independent A-image requirements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from scipy.optimize import least_squares
import cv2
import numpy as np

from software.camera.x3.factory import Kb4Model


FIT_VERSION = 1
FIT_SIDE = 65
VALIDATION_SIDE = 193
SENSOR_RADIUS = 960.


@dataclass(frozen=True)
class ForwardFit:
    fit: dict
    pixels: np.ndarray
    rays: np.ndarray
    conversion: dict


def fit_forward(mei, parameters) -> ForwardFit:
    """Fit and check the common <=85° domain used by calibration and fallback."""
    fitted = fit(mei)
    model = Kb4Model(**fitted['kb4'])
    pixels = grid(1440, 129, .37)
    rays = mei.unproject(pixels)
    domain = (in_disc(pixels, 1440) & np.isfinite(rays).all(axis=1)
              & (rays[:, 2] >= np.cos(np.deg2rad(85.))))
    if np.count_nonzero(domain) < 200 or fitted['views']['1440']['minimum_radial_derivative'] <= 0:
        raise ValueError('工厂 KB4 缺少单调、有效的前向视野')
    residual = np.linalg.norm(model.project(rays[domain])-pixels[domain], axis=1)
    conversion = {key: value for key, value in stats(residual).items() if key not in ('total', 'invalid')}
    if (not np.isfinite(residual).all() or conversion['rms'] > parameters.maximum_fit_rms_px
            or conversion['p99'] > parameters.maximum_fit_p99_px):
        raise ValueError(f'工厂 MEI → KB4 转换误差超限：{conversion}')
    return ForwardFit(fitted, pixels[domain], rays[domain], conversion)


def stats(values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    result = dict(total=len(values), count=len(finite), invalid=int(len(values)-len(finite)))
    if not len(finite):
        return dict(result, rms=None, median=None, p95=None, p99=None, maximum=None)
    return dict(result, rms=float(np.sqrt(np.mean(finite**2))), median=float(np.median(finite)),
                p95=float(np.percentile(finite, 95)), p99=float(np.percentile(finite, 99)),
                maximum=float(finite.max()))


def grid(size, side, phase):
    axis = (np.arange(side)+phase)*(size-1)/side
    x, y = np.meshgrid(axis, axis)
    return np.column_stack((x.ravel(), y.ravel()))


def in_disc(pixels, size):
    return np.linalg.norm(pixels-(size-1)/2, axis=1) <= SENSOR_RADIUS+1e-8


def boundary(size):
    axis = np.linspace(0., size-1., 1025)
    zero = np.zeros_like(axis)
    last = zero+size-1
    edges = np.concatenate([np.column_stack(pair) for pair in
                            ((axis, zero), (axis, last), (zero, axis), (last, axis))])
    angle = np.arange(4096)*2*np.pi/4096
    circle = (size-1)/2 + SENSOR_RADIUS*np.column_stack((np.cos(angle), np.sin(angle)))
    points = np.concatenate((edges, circle))
    inside = in_disc(points, size) & (points >= 0).all(axis=1) & (points <= size-1).all(axis=1)
    return np.unique(points[inside], axis=0)


def minimum_radial_derivative(model, theta_max):
    """Exact minimum of dr/dtheta on [0, theta_max], via cubic stationary roots."""
    coefficients = np.array([1., 3*model.k1, 5*model.k2, 7*model.k3, 9*model.k4])
    roots = np.polynomial.polynomial.polyroots(np.arange(1, 5)*coefficients[1:])
    interior = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < theta_max**2]
    return float(np.min(np.polynomial.polynomial.polyval([0., theta_max**2, *interior], coefficients)))


def angular_error(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a*b, axis=1)))


def errors(mei, kb4, pixels):
    rays = mei.unproject(pixels)
    predicted = kb4.project(rays)
    recovered = kb4.unproject(pixels)
    error = np.linalg.norm(predicted-pixels, axis=1)
    return dict(pixel=error, angle=angular_error(rays, recovered), rays=rays,
                roundtrip=np.linalg.norm(kb4.project(recovered)-pixels, axis=1))


def validate(mei, kb4, size):
    """Independent shifted grid, plus separately reported exact boundary probes."""
    all_pixels = grid(size, VALIDATION_SIDE, .37)
    pixels = all_pixels[in_disc(all_pixels, size)]
    evidence = errors(mei, kb4, pixels)
    border = boundary(size)
    edge_evidence = errors(mei, kb4, border)
    # The 1440 crop is centred in either view. These regions deliberately overlap.
    xy = np.abs(pixels-(size-1)/2)
    masks = dict(all=np.ones(len(pixels), bool), centre=xy.max(axis=1) <= .4*719.5,
                 edge=(xy.max(axis=1) >= .8*719.5) & (xy.max(axis=1) <= 719.5),
                 corners=(xy.min(axis=1) >= .8*719.5) & (xy.max(axis=1) <= 719.5))
    if size == 1920:
        masks['outside_fitted_crop'] = xy.max(axis=1) > 719.5
    rays = np.concatenate((evidence['rays'], edge_evidence['rays']))
    finite = np.isfinite(rays).all(axis=1)
    theta_max = float(np.arctan2(np.linalg.norm(rays[finite, :2], axis=1), rays[finite, 2]).max()) if finite.any() else 0.
    result = dict(grid_side=VALIDATION_SIDE, grid_phase=.37,
        rectangle_samples=len(all_pixels), disc_samples=len(pixels),
        outside_nominal_disc=int(len(all_pixels)-len(pixels)),
        invalid_mei=int((~np.isfinite(evidence['rays']).all(axis=1)).sum()),
        regions={name: dict(pixels=stats(evidence['pixel'][mask]), degrees=stats(evidence['angle'][mask]))
                 for name, mask in masks.items()},
        boundary=dict(pixels=stats(edge_evidence['pixel']), degrees=stats(edge_evidence['angle'])),
        roundtrip_pixels=stats(evidence['roundtrip']), theta_max_degrees=float(np.degrees(theta_max)),
        minimum_radial_derivative=minimum_radial_derivative(kb4, theta_max))
    # Diagnostic comparison only: OpenCV clips theta_d and returns a forward
    # pinhole bearing. Neither this comparison nor OpenCV drives the fit or PnP.
    K = np.array([[kb4.fx, 0, kb4.cx], [0, kb4.fy, kb4.cy], [0, 0, 1.]])
    D = np.array([kb4.k1, kb4.k2, kb4.k3, kb4.k4])
    rays = evidence['rays']
    front = np.isfinite(rays).all(axis=1) & (rays[:, 2] > 1e-6)
    cv_pixels = cv2.fisheye.projectPoints(rays[front, None], np.zeros(3), np.zeros(3), K, D)[0][:, 0] if front.any() else np.empty((0, 2))
    normal = cv2.fisheye.undistortPoints(pixels[:, None], K, D)[:, 0]
    cv_rays = np.column_stack((normal, np.ones(len(normal))))
    cv_rays /= np.linalg.norm(cv_rays, axis=1)[:, None]
    cv_roundtrip = np.linalg.norm(kb4.project(cv_rays)-pixels, axis=1)
    cv_angle = angular_error(cv_rays, kb4.unproject(pixels))
    result['opencv'] = dict(front_facing_samples=int(front.sum()),
        version=cv2.__version__, role='compatibility_diagnostic_only',
        projection_agreement_pixels=stats(np.linalg.norm(cv_pixels-kb4.project(rays[front]), axis=1)),
        inverse_agreement_degrees=stats(cv_angle), roundtrip_pixels=stats(cv_roundtrip),
        inverse_mismatch_samples=int((~np.isfinite(cv_angle) | (cv_angle > 1e-5)
                                      | ~np.isfinite(cv_roundtrip) | (cv_roundtrip > .01)).sum()))
    # The bundled ORB-SLAM3 KannalaBrandt8 inverse also clips theta_d at pi/2
    # and returns (x, y, 1). An atan2 forward projection does not fix its inverse.
    normalized_radius = np.linalg.norm((pixels-[kb4.cx, kb4.cy])/[kb4.fx, kb4.fy], axis=1)
    result['orb_slam3_inverse_limits'] = dict(
        source='src/CameraModels/KannalaBrandt8.cpp:unproject in bundled ORB-SLAM3',
        clipped_radius_samples=int((normalized_radius > np.pi/2).sum()),
        backward_or_horizon_reference_samples=int((np.isfinite(rays).all(axis=1) & (rays[:, 2] <= 0)).sum()),
        engine_roundtrip_tested=False)
    return result


def fit(mei):
    """Return a diagnostic candidate; success does not grant calibration acceptance."""
    all_pixels = grid(1440, FIT_SIDE, .5)
    pixels = all_pixels[in_disc(all_pixels, 1440)]
    rays = mei.unproject(pixels)
    valid = np.isfinite(rays).all(axis=1)
    if valid.sum() < 64:
        raise ValueError('工厂 MEI 有效射线不足，无法拟合 KB4')
    pixels, rays = pixels[valid], rays[valid]
    radius = np.linalg.norm(rays[:, :2], axis=1)
    theta = np.arctan2(radius, rays[:, 2])
    direction = np.divide(rays[:, :2], radius[:, None], out=np.zeros_like(rays[:, :2]), where=radius[:, None] > 0)
    powers = theta[:, None]**np.arange(3, 10, 2)

    def residual(parameters):
        return (Kb4Model(*parameters).project(rays)-pixels).ravel()

    def jacobian(p):
        radial = theta + powers @ p[4:]
        J = np.zeros((len(pixels), 2, 8))
        J[:, 0, 0], J[:, 1, 1] = direction[:, 0]*radial, direction[:, 1]*radial
        J[:, 0, 2], J[:, 1, 3] = 1., 1.
        J[:, 0, 4:] = p[0]*direction[:, :1]*powers
        J[:, 1, 4:] = p[1]*direction[:, 1:]*powers
        return J.reshape(-1, 8)

    # All starts depend only on MEI and training pixels, never measured A/B/C data.
    seed = np.array([mei.fx/(1+mei.xi), mei.fy/(1+mei.xi), mei.cx, mei.cy, 0, 0, 0, 0])
    target_radius = np.sum((pixels-seed[2:4])/seed[:2]*direction, axis=1)
    radial_seed = seed.copy()
    radial_seed[4:] = np.linalg.lstsq(powers, target_radius-theta, rcond=None)[0]
    lower = [1e-3, 1e-3, -1440., -1440., *([-np.inf]*4)]
    upper = [1e6, 1e6, 2880., 2880., *([np.inf]*4)]
    attempts, solutions = [], []
    for name, start in (('equidistant', seed), ('radial_linear', radial_seed)):
        solved = least_squares(residual, start, jac=jacobian, x_scale='jac', bounds=(lower, upper),
                               max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-10)
        model = Kb4Model(*solved.x)
        derivative = minimum_radial_derivative(model, float(theta.max()))
        rms = float(np.sqrt(np.mean(np.sum(residual(solved.x).reshape(-1, 2)**2, axis=1))))
        usable = bool(solved.success and np.isfinite(solved.x).all() and derivative > 0)
        attempts.append(dict(start=name, converged=bool(solved.success), evaluations=int(solved.nfev),
                             training_rms_px=rms, monotonic=derivative > 0, message=str(solved.message)))
        if usable:
            solutions.append((rms, name, model))
    if not solutions:
        raise ValueError('工厂 MEI → KB4 未收敛到单调、有限的模型')
    _, selected, model = min(solutions, key=lambda item: item[0])
    views = {'1440': validate(mei, model, 1440)}
    validation = views['1440']
    domain_complete = (not validation['invalid_mei']
        and not validation['regions']['all']['degrees']['invalid']
        and not validation['boundary']['degrees']['invalid']
        and validation['minimum_radial_derivative'] > 0)
    return dict(kind='factory_mei_kb4_optical_fit', fit_version=FIT_VERSION,
        status='ok' if domain_complete else 'partial', source='insv_factory_mei',
        diagnostic_only=True, acceptance='not_evaluated', resolution=[1440, 1440],
        domain=dict(shape='centred crop intersect nominal sensor disc', sensor_radius_px=SENSOR_RADIUS,
                    centre_px=[719.5, 719.5], production_mask_verified=False),
        objective='unweighted pixel L2; all 8 KB4 intrinsics; fixed optical coordinate frame',
        selection='lowest training RMS among converged monotonic starts; validation is held out',
        sampling=dict(fit_side=FIT_SIDE, phase=.5, total=len(all_pixels), used=len(pixels),
                      outside_nominal_disc=int((~in_disc(all_pixels, 1440)).sum()), invalid_mei=int((~valid).sum())),
        attempts=attempts, selected_start=selected, mei=asdict(mei), kb4=asdict(model), views=views)
