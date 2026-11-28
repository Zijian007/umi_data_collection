"""场景与独立图共用的海报因子；从原场景实现提取，保留解析 Jacobian。"""
import numpy as np
from scipy.spatial.transform import Rotation

from umi.core.geometry import skew
from umi.devices.x3.factory_geometry import Kb4Model
from umi.perception.poster.geometry import (
    PosterGeometryError,
    residual_and_jacobians,
    rotation_correction_jacobian,
    skew_batch,
    tag_camera_points,
    tag_pose_jacobians,
    transform,
)
from umi.perception.poster.quality import corner_sigma_px

_MIN_NATURAL_DEPTH_M = 1e-3
_TAG_DEPTH_BARRIER_Z_MIN_M = 1e-3
_TAG_DEPTH_BARRIER_TAU_M = 2e-4
_TAG_DEPTH_BARRIER_SIGMA_M = 1e-4


def _natural_camera_point(body, point, T_camera_imu):
    in_body = body[:3, :3].T @ (point - body[:3, 3])
    return in_body, T_camera_imu[:3, :3] @ in_body + T_camera_imu[:3, 3]


def natural_track_error(state, pose_key, point_key, T_camera_imu, model, observed):
    """Return one natural-feature reprojection residual and its Jacobians.

    The point is expressed in the graph world frame and the pose maps IMU
    coordinates to that world frame, as in the inertial graph. GTSAM's Pose3 right
    perturbation is used here as in :func:`residual_and_jacobians`, so the
    factor can share the same state with tag corners without introducing a
    second trajectory representation.
    """
    body = state.atPose3(pose_key).matrix()
    point = np.asarray(state.atPoint3(point_key), dtype=float)
    in_body, in_camera = _natural_camera_point(body, point, T_camera_imu)
    if (not np.isfinite(in_camera).all() or in_camera[2] <= _MIN_NATURAL_DEPTH_M):
        # Keep native LM finite while refinement removes the invalid landmark
        # observation from the graph.  This follows Specialist's cheirality
        # handling and avoids aborting an otherwise recoverable solve.
        zero_residual = np.zeros(2, dtype=float)
        zero_pose = np.zeros((2, 6), dtype=float)
        zero_point = np.zeros((2, 3), dtype=float)
        return zero_residual, zero_pose, zero_point
    projection = model.project_jacobian(in_camera[None, ...])[0]
    residual = model.project(in_camera[None, ...])[0] - np.asarray(observed, dtype=float)
    body_twist = np.concatenate((skew(in_body), -np.eye(3)), axis=1)
    pose_jacobian = projection @ T_camera_imu[:3, :3] @ body_twist
    point_jacobian = projection @ T_camera_imu[:3, :3] @ body[:3, :3].T
    return residual, pose_jacobian, point_jacobian


def native_factors():
    try:
        import _umi_gtsam_factors
    except ModuleNotFoundError as error:
        if error.name != '_umi_gtsam_factors':
            raise
        return None
    except ImportError as error:
        # An upgraded wheel can no longer supply the extension's hashed shared library.
        import warnings
        warnings.warn(f'GTSAM 视觉因子扩展加载失败，使用 Python 实现：{error}', RuntimeWarning)
        return None
    return _umi_gtsam_factors


def _kb4_intrinsics(model):
    return (model.fx, model.fy, model.cx, model.cy, model.k1, model.k2, model.k3, model.k4)


def add_natural_track_factors(graph, values, *, pose_key_fn, point_key_fn,
                              T_camera_imu, model, tracks, sigma_px=1.5,
                              huber=2.0):
    """Insert robust reprojection factors for task-local natural tracks.

    Each track dictionary contains ``track_id``, ``node_indices``, ``pixels``
    and ``initial_point_m``.  The landmark is shared by all observations of a
    track, while every observation connects to the arm's inertial pose node.
    """
    import gtsam

    native = native_factors() if type(model) is Kb4Model else None
    intrinsics = _kb4_intrinsics(model) if native is not None else None

    for track in tracks:
        point_key = point_key_fn(int(track['track_id']))
        values.insert(point_key, gtsam.Point3(*np.asarray(track['initial_point_m'], dtype=float)))
        nodes = np.asarray(track['node_indices'], dtype=int)
        pixels = np.asarray(track['pixels'], dtype=float)
        # Both implementations apply one robust weight per batch of up to four observations.
        for start in range(0, len(nodes), 4):
            batch_nodes = nodes[start:start + 4]
            batch_pixels = pixels[start:start + 4]
            pose_keys = [pose_key_fn(int(node)) for node in batch_nodes]
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(float(huber)),
                gtsam.noiseModel.Isotropic.Sigma(2 * len(batch_nodes), float(sigma_px)))

            if intrinsics is not None:
                graph.add(native.NaturalTrackFactor(noise, pose_keys + [point_key],
                                                    T_camera_imu, intrinsics, batch_pixels))
                continue

            def error(factor, state, jacobians, batch_pixels=batch_pixels,
                      pose_keys=pose_keys, point_key=point_key,
                      batch_size=len(batch_nodes)):
                if jacobians is None:
                    point = state.atPoint3(point_key)
                    in_camera = np.array([_natural_camera_point(state.atPose3(key).matrix(), point, T_camera_imu)[1]
                                          for key in pose_keys])
                    valid = np.isfinite(in_camera).all(axis=1) & (in_camera[:, 2] > _MIN_NATURAL_DEPTH_M)
                    residuals = np.zeros_like(batch_pixels)
                    if valid.any():
                        residuals[valid] = model.project(in_camera[valid]) - batch_pixels[valid]
                    return residuals.ravel()

                residuals = []
                pose_derivatives = []
                point_derivatives = []
                for pose_key, observed in zip(pose_keys, batch_pixels):
                    residual, pose_jacobian, point_jacobian = natural_track_error(
                        state, pose_key, point_key, T_camera_imu, model, observed)
                    residuals.append(residual)
                    pose_derivatives.append(pose_jacobian)
                    point_derivatives.append(point_jacobian)
                for index, derivative in enumerate(pose_derivatives):
                    block = np.zeros((2 * batch_size, 6), dtype=float)
                    block[2 * index:2 * index + 2] = derivative
                    jacobians[index] = block
                jacobians[-1] = np.vstack(point_derivatives)
                return np.concatenate(residuals)

            graph.add(gtsam.CustomFactor(noise, pose_keys + [point_key], error))


def rear_rotation_error(state, pose_key, board_key, seat_key, base_ext, corners, observed, model):
    body, board = state.atPose3(pose_key).matrix(), state.atPose3(board_key).matrix()
    correction = state.atRot3(seat_key).matrix()
    ext = base_ext.copy()
    ext[:3, :3] = correction @ base_ext[:3, :3]
    residual, body_jac, board_jac, projection, in_body = residual_and_jacobians(
        body, board, ext, corners, observed, model, return_projection=True)
    seat_jac = rotation_correction_jacobian(correction, base_ext[:3, :3], in_body, projection)
    return residual, body_jac, board_jac, seat_jac


def add_rear_seat(graph, values, rear_key, parameters):
    import gtsam
    if parameters.tag_rear_rotation_refinement_enabled:
        values.insert(rear_key, gtsam.Rot3())
        graph.add(gtsam.PriorFactorRot3(rear_key, gtsam.Rot3(),
            gtsam.noiseModel.Isotropic.Sigma(3, parameters.tag_rear_rotation_prior_sigma_rad)))


def _per_tag_huber_error(project, sigmas, threshold):
    """Use one Huber weight per whitened 8-vector with batched projection.

    For s = ||r|| and w = min(1, k/s), 2*Huber(s) = w*s² + k*max(s-k, 0).
    The extra scalar per tag accounts for the cost offset. During a GTSAM
    linearization the IRLS weights and offsets are frozen, so the normal
    equations match independent native Robust(Huber) factors.
    """
    def error(factor, state, jacobians):
        blocks = None if jacobians is None else [None] * len(jacobians)
        whitened = (project(factor, state, blocks) / sigmas).reshape(-1, 8)
        norms = np.linalg.norm(whitened, axis=1)
        sqrt_weight = np.sqrt(threshold / np.maximum(norms, threshold))
        if blocks is not None:
            scale = np.repeat(sqrt_weight, 8) / sigmas
            for index, block in enumerate(blocks):
                jacobians[index] = np.vstack((scale[:, None] * block,
                                              np.zeros((len(norms), block.shape[1]))))
        offset = np.sqrt(threshold * np.maximum(norms - threshold, 0.0))
        return np.concatenate(((sqrt_weight[:, None] * whitened).ravel(), offset))
    return error


def _tag_rows_by_node(tag_indices):
    rows_by_node = {}
    for row, node in enumerate(tag_indices):
        rows_by_node.setdefault(int(node), []).append(row)
    return rows_by_node


def add_tag_factors(graph, *, pose_key, board_key, rear_key, base_ext, model, points, pixels,
                    tag_indices, parameters, per_tag_huber=False, native=None):
    import gtsam
    enabled = parameters.tag_rear_rotation_refinement_enabled
    corner_factor = getattr(native, 'PosterCornerFactor', None) if type(model) is Kb4Model else None
    intrinsics = _kb4_intrinsics(model) if corner_factor is not None else None
    native_per_tag = getattr(corner_factor, 'per_tag_huber', None)
    points, pixels = np.asarray(points), np.asarray(pixels)
    # A frame usually contains several board tags.  They share one pose key,
    # so one small factor per frame batch avoids entering the Python callback
    # once per tag while retaining every corner residual and Jacobian.  Keep
    # batches bounded to avoid changing the sparsity pattern across many
    # unrelated measurements or creating very large robust residual vectors.
    for node, node_rows in _tag_rows_by_node(tag_indices).items():
        body_key = pose_key(node)
        keys = [body_key, board_key] + ([rear_key] if enabled else [])
        for start in range(0, len(node_rows), 8):
            rows = node_rows[start:start + 8]
            corners = points[rows].reshape(-1, 3)
            observed = pixels[rows].reshape(-1, 2)
            sigmas = np.repeat(corner_sigma_px(observed, model, parameters), 2)

            if per_tag_huber and native_per_tag is not None:
                graph.add(native_per_tag(keys, base_ext, intrinsics, corners, observed, sigmas, parameters.tag_huber))
                continue
            if not per_tag_huber:
                noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(parameters.tag_huber),
                    gtsam.noiseModel.Diagonal.Sigmas(sigmas))
                if corner_factor is not None:
                    graph.add(corner_factor(noise, keys, base_ext, intrinsics, corners, observed))
                    continue

            def error(factor, state, jacobians, corners=corners, observed=observed, body_key=body_key):
                if enabled and jacobians is not None:
                    residual, jacobians[0], jacobians[1], jacobians[2] = rear_rotation_error(
                        state, body_key, board_key, rear_key, base_ext, corners, observed, model)
                    return residual

                body = state.atPose3(body_key).matrix()
                board = state.atPose3(board_key).matrix()
                if jacobians is None:
                    ext = base_ext
                    if enabled:
                        ext = base_ext.copy()
                        ext[:3, :3] = state.atRot3(rear_key).matrix() @ base_ext[:3, :3]
                    _, in_camera = tag_camera_points(body, board, ext, corners)
                    return (model.project(in_camera) - observed).ravel()

                residual, jacobians[0], jacobians[1] = residual_and_jacobians(
                    body, board, base_ext, corners, observed, model)
                return residual

            if per_tag_huber:
                error = _per_tag_huber_error(error, sigmas, parameters.tag_huber)
                noise = gtsam.noiseModel.Unit.Create(len(sigmas) + len(rows))
            graph.add(gtsam.CustomFactor(noise, keys, error))


def _depth_barrier(z):
    """Smooth one-sided penalty for a tag corner approaching the camera plane."""
    deficit = (_TAG_DEPTH_BARRIER_Z_MIN_M - np.asarray(z, dtype=float)) / _TAG_DEPTH_BARRIER_TAU_M
    active = deficit > -8.0
    residual = np.zeros_like(deficit)
    derivative = np.zeros_like(deficit)
    if np.any(active):
        value = deficit[active]
        residual[active] = _TAG_DEPTH_BARRIER_TAU_M * np.logaddexp(0.0, value)
        derivative[active] = -1.0 / (1.0 + np.exp(-value))
    return residual / _TAG_DEPTH_BARRIER_SIGMA_M, derivative / _TAG_DEPTH_BARRIER_SIGMA_M


def depth_barrier_and_jacobians(body, board, T_camera_imu, points, *, compute_jacobians=True):
    """Depth-only residual and Pose3 Jacobians for fixed-board tag corners."""
    in_body, in_camera = tag_camera_points(body, board, T_camera_imu, points)
    residual, derivative = _depth_barrier(in_camera[:, 2])
    if not compute_jacobians:
        return residual, None, None
    depth_projection = np.array([[0., 0., 1.]])
    body_jac, board_jac = tag_pose_jacobians(body, board, T_camera_imu, points, in_body, depth_projection)
    return residual, derivative[:, None] * body_jac[:, 0, :], derivative[:, None] * board_jac[:, 0, :]


def add_tag_depth_barrier_factors(graph, *, pose_key, board_key, base_ext, points, tag_indices, native=None):
    """Add depth-only factors for training tag rows without using held pixels."""
    import gtsam

    depth_factor = getattr(native, 'PosterDepthFactor', None)
    for node, rows in _tag_rows_by_node(tag_indices).items():
        corners = points[rows].reshape(-1, 3)
        keys = [pose_key(node), board_key]
        noise = gtsam.noiseModel.Isotropic.Sigma(len(rows) * 4, 1.0)
        if depth_factor is not None:
            graph.add(depth_factor(noise, keys, base_ext, corners,
                                   _TAG_DEPTH_BARRIER_Z_MIN_M, _TAG_DEPTH_BARRIER_TAU_M,
                                   _TAG_DEPTH_BARRIER_SIGMA_M))
            continue

        def error(factor, state, jacobians, corners=corners):
            body = state.atPose3(factor.keys()[0]).matrix()
            board = state.atPose3(factor.keys()[1]).matrix()
            residuals, body_jacobians, board_jacobians = depth_barrier_and_jacobians(
                body, board, base_ext, corners, compute_jacobians=jacobians is not None)
            if jacobians is not None:
                jacobians[0] = body_jacobians
                jacobians[1] = board_jacobians
            return residuals

        graph.add(gtsam.CustomFactor(noise, keys, error))


def base_tag_corners_m(size_m):
    """Return ArUco corner coordinates in the tag frame, in detector order."""
    if not np.isfinite(size_m) or size_m <= 0:
        raise ValueError('base tag 尺寸必须为正')
    return np.array([[-1., 1., 0.], [1., 1., 0.],
                     [1., -1., 0.], [-1., -1., 0.]]) * float(size_m) / 2.


def _base_tag_noise(pixels, model, parameters):
    import gtsam
    sigma = corner_sigma_px(np.asarray(pixels, dtype=float), model, parameters['scene'])
    return gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(float(parameters['scene'].tag_huber)),
        gtsam.noiseModel.Diagonal.Sigmas(np.repeat(sigma, 2)))


def base_tag_self_error(tag_pose, T_cam_imu, model, pixels, size_m):
    """Eight corner residuals and the analytic right-Pose3 mount Jacobian."""
    corners = base_tag_corners_m(size_m)
    in_imu = transform(tag_pose.matrix(), corners)
    in_camera = transform(T_cam_imu, in_imu)
    projection = model.project_jacobian(in_camera)
    twist = np.concatenate((-skew_batch(corners),
                            np.broadcast_to(np.eye(3), (4, 3, 3))), axis=2)
    jac = np.einsum('nij,jk,nkl->nil', projection,
                   T_cam_imu[:3, :3] @ tag_pose.rotation().matrix(), twist)
    return (model.project(in_camera) - pixels).ravel(), jac.reshape(8, 6)


def cross_base_tag_error(observer_pose, owner_pose, tag_pose, T_cam_imu,
                         model, pixels, size_m):
    """Residual/Jacobians for observer, owner and rigid tag mount variables."""
    corners = base_tag_corners_m(size_m)
    in_owner = transform(tag_pose.matrix(), corners)
    residual, observer_jac, owner_jac = residual_and_jacobians(
        observer_pose.matrix(), owner_pose.matrix(), T_cam_imu, in_owner, pixels, model)
    # A right perturbation of the tag mount is the owner's right perturbation
    # transported by Ad(T_imu_base_tag). Reuse its projection Jacobian.
    tag_jac = owner_jac @ tag_pose.AdjointMap()
    return residual, observer_jac, owner_jac, tag_jac


def add_base_tag_self_factor(graph, *, tag_key, T_cam_imu, model, pixels, parameters):
    """Self sightings measure the mount, never the world trajectory."""
    import gtsam

    def error(factor, state, jacobians):
        residual, jac = base_tag_self_error(state.atPose3(tag_key), T_cam_imu,
                                            model, pixels, parameters['size_m'])
        if jacobians is not None:
            jacobians[0] = jac
        return residual

    graph.add(gtsam.CustomFactor(_base_tag_noise(pixels, model, parameters), [tag_key], error))


def add_cross_base_tag_factor(graph, *, observer_key, owner_key, tag_key,
                              T_cam_imu, model, pixels, parameters):
    """One coupled factor for two moving IMUs and their rigid owner tag mount."""
    import gtsam

    def error(factor, state, jacobians):
        residual, *derivatives = cross_base_tag_error(
            state.atPose3(observer_key), state.atPose3(owner_key), state.atPose3(tag_key),
            T_cam_imu, model, pixels, parameters['size_m'])
        if jacobians is not None:
            for index, jac in enumerate(derivatives):
                jacobians[index] = jac
        return residual

    graph.add(gtsam.CustomFactor(_base_tag_noise(pixels, model, parameters),
                                [observer_key, owner_key, tag_key], error))


def add_base_tag_prior(graph, values, tag_key, initial, *, rotation_sigma_rad,
                       translation_sigma_m):
    """A soft seed prior is regularization, not an independent mount measurement."""
    import gtsam

    pose = gtsam.Pose3(initial)
    values.insert(tag_key, pose)
    graph.add(gtsam.PriorFactorPose3(tag_key, pose, gtsam.noiseModel.Diagonal.Sigmas(
        np.array([rotation_sigma_rad] * 3 + [translation_sigma_m] * 3))))


def solved_rear_extrinsic(optimized, rear_key, base_ext, parameters):
    final_ext = base_ext.copy()
    correction_deg = 0.
    if parameters.tag_rear_rotation_refinement_enabled:
        correction = optimized.atRot3(rear_key).matrix()
        final_ext[:3, :3] = correction @ base_ext[:3, :3]
        correction_deg = float(np.degrees(Rotation.from_matrix(correction).magnitude()))
    return final_ext, correction_deg


def rear_rotation_evidence(correction_deg, parameters):
    evidence = dict(enabled=parameters.tag_rear_rotation_refinement_enabled,
        prior_sigma_rad=parameters.tag_rear_rotation_prior_sigma_rad,
        maximum_correction_deg=parameters.tag_rear_rotation_maximum_correction_deg,
        correction_deg=correction_deg)
    if not np.isfinite(correction_deg) or correction_deg > parameters.tag_rear_rotation_maximum_correction_deg:
        raise PosterGeometryError(f'环境镜头旋转修正超限：{correction_deg:.3f} deg',
                                  metrics=dict(rear_rotation_refinement=evidence))
    return evidence
