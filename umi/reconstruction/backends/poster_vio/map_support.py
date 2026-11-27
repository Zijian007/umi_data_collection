"""Local metric uncertainty of front-map observations in the published poster frame."""

import gtsam
import numpy as np
from scipy.stats import chi2

from umi.reconstruction.backends.poster_vio.front_map import placement_keys
from umi.reconstruction.estimation.fusion.graph import logR, skew


# Six-dimensional unrobustified pose innovation. A downweighted gross outlier
# must not establish coverage merely because it remains in the robust graph.
_INNOVATION_LIMIT = float(chi2.ppf(.99, 6))


def placement_jacobian(board_from_map, map_point_m, map_from_body):
    """Board-frame center and local body angle vs right-local board/map/log-scale."""
    R = board_from_map[:3, :3]
    center = R @ map_point_m + board_from_map[:3, 3]
    J = np.zeros((6, 13))
    J[:3, :3] = -(R @ map_from_body).T
    J[:3, 6:9] = map_from_body.T
    J[3:, :3] = skew(center)
    J[3:, 3:6] = -np.eye(3)
    J[3:, 6:9] = -R @ skew(map_point_m)
    J[3:, 9:12] = R
    J[3:, 12] = R @ map_point_m
    return J


def _worst_axis_std(cov):
    return np.sqrt(np.maximum(np.linalg.eigvalsh(cov)[-1], 0.))


def assess_segment(segment, board_from_map, scale, joint_cov, board_from_bodies, lever, item):
    """Screen each actual ORB exposure, with shared placement uncertainty retained."""
    std_log_scale = float(np.sqrt(joint_cov[-1, -1]))
    report = dict(status='evaluated', relative_scale_std=std_log_scale,
                  maximum_relative_scale_std=item.fusion_parameters.maximum_relative_scale_std,
                  maximum_position_std_m=item.parameters.tag_maximum_position_std_m,
                  maximum_rotation_std_deg=item.parameters.tag_maximum_rotation_std_deg,
                  maximum_innovation_squared=_INNOVATION_LIMIT,
                  supported_imu_times_s=[], frames=[])
    if std_log_scale > item.fusion_parameters.maximum_relative_scale_std:
        report['status'] = 'uncertain_scale'
        return report
    R = board_from_map[:3, :3]
    local_translation = scale * R
    for t, p, M, C, body in zip(segment.times_s, segment.positions, segment.body_rotations,
                                segment.covariances, board_from_bodies):
        J = placement_jacobian(board_from_map, scale * p, M)
        shared = J @ joint_cov @ J.T
        # Worst-axis standard deviations add conservatively when the map
        # placement and the measured pose may be correlated.
        pos_std = _worst_axis_std(shared[3:, 3:]) + _worst_axis_std(local_translation @ C[3:, 3:] @ local_translation.T)
        rot_std = np.degrees(_worst_axis_std(shared[:3, :3]) + _worst_axis_std(C[:3, :3]))
        angle = logR((R @ M).T @ body[:3, :3])
        center = body[:3, 3] + body[:3, :3] @ lever
        displacement = R.T @ (center - board_from_map[:3, 3]) / scale - p
        innovation = np.r_[angle, displacement]
        squared = float(innovation @ np.linalg.solve(C, innovation))
        accepted = bool(pos_std <= item.parameters.tag_maximum_position_std_m
                        and rot_std <= item.parameters.tag_maximum_rotation_std_deg
                        and squared <= _INNOVATION_LIMIT)
        report['frames'].append(dict(imu_time_s=float(t), position_std_bound_m=float(pos_std),
                                     rotation_std_bound_deg=float(rot_std), innovation_squared=squared,
                                     position_supported=accepted))
        if accepted:
            report['supported_imu_times_s'].append(float(t))
    return report


def evaluate(graph, optimized, solved):
    """Bound map uncertainty at the final estimate without credit from natural landmarks."""
    # ORB and natural tracks can use the same image texture. Dropping the
    # natural factors for this diagnostic also removes their nuisance depths,
    # including landmarks with unobservable depth. The fitted trajectory is untouched.
    certificate = gtsam.NonlinearFactorGraph()
    fixed_board = False
    for i in range(graph.graph.size()):
        factor = graph.graph.at(i)
        if isinstance(factor, gtsam.NonlinearEqualityPose3) and list(factor.keys()) == [graph.board_key]:
            fixed_board = True
            # A hard gauge has no Gaussian variance. Use an auxiliary unit
            # prior, then condition on its value exactly in the joint marginal.
            certificate.add(gtsam.PriorFactorPose3(graph.board_key, optimized.atPose3(graph.board_key),
                                                   gtsam.noiseModel.Unit.Create(6)))
        elif not any(chr(gtsam.Symbol(k).chr()) == 'p' for k in factor.keys()):
            certificate.add(factor)
    try:
        marginals = gtsam.Marginals(certificate, optimized)
    except RuntimeError:
        marginals = None
    board = optimized.atPose3(graph.board_key)
    for arm_index, item in enumerate(graph.arms):
        for i, segment in enumerate(item.front_maps):
            evidence = solved.arm_evidence[item.arm]['front_maps'][i]
            map_key, scale_key = placement_keys(arm_index, i)
            keys = [graph.board_key, map_key, scale_key]
            support = dict(status='marginal_unavailable', supported_imu_times_s=[])
            evidence['position_support'] = support
            if marginals is None:
                continue
            try:
                joint = marginals.jointMarginalCovariance(keys)
                covariance = np.block([[joint.at(a, b) for b in keys] for a in keys])
                covariance = (covariance + covariance.T) * .5
                if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance)[0] <= 0:
                    continue
                if fixed_board:
                    cross = covariance[6:, :6]
                    covariance[6:, 6:] -= cross @ np.linalg.solve(covariance[:6, :6], cross.T)
                    covariance[:6, :] = 0.
                    covariance[:, :6] = 0.
                    if np.linalg.eigvalsh(covariance[6:, 6:])[0] <= 0:
                        continue
                placement = board.between(optimized.atPose3(map_key)).matrix()
                nodes = segment.nodes(item.times_s)
                pose_key = graph.keys_by_arm[item.arm].pose
                bodies = [board.between(optimized.atPose3(pose_key(int(n)))).matrix() for n in nodes]
                lever = np.linalg.inv(item.natural_cameras['front'].T_cam_imu)[:3, 3]
                evidence['position_support'] = assess_segment(segment, placement, evidence['final_scale'],
                                                             covariance, bodies, lever, item)
            except (RuntimeError, np.linalg.LinAlgError):
                continue
