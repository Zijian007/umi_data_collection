"""KB4 bearing geometry and bounded batches of task-local landmarks.

The staging follows specialist da29719 (MIT); see NOTICE.md. Geometry and
validation use UMI's KB4 model and native-IMU poses.
"""
from __future__ import annotations

import cv2
import numpy as np


def bearing_inliers(previous, current, model, previous_pose, current_pose, *, threshold_px=3., seed=0):
    """Calibrated epipolar/rotation gate plus deterministic essential RANSAC.

    A dominant moving object can fit its own essential matrix; agreement with
    the metric pose seed is required as well. This is not semantic segmentation.
    """
    first, second = model.unproject(previous), model.unproject(current)
    valid = np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
    valid &= (first[:, 2] > .05) & (second[:, 2] > .05)
    relative = np.linalg.inv(current_pose) @ previous_pose
    rotated = first @ relative[:3, :3].T
    distance = np.linalg.norm(relative[:3, 3])
    tolerance = threshold_px / min(model.fx, model.fy)
    if distance < 1e-5:
        rotated /= np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-12)
        valid &= np.linalg.norm(np.cross(rotated, second), axis=1) <= tolerance
        return valid
    normal = np.cross(np.broadcast_to(relative[:3, 3], rotated.shape), rotated)
    denominator = np.linalg.norm(normal, axis=1)
    valid &= np.abs(np.sum(normal * second, axis=1)) <= tolerance * np.maximum(denominator, 1e-12)
    candidates = np.flatnonzero(valid)
    if len(candidates) >= 8:
        cv2.setRNGSeed(int(seed) % 2147483647)
        a, b = first[candidates], second[candidates]
        _, mask = cv2.findEssentialMat(a[:, :2] / a[:, 2:], b[:, :2] / b[:, 2:],
            focal=1., pp=(0., 0.), method=cv2.USAC_DEFAULT, prob=.999, threshold=tolerance)
        if mask is not None:
            valid[candidates] &= mask.ravel().astype(bool)
    return valid


def triangulate_batch(tracks, poses, camera_from_imu, model, *, maximum_rms_px=3., minimum_parallax_deg=.5):
    """At most 64 equal-length tracks per batch; NaN denotes rejected geometry."""
    result = np.full((len(tracks), 3), np.nan)
    lengths = {}
    for index, track in enumerate(tracks):
        lengths.setdefault(len(track['node_indices']), []).append(index)
    inverse = np.linalg.inv(camera_from_imu)
    for length, members in lengths.items():
        for offset in range(0, len(members), 64):
            selected = np.asarray(members[offset:offset + 64])
            nodes = np.stack([tracks[i]['node_indices'] for i in selected])
            pixels = np.stack([tracks[i]['pixels'] for i in selected])
            cameras = poses[nodes] @ inverse
            rays = model.unproject(pixels.reshape(-1, 2)).reshape(-1, length, 3)
            valid = np.isfinite(rays).all(axis=(1, 2)) & (rays[:, :, 2] > 0).all(axis=1)
            rays = np.nan_to_num(rays, nan=0., posinf=0., neginf=0.)
            directions = np.einsum('blij,blj->bli', cameras[:, :, :3, :3], rays)
            directions /= np.maximum(np.linalg.norm(directions, axis=2, keepdims=True), 1e-12)
            dots = directions @ directions.swapaxes(1, 2)
            parallax = np.degrees(np.arccos(np.clip(dots.min(axis=(1, 2)), -1., 1.)))
            valid &= parallax >= minimum_parallax_deg
            projectors = np.eye(3) - directions[:, :, :, None] * directions[:, :, None, :]
            normal = projectors.sum(axis=1)
            rhs = np.einsum('blij,blj->bi', projectors, cameras[:, :, :3, 3])
            eigenvalues = np.linalg.eigvalsh(normal)
            valid &= eigenvalues[:, 0] > np.maximum(eigenvalues[:, -1], 1e-12) / 1e6
            good = np.flatnonzero(valid)
            if not len(good):
                continue
            points = np.linalg.solve(normal[good], rhs[good, :, None])[:, :, 0]
            in_camera = np.einsum('blji,blj->bli', cameras[good, :, :3, :3], points[:, None] - cameras[good, :, :3, 3])
            predicted = model.project(in_camera.reshape(-1, 3)).reshape(-1, length, 2)
            errors = np.linalg.norm(predicted - pixels[good], axis=2)
            passed = (np.isfinite(points).all(axis=1) & np.isfinite(errors).all(axis=1)
                      & (in_camera[:, :, 2] > 0).all(axis=1) & (errors.max(axis=1) <= maximum_rms_px))
            result[selected[good[passed]]] = points[passed]
    return result


def describe(gray, points):
    """ORB descriptors at existing LK corners; IDs preserve compute() filtering."""
    points = np.asarray(points)
    keypoints = [cv2.KeyPoint(float(point[0]), float(point[1]), 31., class_id=i)
                 for i, point in enumerate(points)]
    returned, descriptors = cv2.ORB_create().compute(gray, keypoints)
    if descriptors is None:
        return {}
    return {point.class_id: descriptor.copy() for point, descriptor in zip(returned, descriptors)}


def match_revisits(points, descriptors, bank, camera_pose, model, time_s, *, maximum_gap_s,
                   ratio=.75, maximum_reprojection_px=4.):
    """Unique descriptor matches independently checked by metric reprojection."""
    available = [(key, track) for key, track in bank.items()
                 if 0 < time_s - track['last_time_s'] <= maximum_gap_s]
    if len(available) < 2 or not descriptors:
        return {}
    rows = list(descriptors)
    query = np.asarray([descriptors[row] for row in rows], dtype=np.uint8)
    target = np.asarray([track['descriptor'] for _, track in available], dtype=np.uint8)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(query, target, k=2)
    inverse = np.linalg.inv(camera_pose)
    result, used = {}, set()
    for pair in sorted(matches, key=lambda pair: pair[0].distance if len(pair) else 256):
        if len(pair) < 2 or pair[0].distance >= ratio * pair[1].distance:
            continue
        best = pair[0]
        row, (key, track) = rows[best.queryIdx], available[best.trainIdx]
        point = inverse[:3, :3] @ track['initial_point_m'] + inverse[:3, 3]
        if (key in used or point[2] <= 0 or not np.isfinite(point).all()
                or np.linalg.norm(model.project(point[None])[0] - points[row]) > maximum_reprojection_px):
            continue
        result[row] = key
        used.add(key)
    return result
