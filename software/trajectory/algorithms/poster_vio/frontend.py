"""Streaming front/rear-camera feature tracks in exact raw-video time cells.

Pixels use the calibrated 1440 crop.  Forward/backward optical flow and geometric
triangulation reject invalid tracks before they enter the joint graph. The KB4
geometry and landmark staging follow specialist da29719 (MIT, see NOTICE.md).
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np

from software.camera.x3.frame import CROP, SRC_H, SRC_W, LENS_HALF, crop_xywh
from software.vision.decoding import configure_decoder
from software.trajectory.algorithms.poster_vio.models import FeatureTrackingParameters


def subsample_observations(node_indices, pixels, maximum_observations):
    """保留首尾和均匀分布的中间观测，限制单条轨迹进入联合图的规模。"""
    nodes = np.asarray(node_indices)
    points = np.asarray(pixels)
    if nodes.ndim != 1 or points.shape != (len(nodes), 2):
        raise ValueError('poster_vio 自然特征轨迹观测形状无效')
    if maximum_observations is not None and (type(maximum_observations) is not int
                                             or maximum_observations < 2):
        raise ValueError('poster_vio 自然特征轨迹观测上限必须至少为 2')
    if maximum_observations is None or len(nodes) <= maximum_observations:
        return nodes, points
    selected = np.rint(np.linspace(0, len(nodes) - 1, maximum_observations)).astype(int)
    selected[0], selected[-1] = 0, len(nodes) - 1
    selected = np.unique(selected)
    if len(selected) != maximum_observations:
        selected = np.linspace(0, len(nodes) - 1, maximum_observations, dtype=int)
    return nodes[selected], points[selected]


def triangulate_track(node_indices, pixels, seed_poses, camera_from_imu, model,
                      *, maximum_rms_px=3.0, minimum_parallax_deg=.5):
    """Return a finite, positive-depth landmark, or None for weak geometry."""
    nodes, pixels, poses = map(np.asarray, (node_indices, pixels, seed_poses))
    if (nodes.ndim != 1 or len(nodes) < 2 or not np.issubdtype(nodes.dtype, np.integer)
            or np.any(np.diff(nodes) <= 0) or np.any(nodes < 0) or np.any(nodes >= len(poses))
            or pixels.shape != (len(nodes), 2) or not np.isfinite(pixels).all()):
        raise ValueError('poster_vio 自然特征轨迹形状或节点索引无效')
    rays = model.unproject(pixels)
    if not np.isfinite(rays).all() or np.any(rays[:, 2] <= 0):
        return None
    cameras = poses[nodes] @ np.linalg.inv(camera_from_imu)
    directions = np.einsum('nij,nj->ni', cameras[:, :3, :3], rays)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    parallax = np.degrees(np.arccos(np.clip(np.min(directions @ directions.T), -1, 1)))
    if parallax < minimum_parallax_deg:
        return None
    projections = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    matrix = projections.sum(axis=0)
    if np.linalg.cond(matrix) > 1e6:
        return None
    point = np.linalg.solve(matrix, np.einsum('nij,nj->i', projections, cameras[:, :3, 3]))
    camera_points = np.einsum('nji,nj->ni', cameras[:, :3, :3], point - cameras[:, :3, 3])
    if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
        return None
    errors = np.linalg.norm(model.project(camera_points) - pixels, axis=1)
    if not np.isfinite(errors).all() or np.max(errors) > maximum_rms_px:
        return None
    return point


@dataclass(frozen=True)
class FeatureCameraInput:
    """One camera's exact source frames, IMU graph nodes and frozen geometry."""

    camera: str
    source_frames: np.ndarray
    video_times_s: np.ndarray
    node_indices: np.ndarray
    seed_poses: np.ndarray
    camera_from_imu: np.ndarray
    model: object
    mask_path: Path | None = None
    excluded_polygons: dict = field(default_factory=dict)


class _FeatureTracker:
    def __init__(self, item, *, max_features, maximum_tracks, minimum_track_length, maximum_track_length,
                 parameters, maximum_observations_per_track=None):

        self.item = item
        self.parameters = parameters
        frames, times, nodes = map(np.asarray, (item.source_frames, item.video_times_s, item.node_indices))
        if (item.camera not in LENS_HALF or frames.ndim != 1 or times.ndim != 1 or nodes.ndim != 1
                or len(frames) != len(times) or len(frames) != len(nodes)
                or not np.issubdtype(frames.dtype, np.integer) or np.any(frames < 0)
                or not np.issubdtype(nodes.dtype, np.integer)
                or np.any(nodes < 0) or np.any(nodes >= len(item.seed_poses))
                or np.any(np.diff(frames) <= 0) or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or np.any(np.diff(nodes) <= 0)):
            raise ValueError('poster_vio 自然特征相机、源帧、时间和节点必须有效且一一对应')
        if item.camera == 'front' and item.mask_path is None:
            raise ValueError('poster_vio front 自然特征必须使用冻结遮手图')
        self.mask = np.full((CROP, CROP), 255, dtype=np.uint8)
        if item.mask_path is not None:
            mask = cv2.imread(str(item.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
                raise ValueError('poster_vio 自然特征遮手图必须是可读取的正方形图像')
            # Existing X3 assets mark occlusion with nonzero pixels (ORB blacks
            # those pixels out). OpenCV's feature mask uses the inverse meaning.
            self.mask = np.where(cv2.resize(mask, (CROP, CROP), interpolation=cv2.INTER_NEAREST) == 0,
                                 255, 0).astype(np.uint8)
        self.frames, self.times, self.nodes = frames, times, nodes
        self.camera_from_imu_inverse = np.linalg.inv(item.camera_from_imu)
        self.max_features, self.maximum_tracks = max_features, maximum_tracks
        self.minimum_track_length, self.maximum_track_length = minimum_track_length, maximum_track_length
        self.maximum_observations_per_track = maximum_observations_per_track
        self.histories, self.active, self.retained = {}, {}, {}
        self.descriptors, self.ages, self.revisits, self.bank = {}, {}, {}, {}
        self.pending = []
        self.stats = dict(created_tracks=0, reassociations=0, geometric_rejections=0, triangulation_batches=0)
        self.next_id = 0
        # Reserve output slots across the entire recording. Active tracks keep
        # renewing even after early intervals have filled their archive quota.
        self.bin_count = min(maximum_tracks, max(1, (len(frames) + maximum_track_length - 1) // maximum_track_length))
        self.bin_quotas = [maximum_tracks // self.bin_count + int(i < maximum_tracks % self.bin_count)
                           for i in range(self.bin_count)]
        self.previous = None
        self.selected_index = 0

    @property
    def done(self):
        return self.selected_index == len(self.frames)

    def consume(self, source_index, gray, timestamp):

        if self.done or source_index != self.frames[self.selected_index]:
            return
        if timestamp is None or not np.isclose(timestamp, self.times[self.selected_index], rtol=0, atol=1e-7):
            raise ValueError(f'poster_vio {self.item.camera} 原片时间戳与冻结观测不一致')
        x, y, width, height = crop_xywh(LENS_HALF[self.item.camera])
        current = np.ascontiguousarray(gray[y:y + height, x:x + width])
        mask = self.mask.copy()
        for polygon in self.item.excluded_polygons.get(int(source_index), ()):
            cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 0)
        node = int(self.nodes[self.selected_index])
        if self.previous is not None and self.active:
            ids = list(self.active)
            points = np.asarray([self.active[index] for index in ids], dtype=np.float32)
            predicted, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, current, points, None)
            following = {}
            if predicted is not None and status is not None:
                backward, back_status, _ = cv2.calcOpticalFlowPyrLK(current, self.previous, predicted, None)
                if backward is not None and back_status is not None:
                    valid = (status.ravel().astype(bool) & back_status.ravel().astype(bool)
                             & (np.linalg.norm(backward - points, axis=1) <= 1.)
                             & np.isfinite(predicted).all(axis=1)
                             & (predicted >= 1).all(axis=1) & (predicted < CROP - 1).all(axis=1))
                    candidates = np.flatnonzero(valid)
                    if len(candidates):
                        consistent = bearing_inliers(points[candidates], predicted[candidates], self.item.model,
                            self.item.seed_poses[self.nodes[self.selected_index - 1]] @ self.camera_from_imu_inverse,
                            self.item.seed_poses[node] @ self.camera_from_imu_inverse,
                            threshold_px=self.parameters.poster_vio_geometric_threshold_px,
                            seed=int(source_index) * 2 + int(self.item.camera == 'front'))
                        self.stats['geometric_rejections'] += int(np.count_nonzero(~consistent))
                        valid[candidates] &= consistent
                    for index, point, good in zip(ids, predicted, valid):
                        if good and mask[int(round(point[1])), int(round(point[0]))]:
                            self.histories[index].append((node, point.astype(float)))
                            self.ages[index] += 1
                            if len(self.histories[index]) > self.maximum_track_length:
                                selected = np.unique(np.linspace(0, len(self.histories[index]) - 1,
                                    self.maximum_track_length).round().astype(int))
                                self.histories[index] = [self.histories[index][i] for i in selected]
                            if self.ages[index] < self.maximum_track_length:
                                following[index] = point
            for index in self.active.keys() - following.keys():
                self.retire(index)
            self.active = following
        if len(self.active) < max(1, self.max_features // 2):
            self.flush()
            for point in self.active.values():
                cv2.circle(mask, tuple(np.round(point).astype(int)), 12, 0, -1)
            found = cv2.goodFeaturesToTrack(
                current, maxCorners=self.max_features - len(self.active),
                qualityLevel=.01, minDistance=10, blockSize=7, mask=mask)
            if found is not None:
                found = found.reshape(-1, 2)
                descriptors = describe(current, found) if self.parameters.poster_vio_revisit_enabled else {}
                self.bank = {key: track for key, track in self.bank.items()
                             if timestamp - track['last_time_s'] <= self.parameters.poster_vio_revisit_maximum_gap_s}
                matches = match_revisits(found, descriptors, self.bank,
                    self.item.seed_poses[node] @ self.camera_from_imu_inverse, self.item.model,
                    timestamp, maximum_gap_s=self.parameters.poster_vio_revisit_maximum_gap_s,
                    ratio=self.parameters.poster_vio_revisit_match_ratio,
                    maximum_reprojection_px=self.parameters.poster_vio_revisit_maximum_reprojection_px)
                for row, point in enumerate(found):
                    if row in matches:
                        track_id = matches[row]
                        track = self.bank.pop(track_id)
                        self.histories[track_id] = list(zip(track['node_indices'].tolist(), track['pixels'].copy()))
                        self.histories[track_id].append((node, point.astype(float)))
                        if len(self.histories[track_id]) > self.maximum_track_length:
                            chosen = np.unique(np.linspace(0, len(self.histories[track_id]) - 1,
                                self.maximum_track_length).round().astype(int))
                            self.histories[track_id] = [self.histories[track_id][i] for i in chosen]
                        for bucket in self.retained.values():
                            bucket[:] = [record for record in bucket if record['track_id'] != track_id]
                        self.revisits[track_id] = track.get('revisit_count', 0) + 1
                        self.stats['reassociations'] += 1
                    else:
                        if self.next_id >= 1_000_000:
                            raise ValueError('poster_vio 自然特征 ID 命名空间耗尽')
                        track_id = self.next_id
                        self.next_id += 1
                        self.histories[track_id] = [(node, point.astype(float))]
                        self.revisits[track_id] = 0
                        self.stats['created_tracks'] += 1
                    self.active[track_id] = point
                    self.descriptors[track_id] = descriptors.get(row)
                    self.ages[track_id] = 1
        if len(self.pending) >= 64:
            self.flush()
        self.previous = current
        self.selected_index += 1

    def retire(self, track_id):
        observations = self.histories.pop(track_id)
        descriptor = self.descriptors.pop(track_id, None)
        self.ages.pop(track_id, None)
        revisit_count = self.revisits.pop(track_id, 0)
        if len(observations) < self.minimum_track_length:
            return
        nodes = np.asarray([item[0] for item in observations], dtype=int)
        pixels = np.asarray([item[1] for item in observations], dtype=float)
        nodes, pixels = subsample_observations(nodes, pixels, self.maximum_observations_per_track)
        self.pending.append(dict(track_id=track_id, node_indices=nodes, pixels=pixels,
            camera=self.item.camera, descriptor=descriptor, revisit_count=revisit_count))

    def flush(self):
        if not self.pending:
            return
        candidates, training = [], []
        stride = self.parameters.poster_vio_natural_holdout_stride
        for track in self.pending:
            held = (track['node_indices'] % stride == stride - 1) if stride else np.zeros(len(track['pixels']), dtype=bool)
            if np.count_nonzero(~held) < self.minimum_track_length:
                continue
            track['holdout_mask'] = held
            candidates.append(track)
            training.append(dict(node_indices=track['node_indices'][~held], pixels=track['pixels'][~held]))
        points = triangulate_batch(training, self.item.seed_poses, self.item.camera_from_imu, self.item.model)
        self.stats['triangulation_batches'] += 1
        for track, point in zip(candidates, points):
            if not np.isfinite(point).all():
                continue
            track['initial_point_m'] = point
            start = int(np.searchsorted(self.nodes, track['node_indices'][0]))
            end = int(np.searchsorted(self.nodes, track['node_indices'][-1]))
            track['last_time_s'] = float(self.times[end])
            fraction = (self.times[start] - self.times[0]) / max(float(self.times[-1] - self.times[0]), 1e-12)
            bin_index = min(self.bin_count - 1, int(fraction * self.bin_count))
            bucket = self.retained.setdefault(bin_index, [])
            bucket.append(track)
            bucket.sort(key=lambda record: (-len(record['node_indices']), record['track_id']))
            del bucket[self.bin_quotas[bin_index]:]
            if track['descriptor'] is not None and self.parameters.poster_vio_revisit_enabled:
                self.bank[track['track_id']] = track
        self.pending.clear()
        while len(self.bank) > self.parameters.poster_vio_revisit_maximum_bank:
            self.bank.pop(next(iter(self.bank)))

    def finish(self):
        if not self.done:
            raise ValueError(f'poster_vio 原片截断或缺少 {self.item.camera} 自然特征源帧')
        for track_id in list(self.histories):
            self.retire(track_id)
        self.flush()
        return tuple({key: value for key, value in track.items() if key not in ('descriptor', 'last_time_s')}
                     for index in sorted(self.retained) for track in self.retained[index])


def track_cameras(video: Path, cameras: tuple[FeatureCameraInput, ...], *, parameters: FeatureTrackingParameters, max_features=200,
                  maximum_tracks=2000, minimum_track_length=3, maximum_track_length=40,
                  maximum_observations_per_track=None, diagnostics=None):
    """Decode once for both lenses, retaining one previous crop per camera.

    Frame/PTS failures reject the input. Valid textureless images may return
    no tracks. A front mask applies to both new detections and ongoing tracks.
    """
    import av

    if (any(type(value) is not int or value < 1 for value in
            (max_features, maximum_tracks, minimum_track_length, maximum_track_length))
            or not 2 <= minimum_track_length <= maximum_track_length):
        raise ValueError('poster_vio 自然特征数量必须为正整数，轨迹长度必须满足 2 <= 最短 <= 最长')
    if (maximum_observations_per_track is not None
            and (type(maximum_observations_per_track) is not int
                 or maximum_observations_per_track < minimum_track_length)):
        raise ValueError('poster_vio 自然特征观测上限必须为整数且不少于最短轨迹长度')
    if not cameras or len({item.camera for item in cameras}) != len(cameras):
        raise ValueError('poster_vio 自然特征需要不重复的相机输入')
    trackers = [_FeatureTracker(item, max_features=max_features, maximum_tracks=maximum_tracks,
                minimum_track_length=minimum_track_length, maximum_track_length=maximum_track_length,
                maximum_observations_per_track=maximum_observations_per_track, parameters=parameters)
                for item in cameras]
    if not all(tracker.done for tracker in trackers):
        with av.open(str(video)) as container:
            if len(container.streams.video) != 1:
                raise ValueError('poster_vio 自然特征需要单路 X3 原片')
            stream = container.streams.video[0]
            if (stream.codec_context.width, stream.codec_context.height) != (SRC_W, SRC_H):
                raise ValueError('poster_vio 自然特征需要原始 3840×1920 X3 录像')
            configure_decoder(stream)
            for source_index, frame in enumerate(container.decode(stream)):
                if any(not tracker.done and source_index == tracker.frames[tracker.selected_index]
                       for tracker in trackers):
                    timestamp = float(frame.pts * frame.time_base) if frame.pts is not None and frame.time_base else None
                    gray = frame.to_ndarray(format='gray')
                    for tracker in trackers:
                        tracker.consume(source_index, gray, timestamp)
                if all(tracker.done for tracker in trackers):
                    break
    result = {tracker.item.camera: tracker.finish() for tracker in trackers}
    if diagnostics is not None:
        diagnostics.update({tracker.item.camera: dict(
            tracker.stats,
            retained_tracks=len(result[tracker.item.camera]),
            retained_observations=sum(len(track['node_indices']) for track in result[tracker.item.camera]),
            maximum_observations_per_track=tracker.maximum_observations_per_track)
                            for tracker in trackers})
    return result


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
