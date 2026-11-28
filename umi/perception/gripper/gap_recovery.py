"""Measure missing gripper tag pairs in native pixels between two accepted exposures."""

import av
import cv2
import numpy as np

from umi.infra.file_store import mark_used, read_npz_cache, write_npz_cache
from umi.infra.provenance import canonical_hash
from umi.perception.gripper import frame_detection as frames
from umi.perception.gripper.marker_detection import MarkerDetector, reconcile_detections

# Each side has six 12-pixel cells, surrounded by a six-pixel margin.
_SQUARE = np.float32([[6, 6], [77, 6], [77, 77], [6, 77]])
_CORNER_TOLERANCE_PX = 1.5
RECIPE = 'raw_border_consensus_v4'
_DIRECT_METHODS = ('native', 'clahe', 'local_contrast', 'consensus')


def _normalize(patch):
    patch = patch.astype(np.float32)
    detail = patch - cv2.GaussianBlur(patch, (0, 0), 6)
    return detail / (cv2.GaussianBlur(abs(detail), (0, 0), 6) + 4)


def _cell_values(patch):
    cells = patch[6:78, 6:78].reshape(6, 12, 6, 12)[:, 3:9, :, 3:9]
    return np.median(cells, axis=(1, 3))


def _reads_code(patch, grid, *, local_contrast=False):
    intensity = _cell_values(patch)
    if np.median(intensity[grid]) - np.median(intensity[~grid]) < 10:
        return False
    levels = intensity.round().astype(np.uint8)
    threshold, _ = cv2.threshold(levels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.array_equal(levels > threshold, grid):
        return True
    if not local_contrast:
        return False
    # A shadow can split white cells into two brightness groups. Re-read
    # the same raw patch with its slowly varying illumination removed.
    levels = cv2.normalize(_cell_values(_normalize(patch)), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    threshold, _ = cv2.threshold(levels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.array_equal(levels > threshold, grid):
        return True
    # Both side borders are black in every row. Estimate the illumination from
    # those cells alone, then require every bit to agree on the corrected native patch.
    weight = np.linspace(0., 1., intensity.shape[1])
    corrected = intensity - (intensity[:, :1] * (1. - weight) + intensity[:, -1:] * weight)
    return bool(corrected[grid].min() - corrected[~grid].max() >= 6.)


def _local_contrast(gray):
    gray = gray.astype(np.float32)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 4)
    scale = cv2.GaussianBlur(abs(detail), (0, 0), 4)
    return np.clip(128. + 70. * detail / (scale + 4.), 0., 255.).astype(np.uint8)


def _rectify(gray, corners):
    transform = cv2.getPerspectiveTransform(_SQUARE, np.float32(corners))
    patch = cv2.warpPerspective(gray, transform, (84, 84), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return patch, transform


def _tag_grids():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return tuple(cv2.aruco.generateImageMarker(dictionary, tag, 72)[6::12, 6::12] for tag in (0, 1))


def _refine_border(gray, corners):
    """Locate all four dark outer edges in the original image, close to their seeds."""
    if not np.isfinite(corners).all() or cv2.contourArea(corners, oriented=True) <= 0:
        return corners
    fractions = np.linspace(.18, .82, 13, dtype=np.float32)
    offsets = np.linspace(-5., 5., 21, dtype=np.float32)
    lines = []
    for first, last in zip(corners, np.roll(corners, -1, axis=0)):
        direction = last - first
        length = np.linalg.norm(direction)
        if length < 8.:
            return corners
        inward = np.array([-direction[1], direction[0]], np.float32) / length
        points = first + fractions[:, None] * direction
        samples = points[:, None] + offsets[None, :, None] * inward
        if np.any(samples < 0) or np.any(samples >= np.array(gray.shape[::-1]) - 1):
            return corners
        intensity = cv2.remap(gray, samples[:, :, 0], samples[:, :, 1], cv2.INTER_LINEAR)
        gradient = intensity[:, :-4] - intensity[:, 4:]
        peak = np.argmax(gradient, axis=1)
        strength = gradient[np.arange(len(fractions)), peak]
        supported = (strength >= 8.) & (peak > 0) & (peak < gradient.shape[1] - 1)
        if np.count_nonzero(supported) < 7:
            return corners
        shifts = offsets[2:-2][peak]
        supported &= np.abs(shifts - np.median(shifts[supported])) <= 1.5
        if np.count_nonzero(supported) < 7 or np.ptp(fractions[supported]) < .32:
            return corners
        edge = points[supported] + shifts[supported, None] * inward
        vx, vy, x, y = cv2.fitLine(edge, cv2.DIST_HUBER, 0, .01, .01).ravel()
        normal = np.array([-vy, vx])
        lines.append((normal, float(normal @ [x, y])))
    refined = []
    for (before, d0), (after, d1) in zip([lines[-1], *lines[:-1]], lines):
        system = np.stack((before, after))
        if abs(np.linalg.det(system)) < .15:
            return corners
        refined.append(np.linalg.solve(system, [d0, d1]))
    refined = np.asarray(refined, dtype=np.float32)
    if (not cv2.isContourConvex(refined) or np.any(refined < 0)
            or np.any(refined >= gray.shape[::-1]) or np.max(np.linalg.norm(refined - corners, axis=1)) > 5.):
        return corners
    return refined


class _Tracker:
    def __init__(self, anchors):
        self.previous = np.float32(anchors).copy()
        self.grids = []
        self.templates = []
        for cells in _tag_grids():
            marker = np.repeat(np.repeat(cells, 12, axis=0), 12, axis=1)
            self.grids.append(cells > 0)
            margin = np.pad(marker, 6, constant_values=180)
            self.templates.append(_normalize(cv2.GaussianBlur(margin, (0, 0), 0.9)))

    def observe(self, gray):
        verified = np.full((2, 4, 2), np.nan, np.float32)
        for tag, previous in enumerate(self.previous):
            patch, transform = _rectify(gray, previous)
            try:
                correlation, warp = cv2.findTransformECC(
                    self.templates[tag],
                    _normalize(patch),
                    np.eye(3, dtype=np.float32),
                    cv2.MOTION_HOMOGRAPHY,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 1e-4),
                    None,
                    3,
                )
            except cv2.error:
                continue  # Singular alignment or no convergence on this exposure.
            corners = cv2.perspectiveTransform(_SQUARE[None], transform @ warp)[0]
            if (
                not np.isfinite(correlation)
                or correlation < 0.5
                or not np.isfinite(corners).all()
                or np.max(np.linalg.norm(corners - previous, axis=1)) > 25
                or not cv2.isContourConvex(corners)
                or np.any(corners < 0)
                or np.any(corners >= [gray.shape[1], gray.shape[0]])
            ):
                continue
            # An alignment is a search seed. Only a fresh read of all 36 cells
            # on raw pixels can supply a measurement.
            self.previous[tag] = corners
            patch, _ = _rectify(gray, corners)
            if _reads_code(patch, self.grids[tag]):
                verified[tag] = corners
        return verified


def _agree(first, second):
    return bool(np.max(np.linalg.norm(first - second, axis=-1)) <= _CORNER_TOLERANCE_PX)


def _compatible(reference, candidate):
    return not reference.has_duplicate_targets and reconcile_detections(reference, candidate) is candidate


class _DirectDetector:
    def __init__(self, roi, estimator):
        left, top, right, bottom = roi
        self.origin = np.float32([left, top])
        self.estimator = estimator
        self.detector = MarkerDetector((0, 0, right - left, bottom - top), fast=True)
        self.contrast = cv2.createCLAHE(3, (10, 4))
        self.grids = tuple(cells > 0 for cells in _tag_grids())

    def _valid(self, gray, corners):
        return np.isfinite(self.estimator.distance(corners)[0]) and all(
            _reads_code(_rectify(gray, c - self.origin)[0], grid, local_contrast=True)
            for c, grid in zip(corners, self.grids)
        )

    def detect(self, gray, references, *, include_native=False):
        candidates = [('native', reference) for reference in references] if include_native else []
        candidates += [(name, self.detector.detect(transform(gray)).translated(self.origin))
                       for name, transform in (('clahe', self.contrast.apply), ('local_contrast', _local_contrast))]
        edge_image = cv2.GaussianBlur(gray, (3, 3), .65).astype(np.float32)
        accepted = []
        geometric = np.full((2, 4, 2), np.nan, np.float32)
        for method, detection in candidates:
            if detection.has_duplicate_targets or any(not _compatible(ref, detection) for ref in references):
                return geometric, 'conflict'
            references.append(detection)
            seed = detection.target_corners()
            if not np.isfinite(seed).all():
                continue
            if not np.isfinite(geometric).all() and np.isfinite(self.estimator.distance(seed)[0]):
                geometric = seed
            fitted = np.asarray([_refine_border(edge_image, c) for c in seed - self.origin]) + self.origin
            if self._valid(gray, fitted):
                accepted.append((fitted, method))
            elif self._valid(gray, seed):
                accepted.append((seed, method))
        if not accepted:
            return geometric, 'missing'
        if len(accepted) == 1:
            return accepted[0]
        if any(not _agree(corners, other) for index, (corners, _) in enumerate(accepted)
               for other, _ in accepted[:index]):
            return geometric, 'conflict'
        combined = np.mean([corners for corners, _ in accepted], axis=0)
        return (combined, 'consensus') if self._valid(gray, combined) else (geometric, 'conflict')


def _measure_gap(cache, begin, end, anchors, estimator):
    """Use two bounded-memory sweeps. Rows [begin, end) exclude both anchors."""
    left, top, right, bottom = cache.roi_px
    origin = np.float32([left, top])
    indices = np.arange(begin - 1, end + 1)
    forward = _Tracker(anchors[0] - origin)
    backward = _Tracker(anchors[1] - origin)
    detector = _DirectDetector(cache.roi_px, estimator)
    conventional = cache.recovery_detector()
    converter = av.video.reformatter.VideoReformatter()
    shape = (len(indices), 2, 4, 2)
    forward_corners = np.full(shape, np.nan, np.float32)
    enhanced = np.full_like(forward_corners, np.nan)
    enhanced_methods = np.full(len(indices), 'missing', dtype='<U16')
    result = np.full_like(forward_corners, np.nan)
    methods = np.full(len(indices), 'missing', dtype='<U16')
    references = {}
    native = cache.native_times_s

    def exposures(rows):
        for i, timestamp, frame in frames.sampled_frames(cache.video, cache.query_times_s, rows):
            if frame is None and not np.isfinite(native[i]):
                continue
            if frame is None or not np.isfinite([native[i], timestamp]).all() or abs(native[i] - timestamp) > 1e-8:
                raise ValueError('双向补检与原始检测的曝光时刻不一致')
            yield i, frame, frames.crop_gray(frame, cache.roi_px, converter)

    for i, frame, gray in exposures(indices):
        j = i - indices[0]
        forward_corners[j] = forward.observe(gray) + origin
        if not begin <= i < end:
            continue
        quick = frames._frame_detection(cache.quick_corners[i])
        refs = [quick]
        if cache.attempted[i]:
            full = conventional(i, frame)
            if not _compatible(quick, full):
                methods[j] = 'conflict'
                continue
            refs.append(full)
        if cache.blocked[i]:
            methods[j] = 'conflict'
            continue
        enhanced[j], enhanced_methods[j] = detector.detect(gray, refs)
        if enhanced_methods[j] == 'conflict':
            methods[j] = 'conflict'
        else:
            references[j] = refs

    # Decode a short block in time order before traversing it backwards.
    for stop in range(len(indices), 0, -32):
        block = [(i, gray) for i, _, gray in exposures(indices[max(0, stop - 32) : stop])]
        for i, gray in reversed(block):
            j = i - indices[0]
            reverse = backward.observe(gray) + origin
            if j not in references:
                continue
            direct = enhanced[j]
            direct_ok = enhanced_methods[j] != 'missing'
            tracked = (reverse + forward_corners[j]) * 0.5
            tracked_ok = _agree(reverse, forward_corners[j]) and np.isfinite(estimator.distance(tracked)[0])
            if tracked_ok:
                candidate = frames._frame_detection(tracked)
                if any(not _compatible(ref, candidate) for ref in references[j]):
                    methods[j] = 'conflict'
                    continue
            if tracked_ok and np.isfinite(direct).all() and not _agree(direct, tracked):
                methods[j] = 'conflict'
                continue
            if direct_ok:
                result[j], methods[j] = direct, enhanced_methods[j]
            elif tracked_ok and all(
                _reads_code(_rectify(gray, c - origin)[0], grid) for c, grid in zip(tracked, forward.grids)
            ):
                result[j], methods[j] = tracked, 'template'
    return result[1:-1], methods[1:-1]


def recover(cache, gaps, valid, estimator, distances, sources, fingerprint, progress):
    """Cache verified native corners separately from all conventional detections."""
    reports = []
    corners = cache.corners
    for begin, end in gaps:
        begin, end = int(begin), int(end)
        if begin == 0 or end == len(valid):
            continue
        anchors = corners[[begin - 1, end]]
        inputs = dict(method=RECIPE, base=fingerprint, rows=[begin, end])
        directory = cache.directory / 'verified_gaps' / canonical_hash(inputs)[:20]
        path, marker = directory / 'corners.npz', directory / 'ready.json'
        saved = read_npz_cache(path, marker, inputs, mismatch='双向夹爪补检缓存不一致')
        if saved is None:
            if progress:
                progress(f'双向夹爪标签补检：第 {begin}–{end - 1} 帧')
            recovered, methods = _measure_gap(cache, begin, end, anchors, estimator)
            write_npz_cache(path, marker, inputs, corners=recovered, methods=methods)
        else:
            arrays, _ = saved
            recovered, methods = arrays['corners'], arrays['methods']
        mark_used(directory)
        for j in np.flatnonzero(np.isin(methods, (*_DIRECT_METHODS, 'template'))):
            i = begin + j
            distances[i], sources[i] = estimator.distance(recovered[j])
            valid[i] = np.isfinite(distances[i]) and sources[i] == 'two_tags'
        sources[begin + np.flatnonzero(methods == 'conflict')] = 'ambiguous_detection'
        labels, counts = np.unique(methods, return_counts=True)
        reports.append(
            dict(
                start_row=begin,
                end_row=end - 1,
                counts=dict(zip(labels.tolist(), counts.tolist())),
                accepted_rows=(begin + np.flatnonzero(valid[begin:end])).tolist(),
                cache_key=directory.name,
            )
        )
    return reports


def recover_short(cache, rows, valid, estimator, distances, sources, fingerprint, progress):
    if not len(rows):
        return None
    inputs = dict(method=RECIPE, base=fingerprint, short_rows=rows.tolist())
    directory = cache.directory / 'short_gaps' / canonical_hash(inputs)[:20]
    path, marker = directory / 'corners.npz', directory / 'ready.json'
    saved = read_npz_cache(path, marker, inputs, mismatch='夹爪短缺口补检缓存不一致')
    if saved is None:
        if progress:
            progress(f'夹爪短缺口补检：{len(rows)} 帧')
        detector = _DirectDetector(cache.roi_px, estimator)
        converter = av.video.reformatter.VideoReformatter()
        recovered, methods = [], []
        for i, timestamp, frame in frames.sampled_frames(cache.video, cache.query_times_s, rows):
            if frame is None or not np.isfinite(timestamp) or abs(timestamp - cache.native_times_s[i]) > 1e-8:
                raise ValueError('夹爪短缺口补检与原始检测的曝光时刻不一致')
            gray = frames.crop_gray(frame, cache.roi_px, converter)
            references = [frames._frame_detection(cache.quick_corners[i])]
            if cache.attempted[i]:
                references.append(frames._frame_detection(cache.recovered_corners[i]))
            corners, method = detector.detect(gray, references, include_native=True)
            recovered.append(corners)
            methods.append(method)
        recovered, methods = np.asarray(recovered), np.asarray(methods)
        write_npz_cache(path, marker, inputs, corners=recovered, methods=methods)
    else:
        arrays, _ = saved
        recovered, methods = arrays['corners'], arrays['methods']
    mark_used(directory)
    for i, corners, method in zip(rows, recovered, methods, strict=True):
        if method in _DIRECT_METHODS:
            distances[i], sources[i] = estimator.distance(corners)
            valid[i] = np.isfinite(distances[i]) and sources[i] == 'two_tags'
        elif method == 'conflict':
            sources[i] = 'ambiguous_detection'
    return dict(attempted_rows=len(rows), accepted_rows=rows[valid[rows]].tolist(),
                conflicting_rows=rows[methods == 'conflict'].tolist(), cache_key=directory.name)
