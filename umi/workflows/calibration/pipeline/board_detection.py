"""Detect AprilGrid corners in Kalibr order and cache sampled EuRoC frames."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
import inspect
import threading
import zipfile

import cv2
import numpy as np

from umi.config import loader as config
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import atomic_file


# Kalibr and Basalt use a two-cell border and BL, BR, TR, TL corners.
BLACK_BORDER_BITS = 2  # Kalibr / Basalt blackTagBorder
# OpenCV ArUco 检测角序 → Kalibr (BL,BR,TR,TL)
_OPENCV_TO_KALIBR = [1, 0, 3, 2]


class GridDetector:
    """detect(gray) -> {tag_id: 4×2 float 像素角点，顺序 BL,BR,TR,TL（Kalibr/Basalt）}。"""

    def __init__(self, border_bits: int = BLACK_BORDER_BITS, refine: bool = True):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        p = cv2.aruco.DetectorParameters()
        p.markerBorderBits = int(border_bits)
        if refine:
            # L 形角点（黑方块外角）用 AprilTag 式边缘直线拟合求交点：合成板实测 max 0.57 px / median 0.40 px；
            # CORNER_REFINE_SUBPIX（鞍点假设）偏 ~0.8–1.4 px，CONTOUR ~1.4 px。
            p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.det = cv2.aruco.ArucoDetector(dictionary, p)

    def detect(self, gray: np.ndarray) -> dict:
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        out = {}
        if ids is None:
            return out
        for c, i in zip(corners, ids.ravel()):
            out[int(i)] = c.reshape(4, 2).astype(float)[_OPENCV_TO_KALIBR]
        return out


def detect_up(gray: np.ndarray, det, up: float | None = None) -> dict:
    """Detection on a bicubic-upsampled image, corners mapped back to native pixels."""
    up = config.load("calibration").values.board.detection_upscale if up is None else up
    g = cv2.resize(gray, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up != 1.0 else gray
    dets = det.detect(g)
    # cv2.resize maps source pixel centre x to (x + 0.5) * up - 0.5 -> invert exactly (no constant +0.25 px bias at 2x)
    return {k: (np.asarray(v, float) + 0.5) / up - 0.5 for k, v in dets.items()} if up != 1.0 else dets


def detect_all(euroc: Path, make_detector=GridDetector, step: int = 1, workers: int = 4, up: float | None = None, *, cancelled=None) -> dict:
    """AprilGrid detections (detect_up) on every `step`-th frame of an EuRoC folder.  Detection is the slow part (close to
    1 s a frame at 2x) and frames are independent, so a few threads share the work, each with its own detector (same
    result as one by one).  Returns {"n_frames": frames listed, "shape": (h, w), "dets": {frame index: {tag_id: 4x2 corners}}}."""
    up = config.load("calibration").values.board.detection_upscale if up is None else up
    rows = [l.split(",") for l in (Path(euroc) / "mav0" / "cam0" / "data.csv").read_text(encoding="utf-8").splitlines()
            if l and not l.startswith("#")]
    loc = threading.local()

    def one(k: int):
        if cancelled is not None and cancelled.is_set():
            raise CancelledError('板检测已取消')
        if not hasattr(loc, "det"):
            loc.det = make_detector()
        g = cv2.imread(str(Path(euroc) / "mav0" / "cam0" / "data" / rows[k][1].strip()), cv2.IMREAD_GRAYSCALE)
        return (k, None, None) if g is None else (k, detect_up(g, loc.det, up), g.shape)

    with ThreadPoolExecutor(max(1, workers)) as ex:
        frame_detections = list(ex.map(one, range(0, len(rows), step)))
    return {"n_frames": len(rows), "shape": next((sh for _, _, sh in frame_detections if sh is not None), None),
            "dets": {k: d for k, d, _ in frame_detections if d is not None},
            "video_time_s": np.array([int(row[0]) / 1e9 - 1.0 for row in rows])}


def detect_cached(euroc: Path | Callable[[], Path], make_detector=GridDetector, *, cache: Path, source_identity: str,
                  step: int, workers: int, up: float, cancelled=None) -> dict:
    """Reuse raw corners independently of calibration; a dataset callable runs only on a cache miss."""
    source_file = Path(__file__)
    code = artifact_identity.source_hash(source_file)
    detector_file = Path(inspect.getfile(make_detector))
    detector_code = code if detector_file == source_file else artifact_identity.source_hash(detector_file)
    key = artifact_identity.canonical_hash(dict(source=source_identity, step=step, upscale=up,
        opencv=cv2.__version__, code=code, detector=detector_code))
    try:
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved['identity'].item()) == key:
                frames, tags, corners = saved['frames'], saved['tags'], saved['corners']
                sampled, count, shape = saved['sampled'], int(saved['n_frames'].item()), saved['shape']
                times = saved['video_time_s']
                if times.shape != (count,) or not np.isfinite(times).all():
                    raise ValueError('板检测缓存缺少有效时间')
                if (frames.shape != tags.shape or corners.shape != (len(tags), 4, 2)
                        or not np.isfinite(corners).all() or sampled.ndim != 1
                        or len(set(sampled.tolist())) != len(sampled)
                        or np.any(sampled < 0) or np.any(sampled >= count)
                        or not np.isin(frames, sampled).all() or shape.shape not in ((0,), (2,))):
                    raise ValueError('板检测缓存不完整')
                dets = {int(index): {} for index in sampled}
                for frame, tag, points in zip(frames, tags, corners):
                    dets[int(frame)][int(tag)] = points
                return dict(n_frames=count, shape=tuple(map(int, shape)) if len(shape) else None, dets=dets, video_time_s=times)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        pass
    found = detect_all(euroc() if callable(euroc) else euroc, make_detector, step=step, workers=workers, up=up,
        cancelled=cancelled)
    rows = [(frame, tag, points) for frame, dets in found['dets'].items() for tag, points in dets.items()]
    with atomic_file(cache) as temporary, temporary.open('wb') as stream:
        np.savez_compressed(stream, identity=key, n_frames=found['n_frames'], video_time_s=found['video_time_s'],
            shape=np.asarray(found['shape'] or (), dtype=np.int64), sampled=np.asarray(list(found['dets']), dtype=np.int64),
            frames=np.asarray([row[0] for row in rows], dtype=np.int64),
            tags=np.asarray([row[1] for row in rows], dtype=np.int64),
            corners=np.asarray([row[2] for row in rows], dtype=float).reshape(-1, 4, 2))
    return found
