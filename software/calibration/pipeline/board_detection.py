"""Read prepared images, schedule board detection and cache raw pixel corners."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
import inspect
import threading
import zipfile

import cv2
import numpy as np

from software import config
from software import provenance as artifact_identity
from software.calibration.algorithms import aprilgrid_detect
from software.storage import atomic_file


def detect_all(euroc: Path, make_detector, step: int = 1, workers: int = 4, up: float | None = None, *, cancelled=None) -> dict:
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
        return (k, None, None) if g is None else (k, aprilgrid_detect.detect_up(g, loc.det, up), g.shape)

    with ThreadPoolExecutor(max(1, workers)) as ex:
        frame_detections = list(ex.map(one, range(0, len(rows), step)))
    return {"n_frames": len(rows), "shape": next((sh for _, _, sh in frame_detections if sh is not None), None),
            "dets": {k: d for k, d, _ in frame_detections if d is not None},
            "video_time_s": np.array([int(row[0]) / 1e9 - 1.0 for row in rows])}


def detect_cached(euroc: Path | Callable[[], Path], make_detector, *, cache: Path, source_identity: str,
                  step: int, workers: int, up: float, cancelled=None) -> dict:
    """Reuse raw corners independently of calibration; a dataset callable runs only on a cache miss."""
    key = artifact_identity.canonical_hash(dict(source=source_identity, step=step, upscale=up, opencv=cv2.__version__,
        code=artifact_identity.source_hash(Path(__file__)), preprocessing=artifact_identity.source_hash(Path(aprilgrid_detect.__file__)),
        detector=artifact_identity.source_hash(Path(inspect.getfile(make_detector)))))
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
