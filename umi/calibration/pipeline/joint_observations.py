"""Extract synchronous optical observations for the offline joint calibration."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import threading

import av
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from umi.calibration.pipeline.board_detection import GridDetector, detect_up
from umi.calibration.pipeline.recordings import LUMA_FORMATS
from umi.calibration.solve.board_pnp import grid_points_3d, pnp_from_detections, read_kb4
from umi.core.frame import CROP, LENS_HALF, SRC_H, SRC_W, crop_xywh
from umi.devices.x3.factory_geometry import Kb4Model
from umi.infra.file_store import read_npz_cache, write_npz_cache
from umi.infra.provenance import canonical_hash
from umi.perception.decoding import configure_decoder
from umi.perception.poster.detection import DetectionSettings, PosterDetector
from umi.perception.poster.pnp import solve_frame


def _poster_pose(detector, half, layout, model, min_tags):
    detections, _, _ = detector.detect(half)
    margin = (SRC_H - CROP) / 2
    pairs = [(i, p - margin) for i, p in detections if np.all((p >= margin) & (p < margin + CROP))]
    if len(pairs) < min_tags:
        return None
    xyz = np.concatenate([layout.corners_m[i] for i, _ in pairs])
    uv = np.concatenate([p for _, p in pairs])
    try:
        pose, _ = solve_frame(xyz, uv, model, maximum_rms_px=4.0)
    except ValueError:
        return None

    def residual(v):
        return (model.project(xyz @ Rotation.from_rotvec(v[:3]).as_matrix().T + v[3:]) - uv).ravel()

    seed = np.r_[Rotation.from_matrix(pose[:3, :3]).as_rotvec(), pose[:3, 3]]
    fit = least_squares(residual, seed, loss='huber', f_scale=1.5, max_nfev=40)
    if not fit.success or np.sqrt(np.mean(np.sum(fit.fun.reshape(-1, 2) ** 2, axis=1))) > 2.5:
        return None
    pose[:3, :3] = Rotation.from_rotvec(fit.x[:3]).as_matrix()
    pose[:3, 3] = fit.x[3:]
    return np.linalg.inv(pose), xyz, uv


def observe(video, signature, lens, calibrations, grid, layout, board, cache, workers):
    """For 'dual', store both poses and actual pixels. Single lenses see an AprilGrid."""
    dual = lens == 'dual'
    grid_lens = 'front' if dual else lens
    kb = read_kb4(calibrations[grid_lens])
    rear = Kb4Model(*read_kb4(calibrations['rear']))
    identity = dict(
        version=1,
        video=signature,
        lens=lens,
        grid=grid,
        board=asdict(board),
        kb4=list(kb),
        rear=asdict(rear),
        layout=layout.geometry_hash,
        opencv=cv2.__version__,
    )
    path = cache / (canonical_hash(identity) + '.npz')
    marker = path.with_suffix('.json')
    cached = read_npz_cache(path, marker, identity)
    if cached is not None:
        return cached[0]
    tls = threading.local()
    geometry = grid_points_3d(grid)
    x, y, w, h = crop_xywh(LENS_HALF[grid_lens])

    def detect(sample):
        time, gray = sample
        if not hasattr(tls, 'grid'):
            tls.grid = GridDetector()
            if dual:
                tls.poster = PosterDetector(
                    DetectionSettings(layout.border_bits, ids=layout.ids, dictionary=layout.dictionary)
                )
        det = detect_up(gray[y : y + h, x : x + w], tls.grid, up=board.detection_upscale)
        ids = sorted(geometry.keys() & det.keys())
        if len(ids) < board.minimum_reference_tags:
            return None
        result = pnp_from_detections(det, kb, geometry, parameters=board)
        if result is None:
            return None
        if not dual:
            return time, result['T_bc']
        posterior = _poster_pose(tls.poster, gray[:, :SRC_H], layout, rear, board.minimum_reference_tags)
        if posterior is None:
            return None
        return (
            time,
            result['T_bc'],
            posterior[0],
            np.concatenate([geometry[i] for i in ids]),
            np.concatenate([det[i] for i in ids]),
            posterior[1],
            posterior[2],
        )

    with av.open(str(video)) as container, ThreadPoolExecutor(workers) as pool:
        stream = container.streams.video[0]
        if (stream.width, stream.height) != (SRC_W, SRC_H):
            raise ValueError(f'双镜头联合标定需要 {SRC_W}×{SRC_H} 原片：{video}')
        configure_decoder(stream)
        step = max(1, round(float(stream.average_rate) / (5 if dual else 10)))
        pending, rows = deque(), []
        for i, frame in enumerate(container.decode(stream)):
            if i % step:
                continue
            if frame.format.name not in LUMA_FORMATS or frame.pts is None:
                raise ValueError(f'{video} 图像格式或时间戳不受支持')
            gray = np.frombuffer(frame.planes[0], np.uint8).reshape(frame.height, -1)[:, : frame.width].copy()
            pending.append(pool.submit(detect, (float(frame.pts * frame.time_base), gray)))
            if len(pending) >= workers * 2:
                rows.append(pending.popleft().result())
        rows.extend(job.result() for job in pending)
    rows = [r for r in rows if r is not None]
    if len(rows) < 12:
        raise ValueError(f'{video.name} 的 {lens} 有效标定板帧不足：{len(rows)}')
    data = dict(t=np.array([r[0] for r in rows]), poses=np.stack([r[1] for r in rows]))
    if not np.all(np.diff(data['t']) > 0):
        raise ValueError(f'{video} 帧时间戳未严格递增')
    if dual:
        data['rear'] = np.stack([r[2] for r in rows])
        for name, index in [('front', 3), ('rear', 5)]:
            data[name + '_ends'] = np.cumsum([len(r[index]) for r in rows])
            data[name + '_xyz'] = np.concatenate([r[index] for r in rows])
            data[name + '_uv'] = np.concatenate([r[index + 1] for r in rows])
    cache.mkdir(parents=True, exist_ok=True)
    write_npz_cache(path, marker, identity, **data)
    return data
