"""场景任务的双标记宽度提取与缓存；复用 main 的检测器和毫米映射。"""
from __future__ import annotations

import csv
from pathlib import Path
import json

import numpy as np

from software import config
from software.camera.x3.frame import SRC_H, SRC_W
from software.gripper_width import calibration, series
from software.gripper_width.width_4k60 import detector, width_mm
from software.storage import atomic_file, write_json


def identity() -> dict:
    import cv2

    directory = Path(__file__).parent
    return dict(opencv=cv2.__version__, files={name: config.file_hash(directory/name)
        for name in ('calibration.py', 'series.py', 'pipeline.py', 'width_4k60.py')})


def detect(video: Path, measurement: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import av

    det = detector()
    times, distances = [], []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if (stream.codec_context.width, stream.codec_context.height) != (SRC_W, SRC_H):
            raise ValueError('夹爪宽度估计需要 X3 原始 3840×1920 前镜头画面')
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('宽度检测原片缺少帧时间戳')
            gray = frame.to_ndarray(format='gray')[:, SRC_W//2:]
            corners, ids, _ = det.detectMarkers(gray)
            centres = {}
            if ids is not None:
                values = ids.flatten()
                for tag in (0, 1):
                    matches = np.flatnonzero(values == tag)
                    if len(matches) == 1:
                        centres[tag] = corners[int(matches[0])][0].mean(axis=0)
            times.append(float(frame.pts * frame.time_base))
            distances.append(float(np.linalg.norm(centres[0] - centres[1])) if len(centres) == 2 else np.nan)
    times = np.asarray(times)
    if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('宽度检测视频时间无效或非递增')
    distance = np.asarray(distances)
    found = np.isfinite(distance)
    widths = np.full(len(times), np.nan)
    above = np.zeros(len(times), dtype=bool)
    widths[found], above[found] = width_mm(distance[found], measurement['anchors'])
    return times, widths, above


def align(times: np.ndarray, widths: np.ndarray, above: np.ndarray,
          query: np.ndarray, max_gap_s: float) -> series.WidthSeries:
    """原片邻近三帧中位数，随后仅补短缺口；超出标定范围不插值掩盖。"""
    indices = np.clip(np.searchsorted(times, query), 1, len(times) - 1)
    indices -= (query - times[indices - 1]) <= (times[indices] - query)
    values = np.full(len(query), np.nan)
    source = np.full(len(query), series.MISSING, dtype=np.uint8)
    tolerance = .5 * float(np.median(np.diff(times))) + 1e-6
    for k, i in enumerate(indices):
        if abs(times[i]-query[k]) > tolerance:
            continue
        neighbours = slice(max(0, i-1), min(len(times), i+2))
        if above[neighbours].any():
            source[k] = series.OUT_OF_RANGE
            continue
        samples = widths[neighbours]
        samples = samples[np.isfinite(samples)]
        if len(samples):
            values[k] = float(np.median(samples))
            source[k] = series.TWO_TAGS
    have = np.flatnonzero(np.isfinite(values))
    for left, right in zip(have[:-1], have[1:]):
        if (right - left > 1 and query[right] - query[left] <= max_gap_s + 1e-8
                and not np.any(source[left:right] == series.OUT_OF_RANGE)):
            values[left + 1:right] = np.interp(query[left + 1:right], query[[left, right]], values[[left, right]])
            source[left + 1:right] = series.INTERPOLATED
    return series.WidthSeries(query, values, source)


def process(video: Path, calibration_path: Path, video_times_s: np.ndarray, offset_s: float,
            output: Path, cache: Path, recording_sha256: str) -> dict:
    measurement = calibration.read(calibration_path)
    provenance = dict(recording_sha256=recording_sha256, calibration_sha256=config.file_hash(calibration_path),
                      code=identity())
    key = config.canonical_hash(provenance)[:20]
    cached = cache / key
    marker = cached / 'ready.json'
    data_path = cached / 'frames.npz'
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if saved['inputs'] != provenance or config.file_hash(data_path) != saved['sha256']:
            raise ValueError('夹爪宽度缓存身份或哈希变化')
        with np.load(data_path, allow_pickle=False) as data:
            times, widths, above = data['times'], data['width_mm'], data['above_top']
    else:
        times, widths, above = detect(video, measurement)
        cached.mkdir(parents=True, exist_ok=True)
        with atomic_file(data_path) as temporary, temporary.open('wb') as stream:
            np.savez_compressed(stream, times=times, width_mm=widths, above_top=above)
        write_json(marker, dict(inputs=provenance, sha256=config.file_hash(data_path)))
    aligned = align(times, widths, above, video_times_s, measurement['max_gap_s'])
    series.write(output / 'gripper_width.csv', aligned, measurement, offset_s)
    details = output / 'details'
    with (details / 'gripper_width_frames.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_video_frame', 't_video_s', 'width_mm', 'above_top_anchor'])
        for k, (time, width, top) in enumerate(zip(times, widths, above)):
            writer.writerow([k, f'{time:.9f}', f'{width:.6f}' if np.isfinite(width) else '', int(top)])
    report = dict(status='ok', calibration=measurement, inputs=provenance,
                  valid_fraction=float(aligned.valid.mean()), rows=len(aligned.times_s),
                  sources={name: int((aligned.source == code).sum()) for code, name in series.SOURCE_NAMES.items()},
                  closed_fraction=float((aligned.width_mm[aligned.valid] <= measurement['closed_threshold_mm']).mean())
                  if aligned.valid.any() else None)
    write_json(details / 'gripper_width.json', report)
    return report
