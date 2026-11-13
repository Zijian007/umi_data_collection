"""固定 ROI 的无界面宽度检测：按视频时间采样，CSV 与可选叠加 MP4 输出。"""

from __future__ import annotations

from software import settings

from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
import csv
import math
import sys
import time

import av
import cv2
import numpy as np

from .detection import MarkerDetector, validate_roi
from .frame_processing import detect_frame
from .preview import PreviewWriter
from software.camera.x3.frame import SRC_H, SRC_W
from software.console import Console
from software.storage import atomic_file


console = Console('宽度检测')
HEADER = ('t_video_s', 'id0_x_px', 'id0_y_px', 'id1_x_px', 'id1_y_px', 'distance_px', 'detect_ms')


def run(video: Path, roi: tuple[int, int, int, int], *, hz: float = 30.,
        output: Path | None = None, benchmark: bool = False, video_output: Path | None = None,
        contrast: float = 1.) -> dict:
    video = video.expanduser().resolve()
    if not video.is_file() or video.suffix.lower() != '.insv':
        raise ValueError(f'请提供已有的原始 INSV：{video}')
    roi = validate_roi(roi)
    if not math.isfinite(hz) or hz <= 0:
        raise ValueError('--hz 必须为正的有限数值')
    if output is not None:
        output = output.expanduser().resolve()
        if output.suffix.lower() != '.csv':
            raise ValueError('--out 需要 CSV 文件路径')
        if output.exists():
            raise FileExistsError(f'输出已存在，请选择新 CSV：{output}')
    if video_output is not None:
        video_output = video_output.expanduser().resolve()
        if video_output.suffix.lower() != '.mp4':
            raise ValueError('--video-out 需要 MP4 文件路径')
        if video_output.exists():
            raise FileExistsError(f'输出已存在，请选择新 MP4：{video_output}')
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(settings.values('runtime')['calibration_opencv_threads'])
    started = time.perf_counter()
    try:
        with av.open(str(video)) as source:
            if len(source.streams.video) != 1:
                raise ValueError('宽度检测需要单路 X3 原始 3840×1920 INSV')
            stream = source.streams.video[0]
            if (stream.width, stream.height) != (SRC_W, SRC_H):
                raise ValueError('宽度检测需要 X3 原始 3840×1920 INSV')
            stream.codec_context.thread_count = settings.values('runtime')['decode_threads']
            stream.thread_type = 'AUTO'
            with ExitStack() as outputs:
                destination = sys.stdout
                if output is not None:
                    csv_temporary = outputs.enter_context(atomic_file(output))
                    destination = outputs.enter_context(csv_temporary.open('w', newline=''))
                video_writer = None
                if video_output is not None:
                    video_temporary = outputs.enter_context(atomic_file(video_output))
                    target = outputs.enter_context(av.open(str(video_temporary), 'w', format='mp4',
                                                          options={'movflags': '+faststart'}))
                    video_writer = PreviewWriter(target, stream, rate=Fraction(str(hz)).limit_denominator(1_000_000),
                                                 preset='ultrafast' if contrast > 1 else 'veryfast', contrast=contrast)
                summary = _measure(source, stream, roi, hz, destination, benchmark, video_writer, contrast)
                if video_writer is not None:
                    video_writer.flush()
        elapsed_s = time.perf_counter() - started
        return dict(summary, output=str(output) if output is not None else 'stdout', roi=roi, contrast=contrast,
                    video_output=str(video_output) if video_output is not None else None,
                    target_hz=hz, mode='benchmark' if benchmark else 'realtime',
                    elapsed_s=round(elapsed_s, 3), sample_hz=round(summary['samples'] / elapsed_s, 2))
    finally:
        cv2.setNumThreads(previous_threads)


def _measure(source, stream, roi, hz, destination, benchmark, video_writer: PreviewWriter | None = None,
             contrast: float = 1.) -> dict:
    left, top, right, bottom = roi
    detector = MarkerDetector((0, 0, right - left, bottom - top), fast=True, contrast=contrast)
    converter = av.video.reformatter.VideoReformatter()
    writer = csv.writer(destination)
    writer.writerow(HEADER)
    destination.flush()
    samples = paired = source_frames = late_dropped = 0
    first_time = previous_time = wall_origin = None
    next_due = 0.
    costs = []
    progress = time.perf_counter()
    for frame in source.decode(stream):
        source_frames += 1
        if frame.pts is None or frame.time_base is None:
            raise ValueError('原片缺少帧时间戳')
        timestamp = float(frame.pts * frame.time_base)
        if previous_time is not None and timestamp <= previous_time:
            raise ValueError('原片帧时间必须严格递增')
        previous_time = timestamp
        if first_time is None:
            first_time = timestamp
            wall_origin = time.perf_counter()
        relative = timestamp - first_time
        if relative + 1e-8 < next_due:
            continue
        next_due = (math.floor((relative + 1e-8) * hz) + 1) / hz
        if not benchmark and time.perf_counter() - wall_origin - relative > 1 / hz:
            late_dropped += 1
            continue
        began = time.perf_counter()
        detected = detect_frame(frame, detector, roi, converter)
        detect_ms = (time.perf_counter() - began) * 1000
        costs.append(detect_ms)
        if not benchmark:
            delay = wall_origin + relative - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        coordinates = []
        for tag in (0, 1):
            centre = detected.centres.get(tag)
            coordinates.extend(('', '') if centre is None else (f'{centre[0]:.6f}', f'{centre[1]:.6f}'))
        distance = detected.distance_px
        valid = np.isfinite(distance)
        writer.writerow((f'{timestamp:.9f}', *coordinates, f'{distance:.6f}' if valid else '', f'{detect_ms:.3f}'))
        destination.flush()
        if video_writer is not None:
            video_writer.write(frame, detected, roi=roi)
        samples += 1
        paired += int(valid)
        if time.perf_counter() - progress >= 5:
            console.write(f'{samples} 次；原片 {relative:.2f} s；双标签 {paired} 次；落后跳过 {late_dropped} 次')
            progress = time.perf_counter()
    if source_frames < 2:
        raise ValueError('原片至少需要两帧视频')
    return dict(source_frames=source_frames, samples=samples, paired_samples=paired, late_dropped=late_dropped,
                video_frames=video_writer.frames if video_writer is not None else 0,
                detect_median_ms=round(float(np.median(costs)), 3) if costs else None,
                detect_p95_ms=round(float(np.percentile(costs, 95)), 3) if costs else None)
