"""夹爪标签预览与采样监测，共用原片检测和叠加视频写入。"""

from __future__ import annotations

import csv
import math
import sys
import time
from contextlib import ExitStack, contextmanager
from fractions import Fraction
from pathlib import Path
from types import EllipsisType

import av
import cv2
import numpy as np

from umi.config import operational_settings as settings
from umi.core.frame import FPS_SRC_RATE, SRC_H, SRC_W
from umi.infra.file_store import atomic_file
from umi.infra.terminal import Console
from umi.perception.decoding import configure_decoder
from umi.perception.gripper import default_tag_roi
from umi.perception.gripper.frame_detection import FrameDetector
from umi.perception.gripper.marker_detection import FrameDetection, validate_roi

PREVIEW_SIZE = 960
HEADER = ('t_video_s', 'id0_x_px', 'id0_y_px', 'id1_x_px', 'id1_y_px', 'distance_px', 'detect_ms')
_preview_console = Console('宽度预览')
_monitor_console = Console('宽度检测')


def _recording_path(video: Path) -> Path:
    video = video.expanduser().resolve()
    if not video.is_file() or video.suffix.lower() != '.insv':
        raise ValueError(f'请提供已有的原始 INSV 文件：{video}')
    return video


@contextmanager
def _open_source(video: Path, purpose: str):
    with av.open(str(video)) as source:
        if len(source.streams.video) != 1:
            raise ValueError(f'{purpose}需要单路 X3 原始 3840×1920 INSV')
        stream = source.streams.video[0]
        if (stream.codec_context.width, stream.codec_context.height) != (SRC_W, SRC_H):
            raise ValueError(f'{purpose}需要 X3 原始 3840×1920 INSV')
        yield source, stream


@contextmanager
def _open_mp4(output: Path):
    with atomic_file(output) as temporary, av.open(str(temporary), 'w', format='mp4',
                                                  options={'movflags': '+faststart'}) as target:
        yield target


def annotate(front: np.ndarray, detected: FrameDetection, *, timestamp_s: float,
             frame_index: int, roi: tuple[int, int, int, int] | None = None, contrast: float = 1.) -> np.ndarray:
    rendered = cv2.resize(front, (PREVIEW_SIZE, PREVIEW_SIZE), interpolation=cv2.INTER_AREA)
    scale = PREVIEW_SIZE / SRC_H
    display_roi = tuple(np.rint(np.asarray(roi) * scale).astype(int)) if roi is not None else None
    if contrast > 1:
        left, top, right, bottom = display_roi or (0, 0, PREVIEW_SIZE, PREVIEW_SIZE)
        rendered[top:bottom, left:right] = cv2.convertScaleAbs(rendered[top:bottom, left:right], alpha=contrast)
    rendered[:85] = (rendered[:85] * .2).astype(np.uint8)
    if display_roi is not None:
        left, top, right, bottom = display_roi
        cv2.rectangle(rendered, (left, top), (right, bottom), (255, 180, 0), 2, cv2.LINE_AA)
        cv2.putText(rendered, 'ROI', (left, max(22, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, .7,
                    (255, 180, 0), 2, cv2.LINE_AA)
    if detected.ids is not None:
        for corners, tag in zip(detected.corners, detected.ids.flatten()):
            polygon = np.rint(corners[0] * scale).astype(np.int32)
            color = (0, 230, 0) if tag in (0, 1) else (150, 150, 150)
            cv2.polylines(rendered, [polygon], True, color, 2, cv2.LINE_AA)
            anchor = tuple(np.maximum(polygon.min(axis=0), (0, 22)))
            cv2.putText(rendered, f'ID {int(tag)}', anchor, cv2.FONT_HERSHEY_SIMPLEX, .7,
                        color, 2, cv2.LINE_AA)
    centres = {tag: tuple(np.rint(centre * scale).astype(int)) for tag, centre in detected.centres.items()}
    for centre in centres.values():
        cv2.circle(rendered, centre, 4, (0, 230, 255), -1, cv2.LINE_AA)
    distance_px = detected.distance_px
    if np.isfinite(distance_px):
        cv2.line(rendered, centres[0], centres[1], (0, 230, 255), 2, cv2.LINE_AA)
        distance_label = f'ID 0 <-> 1: {distance_px:.2f} px  (native 1920x1920)'
        text_color = (0, 230, 255)
    else:
        distance_label = 'Distance: N/A  (ID 0/1 missing or duplicated)'
        text_color = (0, 160, 255)
    header = f'Frame {frame_index} | t_video={timestamp_s:.3f} s'
    if contrast > 1:
        header += f' | contrast x{contrast:g}'
    cv2.putText(rendered, header, (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(rendered, distance_label, (16, 65), cv2.FONT_HERSHEY_SIMPLEX, .7,
                text_color, 2, cv2.LINE_AA)
    return rendered


class PreviewWriter:
    """保存采样帧；显示工作镜头半幅，检测和叠加使用原始像素坐标。"""

    def __init__(self, target, source_stream, *, rate: Fraction | None = None, preset: str = 'fast',
                 contrast: float = 1.) -> None:
        self._target = target
        self._stream = target.add_stream('libx264', rate=rate or source_stream.average_rate or Fraction(FPS_SRC_RATE))
        self._stream.width = self._stream.height = PREVIEW_SIZE
        self._stream.pix_fmt = 'yuv420p'
        self._stream.options = {'crf': '20', 'preset': preset}
        self._stream.codec_context.thread_count = 2
        self._stream.time_base = source_stream.time_base
        self._stream.codec_context.time_base = source_stream.time_base
        self._converter = av.video.reformatter.VideoReformatter()
        self._first_pts = None
        self._contrast = contrast
        self.frames = 0

    def write(self, frame, detected: FrameDetection, *, roi=None) -> None:
        scaled = self._converter.reformat(frame, width=PREVIEW_SIZE * 2, height=PREVIEW_SIZE,
                                         format='bgr24', threads=1).to_ndarray()
        rendered = annotate(scaled[:, PREVIEW_SIZE:], detected,
                            timestamp_s=float(frame.pts * frame.time_base), frame_index=self.frames, roi=roi,
                            contrast=self._contrast)
        if self._first_pts is None:
            self._first_pts = frame.pts
        encoded = av.VideoFrame.from_ndarray(rendered, format='bgr24')
        encoded.pts, encoded.time_base = frame.pts - self._first_pts, frame.time_base
        for packet in self._stream.encode(encoded):
            self._target.mux(packet)
        self.frames += 1

    def flush(self) -> None:
        for packet in self._stream.encode():
            self._target.mux(packet)


def preview(video: Path, output: Path | None = None, *,
           roi: tuple[int, int, int, int] | None | EllipsisType = ...,
           contrast: float = 1.) -> dict:
    video = _recording_path(video)
    output = output.expanduser().resolve() if output is not None else video.with_name(video.stem + '_width_preview.mp4')
    if output.suffix.lower() != '.mp4':
        raise ValueError('--out 需要 MP4 文件路径')
    if output.exists():
        raise FileExistsError(f'预览文件已存在，请使用 --out 选择新文件：{output}')
    if roi is ...:
        roi = default_tag_roi()
    detection_roi = roi if roi is not None else (0, 0, SRC_H, SRC_H)
    detector = FrameDetector(detection_roi, contrast=contrast)
    roi = detector.roi if roi is not None else None
    paired_frames = 0
    previous_time = None
    last_progress = time.monotonic()
    _preview_console.write(f'{video.name} → {output}', state='开始')
    with _open_source(video, '宽度预览') as (source, source_stream), _open_mp4(output) as target:
        writer = PreviewWriter(target, source_stream, contrast=contrast)
        for frame in source.decode(source_stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('预览原片缺少帧时间戳')
            timestamp_s = float(frame.pts * frame.time_base)
            if previous_time is not None and timestamp_s <= previous_time:
                raise ValueError('预览原片时间必须严格递增')
            previous_time = timestamp_s
            detected = detector.detect(frame)
            writer.write(frame, detected, roi=roi)
            paired_frames += int(np.isfinite(detected.distance_px))
            if time.monotonic() - last_progress >= 5:
                _preview_console.write(f'{writer.frames} 帧；原片 {timestamp_s:.1f} s；双标签 {paired_frames} 帧')
                last_progress = time.monotonic()
        frames = writer.frames
        if frames < 2:
            raise ValueError('原片至少需要两帧有效视频')
        writer.flush()
    _preview_console.write(f'{frames} 帧；双标签 {paired_frames} 帧；{output}', state='完成')
    return dict(output=str(output), frames=frames, paired_frames=paired_frames, roi=roi, contrast=contrast,
                distance_units='native_pixels', native_lens_size=SRC_H, preview_size=PREVIEW_SIZE)


def monitor(video: Path, roi: tuple[int, int, int, int], *, hz: float = 30.,
        output: Path | None = None, benchmark: bool = False, video_output: Path | None = None,
        contrast: float = 1.) -> dict:
    video = _recording_path(video)
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
        with _open_source(video, '宽度检测') as (source, stream):
            configure_decoder(stream)
            with ExitStack() as outputs:
                destination = sys.stdout
                if output is not None:
                    csv_temporary = outputs.enter_context(atomic_file(output))
                    destination = outputs.enter_context(csv_temporary.open('w', newline=''))
                video_writer = None
                if video_output is not None:
                    target = outputs.enter_context(_open_mp4(video_output))
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
    detector = FrameDetector(roi, mode='fast', contrast=contrast)
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
        detected = detector.detect(frame)
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
            _monitor_console.write(f'{samples} 次；原片 {relative:.2f} s；双标签 {paired} 次；落后跳过 {late_dropped} 次')
            progress = time.perf_counter()
    if source_frames < 2:
        raise ValueError('原片至少需要两帧视频')
    return dict(source_frames=source_frames, samples=samples, paired_samples=paired, late_dropped=late_dropped,
                video_frames=video_writer.frames if video_writer is not None else 0,
                detect_median_ms=round(float(np.median(costs)), 3) if costs else None,
                detect_p95_ms=round(float(np.percentile(costs, 95)), 3) if costs else None)
