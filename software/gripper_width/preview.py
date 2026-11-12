"""把原片的工作镜头与双标记检测叠加为 MP4；显示原始像素距离，不需要毫米测量。"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import time

import numpy as np

from . import detection
from .frame_processing import detect_frame
from software.camera.x3.frame import SRC_H, SRC_W, FPS_SRC_RATE
from software.console import Console
from software.storage import atomic_file


PREVIEW_SIZE = 960
console = Console('宽度预览')


def annotate(front: np.ndarray, detected: detection.FrameDetection, *, timestamp_s: float,
             frame_index: int, roi: tuple[int, int, int, int] | None = None, contrast: float = 1.) -> np.ndarray:
    import cv2

    rendered = cv2.resize(front, (PREVIEW_SIZE, PREVIEW_SIZE), interpolation=cv2.INTER_AREA)
    scale = PREVIEW_SIZE / SRC_H
    if contrast > 1:
        left, top, right, bottom = (np.rint(np.asarray(roi) * scale).astype(int) if roi is not None
                                    else (0, 0, PREVIEW_SIZE, PREVIEW_SIZE))
        rendered[top:bottom, left:right] = cv2.convertScaleAbs(rendered[top:bottom, left:right], alpha=contrast)
    rendered[:85] = (rendered[:85] * .2).astype(np.uint8)
    if roi is not None:
        left, top, right, bottom = np.rint(np.asarray(roi) * scale).astype(int)
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
    if np.isfinite(detected.distance_px):
        cv2.line(rendered, centres[0], centres[1], (0, 230, 255), 2, cv2.LINE_AA)
        distance_label = f'ID 0 <-> 1: {detected.distance_px:.2f} px  (native 1920x1920)'
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
        import av

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

    def write(self, frame, detected: detection.FrameDetection, *, roi=None) -> None:
        import av

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


def render(video: Path, output: Path | None = None, *,
           roi: tuple[int, int, int, int] | None = detection.DEFAULT_TAG_ROI,
           contrast: float = 1.) -> dict:
    import av

    video = video.expanduser().resolve()
    output = output.expanduser().resolve() if output is not None else video.with_name(video.stem + '_width_preview.mp4')
    if not video.is_file() or video.suffix.lower() != '.insv':
        raise ValueError(f'请提供已有的原始 INSV 文件：{video}')
    if output.suffix.lower() != '.mp4':
        raise ValueError('--out 需要 MP4 文件路径')
    if output.exists():
        raise FileExistsError(f'预览文件已存在，请使用 --out 选择新文件：{output}')
    if roi is not None:
        roi = detection.validate_roi(roi)
    detection_roi = roi if roi is not None else (0, 0, SRC_H, SRC_H)
    left, top, right, bottom = detection_roi
    detector = detection.MarkerDetector((0, 0, right - left, bottom - top), contrast=contrast)
    converter = av.video.reformatter.VideoReformatter()
    frames = paired_frames = 0
    previous_time = None
    last_progress = time.monotonic()
    console.write(f'{video.name} → {output}', state='开始')
    with av.open(str(video)) as source:
        if len(source.streams.video) != 1:
            raise ValueError('宽度预览需要单路 3840×1920 INSV 视频')
        source_stream = source.streams.video[0]
        if (source_stream.codec_context.width, source_stream.codec_context.height) != (SRC_W, SRC_H):
            raise ValueError('宽度预览需要 X3 原始 3840×1920 INSV')
        with atomic_file(output) as temporary, av.open(str(temporary), 'w', format='mp4',
                                                     options={'movflags': '+faststart'}) as target:
            writer = PreviewWriter(target, source_stream, contrast=contrast)
            for frame in source.decode(source_stream):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError('预览原片缺少帧时间戳')
                timestamp_s = float(frame.pts * frame.time_base)
                if previous_time is not None and timestamp_s <= previous_time:
                    raise ValueError('预览原片时间必须严格递增')
                previous_time = timestamp_s
                detected = detect_frame(frame, detector, detection_roi, converter)
                writer.write(frame, detected, roi=roi)
                frames += 1
                paired_frames += int(np.isfinite(detected.distance_px))
                if time.monotonic() - last_progress >= 5:
                    console.write(f'{frames} 帧；原片 {timestamp_s:.1f} s；双标签 {paired_frames} 帧')
                    last_progress = time.monotonic()
            if frames < 2:
                raise ValueError('原片至少需要两帧有效视频')
            writer.flush()
    console.write(f'{frames} 帧；双标签 {paired_frames} 帧；{output}', state='完成')
    return dict(output=str(output), frames=frames, paired_frames=paired_frames, roi=roi, contrast=contrast,
                distance_units='native_pixels', native_lens_size=SRC_H, preview_size=PREVIEW_SIZE)
