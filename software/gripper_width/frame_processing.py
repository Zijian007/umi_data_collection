"""标签输入图像：先按原始像素裁剪，再转灰度，检测后恢复半幅坐标。"""

from __future__ import annotations

from typing import TYPE_CHECKING
from math import lcm

import av
import numpy as np

from software.camera.x3.frame import SRC_W

if TYPE_CHECKING:
    from software.gripper_width.detection import FrameDetection, MarkerDetector


def crop_gray(frame: av.VideoFrame, roi: tuple[int, int, int, int],
              converter: av.video.reformatter.VideoReformatter) -> np.ndarray:
    """直接裁剪源像素平面；对齐色度采样后转换，最后恢复精确 ROI 边界。"""
    left, top, right, bottom = roi
    left, right = left + SRC_W // 2, right + SRC_W // 2
    components = frame.format.components
    alignment_x = max(frame.width // component.width for component in components)
    alignment_y = max(frame.height // component.height for component in components)
    if any(component.bits > 8 for component in components):
        # 高位深转 8 位使用周期抖动；保留原图中的 8×8 相位，避免裁剪造成 1 灰度级差异。
        alignment_x, alignment_y = lcm(alignment_x, 8), lcm(alignment_y, 8)
    aligned_left, aligned_top = left // alignment_x * alignment_x, top // alignment_y * alignment_y
    aligned_right = (right + alignment_x - 1) // alignment_x * alignment_x
    aligned_bottom = (bottom + alignment_y - 1) // alignment_y * alignment_y
    cropped = av.VideoFrame(aligned_right - aligned_left, aligned_bottom - aligned_top, frame.format.name)
    cropped.colorspace, cropped.color_range = frame.colorspace, frame.color_range
    for index, (source, target) in enumerate(zip(frame.planes, cropped.planes)):
        if len(frame.planes) == 1:
            if frame.format.padded_bits_per_pixel % 8:
                raise ValueError(f'标签 ROI 不支持位打包像素格式：{frame.format.name}')
            sample_bytes = frame.format.padded_bits_per_pixel // 8
        else:
            sample_bytes = sum((component.bits + 7) // 8 for component in components if component.plane == index)
        if not sample_bytes:
            raise ValueError(f'标签 ROI 不支持该像素平面格式：{frame.format.name}')
        source_pixels = np.ndarray((source.height, source.width * sample_bytes), np.uint8,
                                   buffer=source, strides=(source.line_size, 1))
        target_pixels = np.ndarray((target.height, target.width * sample_bytes), np.uint8,
                                   buffer=target, strides=(target.line_size, 1))
        source_left = aligned_left * source.width // frame.width * sample_bytes
        source_top = aligned_top * source.height // frame.height
        target_pixels[:] = source_pixels[source_top:source_top + target.height,
                                         source_left:source_left + target.width * sample_bytes]
    gray = converter.reformat(cropped, format='gray', threads=1).to_ndarray()
    return gray[top - aligned_top:bottom - aligned_top, left - aligned_left:right - aligned_left]


def detect_frame(frame: av.VideoFrame, detector: MarkerDetector, roi: tuple[int, int, int, int],
                 converter: av.video.reformatter.VideoReformatter) -> FrameDetection:
    """检测器使用 ROI 局部坐标；所有输出统一为原始工作镜头半幅坐标。"""
    return detector.detect(crop_gray(frame, roi, converter)).translated(roi[:2])
