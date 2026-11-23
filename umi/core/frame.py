"""The one frame shared by the trajectory and the calibration dataset.

An Insta360 X3 360 4K60 recording is one 3840x1920 picture: the working lens
(工作镜头, legacy key front) sees the gripper and occupies the right half; the
environment lens (环境镜头, legacy key rear) occupies the left half.
These image halves do not identify the left/right hand or factory lens indices.
The r1440 calibration belongs to a 1440
centre crop of that half. Calibration extraction keeps every second frame.
Trajectory videos use the crop, frame step and output size from trajectory.yaml.
This module owns recording-mode recognition, calibrated bounds, pixel conversion
and the exact derived video rate. Processing never extends the calibrated field of view.
"""

from __future__ import annotations

from fractions import Fraction


PROFILE = "x3_360_4k_60"
SRC_W, SRC_H = 3840, 1920
CROP = 1440
STEP = 2
T0_S = 1.0                                  # EuRoC files shift the clock by one second
FPS_SRC_RATE = '60000/1001'
FPS_SRC = float(Fraction(FPS_SRC_RATE))
LENS_HALF = {"front": "right", "rear": "left"}
# Recognition/timing tolerances belong to this supported recording contract.
# Loosening these would change when fusion may assume a regular frame grid.
FPS_TOLERANCE = 0.05
REGULAR_GRID_TOLERANCE_MS = 1.0
GAP_INTERVAL_RATIO = 1.5
IMU_COVERAGE_TOLERANCE_S = 0.05


def detect_profile_from_streams(video_streams) -> str:
    """按单路视频的几何识别模式；方形单镜头录像单独标识，供提取入口拒绝。"""
    if not isinstance(video_streams, list) or len(video_streams) != 1:
        return "unknown"
    stream = video_streams[0]
    width, height = (stream.get("w"), stream.get("h")) if isinstance(stream, dict) else stream[:2]
    if not width or not height:
        return "unknown"
    width, height = int(width), int(height)
    if (width, height) == (SRC_W, SRC_H):
        return PROFILE
    if width == height:
        return "x3_360_5k7_30"
    return "unknown"


def crop_xywh(half: str, crop=None) -> tuple[int, int, int, int]:
    """``(x, y, w, h)`` of one lens in the 3840x1920 frame. ``half`` is ``left`` or ``right``."""
    if half not in ("left", "right"):
        raise ValueError(f"lens half must be 'left' or 'right', got {half!r}")
    margin = (SRC_H - CROP) // 2
    x, y, size, _ = calibrated_crop_xywh(crop)
    return margin + (SRC_H if half == "right" else 0) + x, margin + y, size, size


def calibrated_crop_xywh(crop=None) -> tuple[int, int, int, int]:
    """裁切坐标以已标定的 1440 方图为基准，拒绝越出已标定视野。"""
    if crop is None:
        return 0, 0, CROP, CROP
    if type(crop.size_px) is not int or not 2 <= crop.size_px <= CROP or crop.size_px % 2:
        raise ValueError(f'video.crop.size_px 必须是 2–{CROP} 之间的偶数')
    if type(crop.offset_x_px) is not int or type(crop.offset_y_px) is not int:
        raise ValueError('video.crop.offset_x_px / offset_y_px 必须为整数')
    x = (CROP - crop.size_px) // 2 + crop.offset_x_px
    y = (CROP - crop.size_px) // 2 + crop.offset_y_px
    if min(x, y) < 0 or max(x, y) + crop.size_px > CROP:
        raise ValueError('video.crop 超出已标定的 1440×1440 视野')
    return x, y, crop.size_px, crop.size_px


def output_frame_rate(frame_step: int) -> Fraction:
    if type(frame_step) is not int or frame_step < 1:
        raise ValueError('video.frame_step 必须为正整数')
    return Fraction(FPS_SRC_RATE) / frame_step


def validate_video_parameters(video) -> None:
    _, _, crop_size, _ = calibrated_crop_xywh(video.crop)
    validate_video_size(video.size_px)
    if video.size_px > crop_size:
        raise ValueError('video.size_px 不能大于 video.crop.size_px')
    output_frame_rate(video.frame_step)
    if video.codec not in ('libx264', 'libx265'):
        raise ValueError('video.codec 只支持 libx264 或 libx265（MP4）')
    if video.pixel_format not in ('yuv420p', 'yuv444p'):
        raise ValueError('video.pixel_format 只支持 yuv420p 或 yuv444p')
    if video.resize_filter not in ('area', 'bilinear', 'bicubic', 'lanczos', 'neighbor'):
        raise ValueError('video.resize_filter 必须为 area、bilinear、bicubic、lanczos 或 neighbor')


def validate_video_size(size_px: int) -> None:
    """保留正方形视野；yuv420p 需要偶数边长，不放大超过标定裁切图。"""
    if type(size_px) is not int or not 2 <= size_px <= CROP or size_px % 2:
        raise ValueError(f'video.size_px 必须是 2–{CROP} 之间的偶数')


def scale_intrinsics(intrinsics: dict, source_px: int, target_px: int) -> dict:
    """同一裁切视野的等比缩放；按像素中心转换主点，KB4 畸变系数保持原值。"""
    scale = target_px / source_px
    return {**intrinsics,
            'fx': intrinsics['fx'] * scale, 'fy': intrinsics['fy'] * scale,
            'cx': (intrinsics['cx'] + 0.5) * scale - 0.5,
            'cy': (intrinsics['cy'] + 0.5) * scale - 0.5}


def video_intrinsics(intrinsics: dict, video, *, source_px: int = CROP) -> dict:
    native = scale_intrinsics(intrinsics, source_px, CROP)
    x, y, size, _ = calibrated_crop_xywh(video.crop)
    native.update(cx=native['cx'] - x, cy=native['cy'] - y)
    return scale_intrinsics(native, size, video.size_px)


def geometry_record(size_px: int, *, mode: str, video=None) -> dict:
    """结果中的图像规格；与视频生成、标定数据提取共用同一组定义。"""
    validate_video_size(size_px)
    crop = video.crop if video else None
    step = video.frame_step if video else STEP
    return {
        'profile': PROFILE,
        'source_px': [SRC_W, SRC_H],
        'calibration_crop_px': CROP,
        'crop_px': crop.size_px if crop else CROP,
        'front_lens': {'half': LENS_HALF['front'], 'crop_xywh': list(crop_xywh(LENS_HALF['front'], crop))},
        'rear_lens': {'half': LENS_HALF['rear'], 'crop_xywh': list(crop_xywh(LENS_HALF['rear'], crop)),
                      'used': mode == 'dual'},
        'slam_video_px': [size_px, size_px],
        'frame_step': step,
        'fps_src': FPS_SRC,
        'fps_used': float(output_frame_rate(step)),
    }
