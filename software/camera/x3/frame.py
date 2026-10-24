"""The one frame shared by the trajectory and the calibration dataset.

An Insta360 X3 360 4K60 recording is one 3840x1920 picture: the front lens is the
right half, the rear lens the left half. The r1440 calibration belongs to a 1440
centre crop of that half. Every second frame is kept, so 59.94 fps becomes 29.97.
This module owns recording-mode recognition, fixed crop/timing and the pixel
conversion used when trajectory videos are resized. Output size is configured
separately in config/trajectory.yaml; it never changes the calibrated field of view.
"""

from __future__ import annotations

from fractions import Fraction

PROFILE = "x3_360_4k_60"
SRC_W, SRC_H = 3840, 1920
CROP = 1440
STEP = 2
T0_S = 1.0                                  # EuRoC files shift the clock by one second
FPS_SRC_RATE = '60000/1001'
FPS_USED_RATE = str(Fraction(FPS_SRC_RATE) / STEP)
FPS_SRC = float(Fraction(FPS_SRC_RATE))
FPS_USED = float(Fraction(FPS_USED_RATE))
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


def crop_xywh(half: str) -> tuple[int, int, int, int]:
    """``(x, y, w, h)`` of one lens in the 3840x1920 frame. ``half`` is ``left`` or ``right``."""
    if half not in ("left", "right"):
        raise ValueError(f"lens half must be 'left' or 'right', got {half!r}")
    margin = (SRC_H - CROP) // 2
    return margin + (SRC_H if half == "right" else 0), margin, CROP, CROP


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


def geometry_record(size_px: int, *, mode: str) -> dict:
    """结果中的图像规格；与视频生成、标定数据提取共用同一组定义。"""
    validate_video_size(size_px)
    return {
        'profile': PROFILE,
        'source_px': [SRC_W, SRC_H],
        'crop_px': CROP,
        'front_lens': {'half': LENS_HALF['front'], 'crop_xywh': list(crop_xywh(LENS_HALF['front']))},
        'rear_lens': {'half': LENS_HALF['rear'], 'crop_xywh': list(crop_xywh(LENS_HALF['rear'])),
                      'used': mode == 'dual'},
        'slam_video_px': [size_px, size_px],
        'frame_step': STEP,
        'fps_src': FPS_SRC,
        'fps_used': FPS_USED,
    }
