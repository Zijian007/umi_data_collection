"""时间轴边界：算法使用秒，EuRoC 文件的人工 +1 s 只在读写时转换。

video_s：原片视频时钟；imu_s：同机 IMU 时钟。
cam_time_offset_ns 的符号沿用 Basalt：t_imu = t_video + offset。
"""
from __future__ import annotations

import numpy as np

from software.camera.x3.frame import T0_S


def canonical_to_video_seconds(canonical_s):
    return np.asarray(canonical_s) - T0_S


def video_to_canonical_seconds(video_s):
    return T0_S + np.asarray(video_s)


def euroc_ns_to_seconds(timestamp_ns, *, camera_offset_ns: int = 0):
    """去掉文件的 +1 s；给出相机偏移时，将 IMU 样本换到视频时钟。

    保留先减整数纳秒偏移、再换秒的顺序，避免改变历史浮点结果。
    """
    return (np.asarray(timestamp_ns) - camera_offset_ns) / 1e9 - T0_S


def video_to_imu_seconds(video_s, camera_offset_ns: int):
    return np.asarray(video_s) + camera_offset_ns / 1e9
