"""Decode an existing video frame when a browser cannot play H.264.

No transcoded videos or extracted frame collections are written to disk. Requests
are bounded and coalesced by the browser; the server limits concurrent decoders.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import threading
from pathlib import Path

from software.viewer.data import ViewerDataError

_DECODERS = threading.BoundedSemaphore(2)
REPO = Path(__file__).resolve().parents[2]


def find_ffmpeg() -> str:
    deps = Path(os.environ.get('UMI_DEPS', REPO / '.deps'))
    bundled = deps / 'env/bin/ffmpeg'
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled)
    installed = shutil.which('ffmpeg')
    if installed:
        return installed
    raise ViewerDataError('兼容播放需要 FFmpeg，请先安装项目的轨迹组件或将 ffmpeg 加入 PATH。')


def decode_frame(video: Path, seconds: float) -> bytes:
    if not math.isfinite(seconds) or seconds < 0 or seconds > 24 * 60 * 60:
        raise ViewerDataError('视频时间必须为 0 到 86400 秒之间的有限数值。')
    if not _DECODERS.acquire(timeout=5):
        raise ViewerDataError('视频解码忙，请稍后重试。')
    try:
        command = [find_ffmpeg(), '-nostdin', '-hide_banner', '-loglevel', 'error',
                   '-threads', '1', '-ss', f'{seconds:.6f}', '-i', str(video),
                   '-frames:v', '1', '-vf', 'scale=720:-2', '-threads', '1',
                   '-q:v', '4', '-f', 'image2pipe', '-vcodec', 'mjpeg', 'pipe:1']
        try:
            result = subprocess.run(command, capture_output=True, timeout=15)
        except subprocess.TimeoutExpired as error:
            raise ViewerDataError('视频解码超时，请重试或选择其他时间。') from error
        if result.returncode != 0 or not result.stdout.startswith(b'\xff\xd8'):
            reason = result.stderr.decode('utf-8', errors='replace').strip()[-400:]
            raise ViewerDataError(f'无法解码该时间的视频画面：{reason or "没有可用画面"}')
        return result.stdout
    finally:
        _DECODERS.release()
