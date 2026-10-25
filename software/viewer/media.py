"""只读原视频：后台生成浏览器预览，缓存首帧和最近的兼容画面。"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from software.viewer.data import ViewerDataError
from software.viewer import cache

_DECODERS = threading.BoundedSemaphore(2)
REPO = Path(__file__).resolve().parents[2]
_PREVIEWS = ThreadPoolExecutor(max_workers=1, thread_name_prefix='umi-preview')
_jobs = {}
_jobs_lock = threading.Lock()
# 仅控制查看器的派生视频，不改变轨迹算法输入；参数进入缓存身份。
_PREVIEW_OPTIONS = ('-vf', "scale=w='min(720,iw)':h=-2,setpts=PTS-STARTPTS", '-c:v', 'libvpx',
                    '-deadline', 'realtime', '-cpu-used', '12', '-threads', '4', '-lag-in-frames', '0',
                    '-b:v', '1500k', '-crf', '12', '-g', '30', '-pix_fmt', 'yuv420p')
_FRAME_OPTIONS = ('-frames:v', '1', '-vf', 'scale=720:-2', '-threads', '1',
                  '-q:v', '4', '-f', 'image2pipe', '-vcodec', 'mjpeg', 'pipe:1')


def video_version(video: Path) -> str:
    return cache.key(cache.file_identity(video))


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
    identity = (cache.file_identity(video), cache.file_identity(Path(find_ffmpeg())), _FRAME_OPTIONS)
    return _decode_cached(video.resolve(), identity, round(seconds, 6))


@lru_cache(maxsize=64)
def _decode_cached(video: Path, identity: tuple, seconds: float) -> bytes:
    if seconds == 0:
        target = cache.path('posters', identity, '.jpg')
        with cache.locked(target):
            if target.is_file():
                image = target.read_bytes()
                if image.startswith(b'\xff\xd8'):
                    return image
            image = _decode(video, seconds)
            with cache.atomic_output(target) as temporary:
                temporary.write_bytes(image)
            return image
    return _decode(video, seconds)


def _decode(video: Path, seconds: float) -> bytes:
    if not _DECODERS.acquire(timeout=5):
        raise ViewerDataError('视频解码忙，请稍后重试。')
    try:
        command = [find_ffmpeg(), '-nostdin', '-hide_banner', '-loglevel', 'error',
                   '-threads', '1', '-ss', f'{seconds:.6f}', '-i', str(video),
                   *_FRAME_OPTIONS]
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


def _preview_target(video: Path) -> Path:
    identity = (cache.file_identity(video), cache.file_identity(Path(find_ffmpeg())), _PREVIEW_OPTIONS)
    return cache.path('previews', identity, '.webm')


def _valid_preview(path: Path) -> bool:
    try:
        with path.open('rb') as stream:
            return stream.read(4) == b'\x1aE\xdf\xa3'
    except FileNotFoundError:
        return False


def _encode_preview(video: Path, target: Path) -> None:
    with cache.locked(target):
        if _valid_preview(target):
            return
        if target != _preview_target(video):
            raise ViewerDataError('原视频已变化，请重新选择录像。')
        before = cache.file_identity(video)
        with cache.atomic_output(target) as temporary:
            # 预览限制到 720 宽，不放大小图；保持原视频帧率，短 GOP 便于拖动。
            command = [find_ffmpeg(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                       '-threads', '2', '-i', str(video), '-map', '0:v:0', '-an',
                       *_PREVIEW_OPTIONS, str(temporary)]
            try:
                process = subprocess.run(command, capture_output=True, timeout=600)
            except subprocess.TimeoutExpired as error:
                raise ViewerDataError('浏览器预览生成超时，可继续使用原视频或兼容播放。') from error
            if process.returncode or not _valid_preview(temporary):
                reason = process.stderr.decode('utf-8', errors='replace')[-400:]
                raise ViewerDataError(f'浏览器预览生成失败：{reason or "未生成有效视频"}')
            if cache.file_identity(video) != before:
                raise ViewerDataError('原视频已变化，请重新选择录像。')


def preview_status(video: Path) -> dict:
    """立即返回状态；同一视频只启动一份任务，后台最多 1 份运行、1 份排队。"""
    target = _preview_target(video)
    if _valid_preview(target):
        return dict(status='ready', version=target.stem)
    with _jobs_lock:
        future = _jobs.get(target)
        if future is None or (future.done() and future.exception() is None):
            if sum(not job.done() for job in _jobs.values()) >= 2:
                return dict(status='preparing')
            for old, job in list(_jobs.items()):
                if job.done() and old != target:
                    del _jobs[old]
            future = _PREVIEWS.submit(_encode_preview, video.resolve(), target)
            _jobs[target] = future
        if future.done() and future.exception() is not None:
            return dict(status='failed', note=str(future.exception()))
    return dict(status='preparing')


def cached_preview(video: Path) -> Path | None:
    """只检查已有预览，不在读取轨迹时启动转码。"""
    try:
        target = _preview_target(video)
        return target if _valid_preview(target) else None
    except (OSError, ViewerDataError):
        return None


def preview_file(video: Path, version: str) -> Path:
    target = _preview_target(video)
    if target.stem != version or not _valid_preview(target):
        raise ViewerDataError('预览视频不存在或已更新，请重新选择录像。')
    return target
