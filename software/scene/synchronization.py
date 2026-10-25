"""双手录像的显式时间同步；t_episode = t_video + offset_s。

音频延迟必须先由用户提供实测音频到视频偏移，不把解码音频起点直接当成视频起点。
"""
from __future__ import annotations

import subprocess
import shutil

import numpy as np
from scipy.signal import correlate, correlation_lags

from software.scene.models import Episode, Parameters, read_yaml
from software.storage import write_yaml


def audio_offset(reference: np.ndarray, other: np.ndarray, p: Parameters) -> tuple[float, dict]:
    """归一化互相关；正偏移表示 other 视频开始得更晚，合成信号测试锁定符号。"""
    x, y = np.asarray(reference, float), np.asarray(other, float)
    if (x.ndim != 1 or y.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all()
            or min(len(x), len(y)) < p.audio_minimum_overlap_s*p.audio_sample_rate):
        raise ValueError('可用于同步的音频过短或包含无效值')
    if min(np.std(x), np.std(y)) < np.finfo(float).eps:
        raise ValueError('音频静音，无法确定双手时间关系')
    x, y = x-x.mean(), y-y.mean()
    one_x, one_y = np.ones(len(x)), np.ones(len(y))
    cross = lambda a, b: correlate(a, b, mode='full', method='fft')
    count = np.maximum(1., np.round(cross(one_x, one_y)))
    sx, sy = cross(x, one_y), cross(one_x, y)
    vx = np.maximum(0., cross(x*x, one_y)-sx*sx/count)
    vy = np.maximum(0., cross(one_x, y*y)-sy*sy/count)
    correlation = (cross(x, y)-sx*sy/count)/np.maximum(np.sqrt(vx*vy), np.finfo(float).eps)
    lags = correlation_lags(len(x), len(y), mode='full')
    valid = ((np.abs(lags) <= p.audio_maximum_offset_s*p.audio_sample_rate)
             & (count >= p.audio_minimum_overlap_s*p.audio_sample_rate)
             & (vx > .01*np.var(x)*count) & (vy > .01*np.var(y)*count))
    correlation[~valid] = -np.inf
    peak = int(np.argmax(correlation))
    best = float(correlation[peak])
    second_mask = valid & (np.abs(lags-lags[peak]) > p.audio_peak_exclusion_s*p.audio_sample_rate)
    second = max(0., float(np.max(correlation[second_mask]))) if second_mask.any() else 0.
    ratio = best/max(second, np.finfo(float).eps)
    if best < p.audio_minimum_correlation or ratio < p.audio_minimum_peak_ratio:
        raise ValueError(f'音频同步不唯一或相关性不足：相关 {best:.3f}，峰值比 {ratio:.2f}；请使用共同提示音或手动实测偏移')
    return float(lags[peak]/p.audio_sample_rate), dict(correlation=best, peak_ratio=ratio,
        sample_rate=p.audio_sample_rate, audio_lag_s=float(lags[peak]/p.audio_sample_rate),
        clock_model='constant_offset_no_drift_correction')


def decode_audio(video, p: Parameters) -> np.ndarray:
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise ValueError('音频同步需要 FFmpeg；请通过 ./umi-scene 启动')
    command = [executable, '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', str(video),
               '-map', '0:a:0', '-t', str(p.audio_duration_s), '-vn', '-ac', '1', '-ar', str(p.audio_sample_rate),
               '-f', 'f32le', 'pipe:1']
    result = subprocess.run(command, capture_output=True, timeout=180)
    if result.returncode or not result.stdout:
        raise ValueError('音频读取失败：'+result.stderr.decode(errors='replace')[-400:])
    return np.frombuffer(result.stdout, dtype='<f4').astype(float)


def synchronize(episode: Episode, *, right_offset_s: float | None = None,
                audio_to_video: dict[str, float] | None = None) -> dict:
    if set(episode.recordings) != {'left', 'right'}:
        raise ValueError('双手同步需要 left 和 right 两段不同相机的录像')
    evidence = dict(recordings={arm: dict(path=str(video), bytes=video.stat().st_size, mtime_ns=video.stat().st_mtime_ns)
                               for arm, video in episode.recordings.items()})
    if audio_to_video is not None:
        if set(audio_to_video) != {'left', 'right'} or not all(np.isfinite(v) for v in audio_to_video.values()):
            raise ValueError('音频同步需要左右手各自实测的 audio_to_video_s')
        samples = {arm: decode_audio(video, episode.scene.parameters) for arm, video in episode.recordings.items()}
        lag, quality = audio_offset(samples['left'], samples['right'], episode.scene.parameters)
        right_offset_s = lag+audio_to_video['left']-audio_to_video['right']
        evidence.update(quality, audio_to_video_s=audio_to_video)
        method = 'audio'
    else:
        if right_offset_s is None or not np.isfinite(right_offset_s):
            raise ValueError('需要显式指定 --right-offset-s；正值表示右手视频开始得更晚')
        method = 'manual'
    sync = dict(method=method, offsets_s=dict(left=0., right=float(right_offset_s)), evidence=evidence)
    data = read_yaml(episode.path)
    data['synchronization'] = sync
    write_yaml(episode.path, data)
    return sync
