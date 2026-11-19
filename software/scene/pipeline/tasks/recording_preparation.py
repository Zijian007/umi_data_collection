"""任务后端共用的录像准备缓存与公共时间区间检查。"""

from __future__ import annotations

from pathlib import Path

from software.infrastructure import provenance as artifact_identity
from software.scene import layout
from software.scene.models import Episode, CalibratedCamera, PreparedRecording
from software.scene.pipeline import recordings
from software.scene.cache.storage import StageCache


def prepare_with_cache(episode, arm, camera, scene, cores, *, work_dir: Path, cache_audit=None):
    """与建图共用阶段缓存；本次设置和日志写在运行工作目录，缓存只存解码产物。"""
    cache = StageCache(layout.metadata(episode.root, 'cache'), artifact_identity.engine_identity('trajectory'), audit=cache_audit)
    return recordings.prepare_recording(episode.recordings[arm], camera, scene, work_dir / arm, cores=cores, cache=cache)


def prepare_episode_recordings(episode: Episode, cameras: dict[str, CalibratedCamera],
                               video_offsets_s: dict[str, float], *, work_dir: Path, cores: str = '',
                               arm_cores: dict[str, str] | None = None, cache_audit=None) -> dict[str, PreparedRecording]:
    """准备各手录像并检查共同时间区间；不估计跨相机时间偏移。"""
    prepared = {arm: prepare_with_cache(episode, arm, cameras[arm], episode.scene,
                                   arm_cores[arm] if arm_cores else cores, work_dir=work_dir,
                                   cache_audit=cache_audit)
                for arm in episode.recordings}
    start = max(recording.video_times_s[0] + video_offsets_s[arm] for arm, recording in prepared.items())
    end = min(recording.video_times_s[-1] + video_offsets_s[arm] for arm, recording in prepared.items())
    if start >= end:
        raise ValueError('左右手录像在任务时间轴上没有重叠')
    return prepared
