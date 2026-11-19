"""写出可独立读取的训练中间包；失败不发布部分数据集。"""

from __future__ import annotations

from pathlib import Path
import math
import shutil
import tempfile

import numpy as np

from software.infrastructure import provenance as artifact_identity
from software.dataset import gripper, sampling, source
from software.dataset.policy import validate_quality_policy
from software.dataset.coordinates import reference_metadata
from software.scene.result_io import read_arm_settings, scene_reference
from software.infrastructure.file_store import locked, write_json


from software.dataset.schema import FORMAT, SCHEMA_VERSION, EPISODE_FORMAT


def _write_episode(item: source.AcceptedEpisode, staging: Path, *, fps: float, max_gap_s: float,
                   quality_policy: str, unknown_support: str,
                   time_window_s: tuple[float, float] | None) -> dict:
    """在未发布目录写入一集；事务清理和最终源结果复核由 build 负责。"""
    target = staging / 'episodes' / item.name
    target.mkdir(parents=True)
    offsets = item.manifest['synchronization']['offsets_s']
    arms = {arm: sampling.read_arm(item.run / arm, offsets[arm]) for arm in item.manifest['arms']}
    arrays, quality = sampling.sample(arms, fps, max_gap_s,
        quality_policy=quality_policy, unknown_support=unknown_support,
        time_window_s=time_window_s)
    width_arrays, readiness = gripper.sample(item.run, item.manifest, arms, arrays['t_episode_s'], max_gap_s)
    arrays.update(width_arrays)
    np.savez_compressed(target / 'samples.npz', **arrays)
    tcp_geometry = {}
    for arm in arms:
        settings = read_arm_settings(item.run / arm, item.manifest)
        tcp_geometry[arm] = dict(T_camera_tcp=settings['T_camera_tcp'], cad_status=settings['cad_status'])
        video = target / 'videos' / f'{arm}.mp4'
        video.parent.mkdir(exist_ok=True)
        shutil.copy2(item.run / arm / 'front.mp4', video)
        if artifact_identity.file_hash(video) != item.manifest['assets'][f'{arm}/front.mp4']:
            raise ValueError('导出期间源视频发生变化')
    metadata = dict(kind=EPISODE_FORMAT, schema_version=SCHEMA_VERSION,
        instruction=item.manifest['instruction'], synchronization=item.manifest['synchronization'],
        **reference_metadata(scene_reference(item.manifest)), scene=item.manifest['scene'], kit=item.manifest['kit'],
        arms=list(arms), tcp_frames=item.manifest['tcp_frames'],
        tcp_geometry=tcp_geometry, actual_error=item.manifest['actual_error'],
        position_unit='m', fps=fps, sampling_window_s=quality['window_s'],
        rows=len(arrays['t_episode_s']), quality=quality,
        training_ready=readiness['training_ready'], missing_channels=readiness['missing_channels'],
        not_ready_reasons=readiness['not_ready_reasons'], gripper_width=readiness['measurements'],
        review=item.review, source=dict(episode_id=str(item.name), run_id=item.run.name,
            result_sha256=item.result_sha256, inputs=item.manifest['inputs']),
        arrays={name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in arrays.items()},
        videos={arm: f'videos/{arm}.mp4' for arm in arms})
    write_json(target / 'episode.json', metadata)
    hashes = {str(path.relative_to(target)): artifact_identity.file_hash(path)
              for path in sorted(target.rglob('*')) if path.is_file()}
    return dict(path=str(target.relative_to(staging)), rows=metadata['rows'], files=hashes,
                selection=quality['selection'], training_ready=metadata['training_ready'],
                missing_channels=metadata['missing_channels'], not_ready_reasons=metadata['not_ready_reasons'])


def build(paths: list[Path], output: Path, *, fps: float = 30., max_gap_s: float = .25,
          quality_policy: str = 'all', unknown_support: str = 'drop',
          time_window_s: tuple[float, float] | None = None) -> dict:
    if not math.isfinite(fps) or not 0 < fps <= 120:
        raise ValueError('fps 必须为 (0, 120] 内的有限数值')
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError('max_gap_s 必须为有限正秒数')
    if time_window_s is not None and (
            len(time_window_s) != 2 or not all(math.isfinite(value) for value in time_window_s)
            or time_window_s[1] <= time_window_s[0]):
        raise ValueError('time_window_s 必须是有限的 [start, end] 秒区间，且 end 大于 start')
    validate_quality_policy(quality_policy, unknown_support)
    selected = source.select_accepted_episodes(paths)
    output = output.expanduser().resolve()
    if any(output.is_relative_to(item.scene_root) or item.scene_root.is_relative_to(output) for item in selected):
        raise ValueError('数据集输出目录不能位于源场景内或覆盖源场景')
    output.parent.mkdir(parents=True, exist_ok=True)
    with locked(output.parent, name=f'.{output.name}.dataset.lock'):
        if output.exists():
            raise ValueError(f'导出目录已存在，拒绝覆盖：{output}')
        staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
        try:
            entries = [_write_episode(item, staging, fps=fps, max_gap_s=max_gap_s,
                                      quality_policy=quality_policy, unknown_support=unknown_support,
                                      time_window_s=time_window_s)
                       for item in selected]
            for item in selected:
                source.verify_accepted_episode_unchanged(item)
            missing_channels = sorted({name for entry in entries for name in entry['missing_channels']})
            ready = all(entry['training_ready'] for entry in entries)
            manifest = dict(kind=FORMAT, schema_version=SCHEMA_VERSION, fps=fps, max_gap_s=max_gap_s,
                time_window_s=list(time_window_s) if time_window_s is not None else None,
                quality_policy=quality_policy, unknown_support=unknown_support,
                training_ready=ready, missing_channels=missing_channels, episodes=entries)
            write_json(staging / 'export_report.json', dict(episodes=len(entries), rows=sum(entry['rows'] for entry in entries),
                quality_policy=quality_policy, unknown_support=unknown_support,
                quality_selection=[dict(path=entry['path'], **entry['selection']) for entry in entries],
                training_ready=ready, missing_channels=missing_channels,
                not_ready=[dict(path=entry['path'], reasons=entry['not_ready_reasons']) for entry in entries if not entry['training_ready']],
                note='training_ready 仅表示基础观测通道齐全、宽度已核验且采样有效；模型动作切片与部署需另行适配。'))
            write_json(staging / 'manifest.json', manifest)
            if output.exists():
                raise ValueError('导出期间目标目录已被创建，拒绝覆盖')
            staging.rename(output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return dict(output=str(output), episodes=len(entries), rows=sum(e['rows'] for e in entries),
                training_ready=ready, missing_channels=missing_channels)
