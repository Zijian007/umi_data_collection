"""写出可独立读取的训练中间包；失败不发布部分数据集。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import shutil
import tempfile

import numpy as np

from umi.infra import provenance as artifact_identity
from umi.dataset import sampling
from umi.dataset.schema import EPISODE_FORMAT, FORMAT, SCHEMA_VERSION, reference_metadata, validate_quality_policy
from umi.workflows.scene import episodes
from umi.workflows.scene.inputs.batch import read_episode
from umi.store.result_io import result_paths, read_arm_settings, scene_reference
from umi.infra.file_store import locked, write_json


@dataclass(frozen=True)
class AcceptedEpisode:
    path: Path
    name: Path
    scene_root: Path
    run: Path
    manifest: dict
    review: dict
    result_sha256: str


def read_accepted_episode(path: Path) -> AcceptedEpisode:
    episode = read_episode(path)
    run, manifest, review = episodes.accepted_run(episode)
    if not episode.instruction.strip():
        raise ValueError(f'训练导出需要非空 instruction：{path}')
    if not manifest['synchronization']['confirmed']:
        raise ValueError(f'训练导出需要先核验双手 synchronization：{path}')
    if manifest.get('instruction') != episode.instruction:
        raise ValueError(f'结果尚未携带当前指令，请重新 run 和验收：{path}')
    if set(manifest['tcp_frames']) != set(episode.expected_arms):
        raise ValueError('结果 TCP 定义与任务使用手不一致')
    for frame in manifest['tcp_frames'].values():
        if not isinstance(frame, dict) or not isinstance(frame.get('convention'), str) or not frame['convention']:
            raise ValueError('训练导出需要可识别的 TCP 定义')
    name = episode.session.relative_path / episode.root.name
    return AcceptedEpisode(episode.path, name, episode.scene.root, run, manifest, review,
                           artifact_identity.file_hash(result_paths(run).manifest))


def select_accepted_episodes(paths: list[Path]) -> list[AcceptedEpisode]:
    found = episodes.episode_paths(paths)
    if not found:
        raise ValueError('没有找到任务清单')
    selected = [read_accepted_episode(path) for path in found]
    if len({accepted_episode.name for accepted_episode in selected}) != len(selected):
        raise ValueError('任务导出身份重复，请检查场景/套件/批次/任务名称')
    return selected


def verify_accepted_episode_unchanged(accepted_episode: AcceptedEpisode) -> None:
    current = read_accepted_episode(accepted_episode.path)
    if (current.run != accepted_episode.run
            or current.result_sha256 != accepted_episode.result_sha256
            or current.review != accepted_episode.review):
        raise ValueError(f'导出期间源结果或验收发生变化：{accepted_episode.path}')


def _write_episode(item: AcceptedEpisode, staging: Path, *, fps: float, max_gap_s: float,
                   quality_policy: str, unknown_support: str,
                   time_window_s: tuple[float, float] | None) -> dict:
    """在未发布目录写入一集；事务清理和最终源结果复核由 build 负责。"""
    target = staging / 'episodes' / item.name
    target.mkdir(parents=True)
    arms = list(item.manifest['arms'])
    arrays, quality, readiness = sampling.sample_episode(item.run, item.manifest, fps, max_gap_s,
        quality_policy=quality_policy, unknown_support=unknown_support,
        time_window_s=time_window_s)
    np.savez_compressed(target / 'samples.npz', **arrays)
    paths = result_paths(item.run)
    tcp_geometry = {}
    for arm in arms:
        settings = read_arm_settings(paths.data / arm, item.manifest)
        tcp_geometry[arm] = dict(T_camera_tcp=settings['T_camera_tcp'], cad_status=settings['cad_status'])
        video = target / 'videos' / f'{arm}.mp4'
        video.parent.mkdir(exist_ok=True)
        source_video = paths.data / arm / 'front.mp4'
        shutil.copy2(source_video, video)
        asset = source_video.relative_to(paths.published).as_posix()
        if artifact_identity.file_hash(video) != item.manifest['assets'][asset]:
            raise ValueError('导出期间源视频发生变化')
    metadata = dict(kind=EPISODE_FORMAT, schema_version=SCHEMA_VERSION,
        instruction=item.manifest['instruction'], synchronization=item.manifest['synchronization'],
        **reference_metadata(scene_reference(item.manifest)), scene=item.manifest['scene'], kit=item.manifest['kit'],
        arms=arms, tcp_frames=item.manifest['tcp_frames'],
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
    selected = select_accepted_episodes(paths)
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
                verify_accepted_episode_unchanged(item)
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
