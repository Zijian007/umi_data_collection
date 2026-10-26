"""训练导出与场景管理的只读边界。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from software import config
from software.scene import episodes, inputs


@dataclass(frozen=True)
class AcceptedEpisode:
    path: Path
    name: Path
    scene_root: Path
    run: Path
    manifest: dict
    review: dict
    result_sha256: str


def read_accepted(path: Path) -> AcceptedEpisode:
    episode = inputs.read_episode(path)
    run, manifest, review = episodes.accepted_run(episode)
    if not episode.instruction.strip():
        raise ValueError(f'训练导出需要非空 instruction：{path}')
    if not episode.synchronization['confirmed']:
        raise ValueError(f'训练导出需要先核验双手 synchronization：{path}')
    if manifest.get('instruction') != episode.instruction or manifest.get('synchronization') != episode.synchronization:
        raise ValueError(f'结果尚未携带当前指令/同步配置，请重新 run 和验收：{path}')
    if set(manifest['tcp_frames']) != set(episode.expected_arms):
        raise ValueError('结果 TCP 定义与任务使用手不一致')
    for frame in manifest['tcp_frames'].values():
        if not isinstance(frame, dict) or not isinstance(frame.get('convention'), str) or not frame['convention']:
            raise ValueError('训练导出需要可识别的 TCP 定义')
    name = Path(episode.scene.name) / episode.session.kit['id'] / episode.session.name / episode.path.parent.name
    return AcceptedEpisode(episode.path, name, episode.scene.root, run, manifest, review,
                           config.file_hash(run / 'episode.json'))


def select(paths: list[Path]) -> list[AcceptedEpisode]:
    found = episodes.episode_paths(paths)
    if not found:
        raise ValueError('没有找到任务清单')
    selected = [read_accepted(path) for path in found]
    if len({item.name for item in selected}) != len(selected):
        raise ValueError('任务导出身份重复，请检查场景/套件/批次/任务名称')
    return selected


def verify_current(source: AcceptedEpisode) -> None:
    current = read_accepted(source.path)
    if current.run != source.run or current.result_sha256 != source.result_sha256 or current.review != source.review:
        raise ValueError(f'导出期间源结果或验收发生变化：{source.path}')
