"""任务发现与运行状态。"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from umi.infra import provenance as artifact_identity
from umi.infra.file_store import write_json
from umi.scene.collection.batch import batch_task_directories, read_episode
from umi.store import layout
from umi.store.layout import INTERNAL_DIRECTORIES
from umi.store.result_io import task_state


def episode_paths(paths: list[Path], *, exclude: Iterable[Path] = ()) -> list[Path]:
    """只发现任务清单，跳过缓存、运行产物和归档；不启动处理。"""
    found = []
    excluded = {path.expanduser().resolve() for path in exclude}
    for path in paths:
        path = path.expanduser().resolve()
        if any(path.is_relative_to(root) for root in excluded):
            continue
        if path.is_file():
            layout.reject_legacy_user_file(path)
            if path.name not in ('collection.yaml', 'manifest.yaml', 'episode.yaml'):
                raise ValueError(f'请指定场景、套件、批次、任务目录或对应清单：{path}')
            if path.name == 'episode.yaml':
                found.append(path)
                continue
            path = layout.owner(path)
        if not path.is_dir():
            raise ValueError(f'任务路径不存在：{path}')
        for current, children, files in os.walk(path):
            children[:] = sorted(name for name in children if not name.startswith('.')
                                 and name not in INTERNAL_DIRECTORIES
                                 and Path(current) / name not in excluded)
            if layout.is_batch(Path(current)):
                found.extend(layout.manifest(task, 'episode.yaml')
                             for task in batch_task_directories(Path(current))
                             if task not in excluded and layout.manifest(task, 'episode.yaml').is_file())
            manifest = layout.manifest(Path(current), 'episode.yaml')
            if manifest.is_file():
                found.append(manifest)
                children[:] = []
    return list(dict.fromkeys(found))


def invalidate_task_state(path: Path) -> None:
    """原片清单变化后，只保留此前发布的运行引用和历史记录。"""
    read_episode(path, verify=False, allow_pending_map=True)
    root = layout.owner(path)
    state_path = layout.metadata(root, 'state.json')
    state = layout.read_state(root)
    state.pop('attempt', None)
    if state:
        write_json(state_path, state)
    else:
        state_path.unlink(missing_ok=True)


def record_attempt(episode, status: str, target: Path | None, *, reason='') -> None:
    state_path = layout.metadata(episode.root, 'state.json')
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    manifest_sha256 = artifact_identity.file_hash(episode.path)
    session_sha256 = artifact_identity.file_hash(episode.session.path)
    state['attempt'] = dict(status=status, run=layout.run_reference(episode.root, target) if target else None,
        manifest_sha256=manifest_sha256, session_sha256=session_sha256,
        task_configuration_sha256=layout.task_configuration_hash(episode.root))
    if reason and target is None:
        state['attempt']['reason'] = reason
    if status in ('ok', 'skipped', 'local_ok', 'local_skipped'):
        state['published_run'] = state['attempt']['run']
    write_json(state_path, state)


def task_status(path: Path) -> dict:
    episode = read_episode(path, verify=False)
    view = task_state(episode.root, episode.path, episode.session.path, episode.expected_arms, episode.recordings)
    return dict(scene=episode.scene.name, session=episode.session.name, name=episode.root.name,
                path=str(episode.path), kit=episode.session.kit['id'], arms=list(episode.expected_arms),
                **view, reason=view['attempt'].get('reason', ''),
                output=view['published']['path'] if view['current'] else None)
