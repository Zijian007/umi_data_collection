"""任务清单、同步策略和任务输入身份解析。"""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

from software.infrastructure import provenance as artifact_identity
from software.scene import layout
from software.scene.management import camera_assets
from software.scene.management.inputs import kit as kit_inputs
from software.scene.management.collection_schema import normalize_collection
from software.scene.management.inputs.batch import read_batch, read_recordings
from software.scene.management.task_configuration import resolve_task_configuration
from software.scene.asset_identity import recording_identity
from software.scene.management.inputs.scene import (
    require_keys,
    resolve_path,
    scene_definition,
    validate_name,
)
from software.scene.models import Episode
from software.scene.result_io import scene_map_identity
from software.infrastructure.file_store import read_yaml

if TYPE_CHECKING:
    from software.scene.overrides import SceneOverrides


def read_episode(path: Path, *, verify: bool = True, overrides: SceneOverrides | None = None,
                 map_epoch_override: str | None = None) -> Episode:
    path = layout.manifest(path, 'episode.yaml')
    root = layout.owner(path)
    episode_document = read_yaml(path)
    require_keys(episode_document, {'schema_version', 'session', 'recordings'},
                 {'schema_version', 'recordings'}, 'episode')
    if type(episode_document['schema_version']) is not int or episode_document['schema_version'] != 5:
        raise ValueError('任务清单需要 schema_version: 5；旧任务请重新运行 umi-scene run 登记')
    session_path = layout.manifest(root.parent, 'manifest.yaml')
    if 'session' in episode_document and layout.manifest(resolve_path(root, episode_document['session']), 'manifest.yaml') != session_path:
        raise ValueError('任务必须直接位于所属采集批次目录下')
    session = read_batch(session_path, overrides=overrides)
    if root.parent != session.root:
        raise ValueError('任务必须直接位于采集批次的 <任务>/ 目录')
    resolved = resolve_task_configuration(root, session.scene, session.expected_arms, session.map_version,
                                          map_epoch_override=map_epoch_override)
    resolved.require_ready_map()
    session = replace(session, scene=resolved.scene)
    expected = resolved.expected_arms
    recording_paths, serials = read_recordings(root, episode_document['recordings'], verify=verify)
    if set(serials.keys()) - set(expected):
        raise ValueError('实际录像包含本次任务未声明使用的手')
    if any(serial != session.arm_serials[arm] for arm, serial in serials.items()):
        raise ValueError('录像清单的设备身份与套件的左右手配置不一致')
    return Episode(path, session, recording_paths, serials, expected, episode_document['recordings'],
                   resolved.instruction, resolved.synchronization, resolved.map_version, resolved.map_selection)


def algorithm_identity() -> dict:
    identity = artifact_identity.code_identity('scene')
    orchestration = {
        'software/scene/cli.py', 'software/scene/__main__.py',
        'software/scene/management/batch/snapshot.py',
        'software/scene/management/ingest.py', 'software/scene/management/episodes.py',
        'software/scene/management/processing.py', 'software/scene/pipeline/recording_checks.py',
        'software/scene/pipeline/tasks/diagnostics.py',
    }
    return {name: digest for name, digest in identity.items() if name not in orchestration}


def configuration_identity(batch_root: Path) -> str | None:
    legacy = batch_root / 'batch.yaml'
    if legacy.is_file():
        layout.reject_legacy_user_file(legacy)
    collection = batch_root / 'collection.yaml'
    if not collection.is_file():
        return None
    batch = normalize_collection(read_yaml(collection))
    kit = read_yaml(kit_inputs.KITS / validate_name(batch.get('kit')) / 'kit.yaml')
    return artifact_identity.canonical_hash(dict(batch=batch, kit=kit))


def episode_inputs(episode, cameras, map_manifest) -> dict:
    from software.gripper_width import pipeline as width_pipeline
    from software.gripper_width.distance import pipeline as tag_distance
    from software.vision.poster.layout import load as load_tag_board

    width_assets = {arm: dict(sha256=camera.asset_hashes['width'])
              for arm, camera in cameras.items() if 'width' in camera.files}
    tag_distance_assets = {arm: dict(sha256=camera.asset_hashes['tag_distance'])
                 for arm, camera in cameras.items() if 'tag_distance' in camera.files}
    current_configuration = configuration_identity(episode.session.root)
    epoch_plan = layout.metadata(episode.session.root, 'map_epochs.yaml')
    source_hashes = algorithm_identity()
    trajectory_hashes = artifact_identity.code_identity('trajectory')
    if any(source_hashes[name] != trajectory_hashes[name] for name in source_hashes.keys() & trajectory_hashes.keys()):
        raise ValueError('记录任务身份时源代码发生变化，请重新运行')
    source_hashes.update(trajectory_hashes)
    return dict(configuration=current_configuration, map_mode=map_manifest['mode'] if map_manifest else 'shared_front',
        scene_map_sha256=scene_map_identity(map_manifest) if map_manifest else None,
        scene_assets=map_manifest['assets'] if map_manifest else {},
        reconstruction_mode=episode.scene.reconstruction_mode,
        poster_vio_base_tag_enabled=episode.scene.poster_vio_base_tag_enabled,
        scene_map_reference=({key: map_manifest[key] for key in ('version', 'frame_id')}
                             if map_manifest else None),
        session_manifest_sha256=artifact_identity.file_hash(episode.session.path),
        map_epoch_plan_sha256=(artifact_identity.file_hash(epoch_plan)
                              if episode.scene.reconstruction_mode not in ('poster_independent', 'poster_vio')
                              and epoch_plan.is_file() else None),
        episode_manifest_sha256=artifact_identity.file_hash(episode.path),
        task_configuration_sha256=layout.task_configuration_hash(episode.root),
        # Scene.path 总是 collection.yaml（read_scene 已强制）。
        scene_manifest_sha256=artifact_identity.canonical_hash(scene_definition(
            read_yaml(episode.scene.path).get('scene', {}), episode.scene.name)),
        recordings={arm: recording_identity(video) for arm, video in episode.recordings.items()},
        cameras={arm: camera_assets.camera_identity(camera) for arm, camera in cameras.items()},
        synchronization_sha256=artifact_identity.canonical_hash(episode.synchronization),
        gripper_width=width_assets, gripper_width_code=width_pipeline.identity() if width_assets else {},
        tag_distance=tag_distance_assets, tag_distance_code=tag_distance.identity() if tag_distance_assets else {},
        environment_tags=load_tag_board(episode.scene.tag_layout).record() if episode.scene.tag_layout else None,
        parameters={key: value for key, value in asdict(episode.scene.parameters).items()
                    if not key.startswith(('recording_', 'pairing_'))}, trajectory_parameters=episode.scene.trajectory.snapshot()['values'],
        code=source_hashes, engines=artifact_identity.engine_identity('trajectory'))
