"""任务清单、同步策略和任务输入身份解析。"""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, replace
import math
from typing import TYPE_CHECKING

from software.infrastructure import provenance as artifact_identity
from software.camera import registry
from software.configuration import user_settings as user_config
from software.scene import layout
from software.scene.management import camera_assets
from software.scene.management.inputs import kit as kit_inputs
from software.scene.management.collection_schema import normalize_collection
from software.scene.management.inputs.batch import read_arms, read_batch, read_recordings
from software.scene.asset_identity import recording_identity
from software.scene.management.inputs.scene import (
    require_keys,
    resolve_path,
    scene_definition,
    validate_episode_name,
    validate_name,
)
from software.scene.models import Episode
from software.scene.pipeline.maps import epoch_selection as map_epoch_selection
from software.scene.result_io import scene_map_identity
from software.infrastructure.file_store import read_yaml

if TYPE_CHECKING:
    from software.scene.overrides import SceneOverrides


def read_synchronization(synchronization_record: dict | str | None, arms: tuple[str, ...]) -> dict:
    """任务视频时钟：t_episode = t_video + offset；首只手为零偏移参考。"""
    if synchronization_record is None:
        policy = user_config.current().values['scene']['sync_policy']
        synchronization_record = dict(offsets_s={arm: 0. for arm in arms}, confirmed=False, note='')
        if policy == 'audio_auto' and len(arms) == 2:
            synchronization_record['mode'] = 'audio_auto'
    elif isinstance(synchronization_record, str):
        # The user-facing shorthand is intentionally small; normalize it to
        # the full internal record before validation and hashing.
        if synchronization_record not in ('audio_auto', 'manual', 'assumed_simultaneous'):
            raise ValueError('synchronization 只能填写 audio_auto、manual 或 assumed_simultaneous')
        synchronization_record = dict(
            mode=synchronization_record,
            offsets_s={arm: 0. for arm in arms},
            confirmed=False,
            note='',
        )
        if synchronization_record['mode'] == 'manual':
            raise ValueError('manual 同步需要填写 offsets_s 和 confirmed')
    elif not isinstance(synchronization_record, dict):
        raise ValueError('synchronization 必须是简写字符串或配置表')
    # ``audio_auto`` is a request for the offline sync stage.  The task file
    # carries placeholder offsets; the specialist-compatible session-clock
    # offsets are filled from the input-fingerprinted sync record at run time.
    mode = synchronization_record.get('mode')
    allowed = {'offsets_s', 'confirmed', 'note', 'mode'}
    required = {'confirmed'} if mode == 'audio_auto' else {'offsets_s', 'confirmed'}
    require_keys(synchronization_record, allowed, required, 'synchronization')
    if mode is not None and mode not in ('audio_auto', 'manual', 'assumed_simultaneous'):
        raise ValueError('synchronization.mode 必须是 audio_auto、manual 或 assumed_simultaneous')
    if mode == 'audio_auto' and len(arms) != 2:
        raise ValueError('audio_auto 需要左右两路录像')
    if mode == 'audio_auto' and 'offsets_s' not in synchronization_record:
        synchronization_record = dict(synchronization_record,
                                      offsets_s={arm: 0. for arm in arms})
    offsets = synchronization_record['offsets_s']
    if not isinstance(offsets, dict) or set(offsets) != set(arms):
        raise ValueError('synchronization.offsets_s 必须包含全部预期使用手')
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in offsets.values()):
        raise ValueError('同步偏移必须为有限秒数')
    if offsets[arms[0]] != 0:
        raise ValueError(f'参考手 {arms[0]} 的同步偏移必须为零')
    if mode == 'audio_auto' and any(float(value) != 0. for value in offsets.values()):
        raise ValueError('audio_auto 不接受手工 offsets_s；请由 specialist 音频同步阶段生成')
    confirmed, note = synchronization_record['confirmed'], synchronization_record.get('note', '')
    if type(confirmed) is not bool or not isinstance(note, str):
        raise ValueError('同步 confirmed 必须为布尔值，note 必须为文本')
    if len(arms) == 1:
        return dict(method='single_camera', offsets_s={arms[0]: 0.}, confirmed=True, note=note)
    if mode == 'audio_auto':
        return dict(method='audio_auto', mode='audio_auto', offsets_s={arm: float(offsets[arm]) for arm in arms},
                    confirmed=False, note=note)
    if confirmed and not note.strip():
        raise ValueError('确认双手同步时需要填写 note，说明核验依据')
    method = 'manual' if mode == 'manual' or confirmed or any(offsets.values()) else 'assumed_simultaneous'
    return dict(method=method, offsets_s={arm: float(offsets[arm]) for arm in arms}, confirmed=confirmed, note=note)


def read_episode(path: Path, *, verify: bool = True, overrides: SceneOverrides | None = None,
                 map_epoch_override: str | None = None) -> Episode:
    path = layout.manifest(path, 'episode.yaml')
    root = layout.owner(path)
    episode_document = read_yaml(path)
    require_keys(episode_document, {'schema_version', 'session', 'recordings'},
                 {'schema_version', 'recordings'}, 'episode')
    if type(episode_document['schema_version']) is not int or episode_document['schema_version'] != 5:
        raise ValueError('任务清单需要 schema_version: 5；旧任务请重新运行 umi-scene run 登记')
    task_configuration = layout.task_configuration(root)
    episode_document = dict(episode_document, **task_configuration)
    map_version = task_configuration.get('map_version')
    task_map_epoch = task_configuration.get('map_epoch')
    if task_map_epoch is not None and (not isinstance(task_map_epoch, str) or not task_map_epoch.strip()):
        raise ValueError('map_epoch 必须是非空字符串')
    if map_epoch_override is not None:
        if not isinstance(map_epoch_override, str) or not map_epoch_override.strip():
            raise ValueError('命令行 map_epoch 必须是非空字符串')
        if task_map_epoch is not None and task_map_epoch != map_epoch_override:
            raise ValueError('命令行 map_epoch 与 collection.yaml.tasks 的 map_epoch 不一致')
        task_map_epoch = map_epoch_override
    session_path = layout.manifest(root.parent, 'manifest.yaml')
    if 'session' in episode_document and layout.manifest(resolve_path(root, episode_document['session']), 'manifest.yaml') != session_path:
        raise ValueError('任务必须直接位于所属采集批次目录下')
    session = read_batch(session_path, overrides=overrides)
    from software.scene.pipeline.tasks.policy import read_mode, require_poster
    mode = read_mode(task_configuration.get('reconstruction'), default=session.scene.reconstruction_mode)
    session = replace(session, scene=replace(session.scene, reconstruction_mode=mode))
    if mode in ('poster_independent', 'poster_vio'):
        session = replace(session, scene=replace(session.scene, mode='shared_front'))
    if mode != 'scene_map':
        require_poster(session.scene)
    if root.parent != session.root:
        raise ValueError('任务必须直接位于采集批次的 <任务>/ 目录')
    validate_episode_name(root.name)
    epoch_plan_path = layout.metadata(session.root, 'map_epochs.yaml')
    plan = (map_epoch_selection.read_plan(epoch_plan_path)
            if mode not in ('poster_independent', 'poster_vio') and epoch_plan_path.is_file() else None)
    try:
        declared_map = normalize_collection(read_yaml(session.scene.path)).get('map_version')
        map_selection = None if mode in ('poster_independent', 'poster_vio') else map_epoch_selection.resolve_task_map(
            plan, root.name, task_map_version=map_version,
            task_map_epoch=task_map_epoch, batch_map_version=declared_map or session.map_version)
    except map_epoch_selection.MapEpochError as error:
        if mode != 'auto':
            raise
        map_selection = dict(status='unavailable', map_version=None, reason=str(error))
    if map_selection is not None:
        map_version = map_selection['map_version']
    if mode in ('poster_independent', 'poster_vio'):
        map_version = None
    if map_version is not None:
        registry.validate_id(map_version)
    expected = read_arms(episode_document.get('expected_arms', session.expected_arms), session.expected_arms)
    recording_paths, serials = read_recordings(root, episode_document['recordings'], verify=verify)
    if set(serials.keys()) - set(expected):
        raise ValueError('实际录像包含本次任务未声明使用的手')
    if any(serial != session.arm_serials[arm] for arm, serial in serials.items()):
        raise ValueError('录像清单的设备身份与套件的左右手配置不一致')
    if not isinstance(episode_document.get('instruction', ''), str):
        raise ValueError('instruction 必须为操作描述字符串')
    synchronization = read_synchronization(episode_document.get('synchronization'), expected)
    return Episode(path, session, recording_paths, serials, expected, episode_document['recordings'],
                   episode_document.get('instruction', ''), synchronization, map_version, map_selection)


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
