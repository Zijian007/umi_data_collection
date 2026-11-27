"""计算运行输入和发布结果的身份。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from umi.config.collection_schema import normalize_collection
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import read_yaml
from umi.scene.collection import kit as kit_inputs
from umi.scene.collection.configuration import scene_definition, validate_name
from umi.store import layout
from umi.store.cache import camera_identity
from umi.store.result_io import recording_identity, scene_map_identity

if TYPE_CHECKING:
    from umi.scene.run import ResolvedInputs

LEGACY_AUTO_ORDER = ('scene_map', 'poster_vio', 'poster_independent')


def configuration_identity(batch_root: Path, *, collection: dict | None = None) -> str | None:
    legacy = batch_root / 'batch.yaml'
    if legacy.is_file():
        layout.reject_legacy_user_file(legacy)
    if collection is None:
        path = batch_root / 'collection.yaml'
        if not path.is_file():
            return None
        collection = normalize_collection(read_yaml(path))
    kit = read_yaml(kit_inputs.KITS / validate_name(collection['kit']) / 'kit.yaml')
    return artifact_identity.canonical_hash(dict(batch=collection, kit=kit))


def map_epoch_identity(episode, *, selected_only: bool) -> str | None:
    path = layout.metadata(episode.session.root, 'map_epochs.yaml')
    if not path.is_file():
        return None
    if not selected_only:
        return artifact_identity.file_hash(path)
    decision = episode.map_selection or {}
    if 'epoch' not in decision:
        return artifact_identity.file_hash(path)
    plan = read_yaml(path)
    chosen = next((item for item in plan['maps'] if item['epoch'] == decision['epoch']), None)
    task = next((item for item in plan['tasks'] if item['task'] == episode.root.name), None)
    # Other tasks may publish their epochs concurrently; hash only the relevant
    # recording and the decision that determines which epoch this task may use.
    if decision['selection'] == 'latest_prior_map' and task is not None:
        eligible = [item for item in plan['maps'] if item['sequence'] <= task['sequence']]
        chosen = max(eligible, key=lambda item: item['sequence'], default=None)
    return artifact_identity.canonical_hash(dict(map=chosen, task=task))


def map_input_fields(episode, map_manifest, *, auto_map_attempted=False, requested_map_version=None,
                     legacy=False) -> dict:
    mode = episode.scene.reconstruction_mode
    auto = mode == 'auto' and not legacy
    selected_epoch = bool((episode.map_selection or {}).get('epoch')) and not legacy
    result = dict(
        map_mode=map_manifest['mode'] if map_manifest else 'shared_front',
        scene_map_sha256=scene_map_identity(map_manifest) if map_manifest else None,
        scene_assets=map_manifest['assets'] if map_manifest else {},
        scene_map_reference=({key: map_manifest[key] for key in ('version', 'frame_id')} if map_manifest else None),
        session_manifest_sha256=(artifact_identity.canonical_hash(dict(kit=episode.session.kit,
            arms=episode.session.expected_arms)) if auto else artifact_identity.file_hash(episode.session.path)),
        map_epoch_plan_sha256=(map_epoch_identity(episode, selected_only=selected_epoch)
                               if mode == 'scene_map' or auto_map_attempted or (mode == 'auto' and legacy) else None),
    )
    if selected_epoch and mode == 'scene_map':
        result['selected_epoch_identity'] = True
    if auto:
        result['auto_order'] = list(episode.scene.auto_order)
        if auto_map_attempted:
            result['auto_map_attempted'] = True
            if requested_map_version not in (None, episode.map_version, episode.session.map_version):
                result['auto_map_version_override'] = requested_map_version
            decision = episode.map_selection or {}
            result['auto_map_request'] = {
                'epoch': decision.get('epoch'),
                'version': requested_map_version or episode.map_version or episode.session.map_version,
                'mapping': episode.session.mapping,
            }
    return result


def episode_inputs(episode, cameras, map_manifest, *, auto_map_attempted=False, requested_map_version=None,
                   legacy=False) -> dict:
    from umi.perception.poster.layout import load as load_tag_board

    width_assets = {arm: dict(sha256=camera.asset_hashes['width'])
              for arm, camera in cameras.items() if 'width' in camera.files}
    tag_distance_assets = {arm: dict(sha256=camera.asset_hashes['tag_distance'])
                 for arm, camera in cameras.items() if 'tag_distance' in camera.files}
    document = read_yaml(episode.scene.path)
    collection = normalize_collection(document)
    current_configuration = configuration_identity(episode.session.root, collection=collection)
    parameters = {key: value for key, value in asdict(episode.scene.parameters).items()
                  if not key.startswith(('recording_', 'pairing_'))}
    # Keep historical input fingerprints when opt-in algorithms are disabled.
    for optional in ('front_map_pixel_fusion', 'poster_vio_per_tag_huber', 'poster_vio_robust_inertial_seed',
                     'poster_vio_board_tilt_minimum_gap_s', 'tag_distance_verified_gap_recovery'):
        if not parameters[optional]:
            del parameters[optional]
    uses_independent = (episode.scene.reconstruction_mode == 'poster_independent' or
                        episode.scene.reconstruction_mode == 'auto' and 'poster_independent' in episode.scene.auto_order)
    if uses_independent:
        parameters['poster_independent_clock_policy'] = 'cross_checked_frame_shift_v1'
    for key, inherited in (('poster_independent_node_stride', 0), ('poster_independent_imu_white_noise_multiplier', 1.0)):
        if not uses_independent or parameters[key] == inherited:
            del parameters[key]
    return dict(configuration=current_configuration,
        **map_input_fields(episode, map_manifest, auto_map_attempted=auto_map_attempted,
                           requested_map_version=requested_map_version, legacy=legacy),
        reconstruction_mode=episode.scene.reconstruction_mode,
        poster_vio_base_tag_enabled=episode.scene.poster_vio_base_tag_enabled,
        episode_manifest_sha256=artifact_identity.file_hash(episode.path),
        task_configuration_sha256=artifact_identity.canonical_hash(collection['tasks'].get(episode.root.name, {})),
        # Scene.path 总是 collection.yaml（read_scene 已强制）。
        scene_manifest_sha256=artifact_identity.canonical_hash(scene_definition(
            {'description': collection['scene']['description']}, episode.scene.name)),
        recordings={arm: recording_identity(video) for arm, video in episode.recordings.items()},
        cameras={arm: camera_identity(camera) for arm, camera in cameras.items()},
        synchronization_sha256=artifact_identity.canonical_hash(episode.synchronization),
        gripper_width=width_assets, tag_distance=tag_distance_assets,
        environment_tags=load_tag_board(episode.scene.tag_layout).record() if episode.scene.tag_layout else None,
        parameters=parameters, trajectory_parameters=episode.scene.trajectory.snapshot()['values'])


def current_auto_identity(resolved: ResolvedInputs, previous: dict) -> dict:
    from umi.scene.maps import binding

    stored = previous.get('inputs')
    if not isinstance(stored, dict):
        return resolved.identity
    legacy = 'auto_order' not in stored
    if stored.get('auto_order') not in (None, resolved.identity['auto_order']):
        return resolved.identity
    if legacy and list(LEGACY_AUTO_ORDER) != resolved.identity['auto_order']:
        return resolved.identity
    if not legacy and not stored.get('auto_map_attempted'):
        return resolved.identity
    shared_inputs = resolved.identity.keys() - map_input_fields(resolved.episode, None).keys()
    if stored.get('map_mode') == 'dual':
        shared_inputs.discard('cameras')
    if any(stored.get(key) != resolved.identity[key] for key in shared_inputs):
        return resolved.identity
    episode, selection, cameras = binding.resolve_auto_map(
        resolved.episode, resolved.requested_map, resolved.cameras, batch_map=resolved.batch_map, allow_build=False)
    return {**{key: value for key, value in resolved.identity.items() if key != 'auto_order'},
            'cameras': {arm: camera_identity(camera) for arm, camera in cameras.items()},
            **map_input_fields(episode, selection.manifest, auto_map_attempted=not legacy,
                               requested_map_version=resolved.requested_map, legacy=legacy)}
