"""计算运行输入和发布结果的身份。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from umi.config.collection_schema import apply_scene_map_poster_policy, normalize_collection
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import read_yaml
from umi.scene.collection import kit as kit_inputs
from umi.scene.collection.configuration import scene_definition, validate_name
from umi.store import layout
from umi.store.cache import camera_identity
from umi.store.result_io import recording_identity, scene_map_identity

if TYPE_CHECKING:
    from umi.scene.run import ResolvedInputs

RESULT_IDENTITY_VERSION = 2

# Shared fusion inputs used by VIO's metric seed, inertial graph and full-frame recovery.
_VIO_FUSION_FIELDS = frozenset({
    'gravity_m_s2', 'gravity_tolerance_m_s2', 'maximum_acc_bias_m_s2',
    'acc_noise', 'gyro_noise', 'acc_bias_rw', 'gyro_bias_rw', 'integration_covariance',
    'velocity_prior_sigma_m_s', 'acc_bias_prior_sigma', 'gyro_bias_prior_sigma',
    'initial_static_gyro_rad_s', 'initial_bias_min_samples', 'initial_bias_max_samples',
    'split_rotation_deg', 'minimum_alignment_pairs', 'position_sigma_m', 'visual_rotation_sigma_deg',
    'lm_iterations',
})
_VIO_FRONT_MAP_FUSION_FIELDS = frozenset({'node_stride', 'maximum_relative_scale_std'})
# VIO's optional tail fit uses the shared rear pose noise model.
_VIO_TAIL_FIELDS = ('rear_position_sigma_m', 'visual_rotation_sigma_deg', 'huber')


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


def map_input_fields(episode, map_manifest, *, auto_map_attempted=False, requested_map_version=None) -> dict:
    mode = episode.scene.reconstruction_mode
    auto = mode == 'auto'
    selected_epoch = bool((episode.map_selection or {}).get('epoch'))
    result = dict(
        map_mode=map_manifest['mode'] if map_manifest else 'shared_front',
        scene_map_sha256=scene_map_identity(map_manifest) if map_manifest else None,
        scene_assets=map_manifest['assets'] if map_manifest else {},
        scene_map_reference=({key: map_manifest[key] for key in ('version', 'frame_id')} if map_manifest else None),
        session_manifest_sha256=(artifact_identity.canonical_hash(dict(kit=episode.session.kit,
            arms=episode.session.expected_arms)) if auto else artifact_identity.file_hash(episode.session.path)),
        map_epoch_plan_sha256=(map_epoch_identity(episode, selected_only=selected_epoch)
                               if mode == 'scene_map' or auto_map_attempted else None),
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


def episode_inputs(episode, cameras, map_manifest) -> dict:
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
    uses_independent = (episode.scene.reconstruction_mode == 'poster_independent' or
                        episode.scene.reconstruction_mode == 'auto' and 'poster_independent' in episode.scene.auto_order)
    if uses_independent:
        parameters['poster_independent_clock_policy'] = 'cross_checked_frame_shift_v1'
        if episode.scene.trajectory.values.zupt.enabled:
            parameters['poster_independent_zupt_policy'] = 'front_orb_confirmation_v1'
    return dict(configuration=current_configuration,
        **map_input_fields(episode, map_manifest),
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


def _scope_backend_identity(backend, record, source_parameters):
    trajectory = record['trajectory']
    if backend != 'poster_vio':
        # Scene retries use the selected backend's position-support or visual-gap gate.
        retry = {name: value for name, value in trajectory['retry'].items()
                 if name != 'maximum_position_support_gap_s'}
        return dict(record, trajectory=dict(trajectory, retry=retry))

    parameters = {name: value for name, value in record['parameters'].items()
                  if not name.startswith('tag_rear_rotation_')}
    front_map = parameters['poster_vio_front_map_enabled']
    if not front_map:
        parameters.pop('poster_vio_front_map_cauchy_scale', None)
        parameters.pop('poster_vio_front_map_minimum_cull_rounds', None)
    for name in _VIO_TAIL_FIELDS:
        # Older identities omitted these fields. Recover them from the frozen scene inputs.
        if name not in parameters and name in source_parameters:
            parameters[name] = source_parameters[name]
    fusion_fields = _VIO_FUSION_FIELDS | _VIO_FRONT_MAP_FUSION_FIELDS if front_map else _VIO_FUSION_FIELDS
    scoped = dict(input=trajectory['input'], video=trajectory['video'],
                  fusion={name: value for name, value in trajectory['fusion'].items() if name in fusion_fields})
    if front_map:
        scoped['orb'] = {name: value for name, value in trajectory['orb'].items() if name != 'gyro_aided_mapping'}
    return dict(record, parameters=parameters, trajectory=scoped)


def stored_reuse_identity(inputs: dict) -> dict | None:
    """Compare previous receipts in the current scope without rewriting published results."""
    stored = inputs.get('reuse_identity')
    if not isinstance(stored, dict) or stored.get('version') != RESULT_IDENTITY_VERSION:
        return None
    try:
        parameters = {backend: _scope_backend_identity(backend, record, inputs.get('parameters', {}))
                      for backend, record in stored['parameters'].items()}
    except (KeyError, TypeError, AttributeError):
        return None
    return dict(stored, parameters=parameters)


def _backend_inputs(backend, scene, identity):
    parameters = identity['parameters']
    if backend == 'poster_vio':
        prefixes = ('poster_vio_', 'tag_')
    elif backend == 'poster_independent':
        prefixes = ('poster_independent_', 'poster_maximum_', 'tag_')
    else:
        prefixes = ('alignment_', 'localization_', 'placement_', 'minimum_', 'maximum_', 'gravity_',
                    'front_position_', 'rear_position_', 'visual_rotation_', 'huber', 'tag_')
    effective = apply_scene_map_poster_policy(scene) if backend == 'scene_map' else scene
    uses_tags = backend != 'scene_map' or effective.environment_tag_detection
    selected = {k: v for k, v in parameters.items()
                if k.startswith(prefixes) and not k.startswith(('tag_distance_', 'tag_mapping_'))
                and (uses_tags or not k.startswith('tag_'))}
    if backend != 'scene_map':
        selected.pop('tag_fusion_method', None)
    elif uses_tags:
        selected['tag_fusion_enabled'] = effective.parameters.tag_fusion_enabled
    if backend == 'poster_vio' and selected['poster_vio_maximum_cull_rounds'] > 0:
        selected['poster_vio_culling_policy'] = 'single_natural_track_extension_v1'
    record = dict(parameters=selected, trajectory=identity['trajectory_parameters'],
                **(dict(base_tag=scene.poster_vio_base_tag_enabled) if backend == 'poster_vio' else {}),
                **(dict(poster_constraint=scene.scene_map_poster_constraint_enabled,
                        tag_detection=scene.scene_map_tag_detection_enabled,
                        edge_trim=scene.scene_map_edge_trim) if backend == 'scene_map' else {}))
    return _scope_backend_identity(backend, record, parameters)


def result_reuse_identity(episode, identity, backends, *, retry_preparation=False) -> dict:
    scene = episode.scene
    common = {name: identity[name] for name in (
        'recordings', 'synchronization_sha256', 'gripper_width', 'tag_distance', 'environment_tags',
        'scene_manifest_sha256', 'episode_manifest_sha256')}
    common.update(cameras=identity.get('base_cameras', identity['cameras']), instruction=episode.instruction,
                  kit=dict(id=episode.session.kit['id'], arms={arm: episode.session.kit['arms'][arm]
                                                               for arm in episode.expected_arms}),
                  trim=scene.recording_trim,
                  measurements={key: value for key, value in identity['parameters'].items()
                                if key.startswith(('tag_distance_', 'gripper_width_'))})
    if identity['tag_distance']:
        from umi.perception.gripper.gap_recovery import RECIPE

        common['tag_distance_policy'] = dict(coverage='report_separately_v1', recovery=RECIPE)
    backends = tuple(backends)
    maps = {key: identity.get(key) for key in (
        'map_mode', 'scene_map_sha256', 'scene_assets', 'map_epoch_plan_sha256', 'auto_map_request',
        'auto_map_cameras', 'auto_map_version_override')} if 'scene_map' in backends else {}
    return dict(version=RESULT_IDENTITY_VERSION, mode=scene.reconstruction_mode,
                common=common, backends=list(backends),
                parameters={name: _backend_inputs(name, scene, identity) for name in backends},
                auto_order=list(scene.auto_order) if scene.reconstruction_mode == 'auto' else None,
                maps=maps, retry_preparation=retry_preparation)


def current_auto_identity(resolved: ResolvedInputs, previous: dict) -> dict | None:
    from umi.scene.maps import binding

    stored = (previous.get('inputs') or {}).get('reuse_identity')
    if not isinstance(stored, dict) or stored.get('version') != RESULT_IDENTITY_VERSION:
        return None
    order = list(resolved.episode.scene.auto_order)
    backends = stored.get('backends')
    if not isinstance(backends, list) or not backends or order[:len(backends)] != backends:
        return None
    identity = resolved.identity
    if 'scene_map' in backends:
        episode, selection, cameras = binding.resolve_auto_map(
            resolved.episode, resolved.requested_map, resolved.cameras, batch_map=resolved.batch_map, allow_build=False)
        identity = {**identity, **map_input_fields(episode, selection.manifest, auto_map_attempted=True,
                                                   requested_map_version=resolved.requested_map),
                    'auto_map_cameras': {arm: camera_identity(camera) for arm, camera in cameras.items()}}
    return result_reuse_identity(resolved.episode, identity, backends)
