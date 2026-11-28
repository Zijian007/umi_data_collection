"""批次声明、任务有效配置与专家覆盖。"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field as dataclass_field, fields, replace
from pathlib import Path

from umi.config import collection_defaults, reconstruction
from umi.config import loader as config
from umi.config.collection_schema import (
    PROCESSING_FIELDS,
    apply_backend_options,
    apply_task_backend,
    normalize_collection,
)
from umi.core.arms import ARMS, resolve_arms
from umi.devices import registry
from umi.infra import provenance
from umi.infra.file_store import read_yaml
from umi.infra.paths import PROJECT_ROOT
from umi.perception.gripper import default_tag_roi, validate_roi
from umi.scene.maps import epochs as epoch_selection
from umi.store import layout
from umi.store.models import Scene, SceneParameters

ROOT = PROJECT_ROOT
BATCH_FIELDS = {'kit', 'arms', 'map_version', 'map_policy'}


def validate_fusion_method(value, label: str) -> None:
    """合法值来自环境海报估计器注册表；延迟导入以免配置加载拉起数值依赖。"""
    from umi.reconstruction.estimation.fixed_map.poster import FUSION_METHODS

    if not isinstance(value, str) or value not in FUSION_METHODS:
        raise ValueError(f'{label} 必须为 {" / ".join(FUSION_METHODS)} 之一')


def require_keys(document: dict, allowed: set, required: set, label: str) -> None:
    if not isinstance(document, dict) or document.keys() - allowed or required - document.keys():
        raise ValueError(f'{label} 字段缺失或含未知项；允许 {sorted(allowed)}，必需 {sorted(required)}')


def resolve_path(base: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('文件路径必须为非空字符串')
    return (base / Path(value).expanduser()).resolve()


def validate_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[\w-]+', value, re.UNICODE):
        raise ValueError('场景和任务名称只允许字母、数字、中文、下划线和连字符')
    return value


def validate_episode_name(value: str) -> str:
    value = validate_name(value)
    if value in layout.INTERNAL_DIRECTORIES:
        raise ValueError(f'任务名不能使用内部目录名称：{value}')
    return value


def validate_batch_name(value: str) -> str:
    value = validate_name(value)
    if value in layout.INTERNAL_DIRECTORIES | {'kits', 'sessions'}:
        raise ValueError(f'批次名不能使用内部目录名称：{value}')
    return value


def load_scene_parameters(path: Path | None = None, *, overrides: dict | None = None,
                          gap_sources: dict[str, str] | None = None) -> SceneParameters:
    """项目默认值来自 algorithms/reconstruction/。path 是公开覆盖文件；overrides 可以是同结构的
    公开树，也可以是内部扁平字段（运行流程内部合并结果和策略开关使用）。"""
    if gap_sources is not None:
        gap_sources.update(reconstruction.default_gap_sources())
    parameter_values = {'tag_distance_roi_px': default_tag_roi(), 'schema_version': reconstruction.SCENE_SNAPSHOT_VERSION,
                        **reconstruction.defaults().scene, **reconstruction.POSTER_POLICY_DEFAULTS}
    if path is not None:
        overlay = reconstruction.split(read_yaml(path), label=str(path), gap_sources=gap_sources).scene
        parameter_values = config.merge_overrides(parameter_values, overlay, 'parameters.')
    if overrides is not None:
        if set(overrides) & {path[0] for path in reconstruction.GROUPS}:
            overrides = reconstruction.split(overrides, label='parameters', gap_sources=gap_sources).scene
        else:
            unknown = set(overrides) - {field.name for field in fields(SceneParameters)}
            if unknown:
                raise ValueError(f'parameters 含未知字段：{sorted(unknown)}')
            overrides = dict(overrides)
        parameter_values = config.merge_overrides(parameter_values, overrides, 'parameters.')
    return _decode_scene_parameters(parameter_values)


def _decode_scene_parameters(parameter_values: dict) -> SceneParameters:
    if parameter_values.keys() != {field.name for field in fields(SceneParameters)}:
        raise ValueError('场景参数字段不完整')
    for field in fields(SceneParameters):
        value = parameter_values[field.name]
        if field.name == 'tag_distance_roi_px':
            parameter_values[field.name] = list(validate_roi(value))
            continue
        if field.name == 'tag_fusion_method':
            validate_fusion_method(value, 'tag_fusion_method')
            continue
        if field.type == 'bool':
            if type(value) is not bool:
                raise ValueError(f'{field.name} 必须为布尔值')
            continue
        typ = int if field.type == 'int' else float
        allows_zero = field.name in {'gripper_width_max_gap_s', 'gripper_width_closed_threshold_mm',
                                    'tag_distance_max_missing_frames', 'tag_sigma_per_radius_px',
                                    'tag_maximum_cull_fraction', 'poster_vio_maximum_cull_rounds',
                                    'poster_vio_maximum_natural_cull_fraction', 'poster_vio_holdout_stride',
                                    'poster_vio_keyframe_maximum_dt_s', 'poster_vio_natural_holdout_stride',
                                    'poster_vio_board_tilt_minimum_gap_s', 'poster_independent_node_stride',
                                    'poster_independent_keyframe_maximum_dt_s',
                                    'poster_vio_front_map_minimum_cull_rounds'}
        if (type(value) not in ((int,) if typ is int else (int, float)) or not math.isfinite(value)
                or value < 0 or value == 0 and not allows_zero):
            bound = '非负' if allows_zero else '正的'
            raise ValueError(f'{field.name} 必须为{bound} {typ.__name__}')
        parameter_values[field.name] = typ(value)
    parameters = SceneParameters(**parameter_values)
    if parameters.poster_vio_base_tag_minimum_self_frames < 3:
        raise ValueError('backends.poster_vio.base_tag.minimum_self_frames 至少为 3')
    if parameters.poster_vio_base_tag_maximum_rotation_correction_deg > 180:
        raise ValueError('backends.poster_vio.base_tag.maximum_rotation_correction_deg 不能超过 180')
    if parameters.poster_vio_minimum_tag_frames < 3:
        raise ValueError('backends.poster_vio.minimum_training_observation_times 至少为 3')
    if parameters.poster_vio_maximum_gravity_difference_deg > 180:
        raise ValueError('backends.poster_vio.maximum_gravity_difference_deg 不能超过 180')
    if not 2 <= parameters.poster_vio_minimum_track_length <= parameters.poster_vio_maximum_track_length:
        raise ValueError('backends.poster_vio.frontend 轨迹长度必须满足 2 <= minimum <= maximum')
    if parameters.poster_vio_maximum_observations_per_track < parameters.poster_vio_minimum_track_length:
        raise ValueError('backends.poster_vio.frontend.maximum_observations_per_track 不能少于 minimum_track_length')
    if not 0 <= parameters.poster_vio_maximum_natural_cull_fraction < 1:
        raise ValueError('backends.poster_vio.culling.maximum_natural_fraction 必须在 [0, 1) 内')
    if parameters.poster_vio_holdout_stride == 1:
        raise ValueError('backends.poster_vio.holdout.stride 必须为 0 或至少为 2')
    if parameters.poster_vio_natural_holdout_stride == 1:
        raise ValueError('backends.poster_vio.holdout.natural_stride 必须为 0 或至少为 2')
    if not 0 < parameters.poster_vio_revisit_match_ratio < 1:
        raise ValueError('backends.poster_vio.frontend.revisit.match_ratio 必须在 (0, 1) 内')
    if parameters.poster_vio_initialization_maximum_hypotheses < 3:
        raise ValueError('backends.poster_vio.initialization.maximum_hypotheses 至少为 3')
    if (parameters.schema_version != reconstruction.SCENE_SNAPSHOT_VERSION
            or parameters.minimum_alignment_nodes < 5 or parameters.minimum_pose_pairs < 3):
        raise ValueError('场景参数版本无效，或最少对齐节点/位姿对不足')
    if not 0 < parameters.minimum_localization_fraction <= 1 or parameters.maximum_alignment_condition <= 1:
        raise ValueError('定位比例必须在 (0, 1] 内，最大条件数必须大于 1')
    if not 0 < parameters.tag_distance_min_valid_fraction <= 1:
        raise ValueError('标签中心距最低有效比例必须在 (0, 1] 内')
    if parameters.localization_minimum_overlap_frames < 3 or parameters.localization_max_rotation_difference_deg > 180:
        raise ValueError('双向定位至少需要 3 个重叠帧，旋转差上限不能超过 180 度')
    config.validate_environment_tags(parameters)
    if parameters.poster_maximum_gravity_difference_deg > 180:
        raise ValueError('backends.poster_independent.maximum_gravity_difference_deg 不能超过 180 度')
    if parameters.tag_mapping_minimum_frames < 3:
        raise ValueError('海报尺度估计至少需要 3 个有效时刻')
    if parameters.tag_mapping_scale_enabled and not parameters.tag_mapping_enabled:
        raise ValueError('海报尺度修正需要先开启 tag_mapping_enabled')
    return parameters


def scene_definition(scene_document: dict, name: str, *, default_mode: str = 'shared_front') -> dict:
    """在内存中补齐场景默认值，目录是场景身份的来源。"""
    version = scene_document.get('schema_version', 3)
    if type(version) is not int or version != 3:
        raise ValueError('场景配置需要 schema_version: 3')
    if any(key in scene_document for key in ('trajectory', 'parameters')):
        raise ValueError('collection.yaml 的 scene 不承载算法参数；请用 --expert-config，格式见 config/README.md')
    require_keys(scene_document, {'schema_version', 'mode', 'layout', 'reference_arm', 'map_reference_arm', 'description', 'tags'}, set(), 'scene')
    scene_document = dict(scene_document)
    layout = scene_document.pop('layout', None)
    if layout is not None:
        if 'mode' in scene_document:
            raise ValueError('scene 不能同时填写 layout 和 mode')
        layout_modes = {'auto': 'auto', 'shared': 'shared_front', 'per_arm': 'dual'}
        if layout not in layout_modes:
            raise ValueError('scene.layout 必须是 auto、shared 或 per_arm')
        scene_document['mode'] = layout_modes[layout]
    map_reference_arm = scene_document.pop('map_reference_arm', None)
    if map_reference_arm is not None:
        if 'reference_arm' in scene_document:
            raise ValueError('scene 不能同时填写 map_reference_arm 和 reference_arm')
        scene_document['reference_arm'] = map_reference_arm
    normalized = dict(schema_version=3, name=validate_name(name), mode=default_mode, reference_arm='left',
                      description='', trajectory={}, parameters={})
    normalized.update(scene_document)
    if not isinstance(normalized['description'], str):
        raise ValueError('场景 description 必须为字符串')
    return normalized


def _read_tags(tags, base: Path, parameters: SceneParameters) -> tuple[SceneParameters, Path, bool]:
    """解析 scene.tags：布局、检测总开关、建图/尺度/融合开关和融合方法。"""
    from umi.perception.poster.layout import load, resolve_layout_path
    if not isinstance(tags, dict):
        raise ValueError('scene.tags 必须是配置表')
    require_keys(tags, {'enabled', 'layout', 'mapping', 'fusion'}, {'enabled', 'layout'}, 'scene.tags')
    mapping = tags.get('mapping', {})
    fusion = tags.get('fusion', {})
    if not isinstance(mapping, dict) or not isinstance(fusion, dict):
        raise ValueError('scene.tags.mapping 和 scene.tags.fusion 必须是配置表')
    require_keys(mapping, {'enabled', 'scale'}, set(), 'scene.tags.mapping')
    require_keys(fusion, {'enabled', 'method'}, set(), 'scene.tags.fusion')
    scale = mapping.get('scale', {})
    if not isinstance(scale, dict):
        raise ValueError('scene.tags.mapping.scale 必须是配置表')
    require_keys(scale, {'enabled'}, set(), 'scene.tags.mapping.scale')
    method = fusion.get('method', parameters.tag_fusion_method)
    validate_fusion_method(method, 'scene.tags.fusion.method')
    values = {
        'enabled': tags['enabled'],
        'mapping': mapping.get('enabled', False),
        'scale': scale.get('enabled', False),
        'fusion': fusion.get('enabled', False),
    }
    if any(type(value) is not bool for value in values.values()):
        raise ValueError('scene.tags.enabled、mapping.enabled、mapping.scale.enabled、fusion.enabled 必须为布尔值')
    if (values['mapping'] or values['scale'] or values['fusion']) and not values['enabled']:
        raise ValueError('启用 scene.tags.mapping 或 scene.tags.fusion 前必须开启 scene.tags.enabled')
    if values['scale'] and not values['mapping']:
        raise ValueError('启用 scene.tags.mapping.scale.enabled 前必须开启 scene.tags.mapping.enabled')
    parameters = replace(parameters,
        tag_mapping_enabled=values['mapping'],
        tag_mapping_scale_enabled=values['scale'],
        tag_fusion_enabled=values['fusion'],
        tag_fusion_method=method)
    tag_layout = resolve_layout_path(resolve_path(base, tags['layout']))
    load(tag_layout)  # 在创建运行目录或清理旧结果之前验证几何资产。
    return parameters, tag_layout, values['enabled']


def read_scene(path: Path, *, overrides: SceneOverrides | None = None, collection: dict | None = None) -> Scene:
    path = path.expanduser().resolve()
    if path.is_dir():
        path /= 'collection.yaml'
    layout.reject_legacy_user_file(path)
    if path.name != 'collection.yaml':
        raise ValueError('场景配置必须是批次根目录的 collection.yaml')
    if not path.is_file():
        raise ValueError(f'缺少批次配置 collection.yaml：{path}')
    base = path.parent.parent
    if collection is None:
        collection = normalize_collection(read_yaml(path))
    # scene_definition 拒绝 scene 内的 trajectory / parameters，算法参数只来自默认值和 overrides。
    scene_document = scene_definition(collection['scene'], base.name, default_mode='auto')
    if scene_document['mode'] not in ('shared_front', 'dual', 'auto'):
        raise ValueError('场景 mode 必须为 shared_front、dual 或 auto')
    if scene_document['reference_arm'] not in ARMS:
        raise ValueError('map_reference_arm 必须为 left 或 right')
    # 用途开关来自 collection，expert-config 覆盖算法参数。
    gap_sources: dict[str, str] = {}
    parameters = load_scene_parameters(gap_sources=gap_sources)
    sources = dict.fromkeys((*PROCESSING_FIELDS.values(), *reconstruction.FRONT_MAP_FIELDS), 'project')
    switches = collection['reconstruction'].get('processing_options', {})
    if switches:
        parameters = replace(parameters, **switches)
        sources.update(dict.fromkeys(switches, 'collection'))
    tag_layout = None
    environment_tags_enabled = False
    if 'tags' in scene_document:
        parameters, tag_layout, environment_tags_enabled = _read_tags(scene_document['tags'], base, parameters)
    trajectory_config, parameters = apply_overrides(None, parameters, overrides,
                                                    gap_sources=gap_sources, processing_sources=sources)
    if parameters.tag_fusion_enabled or parameters.tag_mapping_enabled:
        if tag_layout is None:
            raise ValueError('启用环境标签需要 collection.yaml 的 scene.tags.layout')
        if not environment_tags_enabled:
            raise ValueError('启用环境标签建图或融合前必须开启 scene.tags.enabled')
    scene = Scene(path, scene_document['name'], scene_document['mode'], scene_document['reference_arm'],
                  recordings={}, rear_calibrations={}, trajectory=trajectory_config, parameters=parameters,
                  tag_layout=tag_layout, overrides=overrides, environment_tags_enabled=environment_tags_enabled,
                  position_support_gap_sources=gap_sources, processing_option_sources=sources)
    return apply_backend_options(scene, collection['reconstruction'], source=f'{path}:backend')


@dataclass(frozen=True)
class SceneOverrides:
    """本次场景使用的专家覆盖。"""
    path: Path
    sha256: str
    values_json: str
    _overlay: reconstruction.Split = dataclass_field(repr=False, compare=False)
    _gap_sources: dict[str, str] = dataclass_field(repr=False, compare=False)

    @classmethod
    def load(cls, path: Path) -> SceneOverrides:
        from umi.infra.file_store import read_yaml
        path = path.expanduser().resolve()
        declaration = read_yaml(path)
        return cls.from_snapshot(dict(path=str(path), sha256=provenance.file_hash(path), values=declaration))

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> SceneOverrides:
        """验证并还原结果中冻结的算法覆盖。"""
        require_keys(snapshot, {'path', 'sha256', 'values'}, {'path', 'sha256', 'values'}, '场景覆盖记录')
        declaration = snapshot['values']
        if not isinstance(declaration, dict):
            raise ValueError('场景 --expert-config 必须是配置表')
        path = Path(snapshot['path'])
        gap_sources: dict[str, str] = {}
        overlay = reconstruction.split(declaration, label=str(path), scene_overrides=True, gap_sources=gap_sources)
        return cls(path, snapshot['sha256'], json.dumps(declaration, sort_keys=True), overlay, gap_sources)

    @property
    def values(self) -> dict:
        return json.loads(self.values_json)

    def overlays(self, *, gap_sources: dict[str, str] | None = None) -> reconstruction.Split:
        """翻译为内部覆盖：轨迹为嵌套内部结构，场景为扁平内部字段。"""
        if gap_sources is not None:
            gap_sources.update(self._gap_sources)
        return deepcopy(self._overlay)

    def snapshot(self) -> dict:
        return dict(path=str(self.path), sha256=self.sha256, values=self.values)


def apply_overrides(trajectory: config.Configuration | None, parameters: SceneParameters,
                    overrides: SceneOverrides | None, *,
                    gap_sources: dict[str, str] | None = None,
                    processing_sources: dict[str, str] | None = None) -> tuple[config.Configuration, SceneParameters]:
    if overrides is None:
        return trajectory or config.load('trajectory', standalone=False), parameters
    overlay = overrides.overlays(gap_sources=gap_sources)
    if trajectory is None:
        trajectory = config.load('trajectory', inline=overlay.trajectory, standalone=False)
        trajectory = replace(trajectory, sources=(*trajectory.sources, str(overrides.path)))
    else:
        trajectory = config.overlay_trajectory(trajectory, overlay.trajectory, overrides.path)
    if processing_sources is not None:
        processing_sources.update(dict.fromkeys(processing_sources.keys() & overlay.scene.keys(), 'expert'))
    parameter_values = config.merge_overrides(asdict(parameters), overlay.scene, 'parameters.')
    return trajectory, _decode_scene_parameters(parameter_values)


def default_audio_sync_enabled(batch_setting: bool | None) -> bool:
    if batch_setting is not None:
        return batch_setting
    return collection_defaults.audio_sync_enabled()


def read_synchronization(synchronization_record: dict | str | None, arms: tuple[str, ...], *,
                         audio_sync_enabled: bool | None = None,
                         audio_sync_algorithm: str | None = None) -> dict:
    """任务视频时钟：t_episode = t_video + offset；首只手为零偏移参考。"""
    if synchronization_record is None:
        synchronization_record = dict(offsets_s={arm: 0. for arm in arms}, confirmed=False, note='')
        if len(arms) == 2 and default_audio_sync_enabled(audio_sync_enabled):
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
    # carries placeholder offsets; the session-clock
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
        raise ValueError('audio_auto 不接受手工 offsets_s；请由音频同步阶段生成')
    confirmed, note = synchronization_record['confirmed'], synchronization_record.get('note', '')
    if type(confirmed) is not bool or not isinstance(note, str):
        raise ValueError('同步 confirmed 必须为布尔值，note 必须为文本')
    if len(arms) == 1:
        return dict(method='single_camera', offsets_s={arms[0]: 0.}, confirmed=True, note=note)
    if mode == 'audio_auto':
        algorithm = collection_defaults.audio_sync_algorithm() if audio_sync_algorithm is None else audio_sync_algorithm
        return dict(method='audio_auto', mode='audio_auto', offsets_s={arm: float(offsets[arm]) for arm in arms},
                    confirmed=False, note=note, algorithm=algorithm)
    if confirmed and not note.strip():
        raise ValueError('确认双手同步时需要填写 note，说明核验依据')
    method = 'manual' if mode == 'manual' or confirmed or any(offsets.values()) else 'assumed_simultaneous'
    return dict(method=method, offsets_s={arm: float(offsets[arm]) for arm in arms}, confirmed=confirmed, note=note)


@dataclass(frozen=True)
class ResolvedTaskConfiguration:
    scene: Scene
    expected_arms: tuple[str, ...]
    instruction: str
    synchronization: dict
    map_selection: dict | None
    map_request: dict

    @property
    def map_version(self) -> str | None:
        return self.map_selection['map_version'] if self.map_selection else None

    def require_ready_map(self, *, allow_pending: bool = False) -> None:
        allowed_statuses = ('ready', 'pending') if allow_pending else ('ready',)
        if (self.scene.reconstruction_mode != 'auto' and self.map_selection
                and self.map_selection['status'] not in allowed_statuses):
            raise epoch_selection.MapEpochError(self.map_selection['reason'])


def resolve_task_configuration(root: Path, scene: Scene, batch_arms: tuple[str, ...],
                               batch_map_version: str | None = None, *,
                               map_epoch_override: str | None = None,
                               collection: dict | None = None,
                               epoch_plan: epoch_selection.EpochPlan | None = None) -> ResolvedTaskConfiguration:
    """Read declarations and the epoch plan, never recordings or map assets."""
    if (root / 'task.yaml').is_file():
        layout.reject_legacy_user_file(root / 'task.yaml')
    validate_episode_name(root.name)
    if collection is None:
        collection = normalize_collection(read_yaml(scene.path))
    declaration = collection['tasks'].get(root.name, {})
    scene = apply_backend_options(scene, declaration.get('reconstruction'),
                                  source=f'{scene.path}:tasks.{root.name}.backend')
    mode = scene.reconstruction_mode
    scene = apply_task_backend(scene, collection.get('disabled_poster_switches', ()))
    arms = resolve_arms(declaration, batch_arms)
    synchronization = read_synchronization(declaration.get('synchronization'), arms,
        audio_sync_enabled=collection.get('audio_sync_enabled'),
        audio_sync_algorithm=collection['audio_sync_algorithm'])
    map_request = {key: declaration[key] for key in ('map_version', 'map_epoch') if key in declaration}
    if map_epoch_override is not None:
        if not isinstance(map_epoch_override, str) or not map_epoch_override.strip():
            raise ValueError('命令行 map_epoch 必须为非空字符串')
        if 'map_epoch' in map_request and map_request['map_epoch'] != map_epoch_override:
            raise ValueError('命令行 map_epoch 与 collection.yaml.tasks 的 map_epoch 不一致')
        if 'map_version' in map_request:
            raise ValueError('命令行 map_epoch 不能与任务 map.version 同时指定')
        map_request['map_epoch'] = map_epoch_override
    selection = None
    if mode == 'scene_map' or (mode == 'auto' and 'scene_map' in scene.auto_order):
        plan_path = layout.metadata(root.parent, 'map_epochs.yaml')
        try:
            if epoch_plan is None and plan_path.is_file():
                epoch_plan = epoch_selection.read_plan(plan_path)
            selection = epoch_selection.resolve_task_map(epoch_plan, root.name,
                task_map_version=map_request.get('map_version'), task_map_epoch=map_request.get('map_epoch'),
                batch_map_version=collection.get('map_version') or batch_map_version)
            if selection and selection['status'] == 'pending':
                selection = dict(selection, reason=f'任务 {root.name} 选择的地图 epoch 尚未构建：{selection["epoch"]}')
        except epoch_selection.MapEpochError as error:
            selection = dict(status='unavailable', map_version=None, reason=str(error))
        if selection and selection['map_version'] is not None:
            registry.validate_id(selection['map_version'])
    return ResolvedTaskConfiguration(scene, arms, declaration.get('instruction', collection['instruction']),
                                     synchronization, selection, map_request)


def read_collection(batch: Path) -> tuple[Path, dict, dict]:
    """Read the raw declaration and its validated values together."""
    collection = batch / 'collection.yaml'
    retired = batch / 'batch.yaml'
    if retired.is_file():
        layout.reject_legacy_user_file(retired)
    if collection.is_file():
        document = read_yaml(collection)
        return collection, document, normalize_collection(document)
    raise ValueError(f'批次缺少 collection.yaml：{batch}')


def batch_declaration(collection: dict) -> dict:
    return {key: value for key, value in collection.items() if key in BATCH_FIELDS}
