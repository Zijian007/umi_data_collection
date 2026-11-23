"""Resolve task intent once for execution and read-only configuration reports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import math

from software.camera import registry
from software.configuration import user_settings as user_config
from software.infrastructure.file_store import read_yaml
from software.scene import layout
from software.scene.arms import resolve_arms
from software.scene.management.collection_schema import normalize_collection
from software.scene.management.inputs.scene import require_keys, validate_episode_name
from software.scene.pipeline.maps import epoch_selection
from software.scene.management.collection_schema import apply_backend_options, apply_task_backend
from software.sync.specialist_audio import DEFAULT_AUDIO_SYNC_ALGORITHM

if TYPE_CHECKING:
    from software.scene.models import Scene


def default_audio_sync_enabled(batch_setting: bool | None) -> bool:
    if batch_setting is not None:
        return batch_setting
    return user_config.current().values['scene']['sync_policy'] == 'audio_auto'


def read_synchronization(synchronization_record: dict | str | None, arms: tuple[str, ...], *,
                         audio_sync_enabled: bool | None = None,
                         audio_sync_algorithm: str = DEFAULT_AUDIO_SYNC_ALGORITHM) -> dict:
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
        return dict(method='audio_auto', mode='audio_auto', offsets_s={arm: float(offsets[arm]) for arm in arms},
                    confirmed=False, note=note, algorithm=audio_sync_algorithm)
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
                               collection: dict | None = None) -> ResolvedTaskConfiguration:
    """Read declarations and the epoch plan, never recordings or map assets."""
    if (root / 'task.yaml').is_file():
        layout.reject_legacy_user_file(root / 'task.yaml')
    validate_episode_name(root.name)
    if collection is None:
        collection = normalize_collection(read_yaml(scene.path))
    declaration = collection['tasks'].get(root.name, {})
    scene = apply_backend_options(scene, declaration.get('reconstruction'))
    mode = scene.reconstruction_mode
    scene = apply_task_backend(scene, collection.get('disabled_poster_switches', ()))
    arms = resolve_arms(declaration, batch_arms)
    synchronization = read_synchronization(declaration.get('synchronization'), arms,
        audio_sync_enabled=collection.get('audio_sync_enabled'),
        audio_sync_algorithm=collection.get('audio_sync_algorithm', DEFAULT_AUDIO_SYNC_ALGORITHM))
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
    if mode not in ('poster_independent', 'poster_vio'):
        plan_path = layout.metadata(root.parent, 'map_epochs.yaml')
        plan = epoch_selection.read_plan(plan_path) if plan_path.is_file() else None
        try:
            selection = epoch_selection.resolve_task_map(plan, root.name,
                task_map_version=map_request.get('map_version'), task_map_epoch=map_request.get('map_epoch'),
                batch_map_version=collection.get('map_version') or batch_map_version)
            if selection and selection['status'] == 'pending':
                selection = dict(selection, reason=f'任务 {root.name} 选择的地图 epoch 尚未构建：{selection["epoch"]}')
        except epoch_selection.MapEpochError as error:
            selection = dict(status='unavailable', map_version=None, reason=str(error))
        if selection and selection['map_version'] is not None:
            registry.validate_id(selection['map_version'])
    return ResolvedTaskConfiguration(scene, arms, declaration.get('instruction', ''),
                                     synchronization, selection, map_request)
