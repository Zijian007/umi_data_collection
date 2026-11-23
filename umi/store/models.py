"""共享场景的参数、清单与阶段数据；此模块不读取清单或设备资产。"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar, TypedDict

import numpy as np

from umi.config import loader as config
from umi.infra import provenance as artifact_identity
from umi.store import layout
from umi.core.arms import ARMS
from umi.devices import selection
from umi.perception.models import LensVideo, RecordingData, VisualContext


if TYPE_CHECKING:
    from umi.devices.rear_fallback import FactoryRearCalibration
    from umi.infra.terminal import Console
    from umi.workflows.scene.parameters import SceneOverrides
    from umi.store.cache import CacheAudit
    from umi.perception.poster.detection import DetectionSnapshot


_AlgorithmParameters = TypeVar('_AlgorithmParameters')


@dataclass(frozen=True)
class SceneParameters:
    schema_version: int
    alignment_node_dt_s: float
    alignment_max_gap_s: float
    minimum_alignment_nodes: int
    maximum_alignment_condition: float
    maximum_scale_relative_std: float
    gravity_tolerance_m_s2: float
    maximum_alignment_rms_m: float
    maximum_acc_bias_m_s2: float
    minimum_pose_pairs: int
    placement_position_sigma_m: float
    placement_rotation_sigma_deg: float
    maximum_placement_rms_m: float
    maximum_placement_rotation_deg: float
    minimum_localization_fraction: float
    maximum_visual_gap_s: float
    poster_maximum_position_support_gap_s: float
    poster_maximum_gravity_difference_deg: float
    poster_vio_minimum_tag_frames: int
    poster_vio_maximum_observation_gap_s: float
    poster_vio_minimum_time_span_s: float
    poster_vio_maximum_position_support_gap_s: float
    poster_vio_maximum_gravity_difference_deg: float
    poster_vio_maximum_features: int
    poster_vio_maximum_tracks: int
    poster_vio_minimum_track_length: int
    poster_vio_maximum_track_length: int
    poster_vio_maximum_observations_per_track: int
    poster_vio_maximum_cull_rounds: int
    poster_vio_maximum_natural_cull_fraction: float
    poster_vio_natural_cull_threshold_px: float
    poster_vio_holdout_stride: int
    poster_vio_natural_holdout_stride: int
    poster_vio_maximum_holdout_rms_px: float
    poster_vio_initialization_maximum_hypotheses: int
    poster_vio_geometric_threshold_px: float
    poster_vio_revisit_enabled: bool
    poster_vio_revisit_maximum_bank: int
    poster_vio_revisit_maximum_gap_s: float
    poster_vio_revisit_match_ratio: float
    poster_vio_revisit_maximum_reprojection_px: float
    poster_vio_keyframe_maximum_dt_s: float
    poster_vio_keyframe_minimum_translation_m: float
    poster_vio_keyframe_minimum_rotation_deg: float
    poster_vio_precull_iterations: int
    poster_vio_base_tag_prior_rotation_sigma_rad: float
    poster_vio_base_tag_prior_translation_sigma_m: float
    poster_vio_base_tag_maximum_rotation_correction_deg: float
    poster_vio_base_tag_maximum_translation_correction_m: float
    poster_vio_base_tag_minimum_self_frames: int
    localization_bidirectional: bool
    localization_reverse_context_s: float
    localization_minimum_overlap_frames: int
    localization_max_position_difference_m: float
    localization_max_rotation_difference_deg: float
    front_position_sigma_m: float
    rear_position_sigma_m: float
    visual_rotation_sigma_deg: float
    huber: float
    tag_fusion_enabled: bool
    tag_fusion_method: str
    tag_minimum_side_px: float
    tag_minimum_frames: int
    tag_corner_sigma_px: float
    tag_sigma_per_radius_px: float
    tag_huber: float
    tag_maximum_reprojection_rms_px: float
    tag_maximum_position_std_m: float
    tag_maximum_rotation_std_deg: float
    tag_maximum_cull_fraction: float
    tag_minimum_retained_time_fraction: float
    tag_rear_rotation_refinement_enabled: bool
    tag_rear_rotation_prior_sigma_rad: float
    tag_rear_rotation_maximum_correction_deg: float
    tag_mapping_enabled: bool
    tag_mapping_scale_enabled: bool
    tag_mapping_minimum_frames: int
    tag_mapping_minimum_motion_m: float
    tag_mapping_minimum_board_extent_m: float
    tag_mapping_maximum_relative_std: float
    tag_mapping_imu_std_floor: float
    tag_mapping_tag_std_floor: float
    tag_mapping_compatibility_sigma: float
    tag_mapping_maximum_scale_difference: float
    pairing_start_tolerance_s: float
    pairing_duration_tolerance_s: float
    gripper_width_max_gap_s: float
    gripper_width_closed_threshold_mm: float
    gripper_width_diagnostics: bool
    export_auxiliary_trajectories: bool
    tag_distance_max_reprojection_error_px: float
    tag_distance_max_center_ambiguity_mm: float
    tag_distance_min_valid_fraction: float
    tag_distance_max_missing_frames: int
    tag_distance_roi_px: list[int]
    recording_min_duration_s: float
    recording_static_window_s: float
    recording_max_static_gyro_rad_s: float
    recording_max_static_acc_dev_m_s2: float
    recording_max_static_acc_std_m_s2: float
    recording_max_shutter_s: float
    recording_max_exposure_cv: float

    def for_algorithm(self, parameter_type: type[_AlgorithmParameters]) -> _AlgorithmParameters:
        """Project the resolved configuration onto the algorithm's declared fields."""
        return parameter_type(**{field.name: getattr(self, field.name) for field in fields(parameter_type)})


@dataclass(frozen=True)
class Scene:
    path: Path
    name: str
    mode: str
    reference_arm: str
    recordings: dict[str, Path]
    rear_calibrations: dict[str, Path]
    trajectory: config.Configuration
    parameters: SceneParameters
    devices: dict[str, dict] = field(default_factory=dict)
    recording_serials: dict[str, str] = field(default_factory=dict)
    tag_layout: Path | None = None
    overrides: SceneOverrides | None = None
    batch_root: Path | None = None
    environment_tags_enabled: bool = False
    reconstruction_mode: str = 'scene_map'
    poster_vio_base_tag_enabled: bool = True
    recording_trim: dict = field(default_factory=dict)
    scene_map_poster_constraint_enabled: bool | None = None
    scene_map_tag_detection_enabled: bool | None = None
    scene_map_edge_trim: bool = False
    position_support_gap_sources: dict[str, str] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return self.path.parent.parent if self.path.name == 'collection.yaml' else self.path.parent

    @property
    def map_root(self) -> Path:
        return layout.maps(self.batch_root or self.root)

    def role(self, arm: str, lens: str) -> str:
        return 'shared_front' if self.mode == 'shared_front' else ('shared_rear' if lens == 'rear' else f'{arm}_front')

    def localization_lenses(self, arm: str) -> tuple[str, ...]:
        """镜头标定可供其他观测使用；只有这些镜头需要地图定位。"""
        if arm not in ARMS:
            raise ValueError(f'未知手：{arm}')
        if self.mode == 'shared_front':
            return ('front',)
        if self.mode == 'dual':
            return ('front', 'rear')
        raise ValueError(f'未知场景模式：{self.mode}')

    @property
    def environment_tag_detection(self) -> bool:
        """是否运行环境标签检测：collection.yaml 的 environment_poster.detection.enabled。"""
        return self.tag_layout is not None and self.environment_tags_enabled

    @property
    def environment_tag_policy(self) -> dict[str, object]:
        """Canonical user-facing policy for environment-tag processing."""
        return {
            'enabled': self.environment_tag_detection,
            'mapping': {
                'enabled': bool(self.parameters.tag_mapping_enabled),
                'scale': {'enabled': bool(self.parameters.tag_mapping_scale_enabled)},
            },
            'fusion': {'enabled': bool(self.parameters.tag_fusion_enabled),
                       'method': self.parameters.tag_fusion_method},
        }


@dataclass(frozen=True)
class Session:
    path: Path
    scene: Scene
    map_version: str | None
    map_manifest_sha256: str | None
    kit: dict
    expected_arms: tuple[str, ...]
    mapping: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return layout.owner(self.path)

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def relative_path(self) -> Path:
        """批次在原片归档和导出包中的路径。"""
        return Path(self.scene.name) / self.name

    @property
    def devices(self) -> dict[str, dict]:
        return {item['serial']: {key: value for key, value in item.items() if key != 'serial'}
                for item in self.kit['arms'].values()}

    @property
    def arm_serials(self) -> dict[str, str]:
        return {arm: item['serial'] for arm, item in self.kit['arms'].items()}


@dataclass(frozen=True)
class Episode:
    path: Path
    session: Session
    recordings: dict[str, Path]
    serials: dict[str, str]
    expected_arms: tuple[str, ...]
    recording_refs: dict
    instruction: str
    synchronization: dict
    map_version: str | None = None
    map_selection: dict | None = None

    @property
    def root(self) -> Path:
        return layout.owner(self.path)

    @property
    def scene(self) -> Scene:
        return self.session.scene

    @property
    def output(self) -> Path:
        return self.root / 'results'

    def offsets(self) -> dict[str, float]:
        missing = set(self.expected_arms) - set(self.recordings.keys())
        if missing:
            raise ValueError('录像不完整，缺少：' + ', '.join(sorted(missing)))
        return {arm: self.synchronization['offsets_s'][arm] for arm in self.recordings}


@dataclass(frozen=True)
class ArmTrajectory:
    """可供双手对齐与元数据写入直接使用的单手结果。"""
    episode_times_s: np.ndarray
    T_reference_tcp: np.ndarray
    frame_id: str
    tcp_frame: dict
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CalibratedCamera:
    serial: str
    selected: selection.SelectedCalibration
    measurements: dict[str, dict]
    files: dict[str, Path]
    width_measurement: dict | None = None
    tag_distance_measurement: dict | None = None
    factory_rear: FactoryRearCalibration | None = None
    base_tag: dict | None = None

    @cached_property
    def asset_hashes(self) -> dict[str, str]:
        hashes = {name: artifact_identity.file_hash(path) for name, path in sorted(self.files.items())}
        if self.base_tag is not None:
            hashes['base_tag'] = artifact_identity.canonical_hash(self.base_tag)
        if self.factory_rear is not None:
            hashes['rear'] = self.factory_rear.sha256
        return hashes


@dataclass(frozen=True)
class PreparedRecording:
    camera: CalibratedCamera
    context: VisualContext
    lens_videos: tuple[LensVideo, ...]
    video_times_s: np.ndarray  # 原始视频时钟，第一帧约为 0；不含 EuRoC 的人工 +1 s。
    imu: np.ndarray  # 本机 IMU 时间、gyro xyz、acc xyz。
    data: RecordingData

    source_frame_start: int = 0  # 裁剪首帧在原始 INSV 中的帧号。
    full_video_times_s: np.ndarray | None = None
    render_lens: Callable[[str, VisualContext, int], LensVideo] | None = field(default=None, repr=False)

    def lens_video(self, lens: str) -> LensVideo:
        for video in self.lens_videos:
            if video.lens == lens:
                return video
        if self.render_lens is None or lens not in self.camera.measurements:
            raise ValueError(f'录像没有可用的 {lens} 镜头')
        return self.render_lens(lens, self.context, self.source_frame_start)


@dataclass(frozen=True)
class TaskInputs:
    """各后端共用的冻结任务输入；场景和任务身份从 episode 读取。"""
    episode: Episode
    recordings: dict[str, PreparedRecording]
    offsets_s: dict[str, float]
    directory: Path
    arm_cpu_sets: dict[str, str] | None
    cache_audit: CacheAudit | None
    console: Console
    detections: dict[str, DetectionSnapshot | None] = field(default_factory=dict)
    poster_pnp_candidates: dict[str, Callable] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmExecutionResult:
    """所有任务后端共同返回的轨迹、测量和实际输出坐标。"""
    tracks: dict[str, ArmTrajectory]
    width_reports: dict[str, dict]
    tag_distance_reports: dict[str, dict]
    coordinate_record: dict


class BuildSceneResult(TypedDict, total=False):
    status: str
    version: str
    frame_id: str
    output: str
    scene: str
    inputs: dict


class EpisodeRunResult(TypedDict, total=False):
    status: str
    output: str
    frame_id: str
    common_interval_s: list[float] | None
    inputs: dict
    synchronization: dict
    quality: dict
