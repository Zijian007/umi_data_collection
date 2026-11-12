"""共享场景的参数、清单与阶段数据；此模块不读取清单或设备资产。"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from software import config
from software import provenance as artifact_identity
from software.scene import layout
from software.trajectory.calibration import selection
from software.vision.models import LensVideo, VisualContext


ARMS = ('left', 'right')

if TYPE_CHECKING:
    from software.camera.rear_fallback import FactoryRearCalibration
    from software.scene.configuration import SceneOverrides


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
    tag_minimum_side_px: float
    tag_minimum_frames: int
    tag_corner_sigma_px: float
    tag_sigma_per_radius_px: float
    tag_huber: float
    tag_maximum_reprojection_rms_px: float
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

    @property
    def root(self) -> Path:
        return self.path.parent

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
class FusedTrajectory:
    """IMU 位姿位于调用方声明的参考系；可为固定场景或单次局部坐标。"""
    T_reference_imu: np.ndarray
    tracked: np.ndarray
    front_tracked: np.ndarray
    evidence: dict



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

    @cached_property
    def asset_hashes(self) -> dict[str, str]:
        hashes = {name: artifact_identity.file_hash(path) for name, path in sorted(self.files.items())}
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

    def lens_video(self, lens: str):
        return next(video for video in self.lens_videos if video.lens == lens)


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
