"""共享场景的参数、清单与阶段数据；此模块不读取清单或设备资产。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from software import config
from software.trajectory import selection
from software.vision.models import LensVideo, VisualContext

ARMS = ('left', 'right')


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
    front_position_sigma_m: float
    rear_position_sigma_m: float
    visual_rotation_sigma_deg: float
    huber: float
    pairing_start_tolerance_s: float
    pairing_duration_tolerance_s: float
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

    @property
    def root(self) -> Path:
        return self.path.parent

    def role(self, arm: str, lens: str) -> str:
        return 'shared_front' if self.mode == 'shared_front' else ('shared_rear' if lens == 'rear' else f'{arm}_front')


@dataclass(frozen=True)
class Session:
    path: Path
    scene: Scene
    map_version: str | None
    map_manifest_sha256: str | None
    kit: dict
    expected_arms: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.path.parent.name

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
    def scene(self) -> Scene:
        return self.session.scene

    @property
    def output(self) -> Path:
        return self.path.parent / 'results'

    def offsets(self) -> dict[str, float]:
        missing = set(self.expected_arms)-self.recordings.keys()
        if missing:
            raise ValueError('录像不完整，缺少：'+', '.join(sorted(missing)))
        return {arm: self.synchronization['offsets_s'][arm] for arm in self.recordings}


@dataclass(frozen=True)
class ArmTrajectory:
    """可供双手对齐与元数据写入直接使用的单手结果。"""
    episode_times_s: np.ndarray
    T_scene_tcp: np.ndarray
    frame_id: str
    tcp_frame: dict


@dataclass(frozen=True)
class CalibratedCamera:
    serial: str
    selected: selection.SelectedCalibration
    measurements: dict[str, dict]
    files: dict[str, Path]


@dataclass(frozen=True)
class PreparedRecording:
    camera: CalibratedCamera
    context: VisualContext
    lens_videos: tuple[LensVideo, ...]
    video_times_s: np.ndarray  # 原始视频时钟，第一帧约为 0；不含 EuRoC 的人工 +1 s。
    imu: np.ndarray  # 本机 IMU 时间、gyro xyz、acc xyz。

    def lens_video(self, lens: str):
        return next(video for video in self.lens_videos if video.lens == lens)
