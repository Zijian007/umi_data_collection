"""轨迹处理的相机/TCP 资产选择，以及匹配视频几何的 ORB 运行设置。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from umi.core.frame import CROP, output_frame_rate, scale_intrinsics, validate_video_size, video_intrinsics
from umi.core.geometry import camera_from_tcp, validate_transform
from umi.devices.calibration_assets import CalibrationError
from umi.infra.paths import ORB_SETTINGS_ROOT, PACKAGE_ROOT, PROJECT_ROOT
from umi.infra.provenance import file_hash

FRONT_SETTINGS_TEMPLATE = ORB_SETTINGS_ROOT / 'front_x3_r1440_720.yaml'
# 独立融合的求解入口；结果元数据记录其内容哈希。
FUSION_SCRIPT = PACKAGE_ROOT / 'estimation/fusion/solver.py'
DEFAULT_FRONT_MASK = PROJECT_ROOT / 'hardware/grippers/masks/front_hand_mask.png'


def _replace_scalars(text: str, values: dict) -> str:
    for key, value in values.items():
        text, count = re.subn(rf'(?m)^({re.escape(key)}: )\S+$', lambda m: m[1] + str(value), text)
        if count != 1:
            raise CalibrationError(f'SLAM 模板应且仅应包含一行 {key}')
    return text


def _image_size(text: str) -> int:
    width = re.findall(r'(?m)^Camera.width: (\d+)$', text)
    height = re.findall(r'(?m)^Camera.height: (\d+)$', text)
    if len(width) != 1 or width != height:
        raise CalibrationError('SLAM 模板必须包含唯一的正方形图像尺寸')
    size = int(width[0])
    validate_video_size(size)
    return size


def yaml_text(calib_info: dict, *, size_px: int | None = None, video=None) -> str:
    """实测标定 → 指定尺寸的 SLAM 内参；省略尺寸仅用于核对旧包的模板规格。"""
    template_text = FRONT_SETTINGS_TEMPLATE.read_text(encoding="utf-8")
    size_px = _image_size(template_text) if size_px is None else size_px
    validate_video_size(size_px)
    T_imu_camera = np.eye(4)
    T_imu_camera[:3, :3] = Rot.from_quat(calib_info["q_xyzw"]).as_matrix()
    T_imu_camera[:3, 3] = calib_info["p"]
    intrinsics = {key: calib_info[key] for key in ('fx', 'fy', 'cx', 'cy')}
    scaled = video_intrinsics(intrinsics, video) if video else scale_intrinsics(intrinsics, CROP, size_px)
    vals = {**scaled,
            "k1": calib_info["k"][0], "k2": calib_info["k"][1], "k3": calib_info["k"][2], "k4": calib_info["k"][3]}
    template_text = _replace_scalars(template_text, {**{f'Camera1.{key}': f'{value:.12f}' for key, value in vals.items()},
                                                     'Camera.width': size_px, 'Camera.height': size_px})
    rows = [", ".join("%.12f" % x for x in T_imu_camera[i]) for i in range(4)]
    block = "data: [" + ",\n          ".join(rows) + "]"
    template_text, match_count = re.subn(r"(?s)(IMU\.T_b_c1:.*?)data: \[.*?\]", lambda m: m.group(1) + block, template_text, count=1)
    if match_count != 1:
        raise CalibrationError("样例 YAML 缺少 IMU.T_b_c1 配置块")
    return template_text


def slam_settings_text(calib_info: dict, parameters, *, rear: bool = False) -> str:
    """Measured camera geometry + this run's algorithm parameters -> ORB YAML."""
    size = parameters.video.size_px
    validate_video_size(size)
    if rear:
        # 环境镜头只有样例模板；按其原始尺寸转换，不借用工作镜头的实测内参。
        txt = (ORB_SETTINGS_ROOT / 'rear_basalt065_720.yaml').read_text()
        intrinsics = {key: float(re.search(rf'(?m)^Camera1\.{key}: (\S+)$', txt)[1])
                      for key in ('fx', 'fy', 'cx', 'cy')}
        scaled = video_intrinsics(intrinsics, parameters.video, source_px=_image_size(txt))
        txt = _replace_scalars(txt, {**{f'Camera1.{key}': f'{value:.12f}' for key, value in scaled.items()},
                                     'Camera.width': size, 'Camera.height': size})
    else:
        txt = yaml_text(calib_info, size_px=size, video=parameters.video)
    # ORB Settings::readParameter<int> requires an integer nominal frame rate.
    # Video encoding and fusion retain their exact fractional rate and timestamps.
    nominal_fps = max(1, round(output_frame_rate(parameters.video.frame_step)))
    txt = _replace_scalars(txt, {'Camera.fps': nominal_fps})
    orb_params = parameters.orb
    knobs = {'ORBextractor.nFeatures': orb_params.features, 'ORBextractor.scaleFactor': orb_params.scale_factor,
             'ORBextractor.nLevels': orb_params.levels, 'ORBextractor.iniThFAST': orb_params.initial_fast,
             'ORBextractor.minThFAST': orb_params.minimum_fast, 'System.thFarPoints': orb_params.far_points_map_units}
    return txt + '\n' + '\n'.join(f'{key}: {value}' for key, value in knobs.items()) + '\n'


def read_tcp_geometry(path: Path) -> dict:
    """在资产入口读取并验证安装外参，计算阶段直接使用已解析的几何。"""
    geometry = json.loads(path.read_text())
    if geometry["units"] != "metres":
        raise ValueError("TCP 变换的 units 必须为 metres")
    validate_transform(camera_from_tcp(geometry))
    return geometry


@dataclass(frozen=True)
class CalibrationAssets:
    camera_json: Path
    tcp_geometry: Path | None
    front_mask: Path

    def hashes(self) -> dict[str, str]:
        return {
            name: file_hash(path)
            for name, path in (
                ("camera_json", self.camera_json), ("tcp_geometry", self.tcp_geometry),
                ("front_mask", self.front_mask),
            ) if path is not None
        }


@dataclass(frozen=True)
class SelectedCalibration:
    assets: CalibrationAssets
    directory: Path
    calibration_id: str
    camera_serial: str | None
    source: str
    selection: str
    status: str
    time_offset_ns: int
    geometry: dict | None = None
    intrinsics_source: str = "measured"

    def summary(self) -> dict:
        """Keep the existing calibration_used fields for report consumers."""
        return dict(source=self.source, id=self.calibration_id, dir=str(self.directory),
                    serial=self.camera_serial,
                    status=self.status, selection=self.selection, intrinsics_source=self.intrinsics_source)

    def postprocess_record(self, hashes: dict[str, str]) -> dict:
        """Build the explicit record consumed by trajectory postprocessing."""
        return dict(self.summary(), json=str(self.assets.camera_json),
                    fuser=str(FUSION_SCRIPT),
                    tcp_json=str(self.assets.tcp_geometry) if self.assets.tcp_geometry else None,
                    tcp_geometry=self.geometry, mask=str(self.assets.front_mask),
                    cam_time_offset_ns=self.time_offset_ns,
                    bundle_sha256=hashes['camera_json'])

def select_calibration(video: Path, *, device: str = "", tcp: Path | None = None,
                       mask: Path = DEFAULT_FRONT_MASK, intrinsics_source: str | None = None) -> SelectedCalibration:
    """独立引擎默认输出相机轨迹；TCP 必须由调用者明确提供安装外参。"""
    from umi.devices.registry import resolve_insv

    record = (resolve_insv(video, device=device or None, intrinsics_source=intrinsics_source)
              if intrinsics_source else resolve_insv(video, device=device or None))
    if record.get("intrinsics_source") == "factory":
        from umi.infra.orb_install import require_forward_kb4
        require_forward_kb4()
        from umi.devices.calibration_assets import check_recording
        check_recording(video, record["intrinsics_provenance"])
    geometry = read_tcp_geometry(tcp) if tcp is not None else None
    return SelectedCalibration(
        assets=CalibrationAssets(Path(record['json']), tcp, mask),
        directory=Path(record['dir']), calibration_id=record.get('calibration_version', record['id']),
        camera_serial=record.get('serial'), source=record['source'], selection=record['selection'],
        status=record.get('status', 'override'), time_offset_ns=int(record['cam_time_offset_ns']),
        geometry=geometry, intrinsics_source=record.get("intrinsics_source", "measured"),
    )
