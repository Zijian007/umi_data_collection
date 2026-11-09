"""Typed boundary between the device registry and trajectory execution.

Selection priority remains UMI_TRAJ_CALIB, --device, then the recording's serial.
The registry validates installed assets; this module names the selected values for
callers and adapts them to the existing numerical postprocessor's dictionary API.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .assets import FUSION_SCRIPT
from software.camera.geometry import camera_from_tcp, validate_transform

DEFAULT_FRONT_MASK = Path(__file__).resolve().parents[1] / "assets/config/front_hand_mask.png"


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
            name: hashlib.sha256(path.read_bytes()).hexdigest()
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

    def summary(self) -> dict:
        """Keep the existing calibration_used fields for report consumers."""
        return dict(source=self.source, id=self.calibration_id, dir=str(self.directory),
                    serial=self.camera_serial,
                    status=self.status, selection=self.selection)

    def as_postprocess_dict(self, hashes: dict[str, str]) -> dict:
        """Compatibility adapter: numerical postprocessing keeps its established API."""
        return dict(self.summary(), json=str(self.assets.camera_json),
                    fuser=str(FUSION_SCRIPT),
                    tcp_json=str(self.assets.tcp_geometry) if self.assets.tcp_geometry else None,
                    tcp_geometry=self.geometry, mask=str(self.assets.front_mask),
                    cam_time_offset_ns=self.time_offset_ns,
                    bundle_sha256=hashes['camera_json'])

def select_calibration(video: Path, *, device: str = "", tcp: Path | None = None,
                       mask: Path = DEFAULT_FRONT_MASK) -> SelectedCalibration:
    """独立引擎默认输出相机轨迹；TCP 必须由调用者明确提供安装外参。"""
    from software.registry import resolve_insv

    record = resolve_insv(video, device=device or None)
    geometry = read_tcp_geometry(tcp) if tcp is not None else None
    return SelectedCalibration(
        assets=CalibrationAssets(Path(record['json']), tcp, mask),
        directory=Path(record['dir']), calibration_id=record.get('calibration_version', record['id']),
        camera_serial=record.get('serial'), source=record['source'], selection=record['selection'],
        status=record.get('status', 'override'), time_offset_ns=int(record['cam_time_offset_ns']),
        geometry=geometry,
    )
