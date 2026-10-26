"""Typed boundary between the device registry and trajectory execution.

Selection priority remains UMI_TRAJ_CALIB, --device, then the recording's serial.
The registry validates installed assets; this module names the selected values for
callers and adapts them to the existing numerical postprocessor's dictionary API.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from software.registry import RegistryError, resolve_insv
from software.trajectory.calibration_assets import CalibrationError, FUSION_SCRIPT


@dataclass(frozen=True)
class CalibrationAssets:
    camera_json: Path
    tcp_geometry: Path
    front_mask: Path

    def hashes(self) -> dict[str, str]:
        return {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("camera_json", self.camera_json), ("tcp_geometry", self.tcp_geometry),
                ("front_mask", self.front_mask),
            )
        }


@dataclass(frozen=True)
class SelectedCalibration:
    assets: CalibrationAssets
    directory: Path
    calibration_id: str
    camera_serial: str | None
    gripper_id: str
    source: str
    selection: str
    status: str
    time_offset_ns: int

    def summary(self) -> dict:
        """Keep the existing calibration_used fields for report consumers."""
        return dict(source=self.source, id=self.calibration_id, dir=str(self.directory),
                    serial=self.camera_serial, gripper=self.gripper_id,
                    status=self.status, selection=self.selection)

    def as_postprocess_dict(self, hashes: dict[str, str]) -> dict:
        """Compatibility adapter: numerical postprocessing keeps its established API."""
        return dict(self.summary(), json=str(self.assets.camera_json),
                    fuser=str(FUSION_SCRIPT),
                    tcp_json=str(self.assets.tcp_geometry), mask=str(self.assets.front_mask),
                    cam_time_offset_ns=self.time_offset_ns,
                    bundle_sha256=hashes['camera_json'])

def select_calibration(video: Path, *, device: str = "", gripper: str = "") -> SelectedCalibration:
    try:
        record = resolve_insv(video, device=device or None, gripper=gripper or None)
        assets = CalibrationAssets(
            camera_json=Path(record['json']), tcp_geometry=Path(record['tcp_json']),
            front_mask=Path(record['mask']),
        )
        return SelectedCalibration(
            assets=assets, directory=Path(record['dir']), calibration_id=record['id'],
            camera_serial=record.get('serial'), gripper_id=record['gripper'],
            source=record['source'], selection=record['selection'], status=record['status'],
            time_offset_ns=int(record['cam_time_offset_ns']),
        )
    except RegistryError:
        raise
    except (CalibrationError, OSError, KeyError, TypeError, ValueError) as error:
        raise RegistryError(f"无法为 {video.name} 选择标定：{error}") from error
