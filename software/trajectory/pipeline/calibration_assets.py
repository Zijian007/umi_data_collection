"""独立轨迹的环境资产选择与 rear 快照；一次选择供计算、缓存和报告共用。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import shutil

from software import provenance, registry
from software.camera import calibration_source, rear_fallback
from software.storage import write_json
from software.trajectory.calibration.selection import SelectedCalibration
from software.trajectory.models import TrajectoryRequest
from software.vision import tag_board


@dataclass(frozen=True)
class RearCalibrationIdentity:
    serial: str
    version: str
    intrinsics_source: str
    sha256: str


@dataclass(frozen=True)
class SelectedRearCalibration:
    path: Path | None
    identity: RearCalibrationIdentity
    factory: rear_fallback.FactoryRearCalibration | None = None


@dataclass(frozen=True)
class SelectedPosterLayout:
    path: Path
    board: tag_board.TagBoardLayout


@dataclass(frozen=True)
class EnvironmentAssets:
    rear: SelectedRearCalibration | None = None
    poster: SelectedPosterLayout | None = None

    def __post_init__(self) -> None:
        if self.poster is not None and self.rear is None:
            raise ValueError('海报布局必须配合已选 rear 标定')

    def fingerprint_assets(self) -> dict[str, str]:
        """保持现有指纹字段与序列化格式，纯计算，不重新选择或读取资产。"""
        if self.rear is None:
            return {}
        rear_identity = self.rear.identity
        asset_hashes = {
            'rear_calibration': rear_identity.sha256,
            'rear_identity': provenance.canonical_hash(asdict(rear_identity)),
        }
        if self.poster is not None:
            asset_hashes['environment_poster'] = provenance.canonical_hash(dict(
                layout=self.poster.board.record(), rear_sha256=rear_identity.sha256,
                rear_version=rear_identity.version))
        return asset_hashes


def select_environment_assets(
    video: Path, front: SelectedCalibration, request: TrajectoryRequest,
) -> EnvironmentAssets:
    """调用方先解析请求；本入口集中检查 rear 与海报的组合及设备身份。"""
    requires_rear = request.mode == 'dual' or request.tag_layout is not None
    if request.rear_calibration and not requires_rear:
        raise ValueError('--rear-calibration 需要 --mode dual 或 --tag-layout，避免选择后未使用')

    selected_rear = None
    if request.rear_calibration:
        calibration_path = registry.rear_path(front.camera_serial, request.rear_calibration)
        measurement, _ = calibration_source.read_rear(calibration_path, serial=front.camera_serial,
            expected_source=front.intrinsics_source, video=video)
        selected_rear = SelectedRearCalibration(calibration_path, RearCalibrationIdentity(
            serial=front.camera_serial, version=request.rear_calibration,
            intrinsics_source=measurement['intrinsics_source'], sha256=provenance.file_hash(calibration_path)))
    elif requires_rear:
        factory = rear_fallback.derive(video, front_path=front.assets.camera_json,
                                      serial=front.camera_serial, front_source=front.intrinsics_source)
        selected_rear = SelectedRearCalibration(None, RearCalibrationIdentity(
            serial=front.camera_serial, version='factory_fallback_'+factory.sha256[:12],
            intrinsics_source='factory', sha256=factory.sha256), factory)
    if request.mode == 'dual' and selected_rear is not None and selected_rear.identity.intrinsics_source == 'factory':
        from software.vision.capabilities import require_forward_kb4
        require_forward_kb4()

    if request.tag_layout is None:
        if request.configuration.values.environment_tags.tag_fusion_enabled:
            raise ValueError('独立海报诊断/融合需要 --tag-layout')
        return EnvironmentAssets(rear=selected_rear)

    layout_path = request.tag_layout.expanduser().resolve()
    return EnvironmentAssets(selected_rear, SelectedPosterLayout(layout_path, tag_board.load(layout_path)))


def freeze_rear_calibration(selected: SelectedRearCalibration, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    frozen_calibration = directory / 'calibration.json'
    if selected.factory is not None:
        selected.factory.freeze(frozen_calibration)
    else:
        shutil.copyfile(selected.path, frozen_calibration)
    if provenance.file_hash(frozen_calibration) != selected.identity.sha256:
        raise ValueError('rear 标定在预检后发生变化')
    manifest_path = directory / 'manifest.json'
    write_json(manifest_path, dict(schema_version=1, **asdict(selected.identity)))
    return manifest_path


def load_frozen_rear(
    manifest_path: Path, *, expected_hash: str | None = None, expected_identity: str | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('不支持的 rear 快照版本')
    identity = RearCalibrationIdentity(**{
        field: manifest[field] for field in RearCalibrationIdentity.__dataclass_fields__})
    calibration_path = manifest_path.parent / 'calibration.json'
    if (provenance.file_hash(calibration_path) != identity.sha256
            or (expected_hash is not None and identity.sha256 != expected_hash)):
        raise ValueError('rear 冻结标定哈希不一致')
    if expected_identity is not None and provenance.canonical_hash(asdict(identity)) != expected_identity:
        raise ValueError('rear 冻结标定身份不一致')
    return rear_fallback.read_snapshot(calibration_path, serial=identity.serial,
                                       expected_source=identity.intrinsics_source)
