"""选择和冻结环境镜头与海报输入，供独立轨迹和场景回退共用。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING
import json
import shutil

from umi.infra import provenance
from umi.devices import registry
from umi.devices import calibration_source, rear_fallback
from umi.devices.calibration_assets import read_basalt
from umi.config.parameter_models import EnvironmentTagParameters
from umi.infra.file_store import write_json
from umi.devices.selection import SelectedCalibration
from umi.estimation.models import TrajectoryRequest
from umi.perception.poster import detection as environment_tags, layout as tag_board

if TYPE_CHECKING:
    from umi.estimation.fusion.poster import PosterInputs


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
        from umi.infra.orb_install import require_forward_kb4
        require_forward_kb4()

    if request.tag_layout is None:
        if request.configuration.values.environment_tags.tag_fusion_enabled:
            raise ValueError('独立海报诊断/融合需要 --tag-layout')
        return EnvironmentAssets(rear=selected_rear)

    layout_path = tag_board.resolve_layout_path(request.tag_layout)
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


@dataclass(frozen=True)
class PosterAssetHashes:
    layout_sha256: str
    rear_sha256: str


def prepare_poster(video: Path, directory: Path, layout: Path, rear: Path, parameters: EnvironmentTagParameters,
            *, expected: PosterAssetHashes | None = None, progress: Callable[[str], None] | None = None,
            detection_cache: Path | None = None,
            detection_snapshot: environment_tags.DetectionSnapshot | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    frozen_layout, frozen_rear = directory / 'layout.json', directory / 'rear.json'
    shutil.copyfile(tag_board.resolve_layout_path(layout), frozen_layout)
    shutil.copyfile(rear, frozen_rear)
    board = tag_board.load(frozen_layout)
    if expected is not None and (board.asset_hash != expected.layout_sha256
                                 or provenance.file_hash(frozen_rear) != expected.rear_sha256):
        raise ValueError('海报布局或 rear 标定在预检后发生变化')
    detection_settings = environment_tags.DetectionSettings(
        border_bits=board.border_bits, ids=board.ids, dictionary=board.dictionary,
        minimum_side_px=parameters.tag_minimum_side_px)
    cache_directory = detection_cache if detection_cache is not None else directory / 'detection'
    if detection_snapshot is None:
        detection_report = environment_tags.cached(video, cache_directory, detection_settings, layout=board, progress=progress)
    else:
        if detection_snapshot.layout.asset_hash != board.asset_hash:
            raise ValueError('预备海报标签的布局与冻结布局不一致')
        cache_directory = detection_snapshot.directory
        detection_report = detection_snapshot.report
    if cache_directory.resolve() != (directory / 'detection').resolve():
        (directory / 'detection').mkdir(exist_ok=True)
        for name in ('observations.npz', 'detection.json', 'report.json'):
            shutil.copyfile(cache_directory / name, directory / 'detection' / name)
    manifest_path = directory / 'input.json'
    write_json(manifest_path, dict(
        schema_version=1, layout='layout.json', rear='rear.json', observations='detection/observations.npz',
        assets={name: provenance.file_hash(directory / name)
                for name in ('layout.json', 'rear.json', 'detection/observations.npz')},
        parameters=asdict(parameters), detection_identity=detection_report['identity_hash']))
    return manifest_path


def load_poster(manifest_path: Path) -> PosterInputs:
    from umi.estimation.fusion.poster import PosterInputs
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema_version'] != 1:
        raise ValueError('未知海报输入版本')
    snapshot_directory = manifest_path.parent.resolve()
    for asset_name, expected_hash in manifest['assets'].items():
        asset_path = (snapshot_directory / asset_name).resolve()
        if not asset_path.is_relative_to(snapshot_directory) or provenance.file_hash(asset_path) != expected_hash:
            raise ValueError('冻结海报输入哈希不匹配')
    for asset_kind in ('layout', 'rear', 'observations'):
        if manifest[asset_kind] not in manifest['assets']:
            raise ValueError('海报输入未被哈希覆盖')
    observations = environment_tags.load_observations(snapshot_directory / manifest['observations'])
    return PosterInputs(tag_board.load(snapshot_directory / manifest['layout']), observations,
                        read_basalt(snapshot_directory / manifest['rear']))
