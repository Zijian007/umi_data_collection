"""设备标定登记、状态查询与退役。"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import NotRequired, TypedDict
from zoneinfo import ZoneInfo

from ruamel.yaml.error import YAMLError

from umi.calibration import reports, result_io
from umi.calibration.models import Decision, Step
from umi.calibration.pipeline import runtime
from umi.devices import calibration_assets, registry
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import locked


def activation_status(doc):
    result = dict(checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'))
    subject = doc['subject']
    if subject['lens'] != 'front' or not subject.get('parameter_sha256'):
        return dict(result, state='unavailable', message='未查询工作镜头默认标定')
    try:
        selected = registry.get(subject['serial'], intrinsics_source=subject.get('intrinsics_source', 'measured'))
        if artifact_identity.file_hash(Path(selected['json'])) == subject['parameter_sha256']:
            return dict(result, state='active', message='本报告已设为该设备此参数来源的当前标定', version=selected['calibration_version'])
        return dict(result, state='inactive', message='当前设备仍使用该来源的其他标定版本；本报告版本未启用')
    except registry.UnknownCameraError:
        return dict(result, state='inactive', message='该相机尚未登记，因此本报告版本未启用')
    except (OSError, ValueError, KeyError, registry.RegistryError) as error:
        return dict(result, state='unavailable', message=f'无法确认：{error}')


class RejectedResult(ValueError):
    """Complete evidence did not pass quality policy; distinct from an invalid result."""


def _installable(output, lens, *, force=False, camera_serial=None):
    doc = result_io.load_result(output, complete=True)
    if doc['operation'] != 'calibrate' or doc['subject']['lens'] != lens:
        raise ValueError('只能安装对应镜头的完整标定结果')
    if camera_serial and camera_serial != doc['subject']['serial']:
        raise ValueError('--serial 与测量所属相机不一致，不能改写设备身份')
    required = (Step.INTRINSICS, Step.CAMERA_IMU, Step.GRAVITY, Step.TRAJECTORY)
    if not all(step in doc['measurements'] for step in required) or not doc.get('evaluation'):
        raise ValueError('缺少完整验收证据')
    decision = Decision(doc['evaluation']['decision'])
    if decision == Decision.INCOMPLETE:
        raise ValueError('证据不完整，--allow-rejected 也不能安装')
    if not decision.installable and not force:
        raise RejectedResult('验收未通过，完整且有效的结果可用 umi-calib install --allow-rejected 登记')
    return doc


def install_bundle(output: Path, force: bool, camera_serial: str | None = None) -> int:
    output = Path(output).expanduser().resolve()
    try:
        with locked(output.parent, name=f'.{output.name}.calibration.lock'):
            doc = _installable(output, 'front', force=force, camera_serial=camera_serial)
            device = registry.register(doc['subject']['serial'], output,
                status='active', note=f'umi-calib install from {output}')
            doc['activation'] = activation_status(doc)
            result_io.write(output, doc)
            reports.write_report(output, doc)
            archived = Path(device['dir']).parent/'report.txt'
            if not archived.exists():
                shutil.copyfile(output/'report.txt', archived)
    except RejectedResult as error:
        runtime.console.write(str(error), state='失败')
        return 1
    except (OSError, ValueError, KeyError, RuntimeError, calibration_assets.CalibrationError) as error:
        runtime.console.write(f'不能安装标定：{error}', state='失败')
        return 2
    runtime.console.write(f"已更新当前标定：相机 {device['serial']}；front / {device['intrinsics_source']}", state='完成')
    return 0


def _install_rear_asset(measurement: Path, *, force: bool = False, camera_serial: str | None = None) -> int:
    measurement = Path(measurement).expanduser().resolve()
    output = measurement.parent
    try:
        with locked(output.parent, name=f'.{output.name}.calibration.lock'):
            doc = _installable(output, 'rear', force=force, camera_serial=camera_serial)
            raw = json.loads(measurement.read_text())
            expected = json.loads(result_io.validate_subject(output, doc).read_text())
            metadata = raw.pop('validation')
            if (raw.pop('kind') != 'umi_rear_calibration' or raw.pop('camera_serial') != doc['subject']['serial'] or
                    raw.pop('lens') != 'rear' or metadata['evidence_fingerprint'] != result_io.fingerprint(doc)):
                raise ValueError('环境镜头文件与当前验收身份不一致')
            raw.pop('parameters')
            origin = raw.pop('provenance', {})
            manifest = json.loads((output/'manifest.json').read_text())
            if origin != manifest.get('provenance', {}):
                raise ValueError('环境镜头来源记录与验收清单不一致')
            if raw != expected or metadata['C'] != result_io.measurement(doc, Step.TRAJECTORY) or metadata['D'] != result_io.measurement(doc, Step.GRAVITY):
                raise ValueError('环境镜头文件与当前参数或证据不一致')
            result = registry.register_rear(measurement)
    except (OSError, ValueError, KeyError, RuntimeError, calibration_assets.CalibrationError) as error:
        runtime.console.write(str(error), state='失败')
        return 2
    runtime.console.write(f"已更新当前标定：相机 {result['serial']}；rear", state='完成')
    return 0


def install(asset: Path, *, force: bool = False, camera_serial: str | None = None) -> int:
    """Install either lens from a result directory or its rear measurement file.

    A rear result directory is accepted directly; the dispatcher selects its
    ``rear_calibration.json`` so front and rear have the same public command.
    """
    asset = Path(asset).expanduser().resolve()
    measurement = asset if asset.is_file() else asset / 'rear_calibration.json'
    if measurement.is_file():
        return _install_rear_asset(measurement, force=force, camera_serial=camera_serial)
    return install_bundle(asset, force=force, camera_serial=camera_serial)


def list_devices(*, kit: str | None = None, json_output: bool = False, require_rear: bool = False) -> int:
    exit_code = None
    try:
        inventory_report = report(kit=kit, require_rear=require_rear)
    except READ_ERRORS as error:
        inventory_report = dict(schema_version=1, ok=False, errors=[str(error)])
        exit_code = 2
    if json_output:
        print(json.dumps(inventory_report, ensure_ascii=False, indent=2))
    else:
        reports.show_inventory(inventory_report)
        if inventory_report.get('devices') == []:
            print('设备档案是空的')
    return exit_code if exit_code is not None else (0 if inventory_report['ok'] else 1)


def retire(camera_serial: str) -> int:
    try:
        registry.retire(camera_serial)
    except registry.RegistryError as error:
        runtime.console.write(str(error), state="失败")
        return 2
    runtime.console.write(f"已停用相机 {camera_serial}", state="完成")
    return 0


READ_ERRORS = (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError, YAMLError)


class LensVersionStatus(TypedDict):
    version: str | None
    lens: str
    ok: bool
    intrinsics_source: NotRequired[str]
    verdict: NotRequired[str]
    error: NotRequired[str]
    default: NotRequired[bool]


class DeviceStatus(TypedDict):
    serial: str
    status: str | None
    front_default: str | None
    calibration_owner: str
    calibration_sources: dict[str, str]
    versions: list[LensVersionStatus]
    errors: list[str]
    ok: bool


class ArmCalibrationStatus(TypedDict):
    serial: str
    status: str | None
    errors: list[str]
    front: LensVersionStatus
    rear: LensVersionStatus
    front_calibration_available: bool
    dual_calibrations_available: bool
    ok: bool


class KitStatus(TypedDict):
    id: str
    arms: dict[str, ArmCalibrationStatus]
    errors: list[str]
    require_rear: bool
    scope: str
    combination_validation: str
    ok: bool


class InventoryReport(TypedDict):
    schema_version: int
    ok: bool
    devices: NotRequired[list[DeviceStatus]]
    kit: NotRequired[KitStatus]
    errors: NotRequired[list[str]]


def lens_version(serial: str, version: str, lens: str, *, expected_source: str | None = None) -> LensVersionStatus:
    lens_status: LensVersionStatus = dict(version=version, lens=lens, ok=False)
    try:
        if lens == 'front':
            camera = registry.read_front_version(serial, version)
            source = camera['intrinsics_source']
            lens_status['intrinsics_source'] = source
            calibration_assets.require_intrinsics_source(camera, expected_source, label=f'{serial} {lens}')
        else:
            path = registry.rear_path(serial, version)
            camera, metadata = calibration_assets.read_rear(path, serial=serial, expected_source=expected_source)
            source = camera['intrinsics_source']
            lens_status['verdict'] = metadata['validation']['verdict']
        lens_status['intrinsics_source'] = source
        lens_status['ok'] = True
    except READ_ERRORS as error:
        lens_status['error'] = str(error)
    return lens_status


def device_inventory() -> list[DeviceStatus]:
    device_statuses: list[DeviceStatus] = []
    for serial, device_record in sorted(registry.load_index()['devices'].items()):
        device_status = dict(serial=serial, status=device_record.get('status'), front_default=device_record.get('calibration'),
                   calibration_owner=device_record.get('calibration_owner', serial),
                   calibration_sources=device_record.get('calibration_sources', {}), versions=[], errors=[])
        try:
            registry.get(serial, allow_retired=True)
            for source, version in device_status['calibration_sources'].items():
                registry.get(serial, calibration=version, intrinsics_source=source, allow_retired=True)
        except READ_ERRORS as error:
            device_status['errors'].append(str(error))
        # Inventory reports every immutable version, while the index only stores
        # one current pointer per lens/source.  A damaged historical asset is
        # visible in the report but does not alter the current kit selection.
        for lens in ('front', 'rear'):
            current = device_record.get('calibration_sources' if lens == 'front' else 'rear_calibration_sources', {})
            current_by_version = {identity: source for source, identity in current.items()}
            for source, directory in registry._iter_version_directories(serial, lens):
                identity = directory.name
                version = lens_version(serial, identity, lens,
                                       expected_source=current_by_version.get(identity, source))
                version['default'] = (lens == 'front' and device_status['calibration_owner'] == serial
                                      and identity == device_status['front_default'])
                device_status['versions'].append(version)
            # Keep a clear failure row when an index pointer references a
            # missing version, instead of silently hiding the broken binding.
            seen = {row['version'] for row in device_status['versions'] if row['lens'] == lens}
            for source, identity in current.items():
                if identity in seen:
                    continue
                version = lens_version(serial, identity, lens, expected_source=source)
                version['default'] = lens == 'front' and identity == device_status['front_default']
                device_status['versions'].append(version)
        device_status['ok'] = not device_status['errors'] and all(version['ok'] for version in device_status['versions'])
        device_statuses.append(device_status)
    return device_statuses


def kit_status(name: str, *, intrinsics_source: str | None = None, require_rear: bool = False) -> KitStatus:
    from umi.scene.collection.kit import read_kit

    kit_report = dict(id=name, arms={}, errors=[], require_rear=require_rear,
                  scope='camera_calibration_bindings', combination_validation='not_checked')
    try:
        kit = read_kit(name, intrinsics_source=intrinsics_source)
        device_index = registry.load_index()['devices']
    except READ_ERRORS as error:
        kit_report.update(ok=False, errors=[str(error)])
        return kit_report
    for arm, arm_binding in kit['arms'].items():
        serial = arm_binding['serial']
        device_record = device_index.get(serial, {})
        arm_status = dict(serial=serial, status=device_record.get('status'), errors=[])
        if device_record.get('status') != 'active':
            arm_status['errors'].append('新采集需要已启用的本设备标定（设备未登记、已停用或为样例）')
        if device_record.get('calibration_owner', serial) != serial:
            arm_status['errors'].append('套件不能借用其他设备或样例的标定')
        expected_source = arm_binding.get('intrinsics_source')
        arm_status['front'] = lens_version(serial, arm_binding['calibration'], 'front', expected_source=expected_source)
        expected_source = arm_binding.get('rear_intrinsics_source') or expected_source or arm_status['front'].get('intrinsics_source')
        rear = arm_binding.get('rear_calibration')
        arm_status['rear'] = (lens_version(serial, rear, 'rear', expected_source=expected_source) if rear else
                       dict(version=None, lens='rear', ok=False, error='未绑定 rear 当前标定'))
        if not rear and arm_binding.get('rear_intrinsics_source') == 'factory':
            arm_status['rear'].update(intrinsics_source='factory',
                error='将从原片工厂参数与 front B 推导；原片尚未检查，rear 未独立验收')
        arm_status['front_calibration_available'] = not arm_status['errors'] and arm_status['front']['ok']
        arm_status['dual_calibrations_available'] = arm_status['front_calibration_available'] and arm_status['rear']['ok']
        arm_status['ok'] = arm_status['front_calibration_available'] and (arm_status['rear']['ok'] if rear or require_rear else True)
        kit_report['arms'][arm] = arm_status
    kit_report['ok'] = all(arm_status['ok'] for arm_status in kit_report['arms'].values())
    return kit_report


def report(*, kit: str | None = None, intrinsics_source: str | None = None,
           require_rear: bool = False) -> InventoryReport:
    if (require_rear or intrinsics_source) and kit is None:
        raise ValueError('来源选择或 --require-rear 需要同时指定 --kit')
    inventory_report: InventoryReport = dict(schema_version=1, devices=device_inventory(), ok=False)
    if kit is not None:
        inventory_report['kit'] = kit_status(kit, intrinsics_source=intrinsics_source, require_rear=require_rear)
        # Other devices' historical problems do not change this kit's binding verdict.
        inventory_report['ok'] = inventory_report['kit']['ok']
    else:
        inventory_report['ok'] = bool(inventory_report['devices']) and all(device['ok'] for device in inventory_report['devices'])
    return inventory_report
