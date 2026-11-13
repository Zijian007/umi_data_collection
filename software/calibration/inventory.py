"""只读检查每个镜头、每种来源的当前参数和套件选择。"""
from __future__ import annotations

from typing import NotRequired, TypedDict

from ruamel.yaml.error import YAMLError

from software import registry
from software.camera import calibration_source


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
            calibration_source.require(camera, expected_source, label=f'{serial} {lens}')
        else:
            path = registry.rear_path(serial, version)
            camera, metadata = calibration_source.read_rear(path, serial=serial, expected_source=expected_source)
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
        front = device_record.get('calibration_sources') or {None: device_record.get('calibration')}
        for lens, selected in (('front', front), ('rear', device_record.get('rear_calibration_sources', {}))):
            for source, identity in selected.items():
                version = lens_version(serial, identity, lens, expected_source=source)
                version['default'] = lens == 'front' and device_status['calibration_owner'] == serial and identity == device_status['front_default']
                device_status['versions'].append(version)
        device_status['ok'] = not device_status['errors'] and all(version['ok'] for version in device_status['versions'])
        device_statuses.append(device_status)
    return device_statuses


def kit_status(name: str, *, intrinsics_source: str | None = None, require_rear: bool = False) -> KitStatus:
    from software.scene.management.inputs import read_kit

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
        raise ValueError('--require-rear / --intrinsics-source 需要同时指定 --kit')
    inventory_report: InventoryReport = dict(schema_version=1, devices=device_inventory(), ok=False)
    if kit is not None:
        inventory_report['kit'] = kit_status(kit, intrinsics_source=intrinsics_source, require_rear=require_rear)
        # Other devices' historical problems do not change this kit's binding verdict.
        inventory_report['ok'] = inventory_report['kit']['ok']
    else:
        inventory_report['ok'] = bool(inventory_report['devices']) and all(device['ok'] for device in inventory_report['devices'])
    return inventory_report
