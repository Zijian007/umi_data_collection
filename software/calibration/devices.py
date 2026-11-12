"""Device registry commands, separate from calibration solving."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import shutil
import tempfile

from software import provenance as artifact_identity
from software import registry
from software.calibration import reports
from software.calibration import result_io
from software.calibration.models import Decision, Step
from software.calibration.pipeline import runtime
from software.storage import locked
from software.trajectory.calibration import assets as trajectory_calibration


def activation_status(doc):
    result = dict(checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'))
    subject = doc['subject']
    if subject['lens'] != 'front' or not subject.get('parameter_sha256'):
        return dict(result, state='unavailable', message='未查询工作镜头默认标定')
    try:
        selected = registry.get(subject['serial'])
        if artifact_identity.file_hash(Path(selected['json'])) == subject['parameter_sha256']:
            return dict(result, state='active', message='已设为该设备默认标定', version=selected['calibration_version'])
        return dict(result, state='inactive', message='设备当前启用的是另一份标定')
    except registry.UnknownCameraError:
        return dict(result, state='inactive', message='该相机尚未登记')
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
        raise ValueError('证据不完整，--force 也不能安装')
    if not decision.installable and not force:
        raise RejectedResult('验收未通过；完整且有效的结果可用 --force 显式覆盖质量判定')
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
    except (OSError, ValueError, KeyError, RuntimeError, trajectory_calibration.CalibrationError) as error:
        runtime.console.write(f'不能安装标定：{error}', state='失败')
        return 2
    runtime.console.write(f"已启用标定：相机 {device['serial']}；版本 {device['calibration_version']}", state='完成')
    return 0


def install_rear(measurement: Path) -> int:
    measurement = Path(measurement).expanduser().resolve()
    output = measurement.parent
    try:
        with locked(output.parent, name=f'.{output.name}.calibration.lock'):
            doc = _installable(output, 'rear')
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
    except (OSError, ValueError, KeyError, RuntimeError, trajectory_calibration.CalibrationError) as error:
        runtime.console.write(str(error), state='失败')
        return 2
    runtime.console.write(f"已登记环境镜头标定：相机 {result['serial']}；版本 {result['version']}", state='完成')
    return 0


def list_devices(*, kit: str | None = None, json_output: bool = False,
                 intrinsics_source: str | None = None, require_rear: bool = False) -> int:
    from software.calibration import inventory

    exit_code = None
    try:
        inventory_report = inventory.report(kit=kit, intrinsics_source=intrinsics_source, require_rear=require_rear)
    except inventory.READ_ERRORS as error:
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


def upgrade_device(serial: str, *, dry_run: bool = False) -> dict:
    serial = registry.validate_id(serial)
    selected = registry.get(serial, allow_retired=True)
    if selected.get('calibration_owner', serial) != serial:
        raise ValueError('样例借用设备不能升级为实测标定，请先完成设备标定')
    source = Path(selected['dir'])
    old_version = selected['calibration_version']
    if selected['bundle_schema_version'] >= 2:
        return dict(status='current', serial=serial, version=old_version, activated=False)
    parent = source.parent.parent
    with nullcontext() if dry_run else locked(registry.ROOT, '.registry.lock'):
        # 原版本必须仍然存在且校验通过；不依赖期间可能改变的当前默认指针。
        registry.get(serial, calibration=old_version, allow_retired=True)
        with tempfile.TemporaryDirectory(prefix='umi-upgrade-') as temporary_bundle:
            bundle = Path(temporary_bundle)/'bundle'
            trajectory_calibration.make_bundle(Path(selected['json']), bundle, note=dict(upgraded_from=old_version))
            if registry.bundle_sha256(bundle) != selected['bundle_sha256']:
                raise ValueError('升级期间源测量发生变化，拒绝发布')
            files = {'bundle/'+p.name: p for p in bundle.iterdir() if p.is_file()}
            version = registry.measurement_version(parent, files,
                identity=dict(kind='umi_camera_calibration', serial=serial), asset_prefix='bundle/')
            plan = dict(status='planned' if dry_run else 'stored', serial=serial, source_version=old_version,
                        version=version, directory=str(parent / version), activated=False)
            if dry_run:
                return plan
            report = source.parent/'report.txt'
            if report.is_file():
                files['report.txt'] = report
            registry.store_version(parent, version, files, kind='umi_camera_calibration', serial=serial,
                                   note='measurement bundle v2; upgraded from '+old_version)
    return plan


def activate_calibration(serial: str, version: str) -> dict:
    """显式变更设备默认值；已有套件与批次的测量绑定保持不变。"""
    serial, version = registry.validate_id(serial), registry.validate_id(version)
    with locked(registry.ROOT, '.registry.lock'):
        selected = registry.get(serial, calibration=version)
        if selected.get('calibration_owner', serial) != serial:
            raise ValueError('不能把样例借用标定作为实测设备版本启用')
        index = registry.load_index()
        previous = index['devices'][serial]['calibration']
        index['devices'][serial]['calibration'] = version
        if 'calibration_sources' in index['devices'][serial]:
            index['devices'][serial]['calibration_sources'][selected['intrinsics_source']] = version
        registry.save_index(index)
    return dict(status='activated', serial=serial, version=version, previous=previous)
