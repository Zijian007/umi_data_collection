"""显式升级旧设备标定：新增不可变版本，默认不切换设备或历史批次。"""
from __future__ import annotations
import tempfile
from pathlib import Path
from software import registry
from software.storage import locked
from software.trajectory import calibration_assets


def upgrade_device(serial: str, *, dry_run: bool = False) -> dict:
    serial = registry.validate_id(serial)
    selected = registry.get(serial, allow_retired=True)
    if selected.get('calibration_owner', serial) != serial:
        raise ValueError('样例借用设备不能升级为实测标定，请先完成设备标定')
    source = Path(selected['dir'])
    old_version = selected['calibration_version']
    if selected['bundle_schema_version'] == 2:
        return dict(status='current', serial=serial, version=old_version, activated=False)
    version = selected['id']+'-v2'
    parent = source.parent.parent
    directory = parent/version
    plan = dict(status='planned' if dry_run else 'stored', serial=serial, source_version=old_version,
                version=version, directory=str(directory), activated=False)
    if dry_run:
        return plan
    with locked(registry.ROOT, '.registry.lock'):
        # 原版本必须仍然存在且校验通过；不依赖期间可能改变的当前默认指针。
        registry.get(serial, calibration=old_version, allow_retired=True)
        with tempfile.TemporaryDirectory(prefix='umi-upgrade-') as temp:
            bundle = Path(temp)/'bundle'
            calibration_assets.make_bundle(Path(selected['json']), bundle, note=dict(upgraded_from=old_version))
            if registry.bundle_sha256(bundle) != selected['bundle_sha256']:
                raise ValueError('升级期间源测量发生变化，拒绝发布')
            files = {'bundle/'+p.name: p for p in bundle.iterdir() if p.is_file()}
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
        registry.save_index(index)
    return dict(status='activated', serial=serial, version=version, previous=previous)
