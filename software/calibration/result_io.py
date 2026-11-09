"""Versioned measurement records. Reads never execute stages, materialize work or query the registry."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from software import config
from software.storage import atomic_file
from software.calibration.models import (RunPaths, Step, IntrinsicsResult, CameraImuResult,
    Kb4)
from software.trajectory.calibration import assets

SCHEMA_VERSION = 3
MEASUREMENT_VERSION = 1


def read(output: Path) -> dict:
    path = RunPaths(Path(output)).record
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(doc, dict) or doc.get('kind') != 'umi_calibration_run':
        raise ValueError(f'标定记录格式无效：{path}')
    if doc.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'不支持旧标定记录 schema {doc.get("schema_version")}；请用新目录重跑')
    if doc.get('measurement_version') != MEASUREMENT_VERSION:
        raise ValueError('测量方法版本不兼容，请重新计算')
    # Reject non-standard NaN/Infinity as well as damaged measurement payloads.
    json.dumps(doc, allow_nan=False)
    try:
        if doc['operation'] not in ('calibrate', 'verify'):
            raise ValueError('未知标定操作')
        for key in ('subject', 'inputs', 'computation', 'measurements', 'execution'):
            if not isinstance(doc[key], dict):
                raise ValueError(f'运行记录分区无效：{key}')
        if doc['subject']['lens'] not in ('front', 'rear') or not doc['subject']['serial']:
            raise ValueError('设备身份无效')
        if doc['execution']['status'] not in ('running', 'complete', 'failed', 'interrupted'):
            raise ValueError('执行状态无效')
        for name, entry in doc['measurements'].items():
            Step(name)
            if not isinstance(entry['data'], dict) or config.canonical_hash(entry['data']) != entry['sha256']:
                raise ValueError(f'测量记录校验失败：{name}')
        if not isinstance(doc['execution']['steps'], dict):
            raise ValueError('检查点分区无效')
        for name, entry in doc['execution']['steps'].items():
            if entry['input_hash'] != doc['measurements'][name]['input_hash'] or not isinstance(entry['artifacts'], dict):
                raise ValueError(f'检查点与测量不一致：{name}')
    except (KeyError, TypeError) as error:
        raise ValueError(f'标定记录缺字段或字段类型错误：{path}') from error
    return doc


def write(output: Path, doc: dict) -> None:
    doc.update(kind='umi_calibration_run', schema_version=SCHEMA_VERSION, measurement_version=MEASUREMENT_VERSION)
    encoded = json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with atomic_file(RunPaths(Path(output)).record) as temporary:
        temporary.write_text(encoded, encoding='utf-8')


def local_path(output, relative):
    root = Path(output).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError('标定产物路径越界')
    return path


def measurement(doc, step):
    return doc['measurements'][str(step)]['data']


def intrinsics_measurement(result):
    return dict(calibration_snapshot=result.calibration_json.read_text(encoding='utf-8'),
                solver=result.solver_metrics, coverage=result.board_metrics)


def intrinsics_result(output, data):
    raw = json.loads(data['calibration_snapshot'])
    params = (raw.get('value0') or raw)['intrinsics'][0]['intrinsics']
    return IntrinsicsResult(RunPaths(Path(output)).work/'camera_intrinsics/calibration.json',
        Kb4(*(params[key] for key in Kb4._fields)), data['solver'], data['coverage'])


def camera_imu_result(output, data):
    path = RunPaths(Path(output)).calibration
    return CameraImuResult(path, assets.read_basalt(path), data['solver'])


def fingerprint(doc):
    return config.canonical_hash(dict(subject=doc['subject'], inputs=doc['inputs'],
        computation=doc['computation'], measurements=doc['measurements']))


def validate_subject(output, doc):
    subject = doc['subject']
    path = (local_path(output, subject['parameter_file']) if doc['operation'] == 'calibrate'
            else Path(subject['parameter_file']))
    if config.file_hash(path) != subject['parameter_sha256']:
        raise ValueError('参数内容已变化，与本次证据不一致')
    assets.read_basalt(path)
    if doc['operation'] == 'calibrate':
        assets.describe(Path(output), 'registry')
        manifest = json.loads(RunPaths(Path(output)).manifest.read_text())
        if manifest['camera_serial'] != subject['serial'] or manifest['lens'] != subject['lens']:
            raise ValueError('参数清单与验收的设备身份不一致')
        provenance = manifest.get('provenance', {})
        if provenance.get('board') != doc['inputs']['grid'] or provenance.get('recordings') != doc['inputs']['recordings']:
            raise ValueError('参数清单与验收的输入来源不一致')
    return path


def load_result(output, *, complete=False):
    doc = read(output)
    if not doc:
        raise ValueError('没有当前格式的运行记录；旧结果需重新运行')
    if complete and doc['execution']['status'] != 'complete':
        raise ValueError('本次标定尚未完成，不能安装或重新验收')
    if doc.get('evaluation'):
        if doc['evaluation']['evidence_fingerprint'] != fingerprint(doc):
            raise ValueError('验收与当前证据不一致')
    if doc['subject'].get('parameter_sha256'):
        validate_subject(output, doc)
    return doc


def commit_evaluation(output, doc, evaluation):
    old = doc.get('evaluation')
    if old:
        doc.setdefault('evaluation_history', []).append(old)
    doc['evaluation'] = dict(evaluation, evidence_fingerprint=fingerprint(doc),
                             evaluated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    write(output, doc)


def publish_candidate(output, calibration, serial, lens, grid, sources):
    return assets.make_result(calibration.calibration_json, Path(output), provenance=dict(
        created_by='umi-calib', camera_serial=serial, lens=lens, board=grid, recordings=sources))


def publish_rear(output, doc):
    from software.calibration.models import Decision
    if not Decision(doc['evaluation']['decision']).installable:
        return None
    raw = json.loads(RunPaths(Path(output)).calibration.read_text())
    evidence = measurement(doc, Step.GRAVITY)
    raw.update(kind='umi_rear_calibration', camera_serial=doc['subject']['serial'], lens='rear',
        validation=dict(verdict=Decision(doc['evaluation']['decision']).label,
                        C=measurement(doc, Step.TRAJECTORY), D=evidence,
                        evidence_fingerprint=doc['evaluation']['evidence_fingerprint']),
        parameters=doc['computation']['configuration'])
    path = Path(output)/'rear_calibration.json'
    with atomic_file(path) as temp:
        temp.write_text(json.dumps(raw, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return path
