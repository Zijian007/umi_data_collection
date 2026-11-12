"""Fresh REPORT_ONLY gravity diagnostics, bound to the exact inputs."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile

import numpy as np

from software import provenance as artifact_identity
from software.calibration.models import GravityResult
from software.calibration.pipeline import runtime


def validate_gravity(dataset: Path, calibration_json: Path, parameters, execution) -> GravityResult:
    """Measure multi-pose gravity residuals; do not modify calibration parameters."""

    execution.paths.work.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix='gravity_check_', dir=execution.paths.work))
    p = parameters
    calibration_hash = artifact_identity.file_hash(calibration_json)
    imu_csv = dataset / 'mav0/imu0/data.csv'
    input_hash = artifact_identity.file_hash(imu_csv)
    return_code = runtime.run_logged([sys.executable, runtime.TOOLS / 'imu_calib_physical_check.py', '--calib', calibration_json,
                '--euroc', dataset,
                '--window-s', str(p.window_s), '--gyro-max-dps', str(p.maximum_gyro_deg_s),
                '--acc-std-max', str(p.maximum_acc_norm_std_m_s2), '--min-window-samples', str(p.minimum_samples),
                '--gravity', str(p.gravity_m_s2), '--distinct-dir-cos', str(p.distinct_direction_cosine),
                '--min-window-span-fraction', str(p.minimum_window_span_fraction), '--out', destination],
               execution.paths.logs / 'gravity_check.log')
    path = destination / 'imu_calib_physical_check.json'
    if return_code not in (0, 4) or not path.is_file():
        raise RuntimeError(f'D 工具执行失败（退出码 {return_code}），见 {execution.paths.logs / "gravity_check.log"}')
    evidence = json.loads(path.read_text(encoding='utf-8'))
    if evidence.get('status') not in ('REPORT_ONLY', 'INSUFFICIENT_QUASI_STATIC_COVERAGE'):
        raise RuntimeError('D 工具没有返回有效诊断状态')
    provenance = evidence.get('provenance', {})
    expected_parameters = dict(window_s=p.window_s, gyro_max_dps=p.maximum_gyro_deg_s,
        acc_std_max=p.maximum_acc_norm_std_m_s2, min_window_samples=p.minimum_samples,
        gravity=p.gravity_m_s2, max_gravity_residual=None, distinct_dir_cos=p.distinct_direction_cosine,
        minimum_span_fraction=p.minimum_window_span_fraction)
    if (provenance.get('calibration_sha256') != calibration_hash
            or artifact_identity.file_hash(calibration_json) != calibration_hash
            or artifact_identity.file_hash(imu_csv) != input_hash
            or provenance.get('params') != expected_parameters
            or provenance.get('euroc_roots') != [str(dataset)]):
        raise RuntimeError('D 诊断来源与当前输入不一致')
    expected_status = 'REPORT_ONLY' if return_code == 0 else 'INSUFFICIENT_QUASI_STATIC_COVERAGE'
    if evidence['status'] != expected_status:
        raise RuntimeError('D 诊断状态与退出码不一致')
    provenance['imu_sha256'] = input_hash
    pooled = evidence.get('pooled', {})
    residual = pooled.get('residual_cal_abs_max')
    if return_code == 0 and (not isinstance(residual, (int, float)) or not np.isfinite(residual)):
        raise RuntimeError('D 残差不是有限数值')
    return GravityResult(status=evidence.get('status'), windows=pooled.get('n_windows_qualifying_total'),
                         directions=(pooled.get('direction_coverage') or {}).get('n_distinct_gravity_directions'),
                         max_residual=residual, evidence=evidence)
