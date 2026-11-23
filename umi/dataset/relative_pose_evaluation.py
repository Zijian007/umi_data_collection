"""对冻结结果做固定夹具相对位姿评估；不改运行清单、验收或训练状态。"""
from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation

from umi.estimation.trajectory_metrics import relative_pose_errors
from umi.core.geometry import relative_pose_series, resample_pose_series, validate_transform
from umi.dataset.sampling import brackets, validate_times
from umi.infra import provenance
from umi.store.result_io import validate_assets
from umi.store.result_io import pose_rows, scene_reference, trajectory_path


def evaluate(path: Path, reference_path: Path) -> dict:
    run = path.expanduser().resolve()
    if not (run / 'episode.json').is_file():
        run = (run / 'results').resolve()
    manifest_path = run / 'episode.json'
    manifest_hash = provenance.file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('kind') not in ('umi_scene_episode', 'umi_poster_episode')
            or manifest.get('schema_version') != 4 or manifest.get('status') != 'ok'
            or manifest.get('arms') != dict(left='left', right='right')):
        raise ValueError('固定夹具评估需要同一坐标系下的完整双手成功结果')
    coordinate_record = scene_reference(manifest)
    validate_assets(run, manifest['assets'])
    reference_path = reference_path.expanduser().resolve()
    reference_hash = provenance.file_hash(reference_path)
    reference = json.loads(reference_path.read_text())
    fields = {'kind', 'schema_version', 'measurement_note', 'T_left_tcp_right_tcp', 'interval_s',
              'maximum_position_error_m', 'maximum_rotation_error_deg', 'maximum_interpolation_gap_s'}
    if (not isinstance(reference, dict) or set(reference) != fields
            or reference.get('kind') != 'umi_rigid_pair_reference' or reference.get('schema_version') != 1
            or not isinstance(reference.get('measurement_note'), str) or not reference['measurement_note'].strip()):
        raise ValueError('需要完整的 umi_rigid_pair_reference v1：位姿、区间、三项上限和实测依据 measurement_note')
    expected = validate_transform(reference['T_left_tcp_right_tcp'])
    interval = np.asarray(reference['interval_s'], float)
    limits = np.array([reference['maximum_position_error_m'], reference['maximum_rotation_error_deg'],
                       reference['maximum_interpolation_gap_s']], float)
    if (interval.shape != (2,) or not np.isfinite(interval).all() or interval[1] <= interval[0]
            or limits.shape != (3,) or not np.isfinite(limits).all() or np.any(limits <= 0)):
        raise ValueError('夹具区间和误差/插值上限必须为有效的有限数值')
    trajectories = {}
    for arm in ('left', 'right'):
        path = trajectory_path(run / arm)
        asset_name = str(path.relative_to(run))
        if asset_name not in manifest['assets']:
            raise ValueError('被评估的 TCP 轨迹未登记在资产哈希中')
        values = np.asarray(list(pose_rows(path)), float)
        if provenance.file_hash(path) != manifest['assets'][asset_name]:
            raise ValueError('评估期间 TCP 轨迹发生变化')
        if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all():
            raise ValueError('TCP 轨迹格式或数值无效')
        times = values[:, 0]
        validate_times(times, f'{arm} TCP 时间')
        if not np.allclose(np.linalg.norm(values[:, 4:], axis=1), 1., atol=1e-5):
            raise ValueError('TCP 四元数不是单位四元数')
        poses = np.tile(np.eye(4), (len(times), 1, 1))
        poses[:, :3, :3] = Rotation.from_quat(values[:, 4:]).as_matrix()
        poses[:, :3, 3] = values[:, 1:4]
        trajectories[arm] = times, poses
    start = max(trajectory[0][0] for trajectory in trajectories.values())
    end = min(trajectory[0][-1] for trajectory in trajectories.values())
    if interval[0] < start or interval[1] > end:
        raise ValueError('夹具评估区间必须完整落在双手共同时间内，禁止截短或外推')
    left_times = trajectories['left'][0]
    query = left_times[(left_times >= interval[0]) & (left_times <= interval[1])]
    if len(query) < 2:
        raise ValueError('夹具区间共同样本不足')
    poses, gaps = {}, []
    for arm, (times, matrices) in trajectories.items():
        a, b = brackets(times, query)
        gaps.extend(times[b]-times[a])
        gaps.extend(np.diff(np.r_[interval[0], times[(times > interval[0]) & (times < interval[1])], interval[1]]))
        poses[arm] = resample_pose_series(times, matrices, query)
    measured = relative_pose_series(poses['left'], poses['right'])
    metrics = relative_pose_errors(expected, measured)
    reasons = []
    if manifest.get('synchronization', {}).get('confirmed') is not True:
        reasons.append('synchronization_unconfirmed')
    if max(gaps) > limits[2]:
        reasons.append('interpolation_gap_exceeded')
    passed = (metrics['position_error_m']['maximum'] <= limits[0]
              and metrics['rotation_error_deg']['maximum'] <= limits[1])
    # 防止离线报告比较到评估期间被替换的输入。
    if (provenance.file_hash(manifest_path) != manifest_hash
            or provenance.file_hash(reference_path) != reference_hash):
        raise ValueError('评估期间结果或参考发生变化')
    return dict(kind='umi_relative_pose_evaluation', schema_version=1,
                status='not_ready' if reasons else 'passed' if passed else 'failed',
                not_ready_reasons=reasons, frame_id=coordinate_record['frame_id'],
                run=str(run), result_sha256=manifest_hash,
                reference=dict(path=str(reference_path), sha256=reference_hash, definition=reference),
                interval_s=interval.tolist(), sampled_interval_s=[float(query[0]), float(query[-1])],
                samples=len(query), maximum_interpolation_gap_s=float(max(gaps)),
                synchronization=manifest.get('synchronization'), **metrics)
