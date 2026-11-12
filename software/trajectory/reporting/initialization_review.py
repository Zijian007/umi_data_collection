"""用保留的视觉/IMU 数据对照初始化，不重建地图、不发布轨迹。

python -m software.trajectory.reporting.initialization_review --run-dir RUN \
    --calibration CAMERA_JSON --output REVIEW_JSON [--attempt 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation

from software import config, provenance
from software.camera.x3.frame import output_frame_rate
from software.trajectory.algorithms.fusion_alignment import AlignmentError, align_recording
from software.trajectory.algorithms.inertial import integrate_gyro_rotation
from software.trajectory.algorithms.fusion_data import load_inputs
from software.trajectory.calibration.assets import read_basalt


def rotation_residuals(inputs, parameters):
    """连续视觉三帧间隔与陀螺积分对照；只使用原始标定偏移，无时间搜索。"""
    obs = inputs.observations['front']
    if not obs:
        return dict(status='no_visual_observations')
    imu = inputs.imu_video_clock
    times = (inputs.video_times_s if inputs.video_times_s is not None else
             np.arange(max(obs)+1)/float(output_frame_rate(parameters.video.frame_step)))
    r_ci = Rotation.from_quat(inputs.camera['q_xyzw']).as_matrix().T
    static = np.linalg.norm(imu[:, 1:4], axis=1) < parameters.fusion.initial_static_gyro_rad_s
    bg = (imu[static, 1:4][:parameters.fusion.initial_bias_max_samples].mean(0)
          if static.sum() > parameters.fusion.initial_bias_min_samples else np.zeros(3))
    rows = []
    for start in sorted(obs):
        end = start+3
        if (not all(k in obs for k in range(start, end+1)) or times[start] < imu[0, 0] or times[end] > imu[-1, 0] or
                len({inputs.map_ids['front'].get(k) for k in range(start, end+1)}) != 1):
            continue
        visual = (obs[start][1]@r_ci).T@(obs[end][1]@r_ci)
        inertial = integrate_gyro_rotation(imu[:, 0], imu[:, 1:4], bg, times[start], times[end])
        rows.append(dict(start_frame=start, end_frame=end, start_s=float(times[start]), end_s=float(times[end]),
                         error_deg=float(np.degrees(Rotation.from_matrix(visual.T@inertial).magnitude()))))
    if not rows:
        return dict(status='insufficient_continuous_observations')
    errors = np.array([r['error_deg'] for r in rows])
    return dict(status='measured', intervals=len(rows), median_deg=float(np.median(errors)),
                rms_deg=float(np.sqrt(np.mean(errors**2))), p95_deg=float(np.percentile(errors, 95)),
                worst_intervals=sorted(rows, key=lambda r: r['error_deg'], reverse=True)[:10],
                actual_error='not_measured')


def review(inputs, parameters):
    result = dict(actual_error='not_measured', rotation=rotation_residuals(inputs, parameters))
    for name, enabled in [('baseline', False), ('refined', True)]:
        try:
            aligned = align_recording(inputs, parameters, refine_initialization=enabled, log=lambda *a, **k: None)
            result[name] = dict(status='accepted', **aligned.diagnostics)
        except AlignmentError as error:
            result[name] = dict(status='rejected', **error.diagnostics)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args(argv)
    if args.attempt < 1:
        parser.error('--attempt 必须为正整数')
    debug = args.run_dir/'debug'
    frames = debug/f'reconstruction/try_{args.attempt}/front/loc/frames.csv'
    params = debug/'settings/parameters.json'
    imu = debug/'dataset/mav0/imu0/data.csv'
    times = debug/'times.csv'
    paths = dict(frames=frames, parameters=params, imu=imu, calibration=args.calibration)
    if times.is_file():
        paths['times'] = times
    if args.output.resolve() in {p.resolve() for p in paths.values()}:
        parser.error('输出不能覆盖输入文件')
    parameters = config.decode_parameters('trajectory', json.loads(params.read_text()))
    if parameters.input.mode != 'handheld':
        parser.error('此复核入口目前仅支持 handheld；不推测环境镜头外参')
    inputs = load_inputs(front_frames=frames, rear_frames=frames, imu_csv=imu,
        camera_json=args.calibration, offset_ns=read_basalt(args.calibration)['cam_time_offset_ns'],
        rear_quaternion=[0, 0, 0, 1], rear_lever=[0, 0, 0], use_rear=False,
        times_csv=times if times.is_file() else None)
    result = review(inputs, parameters)
    result['inputs'] = {key: dict(path=str(path.resolve()), sha256=provenance.file_hash(path)) for key, path in paths.items()}
    result['code'] = provenance.code_identity('trajectory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'原初始化：{result["baseline"]["status"]}；实验细化：{result["refined"]["status"]}；报告：{args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
