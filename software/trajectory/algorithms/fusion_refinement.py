"""固定重力模长的独立链联合细化，以及错位窗口的预测残差复核。

视觉位姿保持不变；只优化尺度、重力方向、加速度零偏和节点速度。
不会修改相机标定，也不能把未通过原始初始化门限的链升级为有效。
"""

from __future__ import annotations

import numpy as np

from .inertial import preintegrate_samples
from scipy.optimize import least_squares


def _intervals(lens, times, imu, gyro_bias, keys):
    rows = []
    for start, end in zip(keys[:-1], keys[1:]):
        p0, r0 = lens['obs'][int(start)]
        p1, r1 = lens['obs'][int(end)]
        _, dv, dp, jv, jp, dt = preintegrate_samples(
            imu[:, 0], imu[:, 1:4], imu[:, 4:7], times[start], times[end], gyro_bias, np.zeros(3))
        ri = r0 @ lens['R_CI']
        rows.append((p1-p0, (r1-r0) @ lens['lever'], ri@dp, ri@dv, ri@jp, ri@jv, dt))
    return rows


def _prediction(rows, scale, gravity, bias):
    """用各区间位移反推起始速度，再检验与前一区间积分速度是否相符。"""
    residuals, previous = [], None
    for displacement, lever, dp, dv, jp, jv, dt in rows:
        velocity = (scale*displacement + lever - .5*gravity*dt**2 - dp - jp@bias)/dt
        if previous is not None:
            residuals.append((velocity-previous)*dt)
        previous = velocity+gravity*dt+dv+jv@bias
    return np.asarray(residuals).reshape(-1, 3)


def refine_chain(lens, times, imu, gyro_bias, parameters, chain, bias):
    """返回候选及可序列化诊断；保留原始自由重力、尺度不确定度门限。

    用 step 间隔的节点拟合，另外两个固定错位起点的视觉端点不参与拟合。
    它们复用同一 IMU，不是独立真值，且不用于挑选候选或调整标定。
    """
    if len(lens['chains']) != 1:
        raise ValueError('联合细化需要恰好一个连续跟踪段')
    start, end = lens['chains'][0]
    keys = np.arange(start, end+1, parameters.step)
    evidence = dict(status='insufficient_evidence', method='gravity_sphere_huber', k=[start, end])
    if len(keys)-1 < parameters.minimum_alignment_pairs:
        return None, evidence
    # 实验求解使用稠密雅可比；长链不能意外分配平方规模内存。
    if len(keys) > 128:
        evidence['status'] = 'size_limit'
        return None, evidence
    if (not np.all(np.diff(times[keys]) > 0) or times[keys[0]] < imu[0, 0] or
            times[keys[-1]] > imu[-1, 0]):
        raise ValueError('联合细化需要递增且由 IMU 完整覆盖的帧时间')
    scale, gravity, bias = float(chain['scale']), np.asarray(chain.get('g_map')), np.asarray(bias)
    if (scale <= 0 or not np.isfinite(scale) or gravity.shape != (3,) or
            not np.isfinite(gravity).all() or np.linalg.norm(gravity) < 1e-9 or
            not np.isfinite(bias).all() or np.max(abs(bias)) >= parameters.maximum_acc_bias_m_s2):
        evidence['status'] = 'invalid_initialization'
        return None, evidence
    rows = _intervals(lens, times, imu, gyro_bias, keys)
    unit = gravity / np.linalg.norm(gravity)
    axis = np.eye(3)[np.argmin(abs(unit))]
    u = np.cross(unit, axis)
    u /= np.linalg.norm(u)
    tangent = np.column_stack([u, np.cross(unit, u)])
    count = len(keys)
    dt_reference = float(np.median([row[-1] for row in rows]))
    velocities = []
    for displacement, lever, dp, dv, jp, jv, dt in rows:
        v = (scale*displacement+lever-.5*gravity*dt**2-dp-jp@bias)/dt
        velocities.append(v)
    velocities.append(v+gravity*dt+dv+jv@bias)
    x0 = np.r_[np.log(scale), 0., 0., bias, np.asarray(velocities).ravel()]

    def unpack(x):
        direction = unit+tangent@x[1:3]
        norm = np.linalg.norm(direction)
        g = parameters.gravity_m_s2*direction/norm
        dg = parameters.gravity_m_s2*(np.eye(3)/norm-np.outer(direction, direction)/norm**3)@tangent
        return np.exp(x[0]), g, x[3:6], x[6:].reshape(count, 3), dg

    def equations(x, jacobian=False):
        s, g, ba, vs, dg = unpack(x)
        residual = np.empty((len(rows), 6))
        jac = np.zeros((6*len(rows), len(x))) if jacobian else None
        for n, (displacement, lever, dp, dv, jp, jv, dt) in enumerate(rows):
            residual[n, :3] = s*displacement+lever-vs[n]*dt-.5*g*dt**2-dp-jp@ba
            residual[n, 3:] = (vs[n+1]-vs[n]-g*dt-dv-jv@ba)*dt_reference
            if jacobian:
                a = slice(6*n, 6*n+3)
                b = slice(6*n+3, 6*n+6)
                jac[a, 0] = s*displacement
                jac[a, 1:3] = -.5*dt**2*dg
                jac[a, 3:6] = -jp
                jac[a, 6+3*n:9+3*n] = -dt*np.eye(3)
                jac[b, 1:3] = -dt*dg*dt_reference
                jac[b, 3:6] = -jv*dt_reference
                jac[b, 6+3*n:9+3*n] = -np.eye(3)*dt_reference
                jac[b, 9+3*n:12+3*n] = np.eye(3)*dt_reference
        return jac if jacobian else residual.ravel()

    lower, upper = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    lower[0], upper[0] = np.log(scale)-np.log(10.), np.log(scale)+np.log(10.)
    lower[3:6], upper[3:6] = -parameters.maximum_acc_bias_m_s2, parameters.maximum_acc_bias_m_s2
    # 全零/匀速运动下，重力与零偏等量耦合；鲁棒损失不能补足缺失的信息。
    singular = np.linalg.svd(equations(x0, True), compute_uv=False)
    if len(singular) < len(x0) or singular[-1] <= singular[0]*1e-10:
        evidence['status'] = 'unobservable'
        return None, evidence
    result = least_squares(equations, x0, jac=lambda x: equations(x, True), bounds=(lower, upper),
                           loss='huber', f_scale=parameters.huber*parameters.position_sigma_m,
                           max_nfev=parameters.lm_iterations, x_scale='jac')
    s, g, ba, _, _ = unpack(result.x)
    evidence.update(converged=bool(result.success), evaluations=int(result.nfev),
                    condition=float(singular[0]/singular[-1]),
                    scale_before=scale, scale_after=float(s),
                    gravity_before=gravity.tolist(), gravity_after=g.tolist(),
                    bias_before=bias.tolist(), bias_after=ba.tolist(),
                    initial_residual_rms_m=float(np.sqrt(np.mean(equations(x0)**2))),
                    final_residual_rms_m=float(np.sqrt(np.mean(equations(result.x)**2))),
                    position_rms_after_m=float(np.sqrt(np.mean(equations(result.x).reshape(-1, 6)[:, :3]**2))))
    validation = []
    for offset in sorted({parameters.step//3, 2*parameters.step//3}-{0}):
        # 不把训练端点之外的外推误差混入比较。
        heldout = np.arange(start+offset, keys[-1]+1, parameters.step)
        if len(heldout) < 3:
            continue
        validation_rows = _intervals(lens, times, imu, gyro_bias, heldout)
        before = _prediction(validation_rows, scale, gravity, bias)
        after = _prediction(validation_rows, s, g, ba)
        validation.append(dict(offset_frames=offset, intervals=len(validation_rows),
            rms_before_m=float(np.sqrt(np.mean(before**2))), rms_after_m=float(np.sqrt(np.mean(after**2)))))
    evidence['validation'] = validation
    evidence['status'] = 'candidate'
    if not result.success or not np.isfinite(result.x).all():
        evidence['status'] = 'solver_failed'
    elif np.any(result.active_mask):
        evidence['status'] = 'bound_reached'
    elif len(validation) < 2:
        evidence['status'] = 'insufficient_validation'
    elif any(v['rms_after_m'] >= v['rms_before_m'] for v in validation):
        evidence['status'] = 'validation_not_improved'
    elif abs(s/scale-1) > parameters.maximum_relative_scale_std or np.linalg.norm(g-gravity) > parameters.gravity_tolerance_m_s2:
        evidence['status'] = 'initialization_disagreement'
    candidate = dict(scale=float(s), g_map=g, ba=ba, pos_rms=evidence['position_rms_after_m'])
    return candidate, evidence


def refine_initialization(lenses, times, imu, gyro_bias, parameters, estimates, stability):
    """只采用原始门限和采样检查均合格、双窗口预测均改善的单链候选。

    多条有效链共用零偏，尚未验证的独立链细化不能替换其联合结果。
    被拒绝的原始初始化也可生成诊断候选，但永远不会被此函数放行。
    """
    reports = {}
    for name, estimate in estimates.items():
        if estimate is None:
            continue
        lens = lenses[name]
        good_count = sum(c['good'] for c in estimate['chains'])
        original_bias = estimate['ba'].copy()
        reports[name] = []
        for index, chain in enumerate(estimate['chains']):
            isolated = dict(lens, chains=[lens['chains'][index]])
            candidate, report = refine_chain(isolated, times, imu, gyro_bias, parameters, chain, original_bias)
            reports[name].append(report)
            report['applied'] = False
            if candidate is None or report['status'] != 'candidate':
                continue
            if not chain['good']:
                report['status'] = 'baseline_rejected'
            elif good_count != 1:
                report['status'] = 'multiple_valid_chains'
            elif stability[name][index]['status'] != 'stable':
                report['status'] = 'baseline_stability_unverified'
            elif chain['rel_std']*chain['scale']/candidate['scale'] > parameters.maximum_relative_scale_std:
                report['status'] = 'scale_uncertainty'
            else:
                # 原始自由重力与尺度不确定度仍保留；归一化后的重力不可冒充独立验收。
                chain.update(rel_std=chain['rel_std']*chain['scale']/candidate['scale'],
                             scale=candidate['scale'], g_map=candidate['g_map'])
                estimate.update(ba=candidate['ba'], pos_rms=candidate['pos_rms'])
                report.update(status='accepted', applied=True)
    return reports
