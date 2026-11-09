"""Human presentation of explicit measurements and evaluation; no execution or registry access."""
from __future__ import annotations
import shlex
from pathlib import Path
from software.calibration.models import Decision, Step


def trajectory_error_lines(result, label='候选标定'):
    lines = [f'C 段轨迹误差（{label}，相机位置）']
    if not result or result.get('status') != 'ok':
        return lines + ['  无法评估：' + ((result or {}).get('reason') or '尚无有效测量')]
    for key, label, digits in (
        ('ate_se3_rmse_mm', '位置误差 RMSE (mm)', 2), ('ate_se3_p95_mm', '位置误差 95 分位 (mm)', 2),
        ('scale_board_over_trajectory', '尺度比（板参考 / 轨迹）', 4),
        ('ate_sim3_rmse_mm', '允许缩放后的 RMSE (mm，仅诊断)', 2)):
        value = result.get(key)
        lines.append(f'  {label}：{value:.{digits}f}' if value is not None else f'  {label}：未记录')
    scale = result.get('scale_board_over_trajectory')
    if scale is not None and scale > 0:
        lines.append(f'  轨迹距离相对板参考的偏差：{(1/scale-1)*100:+.2f}%')
    lines += [f"  板位姿数：{result.get('board_frames', 0)}；实际匹配数：{result.get('n', 0)}",
              f"  板参考运动范围：{result.get('board_extent_m')} m",
              '  仅反映本次共同时间范围内的相机位置；不是姿态角或夹爪尖误差，不保证其他任务精度。']
    return lines


def trajectory_diagnostic_lines(result):
    lines = []
    for evidence in (result or {}).get('diagnostics', []):
        lines.append(f"  第 {evidence['attempt']} 轮诊断（{'采用' if evidence['selected'] else '未采用'}）")
        quality = evidence['quality']
        if quality.get('full_fraction') is not None:
            lines.append(f"    地图位置约束覆盖：{100*quality['full_fraction']:.1f}%；独立地图放置：{quality.get('placements', '未记录')}")
        if evidence['quality'].get('reason'):
            lines.append('    重建检查：' + evidence['quality']['reason'])
        for name, visual in evidence.get('visual_by_map', {}).items():
            value = visual.get('ate_sim3_rmse_mm')
            lines.append(f'    融合前视觉 {name}：Sim(3) RMSE {value:.2f} mm' if value is not None
                         else f"    融合前视觉 {name}：{visual.get('reason', '证据不足')}")
        for lens, alignment in evidence.get('alignment', {}).items():
            if alignment:
                lines.append(f"    {lens} 视觉–IMU 初始化位置残差：{alignment.get('pos_rms_mm')} mm")
        scale = evidence.get('scale_consistency') or {}
        ratio = scale.get('maximum_change_ratio')
        if ratio is not None:
            lines.append(f'    融合尺度最大变化：{ratio:.3f} 倍')
        if evidence.get('visual_unavailable') or evidence.get('fusion_unavailable'):
            lines.append('    部分中间诊断不可用，详见运行记录。')
    return lines


def render(doc):
    subject, execution = doc['subject'], doc['execution']
    evaluation = doc.get('evaluation')
    decision = Decision(evaluation['decision']) if evaluation else Decision.INCOMPLETE
    lines = ['umi-calib 标定报告' if doc['operation'] == 'calibrate' else 'umi-calib 验证报告（只验证，不标定）',
             f"设备：{subject['serial']}；镜头：{subject['lens']}",
             f"执行：{execution['status']}；结论：{decision.label}",
             '启用状态（报告生成时）：' + doc.get('activation', {}).get('message', '未查询'),
             '完整证据、配置和恢复信息：run/record.json']
    if execution.get('error'):
        error = execution['error']
        try:
            label = Step(error['step']).label
        except ValueError:
            label = error['step']
        lines += [f"失败阶段：{label}；原因：{error['reason']}"]
    if evaluation:
        checks = evaluation['checks']
        rank = dict(FAIL=0, UNAVAILABLE=1, WARN=2, OK=3, INFO=4)
        for row in sorted(checks, key=lambda row: (rank[row['result']], row['id'])):
            lines.append(f"  {row['result']} {row['check']}：{row['value']}（{row['limit']}）")
    measurements = {key: item['data'] for key, item in doc['measurements'].items()}
    lines += [''] + trajectory_error_lines(measurements.get(Step.TRAJECTORY),
        label='候选标定' if doc['operation'] == 'calibrate' else '已选标定')
    lines += trajectory_diagnostic_lines(measurements.get(Step.TRAJECTORY))
    if Step.SAMPLE in measurements:
        sample = measurements[Step.SAMPLE]
        lines.append('样例轨迹对照：已运行，仅供比较。' if sample.get('status') == 'ok' else '样例轨迹对照：未成功，不影响候选验收。')
    elif doc['operation'] == 'calibrate' and subject['lens'] == 'front':
        lines.append('样例轨迹对照：未运行（需要时加 --compare-sample）。')
    lines += ['', '阶段耗时']
    for key, timing in execution.get('timings', {}).items():
        lines.append(f"  {Step(key).label}：{timing['elapsed_s']:.3f} 秒（{timing['status']}）")
    # B 检测名称供已有运行记录展示；当前流程只独立检测 A/C。
    labels = {'camera_intrinsics.bag': 'A 求解数据准备（转 bag）', 'camera_intrinsics.solve': 'A 参数求解',
              'camera_imu.bag': 'B 求解数据准备（转 bag）', 'camera_imu.solve': 'B 参数求解',
              'camera_imu.bag_wait': 'B 阶段等待求解数据', 'A.detect_wait': 'A 阶段等待板检测',
              'A.detect': 'A 板检测/缓存读取', 'A.pnp': 'A 覆盖评估',
              'B.detect': 'B 板检测/缓存读取', 'B.pnp': 'B 覆盖评估', 'C.detect': 'C 板检测/缓存读取',
              'candidate.detect_wait': 'C 阶段等待板检测', 'sample.detect_wait': '样例等待板检测',
              'candidate.trajectory': 'C 轨迹重建', 'candidate.reference': 'C 标定板参考计算',
              'candidate.diagnostics': 'C 分层诊断', 'sample.trajectory': '样例轨迹重建',
              'sample.reference': '样例标定板参考计算', 'sample.diagnostics': '样例分层诊断'}
    if execution.get('details'):
        lines += ['', '本轮子步骤耗时（各项可能重叠，不可直接相加作为总耗时）']
        lines += [f"  {labels[key]}：{timing['elapsed_s']:.3f} 秒" for key, timing in execution['details'].items()]
    if doc['operation'] == 'calibrate' and decision.installable and subject['lens'] == 'front':
        lines.append('手动启用：./umi-calib install ' + shlex.quote(doc['output']))
    lines.append('WARN 表示诊断提醒；INCOMPLETE 表示证据不足或执行未完成，不能安装。')
    return '\n'.join(lines)+'\n'


def write_report(output, doc):
    (Path(output)/'report.txt').write_text(render(doc), encoding='utf-8')
