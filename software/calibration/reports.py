"""Human presentation of explicit measurements and evaluation; no execution or registry access."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import shlex

from software.calibration.models import Decision, Step

if TYPE_CHECKING:
    from software.calibration.inventory import InventoryReport


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


def factory_comparison_lines(result):
    lines = ['', 'INSV 工厂镜头参数对照（自动运行，仅诊断，不参与标定验收）']
    if not result or result.get('status') == 'comparison_failed':
        return lines + ['  未完成：' + ((result or {}).get('reason') or '尚未执行或没有可用参数')]
    if result.get('mapping_status') == 'fixed':
        lines.append('  固定对应：lens1 = 工作镜头，lens0 = 环境镜头；保留两组原始参数；实拍对照镜头见本次设备标题。')
    else:
        lines.append('  历史报告：两组工厂参数作为映射候选比较，尚未采用固定的工作/环境镜头对应关系。')
    lines += [
              '  1440×1440 为实测裁切范围；1920×1920 外围的 KB4 是外推，不代表已验证精度。',
              '  留出标签误差是图像重投影误差，不是轨迹或 TCP 真值误差。',
              '  原始参数、方法及数值：factory/comparison.json；射线差异图和角点叠加：factory/。']
    if not result['native']['consistent']:
        lines.append('  注意：A/B/C/D 工厂参数不同；本次使用 C 段参数，不混合其他录像。')
    if result.get('intrinsics_source') == 'factory':
        lines.append('  本次主参数使用工厂拟合 KB4 + 重新实测 B；当前选用模型不是实测 A。')
    if result.get('intrinsics_source', 'measured') == 'measured':
        lines.append('  本次对照未替代实测 A，正式参数仍来自实测标定。')
    if result.get('fits'):
        lines += ['  C 原片两镜头转换诊断：factory/kb4_fit.json；正式参数来源见本次内参来源与 factory/intrinsics.json（工厂模式）。',
                  '  拟合范围为 1440 裁切与名义半径 960 px 鱼眼圆的交集；不代表整个方形画面均有效。',
                  '  拟合误差是相对工厂模型的像素差，不是实拍精度；1920 只评估外推，不重新拟合。']
        for name, fit in result['fits'].items():
            label = '工作镜头' if fit['lens_role'] == 'working' else ('环境镜头（无实拍验收）' if result.get('compared_lens', 'working') == 'working' else '环境镜头')
            if fit['status'] == 'failed':
                lines.append(f"    {label} {name}：拟合失败，{fit['reason']}")
                continue
            validation = fit['views']['1440']
            pixels = validation['regions']['all']['pixels']
            if pixels['rms'] is not None:
                lines.append(f"    {label} {name}：独立网格 RMS {pixels['rms']:.4f} / P95 {pixels['p95']:.4f} / "
                             f"P99 {pixels['p99']:.4f} / 最大 {pixels['maximum']:.4f} px（1440 分辨率）")
            lines.append(f"      圆外样本 {validation['outside_nominal_disc']}；圆内 MEI 无效 {validation['invalid_mei']}；"
                         f"径向导数最小值 {validation['minimum_radial_derivative']:.6f}；"
                         f"OpenCV 反投影不一致 {validation['opencv']['inverse_mismatch_samples']} 个")
            boundary = validation['boundary']['pixels']
            if boundary['maximum'] is not None:
                lines.append(f"      边界探针最大误差 {boundary['maximum']:.4f} px；数值状态 {fit['status']}（非精度验收）")
            limits = validation['orb_slam3_inverse_limits']
            if limits['clipped_radius_samples'] or limits['backward_or_horizon_reference_samples']:
                lines.append(f"      未启用前向域修复的 ORB-SLAM3 反投影范围不兼容：{limits['clipped_radius_samples']} 个半径超出截断范围，"
                             f"{limits['backward_or_horizon_reference_samples']} 个参考方向非前向；不能直接启用全视野候选。")
    if result.get('skipped_frames'):
        lines.append(f"  实拍对照提醒：{result['skipped_frames']}")
    for size, view in result['views'].items():
        board = view['board']
        lines.append(f"  {size}×{size}：板观测采样 {board['sampled_frames']} 帧，共同有效 {board['common_frames']} 帧")
        for name, candidate in view['candidates'].items():
            rays = candidate['rays']
            lines.append(f"    {name}：共同有效射线 {rays['common_valid_points']}/{rays['nominal_disc_points']}；"
                         f"MEI 无效 {rays['invalid_mei_points']}，KB4 无效 {rays['invalid_kb4_points']}")
            for region, metrics in rays['regions'].items():
                label = '实测裁切内' if region == 'measured_crop' else '裁切外围（探索）'
                angle = metrics['angular_degrees']['p95']
                lines.append(f'      {label}射线方向差 P95：{angle:.4f}°' if angle is not None else f'      {label}：无共同有效射线')
        for name, metrics in board['models'].items():
            value = metrics['heldout_pixels']['rms']
            lines.append(f'    {name} 留出角点 RMS：{value:.3f} px' if value is not None else f'    {name}：留出验证证据不足')
            if size == '1920':
                outer = metrics['regions']['outside_measured_crop']
                lines.append(f"      外围留出角点：{outer['count']} 个；" + (
                    f"RMS {outer['rms']:.3f} px（探索）" if outer['count'] else '缺少实拍板证据'))
    return lines


def render(doc):
    subject, execution = doc['subject'], doc['execution']
    evaluation = doc.get('evaluation')
    decision = Decision(evaluation['decision']) if evaluation else Decision.INCOMPLETE
    lens_label = {'front': '工作镜头', 'rear': '环境镜头'}.get(subject['lens'], subject['lens'])
    lines = ['umi-calib 标定报告' if doc['operation'] == 'calibrate' else 'umi-calib 验证报告（只验证，不标定）',
             f"设备：{subject['serial']}；镜头：{lens_label}（{subject['lens']}）",
             f"执行：{execution['status']}；结论：{decision.label}",
             '启用状态（报告生成时）：' + doc.get('activation', {}).get('message', '未查询'),
             '完整证据、配置和恢复信息：run/record.json']
    lines.append('内参来源：' + ('工厂 MEI 拟合 KB4；A 独立验证、B 重新实测' if subject.get('intrinsics_source') == 'factory' else '实测 KB4'))
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
    if doc['operation'] == 'calibrate' and Step.FACTORY in measurements:
        lines += factory_comparison_lines(measurements.get(Step.FACTORY))
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


def show_inventory(inventory_report: InventoryReport) -> None:
    for device in inventory_report.get('devices', []):
        print(f"{device['serial']} [{device['status']}] front 默认：{device['front_default']}；rear 无全局默认")
        if device['calibration_owner'] != device['serial']:
            print(f"  借用来源：{device['calibration_owner']}")
        for version in device['versions']:
            detail = version.get('intrinsics_source', '?') if version['ok'] else version['error']
            print(f"  {version['lens']} {version['version']}：{'OK' if version['ok'] else 'FAIL'} / {detail}")
        if not any(version['lens'] == 'rear' for version in device['versions']):
            print('  rear：未登记')
        for error in device['errors']:
            print(f'  FAIL：{error}')
    if 'kit' in inventory_report:
        kit = inventory_report['kit']
        print(f"套件 {kit['id']}：{'OK' if kit['ok'] else 'FAIL'}（标定绑定检查）")
        for arm, row in kit['arms'].items():
            print(f"  {arm} / {row['serial']}")
            for lens in ('front', 'rear'):
                version = row[lens]
                detail = version.get('intrinsics_source', '?') if version['ok'] else version['error']
                print(f"    {lens} {version['version'] or '未绑定'}：{'OK' if version['ok'] else '缺失/无效'} / {detail}")
            for error in row['errors']:
                print(f'    FAIL：{error}')
        for error in kit['errors']:
            print(f'  FAIL：{error}')
        print('仅检查标定资产与版本绑定；原片匹配、夹爪安装资产及双镜头组合实测未检查。')
    for error in inventory_report.get('errors', []):
        print(f'FAIL：{error}')
