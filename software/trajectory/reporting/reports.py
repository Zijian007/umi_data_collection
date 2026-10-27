"""文字报告与终端展示；不决定重试、复用或清理策略。"""

import json
from pathlib import Path

from software.trajectory.pipeline import runtime
from software.camera.geometry import describe_tcp_frame
from software.trajectory.calibration.selection import SelectedCalibration

STATUS_LABELS = {"ok": "完成", "skipped": "跳过", "failed": "失败", "planned": "预检通过"}
SELECTION_LABELS = {"registry": "设备档案", "UMI_TRAJ_CALIB": "环境变量覆盖"}


def write_result_report(output: Path, summary: dict) -> None:
    selected = summary.get('calibration_used', {})
    success = summary.get('status') == 'ok'
    lines = ['umi-traj 轨迹报告', '=' * 18,
             f"录像：{summary.get('video', '未知')}",
             f"状态：{STATUS_LABELS.get(summary.get('status'), '未知')}",
             f"设备：{selected.get('serial') or '未知'}",
             f"标定：{selected.get('id', '未知')}；来源：{SELECTION_LABELS.get(selected.get('selection'), selected.get('selection', '未知'))}"]
    configuration = summary.get('parameters', {}).get('configuration')
    if configuration:
        lines += [f"算法配置：{configuration['computation_hash'][:12]}；提醒规则：{configuration['evaluation_hash'][:12]}",
                  f"覆盖参数：{json.dumps(configuration['overrides'], ensure_ascii=False)}",
                  '完整参数与来源：details/run_settings.json']
    if success:
        def metric(value, spec):
            return format(value, spec) if isinstance(value, (int, float)) else '未记录'

        lines += [f"时长：{metric(summary.get('duration_s'), '.2f')} 秒；轨迹点数：{summary.get('rows', '未记录')}",
                  f"夹爪尖累计路径长度：{metric(summary.get('tcp_path_length_m'), '.3f')} m",
                  f"有地图位置约束的帧比例：{metric(summary.get('constraint_fraction', {}).get('full'), '.1%')}",
                  f"重建尝试：{len(summary.get('tries', []))} 次；采用第 {summary.get('chosen_try', '未知')} 次",
                  '', '主要结果：traj_tcp.txt（时间秒、位置米、姿态四元数 qx qy qz qw）',
                  '坐标：本段录像独立的世界坐标系，Z 轴朝上，原点和水平朝向由本次处理确定。',
                  'TCP 自身坐标：' + describe_tcp_frame(summary.get('tcp_frame', {})),
                  '实际误差：本次抽取未与独立真值对照，未测量；地图约束比例不代表真实精度。']
        if (output / 'trajectory.png').is_file():
            lines.append('预览：trajectory.png')
        else:
            lines.append('预览：未生成，查看下方提醒或诊断记录。')
    else:
        lines += [f"失败阶段：{summary.get('failed_stage', '未记录')}",
                  f"原因：{summary.get('reason', '详见诊断记录')}",
                  '本次输出不作为成功轨迹使用。']
    notes = summary.get('notes', [])
    if notes:
        lines += ['', '提醒'] + [f'  - {note}' for note in notes]
    if (output / 'front.mp4').is_file():
        size_px = configuration.get('values', {}).get('video', {}).get('size_px') if configuration else None
        dimensions = f'{size_px}×{size_px}，' if size_px else ''
        lines += ['', f'前镜头视频：front.mp4（{dimensions}保留鱼眼畸变）']
    if (output / 'details/map.json').is_file():
        lines.append('环境地图：details/map.json（前镜头稀疏点云与关键帧，umi-view 可查看）')
    lines += ['', '辅助轨迹、逐帧信息、完整配置和摘要：details/', '运行日志：debug/logs/']
    if summary.get('debug_retained'):
        lines.append('中间数据：debug/（已保留）')
    else:
        lines.append('中间数据：成功后清理；--keep-work 可保留完整数据')
    (output / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def show_batch(summaries: list[dict]) -> None:
    print()
    print("%-40s %-8s %s" % ("录像", "状态", "报告 / 原因"))
    for summary in summaries:
        detail = summary.get("reason") if summary["status"] == "failed" else ("只读预检，未创建输出" if summary["status"] == "planned" else str(Path(summary["out"]) / "report.txt"))
        print("%-40s %-8s %s" % (summary["video"], STATUS_LABELS[summary["status"]], detail))


def show_calibration(selected: SelectedCalibration, *, recording: str) -> None:
    source = SELECTION_LABELS.get(selected.selection, selected.selection)
    runtime.console.write(f"相机 {selected.camera_serial or '未知'}；标定 {selected.calibration_id}；"
                          f"夹爪 {selected.gripper_id}；来源：{source}（{selected.directory}）", recording=recording)


def show_recording(summary: dict) -> None:
    """执行结果与质量指标分别展示；准备阶段失败时尚未生成新报告。"""
    name = Path(summary["video"]).stem
    output = Path(summary["out"])
    state = STATUS_LABELS[summary["status"]]
    if summary["status"] == "ok":
        runtime.console.write(f"耗时 {summary['wall_s']:.0f} 秒；TCP 路径 {summary['tcp_path_length_m']:.2f} m；"
                              f"地图位置约束覆盖 {summary['constraint_fraction']['full']:.0%} 的帧",
                              recording=name, state=state)
        if len(summary.get("tries", [])) > 1:
            runtime.console.write(f"重建尝试 {len(summary['tries'])} 次；采用第 {summary['chosen_try']} 次", recording=name)
        for note in summary.get("notes", []):
            runtime.console.write(note, recording=name, state="提醒")
    elif summary["status"] == "skipped":
        runtime.console.write("输入、标定与处理配置一致，复用已有结果；--force 可重新处理", recording=name, state=state)
    else:
        runtime.console.write(f"阶段：{summary.get('failed_stage', '未知')}；原因：{summary.get('reason')}",
                              recording=name, state=state)
    if "wall_s" in summary or summary["status"] == "skipped":
        runtime.console.write(f"报告：{output / 'report.txt'}；日志：{output / 'debug/logs'}", recording=name)


