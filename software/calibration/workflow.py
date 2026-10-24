"""标定流程的阅读入口：cli.main() 直接调用 run() 或 verify()。

全套标定 run：
    检查输入 → 提取四段录像 → A 内参 → B 相机–IMU → D 重力检查
             → 生成标定包 → C 轨迹验证 → 判定并写报告
单段验证 verify：
    选择现有标定 → 提取 C → 检测标定板 → 计算轨迹误差 → 判定并写报告

A/B 求解参数，D/C 验证结果。阶段函数返回测量，checks 将测量转为检查项，
本模块决定最终结论，reports 负责记录和展示。
两个入口均在生成报告后清理工作目录；中断时也通过 finally 执行清理。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from software import config, registry
from software.calibration import (
    camera_imu, checks, devices, inputs, intrinsics, recording_data, reports, runtime, validation,
)
from software.calibration.algorithms import aprilgrid_detect, board_pnp
from software.calibration.models import CalibrationRequest, RunPaths, StageContext, StageError, VerificationRequest
from software.trajectory import calibration_assets as trajectory_calibration


def run(request: CalibrationRequest) -> int:
    """从 A/B/C/D 生成并验证一份标定，默认由用户手动启用。

    返回码：0 = 验收通过，1 = 执行失败或验收未通过，2 = 环境或输入准备失败。
    显式指定 --install 时，通过后返回设备安装操作的退出码。
    """
    # 所有配置校验先于依赖检查、CPU 设置和 --force 清理。
    try:
        configuration = config.load('calibration', request.config)
        trajectory_configuration = configuration.load_trajectory_validation()
        if not request.dry_run and not runtime.prepare(request.cores):
            return 2
        calibration_inputs = inputs.prepare_inputs(request)
        output = request.output or (request.folder or calibration_inputs.recordings.a.parent) / 'umi_calib_out'
        paths = RunPaths(output)
        # 样例参数用于 A/B 的健全性检查；C 段的位置参考来自标定板。
        sample_parameters = trajectory_calibration.read_basalt(trajectory_calibration.FACTORY_JSON)
        provenance = parameter_record(configuration, trajectory_configuration)
        provenance['options'] = dict(compare_sample=request.compare_sample)
        provenance['measurements'] = dict(grid=calibration_inputs.grid, sample_camera=sample_parameters,
                                          recordings={name: dict(path=str(path.resolve()), bytes=path.stat().st_size,
                                                                  mtime_ns=path.stat().st_mtime_ns)
                                                      for name, path in calibration_inputs.recordings.items()})
        provenance['computation_hash'] = computation_hash(provenance)
        if request.dry_run:
            print(json.dumps(dict(camera_serial=calibration_inputs.camera_serial, output=str(output), **provenance),
                             indent=2, ensure_ascii=False))
            return 0
        # 四段输入及序列号确认无误后，才允许 --force 清理旧输出。
        runtime.prepare_output(paths, force=request.force)
        paths.board.write_text(json.dumps(calibration_inputs.grid, indent=1), encoding='utf-8')
        # context 集中传递路径、板尺寸、CPU 和检测并发数。
        context = StageContext(paths, calibration_inputs.grid, request.cores, recording_data.n_workers(),
                               configuration.values, trajectory_configuration)
        report = reports.CalibrationReport(calibration_inputs, request, sample_parameters)
        report.payload["parameters"] = provenance
    except (registry.RegistryError, OSError, ValueError) as error:
        runtime.console.write(f"准备标定失败：{error}", state="失败")
        return 2

    # 主流程持有验收状态。检查项即使出现 FAIL，也继续收集后续阶段证据；
    # 阶段执行抛出异常时才停止，最后统一决定结论。
    check_results = []
    stage_error = None
    stage = '读取录像'
    # 内层捕获执行异常并记录失败；外层 finally 保证报告写入、安装或中断后都能收尾。
    try:
        try:
            # 1. 原始 INSV 转阶段数据：A/B/C 提取时间戳、IMU 和图像，D 不解码图像。
            datasets = recording_data.extract_recordings(calibration_inputs.recordings, context)
            report.record_data(datasets)

            # 2. 用 A 求镜头内参，并检查板覆盖范围与重投影误差。
            stage = 'A 内参'
            intrinsics_result = intrinsics.solve_intrinsics(datasets.segments.a, context)
            report.record_intrinsics(intrinsics_result)
            check_results.extend(checks.assess_intrinsics(intrinsics_result, sample_parameters, configuration.values.checks))

            # 3. 使用 A 的内参，将 B 的视觉运动与 IMU 对齐，求外参和时间差。
            stage = 'B 相机–IMU'
            camera_imu_result = camera_imu.solve_camera_imu(datasets.segments.b, intrinsics_result, context)
            report.record_camera_imu(camera_imu_result)
            check_results.extend(checks.assess_camera_imu(camera_imu_result, datasets.imu_metrics['B'], sample_parameters, configuration.values.checks))

            # 4. 用 D 的多朝向静止数据检查重力一致性，此步骤只检查、不改参数。
            stage = 'D 重力检查'
            gravity_result = validation.validate_gravity(datasets.segments.d, camera_imu_result.calibration_json, context)
            report.record_gravity(gravity_result)
            check_results.extend(checks.assess_gravity(gravity_result, configuration.values.checks))

            # 5. 封装 umi-traj 读取的标定包，让下一步 C 使用真实的轨迹处理入口。
            stage = '生成标定包'
            bundle = camera_imu.build_bundle(camera_imu_result, context, recordings=calibration_inputs.recordings,
                                             tag_size_mm=request.tag_size_mm)
            report.record_bundle(bundle)

            # 6. 用独立的 C 录像对照标定板验证新标定；样例轨迹对照由选项控制。
            stage = 'C 轨迹验证'
            trajectory_result = validation.validate_trajectory(calibration_inputs.recordings.c, datasets.segments.c,
                                                               bundle, context, compare_sample=request.compare_sample)
            report.record_trajectory(trajectory_result)
            check_results.extend(checks.assess_trajectory(trajectory_result, configuration.values.checks))
        except Exception as error:
            # 保留阶段函数给出的详细归因（如“读取 B 段”）；其他异常用当前 stage 归因。
            stage_error = error if isinstance(error, StageError) else StageError(stage, str(error))
            report.record_failure(stage_error)
            runtime.write_error_log(paths.logs / 'stage_error.log', error)

        # 执行异常直接判失败；正常执行结束后，由全部检查项决定 PASS / 提醒 / FAIL。
        conclusion = 'FAIL' if stage_error else checks.verdict(check_results)
        reason = str(stage_error) if stage_error else '；'.join(check['check'] for check in check_results if check['result'] == 'FAIL')
        report.finish(paths.output, conclusion, check_results=check_results, reason=reason)
        if not conclusion.startswith('PASS'):
            return 1
        # 默认只展示启用状态；只有显式 --install 且验收通过才写入设备档案。
        if request.install:
            return devices.install_bundle(paths.output, force=False, camera_serial=calibration_inputs.camera_serial, gripper=request.gripper)
        report.announce_activation(paths.output)
        return 0
    finally:
        # 只清理可再生成的 work；标定包、报告和日志继续保留。
        if not request.keep_work:
            shutil.rmtree(paths.work, ignore_errors=True)


def verify(request: VerificationRequest) -> int:
    """用一段 C 检查现有标定的轨迹误差，不执行 A/B 求解或设备安装。

    返回码同 run()：0 = 验证通过，1 = 验证失败，2 = 环境或输入准备失败。
    """
    try:
        configuration = config.load("calibration", request.config)
        trajectory_configuration = configuration.load_trajectory_validation()
        if not request.dry_run and not runtime.prepare(request.cores):
            return 2
        # 先确定录像、设备身份和使用的标定，再准备输出目录，保护已有结果。
        verification_inputs = inputs.prepare_verification(request)
        video = verification_inputs.video
        selected_calibration = verification_inputs.calibration
        calibration_name = verification_inputs.calibration_name
        paths = RunPaths(request.output or video.parent / (video.stem + '_umi_calib_verify'))
        provenance = parameter_record(configuration, trajectory_configuration)
        provenance['measurements'] = dict(grid=verification_inputs.grid, calibration=selected_calibration,
                                          recording=dict(path=str(video.resolve()), bytes=video.stat().st_size,
                                                         mtime_ns=video.stat().st_mtime_ns))
        provenance['computation_hash'] = computation_hash(provenance)
        if request.dry_run:
            print(json.dumps(dict(camera_serial=verification_inputs.camera_serial, output=str(paths.output), **provenance),
                             indent=2, ensure_ascii=False))
            return 0
        runtime.prepare_output(paths, force=request.force)
        context = StageContext(paths, verification_inputs.grid, request.cores, recording_data.n_workers(),
                               configuration.values, trajectory_configuration)
        report = reports.VerificationReport(verification_inputs, request)
        report.payload["parameters"] = provenance
    except (registry.RegistryError, OSError, ValueError) as error:
        runtime.console.write(f"准备标定失败：{error}", state="失败")
        return 2

    runtime.console.write(f'用 {calibration_name} 验证 {video.name}（序列号 {verification_inputs.camera_serial}）')
    # stage 随执行步骤更新，保证读取、板检测和轨迹计算的异常分别归因。
    check_results = []
    stage_error = None
    stage = '读取 C 段'
    try:
        try:
            # 1. 提取与画面对应的时间戳和 IMU，准备板检测所需图像。
            runtime.console.write('读取 C 段时间戳、IMU 和画面', step=(1, 3))
            recording_data.build(video, paths.work / 'ds', with_images=True)

            # 2. 检测板角点，下一步据此计算独立的参考相机位置。
            stage = 'C 标定板检测'
            runtime.console.write('C 段：检测标定板', step=(2, 3))
            detections = board_pnp.detect_all(paths.work / 'ds', aprilgrid_detect.GridDetector,
                                             step=configuration.values.board.validation_step, workers=context.workers,
                                             up=configuration.values.board.detection_upscale)

            # 3. 用选中的标定跑轨迹，将相机位置与板参考对齐后计算误差。
            stage = 'C 轨迹验证'
            runtime.console.write('C 段：抽取轨迹，对照标定板计算误差', step=(3, 3))
            bundle = 'factory' if selected_calibration['source'] == 'factory' else selected_calibration['dir']
            trajectory_result = validation.trajectory_check(video, detections, context,
                                                             calibration_json=Path(selected_calibration['json']),
                                                             bundle=bundle, tag='current')
            report.record_trajectory(trajectory_result)
            check_results.extend(checks.assess_verification(trajectory_result, calibration_name, configuration.values.checks))
        except Exception as error:
            # 所有阶段采用同一收尾方式，同时保留真实原因和完整异常堆栈。
            stage_error = error if isinstance(error, StageError) else StageError(stage, str(error))
            report.record_failure(stage_error)
            runtime.write_error_log(paths.logs / 'stage_error.log', error)
            # 失败也生成一条验收项，报告中同时显示执行原因与 FAIL 检查项。
            failed_measurement = {'status': 'failed', 'reason': str(stage_error)}
            check_results.extend(checks.assess_verification(failed_measurement, calibration_name, configuration.values.checks))

        # workflow 决定结论，报告对象接收检查项并负责格式化、计时和落盘。
        conclusion = 'FAIL' if stage_error else checks.verdict(check_results)
        report.finish(paths.output, conclusion, check_results=check_results)
        return 0 if conclusion.startswith('PASS') else 1
    finally:
        # 正常完成或已记录失败时，报告先落盘；中断时同样按配置清理工作目录。
        if not request.keep_work:
            shutil.rmtree(paths.work, ignore_errors=True)


def parameter_record(configuration: config.Configuration, trajectory_configuration: config.Configuration) -> dict:
    """Record both the calibration stages and the C trajectory engine configuration."""
    record = dict(configuration=configuration.snapshot(), trajectory_validation=trajectory_configuration.snapshot(),
                  code=config.code_identity('calibration'), engines=config.engine_identity('calibration'),
                  trajectory_code=config.code_identity('trajectory'), trajectory_engines=config.engine_identity('trajectory'))
    return record



def recheck(output: Path, config_path: Path | None) -> int:
    """Re-evaluate complete stored evidence without running solvers or installing."""
    try:
        configuration = config.load('calibration', config_path)
        trajectory_configuration = configuration.load_trajectory_validation()
        report = reports.load_report(output)
        provenance = report['parameters']
        previous = provenance['configuration']
        if (previous['computation_hash'] != configuration.computation_hash or
                provenance['trajectory_validation']['computation_hash'] != trajectory_configuration.computation_hash):
            raise ValueError('算法参数发生变化，需要重新运行标定或 verify；recheck 只更新验收规则')
        check_results = checks.reassess_report(report, configuration.values.checks)
    except (KeyError, TypeError, OSError, ValueError) as error:
        runtime.console.write(f'无法重新验收：{error}；旧报告保持不变', state='失败')
        return 2
    report.setdefault('evaluation_history', []).append(dict(evaluation_hash=previous['evaluation_hash'], verdict=report['verdict']))
    provenance['configuration'] = configuration.snapshot()
    report.update(checks=check_results, verdict=checks.verdict(check_results))
    if report['kind'] == 'umi_calib_verify_report':
        report['advice'] = reports.verification_advice(report['verdict'], True, report['serial'])
    else:
        report['why'] = '；'.join(check['check'] for check in check_results if check['result'] == 'FAIL')
    reports.write_report(output, report, verification=report['kind'] == 'umi_calib_verify_report')
    runtime.console.write(f"已重新验收：{report['verdict']}；报告：{output / 'report.txt'}", state='完成')
    return 0 if report['verdict'].startswith('PASS') else 1


def computation_hash(record: dict) -> str:
    return config.canonical_hash(dict(configuration=record['configuration']['computation_hash'],
                                     trajectory_configuration=record['trajectory_validation']['computation_hash'], options=record.get('options', {}),
                                     **{key: record[key] for key in ('code', 'engines', 'trajectory_code', 'trajectory_engines', 'measurements')}))
