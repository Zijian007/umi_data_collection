"""Shared orchestration: prepare → solve A/B → measure D/C → evaluate → publish.

Measurements, execution checkpoints, policy and device activation have separate owners.
Rear calibration supplies only a trajectory generator and its pinned input identity.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
import json
import os
import shutil
import threading

from software import config
from software import provenance as artifact_identity
from software.calibration import acceptance, devices, reports, result_io
from software.calibration.algorithms import aprilgrid_detect, board_pnp
from software.calibration.models import CalibrationRequest, VerificationRequest, RunPaths, ExecutionOptions, Step, ROLE_STEPS, Decision
from software.calibration.pipeline import board_detection, inputs, recordings, runtime
from software.calibration.pipeline.steps import StageRunner
from software.calibration.solve import basalt, factory as factory_solver
from software.calibration.validation import factory, gravity, trajectory
from software.camera.calibration_source import validate as validate_source
from software.storage import atomic_file, locked
from software.trajectory.calibration import assets
from software.vision.runtime import write_error_log


def parameter_record(configuration, trajectory_configuration):
    return dict(configuration=configuration.snapshot(), trajectory_validation=trajectory_configuration.snapshot(),
        code=artifact_identity.code_identity('calibration'), engines=artifact_identity.engine_identity('calibration'),
        trajectory_code=artifact_identity.code_identity('trajectory'), trajectory_engines=artifact_identity.engine_identity('trajectory'))


def _document(paths, subject, sources, provenance, *, operation, resume):
    previous = result_io.read(paths.output) if resume else {}
    if previous and (previous['operation'] != operation or
            any(previous['subject'][k] != subject[k] for k in ('serial', 'lens'))):
        raise ValueError('恢复目录的操作、设备或镜头不一致，请选择新输出目录')
    if previous and previous['subject'].get('intrinsics_source', 'measured') != subject.get('intrinsics_source', 'measured'):
        raise ValueError('恢复目录的内参来源不一致，请为另一来源使用独立结果目录')
    doc = previous or dict(measurements={}, execution=dict(steps={}), evaluation=None, evaluation_history=[])
    if previous and any(doc[k] != value for k, value in (('inputs', sources), ('computation', provenance), ('subject', subject))):
        if doc.get('evaluation'):
            doc['evaluation_history'].append(doc['evaluation'])
        doc['evaluation'] = None
    doc.update(operation=operation, subject=subject, inputs=sources, computation=provenance, output=str(paths.output))
    doc['execution'].update(status='running', error=None, details={})
    result_io.write(paths.output, doc)
    return doc


def _prepare(role, video, execution, runner, identity, lens):
    directory = execution.paths.work / 'ds' / role
    def build():
        recordings.build(video, directory, with_images=role != 'D', lens=lens, workers=execution.workers)
        return dict(directory=str(directory.relative_to(execution.paths.output)), imu=recordings.imu_stats(directory))
    runner.run(ROLE_STEPS[role], dict(source=identity), build)
    prepared = runner.timings[ROLE_STEPS[role]]['status'] != 'reused'
    def ensure():
        # Work is disposable. Only a stage which actually needs it regenerates it.
        nonlocal prepared
        if not prepared:
            recordings.build(video, directory, with_images=role != 'D', lens=lens, workers=execution.workers)
            prepared = True
        return directory
    return ensure


def _detect_board(dataset, execution, board, role, source_identity, *, cancelled=None, background=False):
    if background:
        # Linux niceness is per thread; the detector pool inherits this lower priority.
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 15)
    step = {'A': board.intrinsics_step, 'C': board.validation_step}[role]
    timings = {}
    label = f'{role} 板检测'
    if background:
        label += '（后台，与 A 求解并行）' if role == 'A' else '（后台，与 A/B 并行）'
    with runtime.timed(timings, 'detect', label=label):
        found = board_detection.detect_cached(dataset, aprilgrid_detect.GridDetector,
            cache=execution.paths.work/f'{role}_board.npz', source_identity=source_identity,
            step=step, workers=execution.workers, up=board.detection_upscale, cancelled=cancelled)
    return found, dict(timings['detect'], background=background, workers=execution.workers)


def _coverage(directory, path, execution, grid, board, *, role, source_identity, detection=None):
    with runtime.timed(execution.timings, role + '.detect_wait'):
        found, timing = (detection.result() if detection is not None else
                        _detect_board(directory, execution, board, role, source_identity))
    execution.timings[role + '.detect'] = timing
    with runtime.timed(execution.timings, role + '.pnp'):
        return board_pnp.measure_board_coverage(found, board_pnp.read_kb4(path), grid, parameters=board)


def _validate(video, dataset, execution, runner, identity, parameter_file, bundle, *,
              grid, board, trajectory_configuration, generator=None, sample=False, detection=None):
    step = Step.SAMPLE if sample else Step.TRAJECTORY
    def measure():
        try:
            with runtime.timed(execution.timings, ('sample' if sample else 'candidate') + '.detect_wait'):
                found, timing = (detection.result() if detection is not None else
                    _detect_board(dataset, execution, board, 'C', identity['source']))
            execution.timings.setdefault('C.detect', timing)
            return trajectory.trajectory_check(video, found, execution, grid=grid, board=board,
                trajectory_configuration=trajectory_configuration, calibration_json=parameter_file,
                bundle=bundle, tag='sample' if sample else 'candidate', generator=generator)
        except Exception as error:
            if not sample:
                raise
            write_error_log(execution.paths.logs/'sample_error.log', error)
            return dict(status='comparison_failed', reason=str(error))
    return runner.run(step, identity, measure,
        reusable=lambda value: value.get('status') in ('ok', 'insufficient'))


def _finish(paths, doc, configuration, *, keep_work):
    parameters = (assets.read_basalt(paths.calibration) if doc['operation'] == 'calibrate'
                  and Step.CAMERA_IMU in doc['measurements'] and doc['subject'].get('parameter_sha256') else None)
    try:
        assessment = acceptance.evaluate(doc, configuration.values.checks, parameters,
                                        factory_parameters=configuration.values.factory)
    except Exception as error:
        doc['execution'].update(status='failed', error=dict(step='evaluation', reason=str(error)))
        write_error_log(paths.logs/'stage_error.log', error)
        assessment = dict(decision=Decision.INCOMPLETE.value, checks=[], policy_version=acceptance.POLICY_VERSION)
    assessment.update(configuration=configuration.snapshot())
    result_io.commit_evaluation(paths.output, doc, assessment)
    doc['activation'] = devices.activation_status(doc)
    result_io.write(paths.output, doc)
    reports.write_report(paths.output, doc)
    accepted = Decision(assessment['decision']).installable
    if doc['subject']['lens'] == 'rear':
        if accepted:
            result_io.publish_rear(paths.output, doc)
        else:
            (paths.output/'rear_calibration.json').unlink(missing_ok=True)
    runtime.console.write(f"{Decision(assessment['decision']).label}；报告：{paths.output/'report.txt'}", state='完成')
    if accepted and not keep_work:
        shutil.rmtree(paths.work, ignore_errors=True)
    return 0 if accepted else 1


def _compare_factory(source, comparison_metrics, execution, runner, identity, board, detection, *, lens='front', intrinsics_source='measured'):
    """Optional evidence runs automatically; optical failures do not fail calibration."""
    def measure():
        native = None
        try:
            native = factory.sources(source.recordings, source.camera_serial)
            found, timing = (detection.result() if detection is not None else
                _detect_board(comparison_metrics['C'], execution, board, 'C', identity['recordings']['C']))
            execution.timings.setdefault('C.detect', timing)
            return factory.compare(source.recordings.c, native, execution.paths.calibration,
                found, source.grid, board, execution.paths.output, lens=lens, intrinsics_source=intrinsics_source)
        except Exception as error:
            write_error_log(execution.paths.logs/'factory_error.log', error)
            runtime.console.write(f'工厂镜头对照未完成：{error}；原因已写入报告', state='提醒')
            return dict(status='comparison_failed', reason=str(error), source='insv_factory',
                        method_version=factory.METHOD_VERSION, native=native)
    return runner.run(Step.FACTORY, identity, measure,
        reusable=lambda result: factory.reusable(execution.paths.output, result))


def run(request: CalibrationRequest, *, trajectory_generator=None, backend_identity=None) -> int:
    resources = ExitStack()
    try:
        if request.resume and request.force:
            raise ValueError('--resume 和 --force 不能同时使用')
        if request.lens not in ('front', 'rear') or (request.lens == 'rear' and trajectory_generator is None):
            raise ValueError('环境镜头需要显式轨迹后端')
        validate_source(request.intrinsics_source)
        configuration = config.load('calibration', request.config)
        source = inputs.prepare_inputs(request)
        factory_native = None
        if request.intrinsics_source == 'factory':
            factory_native = factory.sources(source.recordings, source.camera_serial)
            from software.vision.capabilities import require_forward_kb4
            require_forward_kb4()
        trajectory_configuration = configuration.load_trajectory_validation(tag_size_m=source.grid['tagSize'])
        paths = RunPaths(request.output_directory)
        provenance = parameter_record(configuration, trajectory_configuration)
        provenance['intrinsics_source'] = request.intrinsics_source
        provenance['backend'] = backend_identity or dict(kind='front')
        sample = assets.read_basalt(assets.SAMPLE_CAMERA_JSON) if request.lens == 'front' else None
        sources = dict(grid=source.grid, sample_camera=sample,
            recordings={role: dict(path=str(video.resolve()), sha256=artifact_identity.file_hash(video)) for role, video in source.recordings.items()})
        if request.dry_run:
            print(json.dumps(dict(serial=source.camera_serial, lens=request.lens, output=str(paths.output), inputs=sources,
                                 computation=provenance, automatic_factory_comparison=(
                                     dict(views=[1440, 1920], lens=request.lens, affects_acceptance=False)
                                     if request.lens == 'front' or request.intrinsics_source == 'factory' else None)),
                             ensure_ascii=False, indent=2))
            return 0
        if not runtime.prepare(request.cores, solve=True):
            return 2
        workers = recordings.n_workers(request.workers)
        resources.enter_context(runtime.output_directory(paths, force=request.force, resume=request.resume))
        subject = dict(serial=source.camera_serial, lens=request.lens, parameter_file='calibration.json', parameter_sha256=None)
        subject['intrinsics_source'] = request.intrinsics_source
        if request.resume:
            previous = result_io.read(paths.output)
            subject['parameter_sha256'] = previous.get('subject', {}).get('parameter_sha256')
        doc = _document(paths, subject, sources, provenance, operation='calibrate', resume=request.resume)
        execution = ExecutionOptions(paths, request.cores, workers, doc['execution']['details'])
        runner = StageRunner(paths.output, doc)
    except (OSError, ValueError, RuntimeError) as error:
        resources.close()
        runtime.console.write(f'准备标定失败：{error}', state='失败')
        return 2
    try:
        try:
            identities = {role: recordings.preparation_identity(video, role != 'D', request.lens)
                          for role, video in source.recordings.items()}
            measurement_record = {role: _prepare(role, video, execution, runner, identities[role], request.lens)
                    for role, video in source.recordings.items()}
            detection = None
            available = len(os.sched_getaffinity(0))
            completed = doc['execution']['steps']
            # Only overlap a cold solve. A C-only rerun should use all requested detector workers.
            if (available >= 8 and workers >= 2 and Step.TRAJECTORY not in completed
                    and not all(step in completed for step in (Step.INTRINSICS, Step.CAMERA_IMU))):
                cancelled = threading.Event()
                pool = resources.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix='calibration-C'))
                resources.callback(cancelled.set)
                background = replace(execution, workers=min(workers, available // 4, 8), timings={})
                detection = pool.submit(_detect_board, measurement_record['C'], background, configuration.values.board,
                    'C', identities['C'], cancelled=cancelled, background=True)
            common = dict(code=provenance['code'], engines=provenance['engines'], grid=source.grid,
                          lens=request.lens, intrinsics_source=request.intrinsics_source)
            board = asdict(configuration.values.board)
            def board_identity(selected):
                return {k: v for k, v in board.items() if not k.endswith('_step') or k == selected}
            prepared_bag = None
            def solve_a():
                nonlocal prepared_bag
                directory = measurement_record['A']()
                if request.intrinsics_source == 'factory':
                    solved, origin = factory_solver.generate(source, request.lens, execution,
                        configuration.values.imu, configuration.values.factory, native=factory_native)
                    found, timing = _detect_board(directory, execution, configuration.values.board, 'A', identities['A'])
                    execution.timings['A.detect'] = timing
                    coverage = board_pnp.measure_board_coverage(found, solved.kb4, source.grid,
                        parameters=configuration.values.board)
                    validation = factory_solver.validate_images(found, solved.kb4, source.grid, configuration.values.factory)
                    return dict(result_io.intrinsics_measurement(replace(solved, board_metrics=coverage)),
                                intrinsics_source='factory', factory=origin, image_validation=validation)
                a_detection = None
                if available >= 16 and workers >= 4:
                    # A corners then B bag share one lane, alongside the existing C detector.
                    # Only start here: a reused A must not launch disposable work.
                    cancelled_ab = threading.Event()
                    prefetch = ThreadPoolExecutor(max_workers=1, thread_name_prefix='calibration-AB')
                    resources.callback(prefetch.shutdown, wait=True, cancel_futures=True)
                    resources.callback(cancelled_ab.set)
                    background_ab = replace(execution, workers=min(workers, available // 4, 8), timings={})
                    a_detection = prefetch.submit(_detect_board, directory, background_ab, configuration.values.board,
                        'A', identities['A'], cancelled=cancelled_ab, background=True)
                    prepared_bag = prefetch.submit(basalt.prepare_bag, measurement_record['B'](), background_ab,
                        name='camera_imu', log_name='camera_imu.log', cancelled=cancelled_ab)
                    resources.callback(prepared_bag.cancel)
                solved = basalt.estimate_intrinsics(directory, source.grid, execution)
                return result_io.intrinsics_measurement(replace(solved, board_metrics=
                    _coverage(directory, solved.calibration_json, execution, source.grid, configuration.values.board,
                        role='A', source_identity=identities['A'], detection=a_detection)))
            a_identity = dict(common, source=identities['A'], board=board_identity('intrinsics_step'))
            if request.intrinsics_source == 'factory':
                a_identity.update(recordings=identities, factory=config.computation_values('calibration',
                    {'factory': asdict(configuration.values.factory)})['factory'])
            a = runner.run(Step.INTRINSICS, a_identity, solve_a)
            def solve_b():
                initial = result_io.intrinsics_result(paths.output, a)
                with atomic_file(initial.calibration_json) as temporary_record:
                    temporary_record.write_text(a['calibration_snapshot'], encoding='utf-8')
                directory = measurement_record['B']()
                prepared = None
                if prepared_bag is not None:
                    with runtime.timed(execution.timings, 'camera_imu.bag_wait'):
                        prepared = prepared_bag.result()
                solved = basalt.estimate_camera_imu(directory, source.grid, initial, configuration.values.imu, execution,
                    prepared_bag=prepared)
                return dict(solver=solved.solver_metrics)
            b = runner.run(Step.CAMERA_IMU, dict(common, source=identities['B'], intrinsics=artifact_identity.canonical_hash(a),
                imu=asdict(configuration.values.imu)), solve_b, artifacts=(paths.calibration,))
            solved = result_io.camera_imu_result(paths.output, b)
            result_io.publish_candidate(paths.output, solved, source.camera_serial, request.lens, source.grid,
                sources['recordings'], intrinsics_source=request.intrinsics_source, factory=a.get('factory'))
            digest = artifact_identity.file_hash(paths.calibration)
            doc['subject']['parameter_sha256'] = digest
            if request.lens == 'front' or request.intrinsics_source == 'factory':
                _compare_factory(source, measurement_record, execution, runner,
                    dict(common, recordings=identities, calibration=digest,
                         board=board_identity('validation_step'), method=factory.METHOD_VERSION),
                    configuration.values.board, detection, lens=request.lens, intrinsics_source=request.intrinsics_source)
            runner.run(Step.GRAVITY, dict(common, source=identities['D'], calibration=digest,
                gravity=asdict(configuration.values.gravity)),
                lambda: asdict(gravity.validate_gravity(measurement_record['D'](), paths.calibration, configuration.values.gravity, execution)))
            c_identity = dict(common, source=identities['C'], calibration=digest, board=board_identity('validation_step'),
                trajectory=trajectory_configuration.computation_hash, backend=provenance['backend'],
                trajectory_code=provenance['trajectory_code'], trajectory_engines=provenance['trajectory_engines'],
                assets=trajectory.trajectory_assets() if request.lens == 'front' else backend_identity)
            _validate(source.recordings.c, measurement_record['C'], execution, runner, c_identity, paths.calibration, str(paths.output), grid=source.grid, board=configuration.values.board,
                trajectory_configuration=trajectory_configuration, generator=trajectory_generator, detection=detection)
            if request.compare_sample and request.lens == 'front':
                _validate(source.recordings.c, measurement_record['C'], execution, runner,
                    dict(c_identity, calibration=artifact_identity.file_hash(assets.SAMPLE_CAMERA_JSON)), assets.SAMPLE_CAMERA_JSON, 'factory', grid=source.grid, board=configuration.values.board,
                    trajectory_configuration=trajectory_configuration, sample=True, detection=detection)
            else:
                runner.invalidate(Step.SAMPLE)
            doc['execution'].update(status='complete', error=None)
        except BaseException as error:
            doc['execution'].update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
            if not doc['execution'].get('error'):
                doc['execution']['error'] = dict(step='workflow', reason=str(error))
            write_error_log(paths.logs/'stage_error.log', error)
            if not isinstance(error, Exception):
                _finish(paths, doc, configuration, keep_work=True)
                raise
        code = _finish(paths, doc, configuration, keep_work=request.keep_work)
    finally:
        resources.close()
    if code == 0 and request.install:
        if request.lens == 'rear':
            return devices.install_rear(paths.output/'rear_calibration.json')
        return devices.install_bundle(paths.output, force=False, camera_serial=source.camera_serial)
    return code


def verify(request: VerificationRequest) -> int:
    resources = ExitStack()
    try:
        if request.resume and request.force:
            raise ValueError('--resume 和 --force 不能同时使用')
        configuration = config.load('calibration', request.config)
        source = inputs.prepare_verification(request)
        trajectory_configuration = configuration.load_trajectory_validation(tag_size_m=source.grid['tagSize'])
        paths = RunPaths((request.output or source.video.parent/(source.video.stem+'_calibration_verification')).expanduser().resolve())
        parameter_file = Path(source.calibration['json']).resolve()
        subject = dict(serial=source.camera_serial, lens='front', parameter_file=str(parameter_file),
                       parameter_sha256=artifact_identity.file_hash(parameter_file), name=source.calibration_name)
        provenance = parameter_record(configuration, trajectory_configuration)
        sources = dict(grid=source.grid, recordings=dict(C=dict(path=str(source.video.resolve()), sha256=artifact_identity.file_hash(source.video))),
                       calibration=source.calibration)
        if request.dry_run:
            print(json.dumps(dict(subject=subject, inputs=sources, computation=provenance, output=str(paths.output)), ensure_ascii=False, indent=2))
            return 0
        if not runtime.prepare(request.cores, solve=False):
            return 2
        workers = recordings.n_workers(request.workers)
        resources.enter_context(runtime.output_directory(paths, force=request.force, resume=request.resume))
        doc = _document(paths, subject, sources, provenance, operation='verify', resume=request.resume)
        execution = ExecutionOptions(paths, request.cores, workers, doc['execution']['details'])
        runner = StageRunner(paths.output, doc)
    except (OSError, ValueError, RuntimeError) as error:
        resources.close()
        runtime.console.write(f'准备验证失败：{error}', state='失败')
        return 2
    try:
        try:
            identity = recordings.preparation_identity(source.video, True, 'front')
            dataset = _prepare('C', source.video, execution, runner, identity, 'front')
            _validate(source.video, dataset, execution, runner, dict(source=identity, calibration=subject['parameter_sha256'],
                board=asdict(configuration.values.board), grid=source.grid, trajectory=trajectory_configuration.computation_hash,
                code=provenance['code'], trajectory_code=provenance['trajectory_code'],
                engines=provenance['trajectory_engines'], assets=trajectory.trajectory_assets()), parameter_file, str(source.calibration['dir']), grid=source.grid, board=configuration.values.board,
                trajectory_configuration=trajectory_configuration)
            doc['execution'].update(status='complete', error=None)
        except BaseException as error:
            doc['execution'].update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
            if not doc['execution'].get('error'):
                doc['execution']['error'] = dict(step='workflow', reason=str(error))
            write_error_log(paths.logs/'stage_error.log', error)
            if not isinstance(error, Exception):
                _finish(paths, doc, configuration, keep_work=True)
                raise
        return _finish(paths, doc, configuration, keep_work=request.keep_work)
    finally:
        resources.close()


def recheck(output: Path, config_path: Path | None) -> int:
    output = Path(output).expanduser().resolve()
    try:
        configuration = config.load('calibration', config_path)
        with locked(output.parent, name=f'.{output.name}.calibration.lock'):
            doc = result_io.load_result(output, complete=True)
            trajectory_configuration = configuration.load_trajectory_validation(tag_size_m=doc['inputs']['grid']['tagSize'])
            provenance = doc['computation']
            previous_values = config.computation_values('calibration', provenance['configuration']['values'])
            if 'factory' not in previous_values and doc['subject'].get('intrinsics_source', 'measured') == 'measured':
                previous_values['factory'] = config.computation_values('calibration',
                    {'factory': asdict(configuration.values.factory)})['factory']
            if (artifact_identity.canonical_hash(previous_values) != configuration.computation_hash or
                provenance['trajectory_validation']['computation_hash'] != trajectory_configuration.computation_hash):
                raise ValueError('计算参数发生变化；recheck 只更新验收规则，请重新运行')
            return _finish(RunPaths(output), doc, configuration, keep_work=True)
    except (KeyError, TypeError, OSError, ValueError, RuntimeError) as error:
        runtime.console.write(f'无法重新验收：{error}', state='失败')
        return 2
