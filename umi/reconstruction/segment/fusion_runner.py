"""独立视觉惯性融合的执行边界：函数调用，在一次性工作进程中运行。

GTSAM 的 Python 绑定在优化期间持有 GIL，线程内执行会让左右手融合串行。因此每次融合
都在一个 spawn 工作进程里调用 :func:`fuse`：双手可以并行，``--cores`` 与 nice 仍然生效，
求解器崩溃不会带垮整个批次，进程的 stdout/stderr 写入调用方指定的日志。输入、输出与
失败诊断仍通过文件交换，与重建阶段的缓存和报告保持一致。实时阶段消息通过单向管道传回。
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from umi.reconstruction.estimation.fusion.data import FusionOutput
    from umi.reconstruction.estimation.fusion.poster_initialization import PosterSeed

_worker_progress: Connection | None = None


@dataclass(frozen=True)
class FusionRequest:
    rear_frames: Path
    front_frames: Path
    imu_csv: Path
    offset_ns: int
    rear_quaternion_xyzw: tuple[float, ...]
    rear_lever_m: tuple[float, ...]
    output: Path
    calibration: Path
    parameters: Path
    times: Path | None = None
    poster: Path | None = None
    rear_calibration: Path | None = None
    refine_initialization: bool = False
    correct_poster_clock: bool = False
    poster_keyframe_maximum_dt_s: float = 0.0
    confirm_zupt_visually: bool = False
    early_poster_seed: PosterSeed | None = None


class FusionFailed(RuntimeError):
    """融合没有生成轨迹；结构化诊断（若有）已写入 ``output``。"""

    def __init__(self, message: str, *, diagnostic: dict | None = None, path: Path | None = None):
        super().__init__(message)
        self.diagnostic = diagnostic
        self.path = path


def fuse(request: FusionRequest, *, progress: Callable[[str], None] | None = None) -> FusionOutput:
    """在当前进程求解并写出结果，返回证据；失败时写诊断并抛出 :class:`FusionFailed`。"""
    from dataclasses import replace

    from umi.config import loader as config
    from umi.perception.poster.geometry import PosterGeometryError
    from umi.reconstruction.estimation.fusion.alignment import AlignmentError
    from umi.reconstruction.estimation.fusion.data import load_inputs, write_solution
    from umi.reconstruction.estimation.fusion.solver import solve

    try:
        parameters = config.decode_parameters('trajectory', json.loads(request.parameters.read_text()))
        inputs = load_inputs(front_frames=request.front_frames, rear_frames=request.rear_frames,
                             imu_csv=request.imu_csv, camera_json=request.calibration, offset_ns=request.offset_ns,
                             rear_quaternion=list(request.rear_quaternion_xyzw), rear_lever=list(request.rear_lever_m),
                             use_rear=parameters.input.mode == 'dual', times_csv=request.times)
        if request.poster is not None or request.rear_calibration is not None:
            from umi.reconstruction.segment.environment_assets import load_frozen_rear, load_poster
            if request.poster is not None:
                inputs = replace(inputs, poster=load_poster(request.poster))
            if request.rear_calibration is not None:
                inputs = replace(inputs, rear_camera=load_frozen_rear(request.rear_calibration))
        timing = None
        if request.correct_poster_clock:
            from umi.reconstruction.estimation.fusion.clock_check import correct_recording_clock
            if inputs.poster is None or parameters.input.mode != 'handheld' or not parameters.environment_tags.tag_fusion_enabled:
                raise ValueError('海报录像对时需要独立工作镜头轨迹和环境海报观测')
            inputs, timing = correct_recording_clock(inputs, parameters.environment_tags)
            _write_diagnostic(request.output / 'front_imu_timing.json', timing, allow_nan=False)
        elif request.poster is not None and parameters.input.mode == 'handheld' and parameters.environment_tags.tag_fusion_enabled:
            from umi.reconstruction.estimation.fusion.clock_check import front_imu_timing
            _write_diagnostic(request.output / 'front_imu_timing.json', front_imu_timing(inputs), allow_nan=False)
        seed = request.early_poster_seed
        if (seed is not None and seed.covers_recording and timing is not None
                and timing['decision']['applied_imu_timestamp_shift_ns']):
            from umi.reconstruction.estimation.fusion.poster_initialization import estimate_poster_seed

            print('poster_initialization_recompute: IMU timestamp shift confirmed by both cameras', flush=True)
            audit = {}
            try:
                seed = estimate_poster_seed(inputs, parameters, audit)
            except ValueError as error:
                print('poster_initialization_rejected', dict(audit, reason=str(error)), flush=True)
                seed = None
        result = solve(inputs, parameters, refine_initialization=request.refine_initialization,
                       poster_keyframe_maximum_dt_s=request.poster_keyframe_maximum_dt_s,
                       confirm_zupt_visually=request.confirm_zupt_visually, early_poster_seed=seed, progress=progress)
        if timing is not None:
            result = replace(result, evidence=dict(result.evidence, recording_clock=timing['decision']))
        output = write_solution(result, request.output)
    except AlignmentError as error:
        evidence = dict(status='failed', code='NO_USABLE_LENS', **error.diagnostics)
        diagnostic = _write_diagnostic(request.output / 'alignment_failure.json', evidence, allow_nan=False)
        raise FusionFailed(f'{error}；诊断：{diagnostic}', diagnostic=evidence, path=diagnostic) from None
    except PosterGeometryError as error:
        evidence = dict(status='failed', code='POSTER_GEOMETRY_FAILURE', message=str(error), metrics=error.metrics)
        diagnostic = _write_diagnostic(request.output / 'poster_failure.json', evidence, allow_nan=True)
        raise FusionFailed(f'{error}；诊断：{diagnostic}', diagnostic=evidence, path=diagnostic) from None
    except (OSError, ValueError, KeyError) as error:
        raise FusionFailed(str(error)) from None
    print(json.dumps(result.evidence), flush=True)
    return output


def _write_diagnostic(path: Path, document: dict, *, allow_nan: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=allow_nan), encoding='utf-8')
    return path


def execute(request: FusionRequest, *, log: Path, environment: dict[str, str], cores: str = '',
            progress: Callable[[str], None] | None = None) -> FusionOutput:
    """在独立工作进程中调用 :func:`fuse`；失败统一抛出 :class:`FusionFailed`。"""
    from umi.config import operational_settings as settings

    cpus = tuple(settings.available_cpus(cores)) if cores else ()
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text('', encoding='utf-8')
    spawn = multiprocessing.get_context('spawn')
    with ExitStack() as resources:
        reader = writer = None
        if progress is not None:
            reader, writer = spawn.Pipe(duplex=False)
            resources.enter_context(reader)
            resources.enter_context(writer)
        # Give the worker its pipe handle during spawn, alongside its log and
        # affinity settings. No descriptor needs to be transferred after startup.
        pool = resources.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=spawn,
            initializer=_initialize_worker, initargs=(settings.child_environment(environment), cpus, str(log), writer)))
        try:
            future = pool.submit(_fuse_in_worker, request)
            if progress is not None:
                writer.close()
                _relay_progress(reader, future, progress)
            return future.result()
        except BrokenProcessPool as error:
            raise FusionFailed(f'融合工作进程异常退出；日志：{log}') from error


def _fuse_in_worker(request: FusionRequest) -> FusionOutput:
    writer = _worker_progress
    if writer is None:
        return fuse(request)

    def report(message: str) -> None:
        try:
            writer.send(message)
        except (EOFError, OSError):
            pass

    with writer:
        return fuse(request, progress=report)


def _relay_progress(reader: Connection, future: Future, progress: Callable[[str], None]) -> None:
    while True:
        finished = future.done()
        try:
            if not reader.poll(0 if finished else 0.1):
                if finished:
                    return
                continue
            message = reader.recv()
        except (EOFError, OSError):
            return
        try:
            progress(message)
        except Exception:
            # A terminal sink can fail independently of the worker's result.
            pass


def _initialize_worker(environment: dict[str, str], cpus: tuple[int, ...], log: str,
                       progress_writer: Connection | None) -> None:
    global _worker_progress
    _worker_progress = progress_writer
    os.environ.clear()
    os.environ.update(environment)
    if cpus and hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, cpus)
    os.nice(10)
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(descriptor, 1)
    os.dup2(descriptor, 2)
    os.close(descriptor)
    sys.stdout = open(1, 'w', encoding='utf-8', buffering=1, closefd=False)
    sys.stderr = open(2, 'w', encoding='utf-8', buffering=1, closefd=False)
