"""独立视觉惯性融合的执行边界：函数调用，在一次性工作进程中运行。

GTSAM 的 Python 绑定在优化期间持有 GIL，线程内执行会让左右手融合串行。因此每次融合
都在一个 spawn 工作进程里调用 :func:`fuse`：双手可以并行，``--cores`` 与 nice 仍然生效，
求解器崩溃不会带垮整个批次，进程的 stdout/stderr 写入调用方指定的日志。输入、输出与
失败诊断仍通过文件交换，与重建阶段的缓存和报告保持一致。
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path


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


class FusionFailed(RuntimeError):
    """融合没有生成轨迹；结构化诊断（若有）已写入 ``output``。"""


def fuse(request: FusionRequest) -> dict:
    """在当前进程求解并写出结果，返回证据；失败时写诊断并抛出 :class:`FusionFailed`。"""
    from dataclasses import replace

    from umi.config import loader as config
    from umi.estimation.fusion.alignment import AlignmentError
    from umi.estimation.fusion.data import load_inputs, write_solution
    from umi.estimation.fusion.solver import solve
    from umi.perception.poster.geometry import PosterGeometryError

    try:
        parameters = config.decode_parameters('trajectory', json.loads(request.parameters.read_text()))
        inputs = load_inputs(front_frames=request.front_frames, rear_frames=request.rear_frames,
                             imu_csv=request.imu_csv, camera_json=request.calibration, offset_ns=request.offset_ns,
                             rear_quaternion=list(request.rear_quaternion_xyzw), rear_lever=list(request.rear_lever_m),
                             use_rear=parameters.input.mode == 'dual', times_csv=request.times)
        if request.poster is not None or request.rear_calibration is not None:
            from umi.workflows.segment.environment_assets import load_frozen_rear, load_poster
            if request.poster is not None:
                inputs = replace(inputs, poster=load_poster(request.poster))
            if request.rear_calibration is not None:
                inputs = replace(inputs, rear_camera=load_frozen_rear(request.rear_calibration))
        result = solve(inputs, parameters, refine_initialization=request.refine_initialization)
        write_solution(result, request.output)
    except AlignmentError as error:
        diagnostic = _write_diagnostic(request.output / 'alignment_failure.json', dict(
            status='failed', code='NO_USABLE_LENS', **error.diagnostics), allow_nan=False)
        raise FusionFailed(f'{error}；诊断：{diagnostic}') from None
    except PosterGeometryError as error:
        diagnostic = _write_diagnostic(request.output / 'poster_failure.json', dict(
            status='failed', code='POSTER_GEOMETRY_FAILURE', message=str(error), metrics=error.metrics),
            allow_nan=True)
        raise FusionFailed(f'{error}；诊断：{diagnostic}') from None
    except (OSError, ValueError, KeyError) as error:
        raise FusionFailed(str(error)) from None
    print(json.dumps(result.evidence), flush=True)
    return result.evidence


def _write_diagnostic(path: Path, document: dict, *, allow_nan: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=allow_nan), encoding='utf-8')
    return path


def execute(request: FusionRequest, *, log: Path, environment: dict[str, str], cores: str = '') -> None:
    """在独立工作进程中调用 :func:`fuse`；失败统一抛出 :class:`FusionFailed`。"""
    from umi.config import operational_settings as settings

    cpus = tuple(settings.available_cpus(cores)) if cores else ()
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text('', encoding='utf-8')
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_initialize_worker,
                             initargs=(settings.child_environment(environment), cpus, str(log))) as pool:
        try:
            pool.submit(fuse, request).result()
        except BrokenProcessPool as error:
            raise FusionFailed(f'融合工作进程异常退出；日志：{log}') from error


def _initialize_worker(environment: dict[str, str], cpus: tuple[int, ...], log: str) -> None:
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
