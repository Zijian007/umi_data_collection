"""Process execution, dependency checks and runtime paths for calibration."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from concurrent.futures import CancelledError
import subprocess
from pathlib import Path

from software.console import Console
from software.calibration.models import RunPaths
from software.storage import locked

HERE = Path(__file__).resolve().parents[1]
TOOLS = HERE / "tools"
REPO = HERE.parents[1]
DEPS = Path(os.environ.get("UMI_DEPS", REPO / ".deps"))
BASALT = Path(os.environ.get("UMI_BASALT_DIR", DEPS / "basalt" / "umi-basalt-0.1.7-fsoffset-linux-x86_64"))
console = Console("标定")


@contextmanager
def timed(timings: dict, name: str, *, label: str | None = None):
    """Record wall time; optionally announce the subphase lifecycle in the terminal."""
    start = time.perf_counter()
    state = '结束'
    if label:
        console.write(label, state='开始')
    try:
        yield
    except BaseException as error:
        state = ('中断' if isinstance(error, KeyboardInterrupt) else
                 '取消' if isinstance(error, CancelledError) else '失败')
        raise
    finally:
        elapsed = round(time.perf_counter() - start, 3)
        timings[name] = dict(elapsed_s=elapsed)
        if label:
            console.write(f'{label}；耗时 {elapsed:.1f} 秒', state=state)


def run_logged(command: list, log: Path, environment: dict | None = None, cwd: Path | None = None, *,
               append: bool = False, cancelled=None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a" if append else "w", encoding="utf-8") as output:
        output.write("$ " + " ".join(map(str, command)) + "\n")
        output.flush()
        if cancelled is None:
            return subprocess.run(list(map(str, command)), stdout=output, stderr=subprocess.STDOUT, env=environment, cwd=cwd).returncode
        if cancelled.is_set():
            raise CancelledError('后台命令已取消')
        with subprocess.Popen(list(map(str, command)), stdout=output, stderr=subprocess.STDOUT,
                              env=environment, cwd=cwd) as process:
            try:
                while True:
                    if cancelled.is_set():
                        raise CancelledError('后台命令已取消')
                    try:
                        return process.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()


def pin_cores(cores: str | None) -> bool:
    if cores and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {c for part in cores.split(",") for lo, _, hi in [part.partition("-")] for c in range(int(lo), int(hi or lo) + 1)})
        except (ValueError, OSError):
            console.write(f"--cores {cores} 无效；请指定可用的 CPU，例如 --cores 0-7", state="失败")
            return False
    return True




def check_environment(quiet: bool = False, *, solve: bool = True) -> bool:
    ok = True

    def line(good: bool, what: str, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        if not quiet:
            console.write(what + (f"；{detail}" if detail else ""), state="正常" if good else "缺失")

    for module_name in ("numpy", "scipy", "cv2", "av") + (("rosbags",) if solve else ()):
        try:
            module = __import__(module_name)
            line(True, "Python 模块 " + module_name, getattr(module, "__version__", ""))
        except Exception as error:
            line(False, "Python 模块 " + module_name, repr(error))
    try:
        import cv2
        line(hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector"), "OpenCV ArUco（contrib）")
    except Exception as error:
        line(False, "OpenCV ArUco（contrib）", repr(error))
    if solve:
        from software.calibration.solve.basalt import basalt_command
    for tool in (("basalt_calibrate", "basalt_calibrate_imu") if solve else ()):
        try:
            result = subprocess.run(basalt_command(tool) + ["--help"], capture_output=True, text=True, timeout=10)
            good = result.returncode == 0 and (tool != "basalt_calibrate_imu" or "--fixed-scale-opt-offset" in result.stdout)
            line(good, tool, "" if good else (result.stderr or result.stdout)[-200:])
        except Exception as error:
            line(False, tool, repr(error))
    return ok


def prepare_output(paths: RunPaths, *, force: bool, resume: bool = False) -> None:
    import shutil

    if resume and force:
        raise ValueError('--resume 和 --force 不能同时使用')
    if paths.output.exists() and any(paths.output.iterdir()):
        if resume:
            if not paths.record.is_file():
                raise ValueError('此目录没有可恢复的阶段记录；请换一个 --out，或用 --force 重新运行')
            paths.logs.mkdir(parents=True, exist_ok=True)
            return
        if not force:
            raise ValueError(f"{paths.output} 已经有内容；用 --resume 继续、换一个 --out，或加 --force 覆盖。")
        shutil.rmtree(paths.output)
    paths.logs.mkdir(parents=True)


@contextmanager
def output_directory(paths: RunPaths, *, force: bool, resume: bool):
    # 锁放在输出旁，--force 清理输出时也不能删除活跃的锁。
    root = paths.output.expanduser().resolve()
    with locked(root.parent, name=f'.{root.name}.calibration.lock'):
        prepare_output(paths, force=force, resume=resume)
        yield


def prepare(cores: str | None, *, solve: bool = True) -> bool:
    """Check installed components, then apply the requested CPU affinity."""
    if not check_environment(quiet=True, solve=solve):
        console.write('标定依赖不完整；请运行 bash software/calibration/setup/install.sh', state='失败')
        return False
    if not pin_cores(cores):
        return False
    # 帧间已经并行；避免每个工作线程再启动一整组 OpenCV 线程。
    import cv2
    cv2.setNumThreads(1)
    return True


def check() -> int:
    console.write("环境检查")
    good = check_environment()
    console.write("环境检查通过" if good else "依赖不完整；请运行 bash software/calibration/setup/install.sh",
                  state="完成" if good else "失败")
    return 0 if good else 1
