"""Process execution, dependency checks and runtime paths for calibration."""

from __future__ import annotations

import os
import re
from functools import wraps
import subprocess
import traceback
from pathlib import Path

from software.console import Console
from software.calibration.models import RunPaths, StageError

HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
REPO = HERE.parents[1]
DEPS = Path(os.environ.get("UMI_DEPS", REPO / ".deps"))
BASALT = Path(os.environ.get("UMI_BASALT_DIR", DEPS / "basalt" / "umi-basalt-0.1.7-fsoffset-linux-x86_64"))
BOARD_JSON = REPO / "hardware" / "calibration_targets" / "aprilgrid_6x6_24mm_A4.json"
console = Console("标定")


def basalt_command(tool: str) -> list[str]:
    """The shipped Basalt build runs through its own dynamic loader + libraries, so it does not depend on the system glibc."""
    return [str(BASALT / "lib" / "ld-linux-x86-64.so.2"), "--library-path", str(BASALT / "lib"), str(BASALT / "bin" / tool)]


def run_logged(command: list, log: Path, environment: dict | None = None, cwd: Path | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as output:
        output.write("$ " + " ".join(map(str, command)) + "\n")
        output.flush()
        return subprocess.run(list(map(str, command)), stdout=output, stderr=subprocess.STDOUT, env=environment, cwd=cwd).returncode


def write_error_log(path: Path, error: BaseException) -> None:
    """Preserve the exception and its cause chain, including failures outside stage functions."""
    path.write_text(''.join(traceback.format_exception(error)), encoding='utf-8')


def pin_cores(cores: str | None) -> bool:
    if cores and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {c for part in cores.split(",") for lo, _, hi in [part.partition("-")] for c in range(int(lo), int(hi or lo) + 1)})
        except (ValueError, OSError):
            console.write(f"--cores {cores} 无效；请指定可用的 CPU，例如 --cores 0-7", state="失败")
            return False
    return True


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def check_environment(quiet: bool = False) -> bool:
    ok = True

    def line(good: bool, what: str, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        if not quiet:
            console.write(what + (f"；{detail}" if detail else ""), state="正常" if good else "缺失")

    for module_name in ("numpy", "scipy", "cv2", "av", "rosbags"):
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
    for tool in ("basalt_calibrate", "basalt_calibrate_imu"):
        try:
            result = subprocess.run(basalt_command(tool) + ["--help"], capture_output=True, text=True, timeout=60)
            good = result.returncode == 0 and (tool != "basalt_calibrate_imu" or "--fixed-scale-opt-offset" in result.stdout)
            line(good, tool, "" if good else (result.stderr or result.stdout)[-200:])
        except Exception as error:
            line(False, tool, repr(error))
    line(BOARD_JSON.is_file(), "标定板配置 " + str(BOARD_JSON.relative_to(REPO)))
    return ok


def last_reprojection(log: Path) -> dict:
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    reprojections = re.findall(r"num_points (\d+) .*?mean reprojection ([0-9.eE+-]+)", text)
    image_sizes = re.findall(r"Image size: (\d+)", text)
    num_points = int(reprojections[-1][0]) if reprojections else None
    return {"converged": text.count("Optimization Converged"), "num_points": num_points,
            "mean_reprojection_px": float(reprojections[-1][1]) if reprojections else None,
            "corners_per_frame": round(num_points / int(image_sizes[0]), 1) if num_points and image_sizes and int(image_sizes[0]) else None}


def prepare_output(paths: RunPaths, *, force: bool) -> None:
    import shutil

    if paths.output.exists() and any(paths.output.iterdir()):
        if not force:
            raise ValueError(f"{paths.output} 已经有内容；换一个 --out，或加 --force 覆盖。")
        shutil.rmtree(paths.output)
    paths.logs.mkdir(parents=True)


def stage_errors(name: str):
    """Attach stage identity to failures; the workflow writes the traceback once."""
    def decorate(operation):
        @wraps(operation)
        def execute(*args, **kwargs):
            try:
                return operation(*args, **kwargs)
            except StageError:
                raise
            except Exception as error:
                raise StageError(name, str(error)) from error
        return execute
    return decorate


def prepare(cores: str | None) -> bool:
    """Check installed components, then apply the requested CPU affinity."""
    if not check_environment(quiet=True):
        console.write('标定依赖不完整；请运行 bash software/calibration/setup/install.sh', state='失败')
        return False
    return pin_cores(cores)


def check() -> int:
    console.write(f"环境检查；版本 {git_commit()}")
    good = check_environment()
    console.write("环境检查通过" if good else "依赖不完整；请运行 bash software/calibration/setup/install.sh",
                  state="完成" if good else "失败")
    return 0 if good else 1
