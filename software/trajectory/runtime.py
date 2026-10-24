"""Thread-safe logging, runtime paths and dependency diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import traceback
from pathlib import Path

from software import config
from software.console import Console
from software.trajectory import calibration_assets
from software.registry import list_devices
from software.trajectory.models import ReconstructionContext, ReconstructionError, ReconstructionSettings

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TOOLS = HERE / "tools"
console = Console("轨迹")


def write_error_log(path: Path, error: BaseException) -> None:
    path.write_text("".join(traceback.format_exception(error)), encoding="utf-8")


def reconstruction_settings(environment: dict[str, str], *, camera_json: Path, front_mask: Path,
                            output: Path, parameters: config.TrajectoryParameters) -> ReconstructionSettings:
    """Validate engines, then combine explicit measurements and algorithm settings."""
    from dataclasses import asdict
    orb, vocabulary = validate_engines(environment)
    camera = calibration_assets.read_basalt(camera_json)
    rear = json.loads((HERE / 'assets/calib/rear_sample.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    front_settings, rear_settings = output / 'front.yaml', output / 'rear.yaml'
    front_settings.write_text(calibration_assets.slam_settings_text(camera, parameters), encoding='utf-8')
    rear_settings.write_text(calibration_assets.slam_settings_text(camera, parameters, rear=True), encoding='utf-8')
    front_mask = calibration_assets.prepare_mask(front_mask, output / 'front_mask.png', parameters.video.size_px)
    parameters_file = output / 'parameters.json'
    parameters_file.write_text(json.dumps(asdict(parameters)), encoding='utf-8')
    return ReconstructionSettings(
        environment=dict(environment), tools_python=environment.get('KIWI_PY_TOOLS') or environment['KIWI_PY'],
        fusion_python=environment['KIWI_PY'], orb_directory=orb, vocabulary=vocabulary,
        front_settings=front_settings, rear_settings=rear_settings, front_mask=front_mask,
        fusion_script=calibration_assets.FACTORY_FUSER, parameters_file=parameters_file,
        camera_json=camera_json, time_offset_ns=str(camera['cam_time_offset_ns']),
        rear_q_ic=','.join(map(str, rear['q_imu_camera_xyzw'])),
        rear_lever=','.join(map(str, rear['lever_camera_imu_m'])),
    )


def run_logged(command: list[str], log: Path, *, stage: str, environment: dict[str, str],
               cwd: Path | None = None, stderr: Path | None = None, check: bool = True,
               failure_code: int | None = None) -> int:
    """所有外部进程走同一日志边界；保留原始 stdout/stderr 和阶段归因。"""
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open('w', encoding='utf-8') as stdout:
            if stderr is None:
                result = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT, env=environment, cwd=cwd)
            else:
                with stderr.open('w', encoding='utf-8') as errors:
                    result = subprocess.run(command, stdout=stdout, stderr=errors, env=environment, cwd=cwd)
    except OSError as error:
        raise ReconstructionError(stage, f'{error}；日志：{log}') from error
    if check and result.returncode:
        raise ReconstructionError(stage, f'外部程序退出码 {result.returncode}；日志：{log}',
                                  return_code=failure_code or result.returncode)
    return result.returncode


def engine_command(context: ReconstructionContext, command: list[str], *, timeout_s: int | None = None) -> list[str]:
    """沿用 nice、可选 taskset，以及 SLAM 的外部 timeout。"""
    prefix = ['nice', '-n', '10']
    if context.request.cores:
        prefix += ['taskset', '-c', context.request.cores]
    if timeout_s is not None:
        prefix += ['timeout', str(timeout_s)]
    return prefix + command


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or os.environ.get("UMI_COMMIT", "unknown")
    except Exception:
        return os.environ.get("UMI_COMMIT", "unknown")


def check() -> int:
    ok = True

    def line(good: bool, what: str, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        console.write(what + (f"；{detail}" if detail else ""), state="正常" if good else "缺失")

    console.write(f"环境检查；版本 {git_commit()}")
    for module_name in ("numpy", "scipy", "gtsam", "av", "matplotlib"):
        try:
            module = __import__(module_name)
            line(True, "Python 模块 " + module_name, getattr(module, "__version__", ""))
        except Exception as error:
            line(False, "Python 模块 " + module_name, repr(error))
    for tool in ("ffmpeg", "ffprobe"):
        executable = shutil.which(tool)
        line(executable is not None, tool, executable or "PATH 中未找到")
    try:
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
        line("libx264" in enc, "FFmpeg 编码器 libx264")
    except Exception as error:
        line(False, "FFmpeg 编码器 libx264", repr(error))
    orb = Path(os.environ.get("ORB3_DIR", ""))
    exe = orb / "Examples" / "Monocular-Inertial" / "x3_mono_slam"
    voc = Path(os.environ.get("ORB3_VOCAB", orb / "Vocabulary" / "ORBvoc.txt"))
    line(exe.is_file(), "x3_mono_slam", str(exe))
    line(voc.is_file(), "ORB 词袋", str(voc))
    if exe.is_file():
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = ":".join([str(orb / "lib"), str(orb / "Thirdparty/DBoW2/lib"), str(orb / "Thirdparty/g2o/lib"),
                                           os.environ.get("ORB3_EXTRA_LD", ""), environment.get("LD_LIBRARY_PATH", "")])
        result = subprocess.run([str(exe), "--help"], capture_output=True, text=True, env=environment, timeout=60)
        line(result.returncode == 0, "x3_mono_slam 可运行", "" if result.returncode == 0 else (result.stderr or result.stdout)[-300:])
    for path in (HERE / "assets" / "calib" / "calibration_r1440.json", HERE / "assets" / "calib" / "t_tcp_camera_nominal_cad.json",
              HERE / "assets" / "config" / "front_x3_r1440_720.yaml", HERE / "assets" / "config" / "front_hand_mask.png"):
        line(path.is_file(), "配置文件 " + str(path.relative_to(REPO)))
    try:
        rows = list_devices()
        line(bool(rows), "设备档案", "%d 台相机" % len(rows))
        for device in rows:
            line(bool(device.get("ok")), "  " + str(device.get("serial")),
                 ("状态 %s；标定 %s；夹爪 %s" % (device.get("status"), device.get("calib_id"), device.get("gripper")))
                 if device.get("ok") else device.get("error", ""))
    except Exception as error:
        line(False, "设备档案", repr(error))
    console.write("环境检查通过" if ok else "依赖不完整；请运行 install.sh，安装日志见 <deps>/logs/",
                  state="完成" if ok else "失败")
    return 0 if ok else 1




def validate_engines(environment: dict[str, str]) -> tuple[Path, Path]:
    """Check engine inputs read-only before a workflow can replace old output."""
    for name in ('KIWI_PY', 'ORB3_DIR'):
        if not environment.get(name):
            raise ReconstructionError('准备运行环境', f'缺少环境变量 {name}；请通过 ./umi-traj 启动')
    orb = Path(environment['ORB3_DIR']).resolve()
    vocabulary = Path(environment.get('ORB3_VOCAB') or orb / 'Vocabulary/ORBvoc.txt').resolve()
    for path in (orb / 'Examples/Monocular-Inertial/x3_mono_slam', vocabulary):
        if not path.is_file():
            raise ReconstructionError('准备运行环境', f'ORB-SLAM3 文件不存在：{path}')
    return orb, vocabulary
