"""Thread-safe logging, runtime paths and dependency diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from software import config
from software.environment import python_executable
from software.console import Console
from software.trajectory.calibration import assets as calibration_assets
from software.trajectory.models import ReconstructionSettings
from software.vision import runtime as visual_runtime
from software.vision.runtime import write_error_log

__all__ = [
    "console",
    "reconstruction_settings",
    "check",
    "write_error_log",
]

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
console = Console("轨迹")


def reconstruction_settings(environment: dict[str, str], *, camera_json: Path, front_mask: Path,
                            output: Path, parameters: config.TrajectoryParameters) -> ReconstructionSettings:
    """Validate engines, then combine explicit measurements and algorithm settings."""
    from dataclasses import asdict
    camera = calibration_assets.read_basalt(camera_json)
    rear = json.loads((HERE / 'assets/calib/rear_sample.json').read_text())
    visual = visual_runtime.visual_settings(environment, lens_settings={
        'front': calibration_assets.slam_settings_text(camera, parameters),
        'rear': calibration_assets.slam_settings_text(camera, parameters, rear=True)},
        front_mask=front_mask, output=output, parameters=parameters)
    parameters_file = output / 'parameters.json'
    parameters_file.write_text(json.dumps(asdict(parameters)), encoding='utf-8')
    return ReconstructionSettings(
        **asdict(visual), fusion_python=python_executable(environment),
        fusion_script=calibration_assets.FUSION_SCRIPT, parameters_file=parameters_file,
        camera_json=camera_json, time_offset_ns=str(camera['cam_time_offset_ns']),
        rear_q_ic=','.join(map(str, rear['q_imu_camera_xyzw'])),
        rear_lever=','.join(map(str, rear['lever_camera_imu_m'])),
    )


def check() -> int:
    ok = True

    def line(good: bool, what: str, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        console.write(what + (f"；{detail}" if detail else ""), state="正常" if good else "缺失")

    console.write("环境检查")
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
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10).stdout
        line("libx264" in enc, "FFmpeg 编码器 libx264")
    except Exception as error:
        line(False, "FFmpeg 编码器 libx264", repr(error))
    orb = Path(os.environ.get("ORB3_DIR", ""))
    exe = visual_runtime.orb_executable(orb)
    voc = Path(os.environ.get("ORB3_VOCAB", orb / "Vocabulary" / "ORBvoc.txt"))
    line(exe.is_file(), "x3_mono_slam", str(exe))
    line(voc.is_file(), "ORB 词袋", str(voc))
    map_exporter = visual_runtime.orb_executable(orb, 'umi_export_atlas')
    line(map_exporter.is_file(), '地图读取程序', str(map_exporter))
    if exe.is_file():
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = ":".join([str(orb / "lib"), str(orb / "Thirdparty/DBoW2/lib"), str(orb / "Thirdparty/g2o/lib"),
                                           os.environ.get("ORB3_EXTRA_LD", ""), environment.get("LD_LIBRARY_PATH", "")])
        result = subprocess.run([str(exe), "--help"], capture_output=True, text=True, env=environment, timeout=10)
        line(result.returncode == 0, "x3_mono_slam 可运行", "" if result.returncode == 0 else (result.stderr or result.stdout)[-300:])
    for path in (HERE / "assets" / "calib" / "calibration_r1440.json", HERE / "assets" / "calib" / "t_tcp_camera_nominal_cad.json",
              HERE / "assets" / "config" / "front_x3_r1440_720.yaml", HERE / "assets" / "config" / "front_hand_mask.png"):
        line(path.is_file(), "配置文件 " + str(path.relative_to(REPO)))
    try:
        from software.registry import list_devices
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
