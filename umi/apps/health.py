"""场景运行环境与设备档案检查。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from umi.calibration import devices as inventory
from umi.devices import selection as trajectory_calibration
from umi.devices.calibration_assets import SAMPLE_CAMERA_JSON
from umi.infra.paths import ORB_SETTINGS_ROOT, PROJECT_ROOT
from umi.infra.terminal import Console
from umi.perception import runtime as visual_runtime

REPO = PROJECT_ROOT
console = Console("轨迹")


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
    for path in (SAMPLE_CAMERA_JSON,
              ORB_SETTINGS_ROOT / "front_x3_r1440_720.yaml", trajectory_calibration.DEFAULT_FRONT_MASK):
        line(path.is_file(), "配置文件 " + str(path.relative_to(REPO)))
    try:
        devices = inventory.device_inventory()
        line(bool(devices), "设备档案", f"{len(devices)} 台相机")
        for device in devices:
            problems = device['errors'] + [version['error'] for version in device['versions'] if not version['ok']]
            line(device['ok'], '  ' + device['serial'], '; '.join(problems) if problems else
                 f"状态 {device['status']}；工作镜头默认标定 {device['front_default']}")
    except Exception as error:
        line(False, "设备档案", repr(error))
    console.write("环境检查通过" if ok else "环境或设备档案检查失败；请按上方缺失项修复，依赖安装日志见 <deps>/logs/",
                  state="完成" if ok else "失败")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(check())
