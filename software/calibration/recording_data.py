"""One X3 360 4K60 recording -> EuRoC folder for calibration (timestamps, IMU and, optionally, the front-lens frames).

  * timestamps + IMU: software.camera.x3.extraction;
  * frames: every 2nd decoded frame, front lens = the right-half crop from software.camera.x3.frame, written as <t_ns>.png.
    The grey image is the decoder's luma (Y) plane, read directly. frame.to_ndarray(format="gray") rescales on newer
    libswscale, so the pixels would change with the PyAV version.
    PNG encoding is the slow part (about 0.12 s a frame on one core), so the files are written by a few threads; the
    bytes of each file do not depend on that.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from software.calibration.models import RecordingData, SegmentPaths, StageContext, StageError
from software.calibration import runtime

import cv2
import numpy as np

from software.camera.x3.frame import SRC_H, SRC_W, STEP, T0_S, crop_xywh
from software.camera.x3 import extraction

CROP_X0, CROP_Y0, CROP_W, CROP_H = crop_xywh("right")
LUMA_FORMATS = ("yuvj420p", "yuv420p")       # 8-bit planar, plane 0 = luma


def _write_png(path: Path, g: np.ndarray) -> None:
    if not cv2.imwrite(str(path), g, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"无法写入图像 {path}；请检查磁盘空间和目录权限")


def n_workers() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 2
    return max(1, min(8, n - 1))


def build(insv: Path, out: Path, with_images: bool = True) -> None:
    """Writes <out>/mav0/{cam0/data.csv, imu0/data.csv[, cam0/data/*.png]} and preserves the extraction manifest."""
    insv, out = Path(insv), Path(out)
    extraction.extract(insv, out, with_sha256=True)           # raises software.camera.x3.extraction.Refused with the reason
    if not with_images:
        return
    names = [l.split(",")[1].strip() for l in (out / "mav0" / "cam0" / "data.csv").read_text(encoding="utf-8").splitlines()
             if l and not l.startswith("#")]
    img_dir = out / "mav0" / "cam0" / "data"
    img_dir.mkdir(parents=True, exist_ok=True)
    import av

    n_out, nw, pending = 0, n_workers(), []
    with ThreadPoolExecutor(nw) as pool, av.open(str(insv)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        tb = float(st.time_base)
        if st.codec_context.width != SRC_W or st.codec_context.height != SRC_H:
            raise RuntimeError(f"{insv.name} 的分辨率为 {st.codec_context.width}×{st.codec_context.height}，需要 {SRC_W}×{SRC_H} 的 4K60 全景录像")
        for k, fr in enumerate(c.decode(st)):
            if k % STEP:
                continue
            if n_out >= len(names):
                break
            name = "%d.png" % int(round((float(fr.pts * tb) + T0_S) * 1e9))
            if name != names[n_out]:
                raise RuntimeError(f"{insv.name} 第 {k} 帧的解码时间戳 {name} 与容器时间戳 {names[n_out]} 不一致")
            if fr.format.name not in LUMA_FORMATS:
                raise RuntimeError(f"{insv.name} 的像素格式为 {fr.format.name}，需要 {LUMA_FORMATS} 之一（4K60 全景录像）")
            y = fr.planes[0]
            g = np.frombuffer(y, np.uint8).reshape(y.height, y.line_size)[CROP_Y0:CROP_Y0 + CROP_H, CROP_X0:CROP_X0 + CROP_W].copy()
            pending.append(pool.submit(_write_png, img_dir / name, g))
            if len(pending) >= 4 * nw:        # bounded backlog: wait for the oldest half (re-raises a write error)
                for f in pending[: 2 * nw]:
                    f.result()
                del pending[: 2 * nw]
            n_out += 1
        for f in pending:
            f.result()
    if n_out != len(names):
        raise RuntimeError(f"{insv.name} 容器记录 {len(names)} 帧，实际解码 {n_out} 帧")


def imu_stats(euroc: Path) -> dict:
    a = np.loadtxt(euroc / "mav0" / "imu0" / "data.csv", delimiter=",", comments="#")
    acc, gyr = np.linalg.norm(a[:, 4:7], axis=1), np.linalg.norm(a[:, 1:4], axis=1)
    return {"duration_s": round(float((a[-1, 0] - a[0, 0]) / 1e9), 1), "accel_norm_std_m_s2": round(float(acc.std()), 3),
            "accel_norm_p95_minus_p5_m_s2": round(float(np.percentile(acc, 95) - np.percentile(acc, 5)), 3),
            "gyro_norm_p95_rad_s": round(float(np.percentile(gyr, 95)), 3)}


@runtime.stage_errors("读取录像")
def extract_recordings(recordings: SegmentPaths, context: StageContext) -> RecordingData:
    """INSV -> EuRoC timestamps/IMU/images; D deliberately has no decoded images."""
    datasets, metrics = {}, {}
    for stage, video in recordings.items():
        runtime.console.write(f"读取 {stage} 段" + ("（时间戳、IMU 和画面）" if stage != "D" else "（IMU）"), step=(1, 6))
        destination = context.paths.work / "ds" / stage
        try:
            build(video, destination, with_images=stage != "D")
            metrics[stage] = imu_stats(destination)
        except (RuntimeError, OSError, ValueError) as error:
            raise StageError(f"读取 {stage} 段", f"{stage} 段读取失败：{error}", imu_metrics=metrics) from error
        datasets[stage.lower()] = destination
    return RecordingData(SegmentPaths(**datasets), metrics)
