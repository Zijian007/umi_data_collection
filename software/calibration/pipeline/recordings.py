"""One X3 360 4K60 recording -> EuRoC folder for calibration (timestamps, IMU and optional lens frames).

  * timestamps + IMU: software.camera.x3.extraction;
  * frames: every 2nd decoded frame, default front/right; rear/left is explicit, written as <t_ns>.png.
    The grey image is the decoder's luma (Y) plane, read directly. frame.to_ndarray(format="gray") rescales on newer
    libswscale, so the pixels would change with the PyAV version.
    PNG encoding is the slow part (about 0.12 s a frame on one core), so the files are written by a few threads; the
    bytes of each file do not depend on that.
"""

from __future__ import annotations

import os
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from software import config
from software.storage import write_json
from software.calibration.pipeline import runtime

import cv2
import numpy as np

from software.camera.x3.frame import SRC_H, SRC_W, STEP, T0_S, crop_xywh
from software.camera.x3 import extraction

LUMA_FORMATS = ("yuvj420p", "yuv420p")       # 8-bit planar, plane 0 = luma


def _write_png(path: Path, g: np.ndarray) -> None:
    if not cv2.imwrite(str(path), g, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"无法写入图像 {path}；请检查磁盘空间和目录权限")


def n_workers(requested: int | None = None) -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 2
    if requested is not None and (type(requested) is not int or requested < 1):
        raise ValueError('--workers 必须为正整数')
    return max(1, min(requested, n)) if requested is not None else max(1, min(24, n - 1))


def preparation_identity(insv: Path, with_images: bool, lens: str) -> str:
    return config.canonical_hash(dict(path=str(insv.resolve()), sha256=config.file_hash(insv),
        images=with_images, lens=lens, code={str(path): config.file_hash(path) for path in (
            Path(__file__), *sorted(Path(extraction.__file__).parent.glob('*.py')))}))


def _reuse(out: Path, identity: str) -> bool:
    try:
        saved = json.loads((out / 'extract_manifest.json').read_text())['preparation']
        if saved['input_hash'] != identity:
            return False
        for relative, expected in saved['files'].items():
            path = (out / relative).resolve()
            if not path.is_relative_to(out.resolve()):
                return False
            stat = path.stat()
            if [stat.st_size, stat.st_mtime_ns] != expected:
                return False
        return bool(saved['files'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _mark_ready(out: Path, identity: str, images: list[Path]) -> None:
    manifest = out / 'extract_manifest.json'
    saved = json.loads(manifest.read_text())
    paths = [out / 'mav0/cam0/data.csv', out / 'mav0/imu0/data.csv', *images]
    saved['preparation'] = dict(input_hash=identity, files={str(path.relative_to(out)):
        [path.stat().st_size, path.stat().st_mtime_ns] for path in paths})
    write_json(manifest, saved)


def build(insv: Path, out: Path, with_images: bool = True, *, lens: str = 'front', workers: int | None = None) -> None:
    """Writes <out>/mav0/{cam0/data.csv, imu0/data.csv[, cam0/data/*.png]} and preserves the extraction manifest."""
    insv, out = Path(insv), Path(out)
    if lens not in ('front', 'rear'):
        raise ValueError('lens 必须为 front 或 rear')
    identity = preparation_identity(insv, with_images, lens)
    if _reuse(out, identity):
        runtime.console.write(f'复用录像准备：{insv.name}', state='复用')
        return
    crop_x, crop_y, crop_w, crop_h = crop_xywh('right' if lens == 'front' else 'left')
    extraction.extract(insv, out, with_sha256=True)           # raises software.camera.x3.extraction.Refused with the reason
    if not with_images:
        _mark_ready(out, identity, [])
        return
    names = [l.split(",")[1].strip() for l in (out / "mav0" / "cam0" / "data.csv").read_text(encoding="utf-8").splitlines()
             if l and not l.startswith("#")]
    img_dir = out / "mav0" / "cam0" / "data"
    img_dir.mkdir(parents=True, exist_ok=True)
    import av

    n_out, nw, pending = 0, n_workers(workers), []
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
            g = np.frombuffer(y, np.uint8).reshape(y.height, y.line_size)[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w].copy()
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
    _mark_ready(out, identity, [img_dir / name for name in names])


def imu_stats(euroc: Path) -> dict:
    a = np.loadtxt(euroc / "mav0" / "imu0" / "data.csv", delimiter=",", comments="#")
    acc, gyr = np.linalg.norm(a[:, 4:7], axis=1), np.linalg.norm(a[:, 1:4], axis=1)
    return {"duration_s": round(float((a[-1, 0] - a[0, 0]) / 1e9), 1), "accel_norm_std_m_s2": round(float(acc.std()), 3),
            "accel_norm_p95_minus_p5_m_s2": round(float(np.percentile(acc, 95) - np.percentile(acc, 5)), 3),
            "gyro_norm_p95_rad_s": round(float(np.percentile(gyr, 95)), 3)}
