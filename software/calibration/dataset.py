"""One X3 360 4K60 recording -> EuRoC folder for calibration (timestamps, IMU and, optionally, the front-lens frames).

  * timestamps + IMU: software.trajectory.x3_extract;
  * frames: every 2nd decoded frame, front lens = the right-half crop from software.trajectory.frame, written as <t_ns>.png.
    The grey image is the decoder's luma (Y) plane, read directly. frame.to_ndarray(format="gray") rescales on newer
    libswscale, so the pixels would change with the PyAV version.
    PNG encoding is the slow part (about 0.12 s a frame on one core), so the files are written by a few threads; the
    bytes of each file do not depend on that.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from software.trajectory.frame import SRC_H, SRC_W, STEP, T0_S, crop_xywh  # noqa: E402
from software.trajectory.x3_extract import extract  # noqa: E402

CROP_X0, CROP_Y0, CROP_W, CROP_H = crop_xywh("right")
LUMA_FORMATS = ("yuvj420p", "yuv420p")       # 8-bit planar, plane 0 = luma


def _write_png(path: Path, g: np.ndarray) -> None:
    if not cv2.imwrite(str(path), g, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"cannot write {path} (disk full?)")


def n_workers() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 2
    return max(1, min(8, n - 1))


def build(insv: Path, out: Path, with_images: bool = True, progress=None) -> dict:
    """Writes <out>/mav0/{cam0/data.csv, imu0/data.csv[, cam0/data/*.png]} and returns the extract manifest."""
    insv, out = Path(insv), Path(out)
    man = extract(insv, out, with_sha256=True)           # raises software.trajectory.x3_extract.Refused with the reason
    if not with_images:
        return man
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
            raise RuntimeError(f"{insv.name}: {st.codec_context.width}x{st.codec_context.height}, expected {SRC_W}x{SRC_H} (360 video, 4K60)")
        for k, fr in enumerate(c.decode(st)):
            if k % STEP:
                continue
            if n_out >= len(names):
                break
            name = "%d.png" % int(round((float(fr.pts * tb) + T0_S) * 1e9))
            if name != names[n_out]:
                raise RuntimeError(f"{insv.name}: decoded frame {k} has timestamp {name}, container list says {names[n_out]}")
            if fr.format.name not in LUMA_FORMATS:
                raise RuntimeError(f"{insv.name}: pixel format {fr.format.name}, expected one of {LUMA_FORMATS} (360 video, 4K60)")
            y = fr.planes[0]
            g = np.frombuffer(y, np.uint8).reshape(y.height, y.line_size)[CROP_Y0:CROP_Y0 + CROP_H, CROP_X0:CROP_X0 + CROP_W].copy()
            pending.append(pool.submit(_write_png, img_dir / name, g))
            if len(pending) >= 4 * nw:        # bounded backlog: wait for the oldest half (re-raises a write error)
                for f in pending[: 2 * nw]:
                    f.result()
                del pending[: 2 * nw]
            n_out += 1
            if progress and n_out % 300 == 0:
                progress(n_out, len(names))
        for f in pending:
            f.result()
    if n_out != len(names):
        raise RuntimeError(f"{insv.name}: decoded {n_out} of {len(names)} frames listed by the container timestamps")
    man["images"] = {"n": n_out, "crop_xywh": [CROP_X0, CROP_Y0, CROP_W, CROP_H], "format": "luma plane, gray png", "step": STEP}
    return man
