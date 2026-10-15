"""X3 fixed projection v0: official Media SDK equirect -> deterministic pinhole view + projection_id.

Why this shape: the raw fisheye intrinsics are NOT calibrated, so a "fisheye -> pinhole" projection
of our own is not available.  The official SDK's 2:1 equirect output was measured to be
byte-deterministic across repeated runs with the same parameters (same host, same binaries), so v0 =
fixed SDK parameters + a purely geometric pinhole cut out of the equirect (equirect -> pinhole needs
no intrinsics: pixel <-> lon/lat <-> bearing is analytic).  Once intrinsics are calibrated, v1 can
project the fisheye directly and ``projection_id`` changes with it = a new dataset version.

Everything that affects pixels goes into ``projection_id``:
  * upstream: SDK version / stitch_type / output_size / any other switch (all ``--enable_*`` OFF);
  * virtual pinhole: yaw/pitch/roll (deg, panorama frame), horizontal FOV, intermediate and final
    resolution, the two-stage anti-aliasing (remap bilinear -> INTER_AREA downscale; ``cv2.remap``
    silently ignores INTER_AREA), LUT type (CV_32FC2, unquantised);
  * the implementation version ``IMPL_VERSION``.

``projection_id = sha256(canonical JSON of the above)[:16]``.  The LUT is uniquely determined by
those parameters, so the same parameters and the same input give byte-identical output (asserted by
tests).

Not done here: SDK output pixels are not altered; no stabilisation or colour work; no claim that
this equals the final calibrated projection.

Provenance: a logic-level COPY of ``software/postprocess/x3_projection.py`` (not sha-pinned, so that
path is now a thin deprecated wrapper around this module).  ``IMPL_VERSION``, the dataclass fields
and their defaults are unchanged, so ``projection_id`` is unchanged -- asserted by
``software/camera/tests/test_x3_projection_equivalence.py``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMPL_VERSION = "equirect_pinhole_v0.2"  # v0.2: explicit guard exceptions + richer provenance


class ProjectionError(ValueError):
    """Invalid projection parameters or input (explicit exception; never ``assert``)."""


@dataclass(frozen=True)
class SdkExportSpec:
    sdk_version: str = "MediaSDK-3.1.3-20260128-win64"
    stitch_type: str = "optflow"
    output_size: str = "3840x1920"
    image_type: str = "png"
    enable_flowstate: bool = False
    enable_directionlock: bool = False
    other_flags: tuple = ()  # any extra switch, recorded explicitly (default: all OFF)


@dataclass(frozen=True)
class PinholeSpec:
    yaw_deg: float = 0.0  # panorama longitude centre (0 = the middle column of the equirect)
    pitch_deg: float = -25.0  # negative = looking down, towards the gripper / workspace
    roll_deg: float = 0.0
    hfov_deg: float = 100.0  # target pinhole horizontal FOV (hard limit 120 deg)
    inter_size: int = 448  # intermediate resolution (first stage of the anti-aliasing)
    out_size: int = 224
    lut_dtype: str = "CV_32FC2"
    remap_interp: str = "INTER_LINEAR"
    downscale_interp: str = "INTER_AREA"


@dataclass(frozen=True)
class ProjectionSpec:
    sdk: SdkExportSpec = field(default_factory=SdkExportSpec)
    pinhole: PinholeSpec = field(default_factory=PinholeSpec)
    impl_version: str = IMPL_VERSION

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def projection_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]


def _rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Virtual camera orientation in the panorama frame (x forward = lon 0, y right, z down):
    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.  The convention is fixed and enters ``projection_id``
    (a change of ``IMPL_VERSION`` is a change of semantics)."""
    cy, sy = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
    cp, sp = np.cos(np.deg2rad(pitch)), np.sin(np.deg2rad(pitch))
    cr, sr = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def build_lut(spec: ProjectionSpec, eq_w: int, eq_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Remap LUT (float32 map_x / map_y at the intermediate resolution): pinhole px -> equirect px."""
    ph = spec.pinhole
    n = ph.inter_size
    f = (n / 2) / np.tan(np.deg2rad(ph.hfov_deg) / 2)
    cx = cy = (n - 1) / 2
    u, v = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64))
    # camera frame: x forward, y right, z down (image u right = +y, v down = +z)
    d = np.stack([np.full_like(u, f), u - cx, v - cy], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    R = _rot(ph.yaw_deg, ph.pitch_deg, ph.roll_deg)
    w = d @ R.T  # bearing in the world (panorama) frame
    lon = np.arctan2(w[..., 1], w[..., 0])  # [-pi, pi], 0 = forward
    lat = np.arcsin(np.clip(-w[..., 2], -1, 1))  # positive = up
    map_x = ((lon + np.pi) / (2 * np.pi) * eq_w - 0.5).astype(np.float32)
    map_y = ((np.pi / 2 - lat) / np.pi * eq_h - 0.5).astype(np.float32)
    return map_x, map_y


class EquirectPinholeProjector:
    def __init__(self, spec: ProjectionSpec, eq_w: int, eq_h: int) -> None:
        ph = spec.pinhole
        # Message strings are kept VERBATIM from the original module: the existing suite
        # software/postprocess/tests/test_x3_projection.py matches on them.
        if not (0.0 < ph.hfov_deg <= 120.0):
            raise ProjectionError(f"针孔目标 hfov={ph.hfov_deg} 不在 (0,120]（设计 MD §5 上限）")
        if ph.lut_dtype != "CV_32FC2":
            raise ProjectionError("v0 固定 CV_32FC2（无量化）")
        if ph.inter_size < ph.out_size or ph.out_size <= 0:
            raise ProjectionError(f"inter_size {ph.inter_size} 必须 >= out_size {ph.out_size} > 0")
        if eq_w <= 0 or eq_h <= 0 or eq_w != 2 * eq_h:
            raise ProjectionError(f"equirect 必须 2:1，收到 {eq_w}x{eq_h}")
        self.spec = spec
        self.eq_w, self.eq_h = eq_w, eq_h
        self.map_x, self.map_y = build_lut(spec, eq_w, eq_h)

    def project(self, equirect_bgr: np.ndarray) -> np.ndarray:
        if equirect_bgr is None or equirect_bgr.ndim != 3 or equirect_bgr.shape[2] != 3:
            raise ProjectionError("输入必须为 HxWx3 图像（imread 失败会得到 None）")
        if equirect_bgr.shape[1] != self.eq_w or equirect_bgr.shape[0] != self.eq_h:
            raise ProjectionError(f"equirect 尺寸 {equirect_bgr.shape[:2]} != LUT 尺寸 {(self.eq_h, self.eq_w)}")
        inter = cv2.remap(equirect_bgr, self.map_x, self.map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_WRAP)
        out = cv2.resize(inter, (self.spec.pinhole.out_size, self.spec.pinhole.out_size),
                         interpolation=cv2.INTER_AREA)
        return out

    def lut_sha256(self) -> str:
        h = hashlib.sha256()
        h.update(self.map_x.tobytes())
        h.update(self.map_y.tobytes())
        return h.hexdigest()


def write_provenance(out_dir: Path, spec: ProjectionSpec, proj: EquirectPinholeProjector,
                     inputs: dict, sdk_manifest: dict | None = None) -> Path:
    import platform

    prov = {
        "projection_id": spec.projection_id,
        "spec": asdict(spec),
        "canonical_json": spec.canonical_json(),
        "lut_sha256": proj.lut_sha256(),
        "equirect_size": [proj.eq_w, proj.eq_h],
        "software": {"opencv": cv2.__version__, "numpy": np.__version__,
                     "python": platform.python_version(), "platform": platform.platform(),
                     "node": platform.node()},
        "sdk_manifest": ({
            "MediaSDKTest.exe_sha256": sdk_manifest["sdk"]["MediaSDKTest.exe_sha256"],
            "MediaSDK.dll_sha256": sdk_manifest["sdk"]["MediaSDK.dll_sha256"],
            "sdk_dir_name": sdk_manifest["sdk"]["sdk_dir_name"],
            "params": sdk_manifest["params"],
            "inputs": [{"name": i["name"], "sha256": i["sha256"]} for i in sdk_manifest["inputs"]],
            "returncode": sdk_manifest["returncode"],
        } if sdk_manifest else None),
        "inputs": inputs,
        "determinism_scope": ("byte-identity demonstrated ONLY for: same host, same SDK binaries "
                              "(hashes above), same OpenCV build, the listed sampled frames, repeated "
                              "runs. NOT claimed: cross-host, other OpenCV builds, or full-sequence "
                              "identity without a per-frame check."),
        "status": "v0: equirect(SDK)->pinhole; NOT a calibrated fisheye projection; "
                  "projection_id change = new dataset version",
    }
    p = out_dir / "projection_provenance.json"
    p.write_text(json.dumps(prov, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


__all__ = ["IMPL_VERSION", "ProjectionError", "SdkExportSpec", "PinholeSpec", "ProjectionSpec",
           "build_lut", "EquirectPinholeProjector", "write_provenance"]
