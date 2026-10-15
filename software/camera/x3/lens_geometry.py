"""X3 lens geometry: the factory ``offset_v3`` prior, and how it maps onto a stored image.

The camera firmware writes a string into metadata record 1 field 54 (``offset_v3``) describing both
lenses.  Following the Gyroflow ``insta360`` model, each lens is a Unified Camera Model (UCM/Mei,
parameter ``xi``) plus radial ``(k1,k2,k3)`` and tangential ``(p1,p2)`` distortion, in the
coordinates of the stitching canvas:

    num _ xi _ fx _ fy _ cx _ cy _ yaw _ pitch _ roll _ tx _ ty _ tz _ k1 _ k2 _ k3 _ p1 _ p2 _
    width _ height _ lensType   (repeated per lens; the second lens' cx carries the canvas offset)

Projection (UCM + rad-tan, same as Gyroflow's insta360.rs)::

    d = |X|;  xu = X/(Z + xi*d),  yu = Y/(Z + xi*d);  r2 = xu^2 + yu^2
    radial = 1 + k1 r2 + k2 r4 + k3 r6
    xd = xu*radial + 2 p1 xu yu + p2 (r2 + 2 xu^2)
    yd = yu*radial + p1 (r2 + 2 yu^2) + 2 p2 xu yu
    u = fx xd + cx;  v = fy yd + cy

Unprojection inverts the distortion iteratively and the UCM analytically.

**STATUS: FACTORY_UNVERIFIED.**  These numbers come from camera firmware and are recorded as they
are found.  This project has never verified them with its own calibration, so they may be used for
smoke-level VO and for filling a ``CameraModel`` slot -- never as a validated projection contract
and never as the basis of an accuracy claim.  A real calibration must estimate intrinsics
independently.

Provenance: ``UcmLens`` and ``parse_offset_v3`` are a logic-level COPY of
``software/data_collection/probes/x3_camera_model.py`` (not sha-pinned, so that path is now a thin
deprecated wrapper around this module).  :func:`derive_camera_model` moved here from
``software/camera/x3/export_capture.py`` (Phase B).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from software.camera.contracts import CameraModel, PixelTransform, sha256_bytes


@dataclass
class UcmLens:
    xi: float
    fx: float
    fy: float
    cx: float
    cy: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    canvas_w: int
    canvas_h: int
    lens_type: int
    lens_index: int  # 0/1 in offset_v3 order
    scale: float = 1.0  # scaling already applied (recorded frame / canvas)
    source: str = "factory_metadata_offset_v3 (FACTORY_UNVERIFIED)"

    def scaled(self, frame_w: int) -> "UcmLens":
        """Scale canvas parameters to the recorded frame (per-lens canvas assumed square)."""
        s = frame_w / float(self.canvas_h)
        d = asdict(self)
        d.update({"fx": self.fx * s, "fy": self.fy * s, "cx": self.cx * s, "cy": self.cy * s,
                  "scale": self.scale * s})
        return UcmLens(**d)

    # ---- projection / unprojection ----
    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float).reshape(-1, 3)
        d = np.linalg.norm(X, axis=1)
        den = X[:, 2] + self.xi * d
        xu = X[:, 0] / den
        yu = X[:, 1] / den
        r2 = xu * xu + yu * yu
        radial = 1 + self.k1 * r2 + self.k2 * r2 ** 2 + self.k3 * r2 ** 3
        xd = xu * radial + 2 * self.p1 * xu * yu + self.p2 * (r2 + 2 * xu * xu)
        yd = yu * radial + self.p1 * (r2 + 2 * yu * yu) + 2 * self.p2 * xu * yu
        return np.stack([self.fx * xd + self.cx, self.fy * yd + self.cy], axis=1)

    def _undistort(self, xd: np.ndarray, yd: np.ndarray, iters: int = 12) -> tuple[np.ndarray, np.ndarray]:
        xu, yu = xd.copy(), yd.copy()
        for _ in range(iters):
            r2 = xu * xu + yu * yu
            radial = 1 + self.k1 * r2 + self.k2 * r2 ** 2 + self.k3 * r2 ** 3
            dx = 2 * self.p1 * xu * yu + self.p2 * (r2 + 2 * xu * xu)
            dy = self.p1 * (r2 + 2 * yu * yu) + 2 * self.p2 * xu * yu
            xu = (xd - dx) / radial
            yu = (yd - dy) / radial
        return xu, yu

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        """Pixels -> unit bearing vectors (camera frame, z forward).

        UCM inverse: given ``m = (xu, yu)``,
        ``X = (xi + sqrt(1 + (1-xi^2) r2)) / (r2 + 1) * (xu, yu, 1) - (0, 0, xi)``, then normalised.
        """
        uv = np.asarray(uv, float).reshape(-1, 2)
        xd = (uv[:, 0] - self.cx) / self.fx
        yd = (uv[:, 1] - self.cy) / self.fy
        xu, yu = self._undistort(xd, yd)
        r2 = xu * xu + yu * yu
        disc = 1 + (1 - self.xi ** 2) * r2
        disc = np.maximum(disc, 0.0)
        fac = (self.xi + np.sqrt(disc)) / (r2 + 1)
        X = np.stack([fac * xu, fac * yu, fac - self.xi], axis=1)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(n, 1e-12)

    def theta_of_radius(self, r_px: float, n: int = 2000) -> float:
        """Numerically solve for the polar angle (degrees) at image radius ``r_px``; FOV sanity."""
        th = np.linspace(0, np.pi * 0.75, n)
        X = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=1)
        uv = self.project(X)
        r = uv[:, 0] - self.cx
        idx = np.argmax(r >= r_px) if np.any(r >= r_px) else None
        return float(np.degrees(th[idx])) if idx is not None else float("nan")


def parse_offset_v3(s: str) -> list[UcmLens]:
    """Parse an ``offset_v3`` string into one :class:`UcmLens` per lens (canvas coordinates).

    The i-th lens' ``cx`` has the canvas offset of its half removed, so each lens is expressed in
    its own square canvas.
    """
    toks = s.split("_")
    n = int(float(toks[0]))
    vals = [float(t) for t in toks[1:]]
    per = 19  # xi fx fy cx cy yaw pitch roll tx ty tz k1 k2 k3 p1 p2 width height lensType
    lenses = []
    for i in range(n):
        v = vals[i * per:(i + 1) * per]
        if len(v) < per:
            break
        (xi, fx, fy, cx, cy, yaw, pitch, roll, _tx, _ty, _tz, k1, k2, k3, p1, p2, w, h, lt) = v
        w, h = int(w), int(h)
        cx_local = cx - (w / n) * i  # horizontal offset of lens i inside the canvas
        lenses.append(UcmLens(xi, fx, fy, cx_local, cy, yaw, pitch, roll, k1, k2, k3, p1, p2,
                              w, h, int(lt), i))
    return lenses


def derive_camera_model(off_v3: str, lens_index: int, plan: dict, src_wh: tuple[int, int],
                        out_size: int, profile_id: str) -> CameraModel:
    """Map the factory UCM prior onto the STORED image geometry.

    Chain (pure geometry, no fitting)::

      canvas (canvas_h x canvas_h) --scale s1 = per_lens_w / canvas_h--> native per-lens frame
      native frame --translate by the crop origin relative to the selected half--> cropped frame
      cropped frame --scale s2 = out_size / crop_size--> stored image

    Status stays ``FACTORY_UNVERIFIED``: these numbers come from camera firmware metadata and this
    project has never calibration-verified them.
    """
    per_lens_w, per_lens_h = plan["per_lens_size"]
    lens = parse_offset_v3(off_v3)[lens_index].scaled(per_lens_w)
    hx, hy, _hw, _hh = plan["half_xywh"]
    crop = plan["crop_xywh_abs"]
    if crop is None:
        cx, cy = lens.cx - 0.0, lens.cy - 0.0
        crop_size = min(per_lens_w, per_lens_h)
    else:
        cx = lens.cx - (crop[0] - hx)
        cy = lens.cy - (crop[1] - hy)
        crop_size = crop[2]
    s2 = out_size / float(crop_size)
    fx, fy, cx, cy = lens.fx * s2, lens.fy * s2, cx * s2, cy * s2
    pt = PixelTransform(
        source_size_px=[int(src_wh[0]), int(src_wh[1])],
        lens_half=plan["lens_half"],
        half_xywh=[int(v) for v in plan["half_xywh"]],
        crop_xywh_abs=None if crop is None else [int(v) for v in crop],
        output_size_px=[int(out_size), int(out_size)],
        resampled=bool(out_size != crop_size),
        resample_interpolation="cv2.INTER_AREA" if out_size != crop_size else "none")
    projection_id = "x3pt-" + sha256_bytes(
        json.dumps(pt.to_dict(), sort_keys=True).encode("utf-8"))[:12]
    return CameraModel(
        model_id=f"x3_factory_ucm/lens{lens_index}/{profile_id}/{out_size}px",
        model_type="ucm_radtan",
        resolution_px=[int(out_size), int(out_size)],
        intrinsic_names=["xi", "fx", "fy", "cx", "cy"],
        intrinsics=[float(lens.xi), float(fx), float(fy), float(cx), float(cy)],
        distortion_names=["k1", "k2", "k3", "p1", "p2"],
        distortion=[float(lens.k1), float(lens.k2), float(lens.k3), float(lens.p1), float(lens.p2)],
        calibration_status="FACTORY_UNVERIFIED",
        pixel_transform=pt,
        roi_xywh=None if crop is None else [int(v) for v in crop],
        lens_half=plan["lens_half"],
        projection_id=projection_id,
        source="Insta360 firmware metadata record 1 field 54 (offset_v3), Gyroflow 'insta360' "
               "UCM + rad-tan interpretation; parsed by "
               "software/data_collection/probes/x3_camera_model.py:parse_offset_v3",
        note="FACTORY_UNVERIFIED prior only; a calibration must estimate intrinsics independently. "
             "Geometry chain: canvas -> per-lens frame (scale) -> crop (translate) -> stored image "
             "(scale).")


__all__ = ["UcmLens", "parse_offset_v3", "derive_camera_model", "asdict"]
