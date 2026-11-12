"""Per-frame camera pose from the AprilGrid board (KB4 intrinsics + PnP), coverage and a metric trajectory reference.

SQPnP and the two IPPE solutions are candidates. Outliers are rejected on normalised reprojection error, then SQPnP
is solved again and refined with Levenberg-Marquardt. The board pose is metric and independent of the IMU and of the
trajectory, so the C recording can check a new calibration against it.
"""

from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np

from software import config
from software import parameters as parameter_models
from software.calibration.models import Kb4
from software.camera.x3.factory import Kb4Model
from software.camera.x3.frame import SRC_H


# Half the original square lens image; independent of the 1440 calibration crop.
COVERAGE_RADIUS_PX = SRC_H / 2


def read_kb4(calib_json: Path) -> Kb4:
    d = json.loads(Path(calib_json).read_text(encoding="utf-8"))
    k = (d.get("value0") or d)["intrinsics"][0]["intrinsics"]
    return Kb4(*(float(k[name]) for name in Kb4._fields))


def kb4_unproject(uv: np.ndarray, kb4: Kb4) -> np.ndarray:
    """Unit bearings, preserving theta > pi/2 and negative z (no OpenCV clipping)."""
    return Kb4Model(*kb4).unproject(uv)


def grid_points_3d(grid: dict) -> dict:
    """tag_id -> 4x3 corners in the board frame (metres); order BL,BR,TR,TL; tag0 bottom-left, +x right, +y up."""
    s = grid["tagSize"]
    pitch = s * (1 + grid["tagSpacing"])
    out = {}
    for r in range(grid["tagRows"]):
        for c in range(grid["tagCols"]):
            x, y = c * pitch, r * pitch
            out[r * grid["tagCols"] + c] = np.array([[x, y, 0], [x + s, y, 0], [x + s, y + s, 0], [x, y + s, 0]], float)
    return out


def pnp_from_detections(dets: dict, kb4: Kb4, pts3d: dict, *, parameters: parameter_models.BoardParameters | None = None) -> dict | None:
    """dets = {tag_id: 4x2 pixel corners (BL,BR,TR,TL)} -> camera pose in the board frame T_bc (metres)."""
    parameters = parameters or config.load("calibration").values.board
    min_tags, inlier_thresh = parameters.minimum_pnp_tags, parameters.normalized_inlier_threshold
    ids = [i for i in dets if i in pts3d]
    if len(ids) < min_tags:
        return None
    board_object_points = np.concatenate([pts3d[i] for i in ids], 0)
    img = np.concatenate([dets[i] for i in ids], 0)
    b = kb4_unproject(img, kb4)
    # The PnP initializer is a forward virtual pinhole. Exclude backward rays
    # explicitly here, rather than folding them into fabricated forward bearings.
    keep = np.isfinite(b).all(axis=1) & (b[:, 2] > parameters.minimum_bearing_z)
    if keep.sum() < 4 * min_tags:
        return None
    n = b[keep, :2] / b[keep, 2:3]
    objk = board_object_points[keep]
    cands = []
    ok, rvec, tvec = cv2.solvePnP(objk, n, np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
    if ok:
        cands.append((rvec, tvec))
    try:
        n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(objk, n, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
        cands += [(rvecs[i], tvecs[i]) for i in range(int(n_sol))]
    except cv2.error:
        pass
    if not cands:
        return None

    def _err(rv, tv, o, m):
        proj, _ = cv2.projectPoints(o, rv, tv, np.eye(3), None)
        return np.linalg.norm(proj.reshape(-1, 2) - m, axis=1)

    rvec, tvec = min(cands, key=lambda c: float(np.sqrt(np.mean(_err(c[0], c[1], objk, n) ** 2))))
    inl = np.ones(len(objk), bool)
    for _ in range(parameters.refinement_iterations):
        inl = _err(rvec, tvec, objk, n) < inlier_thresh
        if inl.sum() < 4 * min_tags // 2:
            return None
        ok, rvec, tvec = cv2.solvePnP(objk[inl], n[inl], np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
    rvec, tvec = cv2.solvePnPRefineLM(objk[inl], n[inl], np.eye(3), None, rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    Xc = (R @ board_object_points[keep].T + tvec).T
    bp = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((bp * b[keep]).sum(1), -1, 1)))
    T_cb = np.eye(4)
    T_cb[:3, :3] = R
    T_cb[:3, 3] = tvec.reshape(3)
    T_bc = np.linalg.inv(T_cb)
    return {"T_bc": T_bc, "n_tags": len(ids), "n_pts": int(keep.sum()), "n_inl": int(inl.sum()),
            "reproj_deg_med": float(np.median(ang)), "reproj_deg_p95": float(np.percentile(ang, 95))}


def board_reference(found: dict, kb4: Kb4, grid: dict, *, parameters: parameter_models.BoardParameters | None = None) -> dict:
    """PnP camera pose for every frame of `found` (raw board detections) with at least `min_tags_keep` decoded tags (PnP itself
    runs with minimum_pnp_tags and the pose is kept or dropped afterwards).
    Returns {"idx": frame indices, "p": Nx3 camera positions in the board frame, "reproj_deg_med": ..., "n_frames": total}."""
    parameters = parameters or config.load("calibration").values.board
    min_tags_keep = parameters.minimum_reference_tags
    pts3d = grid_points_3d(grid)
    idx, pos, rep = [], [], []
    for k in sorted(found["dets"]):
        r = pnp_from_detections(found["dets"][k], kb4, pts3d, parameters=parameters)
        if r is None or r["n_tags"] < min_tags_keep:
            continue
        idx.append(k)
        pos.append(r["T_bc"][:3, 3])
        rep.append(r["reproj_deg_med"])
    return {"idx": np.array(idx, int), "p": np.array(pos, float).reshape(-1, 3), "reproj_deg_med": rep, "n_frames": found["n_frames"],
            "video_time_s": np.asarray(found["video_time_s"])[idx]}


def measure_board_coverage(found: dict, kb4: Kb4, grid: dict, *, parameters: parameter_models.BoardParameters | None = None) -> dict:
    """How well a recording shows the board (found = raw board detections).

    Every radius is r = corner distance from the principal point / coverage_radius_px.  Per frame the farthest corner
    counts; ``reach_radius_percentile`` is the ``coverage_percentile``-th percentile of those per-frame maxima, and
    ``frames_reaching_by_direction`` counts frames whose farthest corner has r >= ``coverage_reach``, by image direction.
    Both parameter values are stored next to the metrics.  Also returns the share of sampled frames with a board pose and
    the camera-to-board distance range.  A recording whose board stays near the lens centre (r about 0.23) produces
    degenerate intrinsics."""
    parameters = parameters or config.load("calibration").values.board
    ref = board_reference(found, kb4, grid, parameters=parameters)
    n = len(found["dets"])
    dist = np.linalg.norm(ref["p"], axis=1) if len(ref["idx"]) else np.array([])
    reach, by_dir = [], {"right": 0, "left": 0, "down": 0, "up": 0}
    for dets in found["dets"].values():
        if not dets:
            continue
        uv = np.concatenate(list(dets.values()), 0) - np.array(kb4.principal_point)
        r = np.linalg.norm(uv, axis=1) / COVERAGE_RADIUS_PX
        i = int(np.argmax(r))
        reach.append(float(r[i]))
        if r[i] >= parameters.coverage_reach:
            x, y = uv[i]
            by_dir[("right" if x > 0 else "left") if abs(x) >= abs(y) else ("down" if y > 0 else "up")] += 1
    return {"frames_sampled": n, "frames_with_board_pose": int(len(ref["idx"])),
            "board_pose_fraction": round(len(ref["idx"]) / max(1, n), 3),
            "reach_radius_percentile": round(float(np.percentile(reach, parameters.coverage_percentile)), 3) if reach else 0.0,
            "coverage_percentile": parameters.coverage_percentile,
            "coverage_radius_px": COVERAGE_RADIUS_PX,
            "frames_reaching_by_direction": by_dir, "coverage_reach_threshold": parameters.coverage_reach,
            "directions_reached": int(sum(v >= parameters.minimum_direction_frames for v in by_dir.values())),
            "board_distance_m": [round(float(dist.min()), 2), round(float(dist.max()), 2)] if len(dist) else None}
