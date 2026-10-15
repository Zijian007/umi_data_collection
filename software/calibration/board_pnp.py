"""Per-frame camera pose from the AprilGrid board (KB4 intrinsics + PnP), and comparison of a trajectory against it.

The solver functions are the validated ones of software/data_collection/probes/vio/calib/calib_validate_pnp.py (dev branch),
copied without change: SQPnP + the two IPPE solutions as candidates, outlier rejection on normalised reprojection error,
SQPnP re-solve, LM refinement.  The board pose is metric and independent of the IMU and of the trajectory tool, which makes
it the ruler for checking a new calibration on the C recording.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def kb4_unproject(uv: np.ndarray, kb4: list[float]) -> np.ndarray:
    fx, fy, cx, cy, k1, k2, k3, k4 = kb4
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    D = np.array([k1, k2, k3, k4], float)
    pts = uv.reshape(-1, 1, 2).astype(np.float64)
    und = cv2.fisheye.undistortPoints(pts, K, D)  # normalised coordinates (x/z, y/z)
    n = und.reshape(-1, 2)
    b = np.concatenate([n, np.ones((len(n), 1))], 1)
    return b / np.linalg.norm(b, axis=1, keepdims=True)


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


def pnp_from_detections(dets: dict, kb4: list[float], pts3d: dict, min_tags: int = 6, inlier_thresh: float = 4e-3) -> dict | None:
    """dets = {tag_id: 4x2 pixel corners (BL,BR,TR,TL)} -> camera pose in the board frame T_bc (metres)."""
    ids = [i for i in dets if i in pts3d]
    if len(ids) < min_tags:
        return None
    obj = np.concatenate([pts3d[i] for i in ids], 0)
    img = np.concatenate([dets[i] for i in ids], 0)
    b = kb4_unproject(img, kb4)
    keep = b[:, 2] > 0.2
    if keep.sum() < 4 * min_tags:
        return None
    n = b[keep, :2] / b[keep, 2:3]
    objk = obj[keep]
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
    for _ in range(2):
        inl = _err(rvec, tvec, objk, n) < inlier_thresh
        if inl.sum() < 4 * min_tags // 2:
            return None
        ok, rvec, tvec = cv2.solvePnP(objk[inl], n[inl], np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
    rvec, tvec = cv2.solvePnPRefineLM(objk[inl], n[inl], np.eye(3), None, rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    Xc = (R @ obj[keep].T + tvec).T
    bp = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((bp * b[keep]).sum(1), -1, 1)))
    T_cb = np.eye(4)
    T_cb[:3, :3] = R
    T_cb[:3, 3] = tvec.reshape(3)
    T_bc = np.linalg.inv(T_cb)
    return {"T_bc": T_bc, "n_tags": len(ids), "n_pts": int(keep.sum()), "n_inl": int(inl.sum()),
            "reproj_deg_med": float(np.median(ang)), "reproj_deg_p95": float(np.percentile(ang, 95))}


def umeyama(src, dst, with_scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    H = S.T @ D / len(src)
    U, sig, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1, 1, d])
    R = Vt.T @ Dm @ U.T
    s = float(np.trace(np.diag(sig) @ Dm) / ((S ** 2).sum() / len(src))) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


UPSCALE = 2.0        # 4K60: tags are 13-30 px wide in the 1440 crop, too small for the detector at native size
MIN_TAGS_KEEP = 12   # poses from fewer decoded tags jitter by 5-8 mm at this resolution


def detect_up(gray: np.ndarray, det, up: float = UPSCALE) -> dict:
    """Detection on a bicubic-upsampled image, corners mapped back to native pixels.  Same as the project's audited 4K60
    reference generator (dev: experiments/reviews/data/20260904_x3_4k60_calibration_rerunA/tools/p3_make_reference_up2.py)."""
    g = cv2.resize(gray, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up != 1.0 else gray
    dets = det.detect(g)
    # cv2.resize maps source pixel centre x to (x + 0.5) * up - 0.5 -> invert exactly (no constant +0.25 px bias at 2x)
    return {k: (np.asarray(v, float) + 0.5) / up - 0.5 for k, v in dets.items()} if up != 1.0 else dets


def detect_all(euroc: Path, make_detector, step: int = 1, workers: int = 4, up: float = UPSCALE) -> dict:
    """AprilGrid detections (detect_up) on every `step`-th frame of an EuRoC folder.  Detection is the slow part (close to
    1 s a frame at 2x) and frames are independent, so a few threads share the work, each with its own detector (same
    result as one by one).  Returns {"n_frames": frames listed, "shape": (h, w), "dets": {frame index: {tag_id: 4x2 corners}}}."""
    rows = [l.split(",") for l in (Path(euroc) / "mav0" / "cam0" / "data.csv").read_text(encoding="utf-8").splitlines()
            if l and not l.startswith("#")]
    loc = threading.local()

    def one(k: int):
        if not hasattr(loc, "det"):
            loc.det = make_detector()
        g = cv2.imread(str(Path(euroc) / "mav0" / "cam0" / "data" / rows[k][1].strip()), cv2.IMREAD_GRAYSCALE)
        return (k, None, None) if g is None else (k, detect_up(g, loc.det, up), g.shape)

    with ThreadPoolExecutor(max(1, workers)) as ex:
        res = list(ex.map(one, range(0, len(rows), step)))
    return {"n_frames": len(rows), "shape": next((sh for _, _, sh in res if sh is not None), None),
            "dets": {k: d for k, d, _ in res if d is not None}}


def board_reference(found: dict, kb4: list[float], grid: dict, min_tags_keep: int = MIN_TAGS_KEEP) -> dict:
    """PnP camera pose for every frame of `found` (from detect_all) with at least `min_tags_keep` decoded tags (PnP itself
    runs with min_tags=4 and the pose is kept or dropped afterwards, as in the audited reference generator).
    Returns {"idx": frame indices, "p": Nx3 camera positions in the board frame, "reproj_deg_med": ..., "n_frames": total}."""
    pts3d = grid_points_3d(grid)
    idx, pos, rep = [], [], []
    for k in sorted(found["dets"]):
        r = pnp_from_detections(found["dets"][k], kb4, pts3d, min_tags=4)
        if r is None or r["n_tags"] < min_tags_keep:
            continue
        idx.append(k)
        pos.append(r["T_bc"][:3, 3])
        rep.append(r["reproj_deg_med"])
    return {"idx": np.array(idx, int), "p": np.array(pos, float).reshape(-1, 3), "reproj_deg_med": rep, "n_frames": found["n_frames"]}


def compare(ref: dict, p_cam_traj: np.ndarray) -> dict:
    """ref from board_reference; p_cam_traj[k] = camera position of frame k from the trajectory tool (its own world frame).
    One global rigid alignment without scale (the tool's output is supposed to be metric), plus the Sim(3) scale as a check."""
    idx = ref["idx"][ref["idx"] < len(p_cam_traj)]
    ok = np.all(np.isfinite(p_cam_traj[idx]), axis=1)
    idx = idx[ok]
    pr = ref["p"][: len(ref["idx"])][np.isin(ref["idx"], idx)]
    pt = p_cam_traj[idx]
    if len(idx) < 30:
        return {"n": int(len(idx)), "note": "too few frames with both a board pose and a trajectory pose"}
    s3, R3, t3 = umeyama(pt, pr, with_scale=True)
    _, R1, t1 = umeyama(pt, pr, with_scale=False)
    e1 = np.linalg.norm((R1 @ pt.T).T + t1 - pr, axis=1)
    e3 = np.linalg.norm((s3 * (R3 @ pt.T)).T + t3 - pr, axis=1)
    return {"n": int(len(idx)), "ate_se3_rmse_mm": float(np.sqrt(np.mean(e1 ** 2)) * 1e3), "ate_se3_p95_mm": float(np.percentile(e1, 95) * 1e3),
            "ate_sim3_rmse_mm": float(np.sqrt(np.mean(e3 ** 2)) * 1e3), "scale_board_over_trajectory": float(s3),
            "board_extent_m": float(np.linalg.norm(pr.max(0) - pr.min(0)))}
