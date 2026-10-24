"""Per-frame camera pose from the AprilGrid board (KB4 intrinsics + PnP), and comparison of a trajectory against it.

SQPnP and the two IPPE solutions are candidates. Outliers are rejected on normalised reprojection error, then SQPnP
is solved again and refined with Levenberg-Marquardt. The board pose is metric and independent of the IMU and of the
trajectory, so the C recording can check a new calibration against it.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from software import config


def read_kb4(calib_json: Path) -> list[float]:
    d = json.loads(Path(calib_json).read_text(encoding="utf-8"))
    k = (d.get("value0") or d)["intrinsics"][0]["intrinsics"]
    return [k["fx"], k["fy"], k["cx"], k["cy"], k["k1"], k["k2"], k["k3"], k["k4"]]



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


def pnp_from_detections(dets: dict, kb4: list[float], pts3d: dict, *, parameters: config.BoardParameters | None = None) -> dict | None:
    """dets = {tag_id: 4x2 pixel corners (BL,BR,TR,TL)} -> camera pose in the board frame T_bc (metres)."""
    parameters = parameters or config.load("calibration").values.board
    min_tags, inlier_thresh = parameters.minimum_pnp_tags, parameters.normalized_inlier_threshold
    ids = [i for i in dets if i in pts3d]
    if len(ids) < min_tags:
        return None
    obj = np.concatenate([pts3d[i] for i in ids], 0)
    img = np.concatenate([dets[i] for i in ids], 0)
    b = kb4_unproject(img, kb4)
    keep = b[:, 2] > parameters.minimum_bearing_z
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
    for _ in range(parameters.refinement_iterations):
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


def detect_up(gray: np.ndarray, det, up: float | None = None) -> dict:
    """Detection on a bicubic-upsampled image, corners mapped back to native pixels."""
    up = config.load("calibration").values.board.detection_upscale if up is None else up
    g = cv2.resize(gray, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up != 1.0 else gray
    dets = det.detect(g)
    # cv2.resize maps source pixel centre x to (x + 0.5) * up - 0.5 -> invert exactly (no constant +0.25 px bias at 2x)
    return {k: (np.asarray(v, float) + 0.5) / up - 0.5 for k, v in dets.items()} if up != 1.0 else dets


def detect_all(euroc: Path, make_detector, step: int = 1, workers: int = 4, up: float | None = None) -> dict:
    """AprilGrid detections (detect_up) on every `step`-th frame of an EuRoC folder.  Detection is the slow part (close to
    1 s a frame at 2x) and frames are independent, so a few threads share the work, each with its own detector (same
    result as one by one).  Returns {"n_frames": frames listed, "shape": (h, w), "dets": {frame index: {tag_id: 4x2 corners}}}."""
    up = config.load("calibration").values.board.detection_upscale if up is None else up
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


def board_reference(found: dict, kb4: list[float], grid: dict, *, parameters: config.BoardParameters | None = None) -> dict:
    """PnP camera pose for every frame of `found` (from detect_all) with at least `min_tags_keep` decoded tags (PnP itself
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
    return {"idx": np.array(idx, int), "p": np.array(pos, float).reshape(-1, 3), "reproj_deg_med": rep, "n_frames": found["n_frames"]}


def compare(ref: dict, p_cam_traj: np.ndarray, *, parameters: config.BoardParameters | None = None) -> dict:
    """ref from board_reference; p_cam_traj[k] = camera position of frame k from the trajectory tool (its own world frame).
    One global rigid alignment without scale (the tool's output is supposed to be metric), plus the Sim(3) scale as a check."""
    parameters = parameters or config.load("calibration").values.board
    idx = ref["idx"][ref["idx"] < len(p_cam_traj)]
    ok = np.all(np.isfinite(p_cam_traj[idx]), axis=1)
    idx = idx[ok]
    pr = ref["p"][: len(ref["idx"])][np.isin(ref["idx"], idx)]
    pt = p_cam_traj[idx]
    if len(idx) < parameters.minimum_comparison_frames:
        return {"n": int(len(idx)), "note": "too few frames with both a board pose and a trajectory pose"}
    s3, R3, t3 = umeyama(pt, pr, with_scale=True)
    _, R1, t1 = umeyama(pt, pr, with_scale=False)
    e1 = np.linalg.norm((R1 @ pt.T).T + t1 - pr, axis=1)
    e3 = np.linalg.norm((s3 * (R3 @ pt.T)).T + t3 - pr, axis=1)
    return {"n": int(len(idx)), "ate_se3_rmse_mm": float(np.sqrt(np.mean(e1 ** 2)) * 1e3), "ate_se3_p95_mm": float(np.percentile(e1, 95) * 1e3),
            "ate_sim3_rmse_mm": float(np.sqrt(np.mean(e3 ** 2)) * 1e3), "scale_board_over_trajectory": float(s3),
            "board_extent_m": float(np.linalg.norm(pr.max(0) - pr.min(0)))}


def measure_board_coverage(found: dict, kb4: list[float], grid: dict, *, parameters: config.BoardParameters | None = None) -> dict:
    """How well a recording shows the board (found = board_pnp.detect_all): share of sampled frames with a board pose, how
    far from the lens centre the board corners reach (r = distance from the principal point / lens radius) and in how many
    of the four image directions, distance range.  A recording whose board stays near the lens centre (r about 0.23)
    produces degenerate intrinsics."""
    parameters = parameters or config.load("calibration").values.board
    ref = board_reference(found, kb4, grid, parameters=parameters)
    n = len(found["dets"])
    dist = np.linalg.norm(ref["p"], axis=1) if len(ref["idx"]) else np.array([])
    reach, by_dir = [], {"right": 0, "left": 0, "down": 0, "up": 0}
    for dets in found["dets"].values():
        if not dets:
            continue
        uv = np.concatenate(list(dets.values()), 0) - np.array(kb4[2:4])
        r = np.linalg.norm(uv, axis=1) / parameters.coverage_radius_px
        i = int(np.argmax(r))
        reach.append(float(r[i]))
        if r[i] >= parameters.coverage_reach:
            x, y = uv[i]
            by_dir[("right" if x > 0 else "left") if abs(x) >= abs(y) else ("down" if y > 0 else "up")] += 1
    return {"frames_sampled": n, "frames_with_board_pose": int(len(ref["idx"])), "board_pose_fraction": round(len(ref["idx"]) / max(1, n), 3),
            "reach_r_p99": round(float(np.percentile(reach, parameters.coverage_percentile)), 3) if reach else 0.0, "frames_reaching_r0p5_by_direction": by_dir,
            "coverage_reach_threshold": parameters.coverage_reach, "coverage_percentile": parameters.coverage_percentile,
            "directions_reached": int(sum(v >= parameters.minimum_direction_frames for v in by_dir.values())),
            "board_distance_m": [round(float(dist.min()), 2), round(float(dist.max()), 2)] if len(dist) else None}
