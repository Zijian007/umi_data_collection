"""惯性尺度/重力对齐与坏跟踪段细分；输出进入地图放置阶段的显式数据。"""
from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from software import config
from software.camera.x3.frame import FPS_USED
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from software.trajectory.algorithms.fusion_data import FusionInputs


@dataclass(frozen=True)
class AlignmentResult:
    lenses: dict
    video_times_s: np.ndarray
    imu_video_times_s: np.ndarray
    gyro: np.ndarray
    acc: np.ndarray
    initial_gyro_bias: np.ndarray
    estimates: dict
    initial_lens: str
    diagnostics: dict


def summarize_alignment(s, chains):
    if s is None:
        return None
    return {"ba": np.round(s["ba"], 4).tolist(), "pos_rms_mm": round(1e3 * s["pos_rms"], 3), "pairs": s["pairs"],
            "chains": [{"k": list(chains[i]), "scale": round(c["scale"], 5), "rel_std": None if c["rel_std"] is None else round(c["rel_std"], 4),
                        "g_free": None if c["g_free"] is None else round(c["g_free"], 3), "pairs": c["pairs"], "good": c["good"]}
                       for i, c in enumerate(s["chains"])]}


def chains_of(obs):
    ch = []
    for k in sorted(obs):
        if ch and k == ch[-1][1] + 1:
            ch[-1][1] = k
        else:
            ch.append([k, k])
    return [tuple(c) for c in ch]


def integrate_gyro_rotation(imu_video_times_s, gyro, initial_gyro_bias, t0, t1):
    i0, i1 = np.searchsorted(imu_video_times_s, t0), np.searchsorted(imu_video_times_s, t1)
    gyro_rotation = np.eye(3)
    tprev = t0
    for i in range(max(i0 - 1, 0), min(i1 + 1, len(imu_video_times_s))):
        tn = min(imu_video_times_s[i + 1], t1) if i + 1 < len(imu_video_times_s) else t1
        dt = tn - max(tprev, imu_video_times_s[i])
        if dt > 0:
            gyro_rotation = gyro_rotation @ Rot.from_rotvec((gyro[i] - initial_gyro_bias) * dt).as_matrix()
        tprev = tn
        if tn >= t1:
            break
    return gyro_rotation


def preintegrate_samples(imu_video_times_s, gyro, acc, t0, t1, gyro_bias, acc_bias):
    i0, i1 = np.searchsorted(imu_video_times_s, t0), np.searchsorted(imu_video_times_s, t1)
    dR = np.eye(3)
    dv = np.zeros(3)
    dp = np.zeros(3)
    Jv = np.zeros((3, 3))
    Jp = np.zeros((3, 3))
    tprev = t0
    for i in range(max(i0 - 1, 0), min(i1 + 1, len(imu_video_times_s))):
        tn = min(imu_video_times_s[i + 1], t1) if i + 1 < len(imu_video_times_s) else t1
        dt = tn - max(tprev, imu_video_times_s[i])
        if dt <= 0:
            continue
        aa = acc[i] - acc_bias
        dp = dp + dv * dt + 0.5 * dR @ aa * dt * dt
        Jp = Jp + Jv * dt - 0.5 * dR * dt * dt
        dv = dv + dR @ aa * dt
        Jv = Jv - dR * dt
        dR = dR @ Rot.from_rotvec((gyro[i] - gyro_bias) * dt).as_matrix()
        tprev = tn
        if tn >= t1:
            break
    return dR, dv, dp, Jv, Jp, t1 - t0


def estimate_lens_alignment(lens, video_times_s, imu_video_clock, initial_gyro_bias, fusion_parameters, use_chains=None):
    """per-chain scale + gravity, shared b_a. Returns dict or None."""
    gravity_m_s2 = fusion_parameters.gravity_m_s2
    alignment_step = fusion_parameters.step
    maximum_scale_relative_std = fusion_parameters.maximum_relative_scale_std
    imu_video_times_s, gyro, acc = imu_video_clock[:, 0], imu_video_clock[:, 1:4], imu_video_clock[:, 4:7]
    obs, R_CI, lever, chains = lens["obs"], lens["R_CI"], lens["lever"], lens["chains"]
    pairs = []
    for c, (k0, k1) in enumerate(chains):
        if use_chains is not None and c not in use_chains:
            continue
        k = k0
        while k + alignment_step <= k1:
            pairs.append((k, c))
            k += alignment_step
    if len(pairs) < fusion_parameters.minimum_alignment_pairs:
        return None
    nc = len(chains)
    # 未知量依次为：每段尺度、每段重力、共享加速度计零偏、各节点速度。
    scale_offset, gravity_offset, bias_offset, velocity_offset = 0, nc, 4 * nc, 4 * nc + 3
    velocity_indices = {}
    for k, c in pairs:
        for kk in (k, k + alignment_step):
            velocity_indices.setdefault(kk, len(velocity_indices))
    unknown_count = velocity_offset + 3 * len(velocity_indices)
    pre = {k: preintegrate_samples(imu_video_times_s, gyro, acc, video_times_s[k], video_times_s[k + alignment_step], initial_gyro_bias, np.zeros(3)) for k, c in pairs}
    pairs_per_chain = np.bincount([c for k, c in pairs], minlength=nc)

    def build(fixed_gravity):
        rows, cols, vals, rhs = [], [], [], []

        def put(r0, c0, Mx):
            Mx = np.atleast_2d(Mx)
            for i in range(Mx.shape[0]):
                for j in range(Mx.shape[1]):
                    if Mx[i, j] != 0:
                        rows.append(r0 + i)
                        cols.append(c0 + j)
                        vals.append(Mx[i, j])
        r = 0
        prow = []
        for k, c in pairs:
            (pk, Rk), (pk1, Rk1) = obs[k], obs[k + alignment_step]
            RMI = Rk @ R_CI
            dR, dv, dp, Jv, Jp, dt = pre[k]
            put(r, scale_offset + c, (pk1 - pk).reshape(3, 1))
            put(r, velocity_offset + 3 * velocity_indices[k], -np.eye(3) * dt)
            b = RMI @ dp - (Rk1 - Rk) @ lever
            if fixed_gravity is None:
                put(r, gravity_offset + 3 * c, -0.5 * np.eye(3) * dt * dt)
            else:
                b = b + 0.5 * fixed_gravity[c] * dt * dt
            put(r, bias_offset, -RMI @ Jp)
            rhs.extend(b)
            prow.extend([r, r + 1, r + 2])
            r += 3
            put(r, velocity_offset + 3 * velocity_indices[k + alignment_step], np.eye(3))
            put(r, velocity_offset + 3 * velocity_indices[k], -np.eye(3))
            b = RMI @ dv
            if fixed_gravity is None:
                put(r, gravity_offset + 3 * c, -np.eye(3) * dt)
            else:
                b = b + fixed_gravity[c] * dt
            put(r, bias_offset, -RMI @ Jv)
            rhs.extend(b)
            r += 3
        # tiny ridge rows keep columns of chains without pairs (and g of fixed solves) determined
        for j in range(unknown_count):
            rows.append(r)
            cols.append(j)
            vals.append(1e-6)
            rhs.append(0.0)
            r += 1
        return sp.csr_matrix((vals, (rows, cols)), shape=(r, unknown_count)), np.array(rhs), np.array(prow)
    A, bb, _ = build(None)
    x = spla.lsqr(A, bb, atol=fusion_parameters.linear_tolerance, btol=fusion_parameters.linear_tolerance, iter_lim=fusion_parameters.linear_iterations)[0]
    free_gravity = x[gravity_offset:bias_offset].reshape(nc, 3)
    fixed_gravity = np.array([g / max(np.linalg.norm(g), 1e-9) * gravity_m_s2 for g in free_gravity])
    A, bb, prow = build(fixed_gravity)
    x2 = spla.lsqr(A, bb, atol=fusion_parameters.linear_tolerance, btol=fusion_parameters.linear_tolerance, iter_lim=fusion_parameters.linear_iterations)[0]
    res = A @ x2 - bb
    dof = max(len(pairs) * 6 - (unknown_count - 3 * nc), 1)
    sig2 = float((res[:len(pairs) * 6] ** 2).sum() / dof)
    AtA = (A.T @ A).tocsc()
    lu = spla.splu(AtA + sp.identity(unknown_count, format="csc") * 1e-12)
    out_ch = []
    for c in range(nc):
        s = float(x2[scale_offset + c])
        if pairs_per_chain[c] == 0:
            out_ch.append({"scale": s, "rel_std": None, "g_free": None, "pairs": 0, "good": False})
            continue
        e = np.zeros(unknown_count)
        e[scale_offset + c] = 1.0
        var = float(lu.solve(e)[scale_offset + c]) * sig2
        rel = float(np.sqrt(max(var, 0)) / abs(s)) if s != 0 else np.inf
        gn = float(np.linalg.norm(free_gravity[c]))
        good = bool(s > 0 and pairs_per_chain[c] >= fusion_parameters.minimum_chain_pairs
                    and rel <= maximum_scale_relative_std
                    and abs(gn - gravity_m_s2) <= fusion_parameters.gravity_tolerance_m_s2)
        out_ch.append({"scale": s, "rel_std": rel, "g_free": gn, "g_map": fixed_gravity[c], "pairs": int(pairs_per_chain[c]), "good": good})
    acc_bias = x2[bias_offset:bias_offset + 3]
    pos_rms = float(np.sqrt((res[prow] ** 2).mean()))
    return {"chains": out_ch, "ba": acc_bias, "pos_rms": pos_rms, "pairs": len(pairs)}


def align_recording(inputs: FusionInputs, parameters: config.TrajectoryParameters, *, log=print) -> AlignmentResult:
    """细分不一致观测，估计各连续跟踪段的尺度与重力，选择初始化镜头。"""
    fusion_parameters = parameters.fusion
    gravity_m_s2, frame_rate_hz = fusion_parameters.gravity_m_s2, FPS_USED
    R_imu_rear_camera = inputs.rear_rotation
    alignment_step = fusion_parameters.step
    rear_lever_m = inputs.rear_lever
    enabled_lenses = {"rear": parameters.input.mode == "dual", "front": True}
    camera = inputs.camera
    R_imu_front_camera = Rot.from_quat(camera['q_xyzw']).as_matrix()
    p_imu_front_camera_m = np.array(camera['p'])
    front_lever_m = -R_imu_front_camera.T @ p_imu_front_camera_m
    lenses = {"rear": {"R_CI": R_imu_rear_camera.T, "lever": rear_lever_m}, "front": {"R_CI": R_imu_front_camera.T, "lever": front_lever_m}}


    for m in ("rear", "front"):
        lenses[m]["obs"] = inputs.observations[m] if enabled_lenses[m] else {}
        lenses[m]["mapid"] = inputs.map_ids[m] if enabled_lenses[m] else {}
        lenses[m]["chains"] = chains_of(lenses[m]["obs"])
    input_video_times_s = inputs.video_times_s
    if input_video_times_s is not None:
        input_times_s = input_video_times_s
        frame_count = len(input_times_s)
    else:
        frame_count = max(max(lenses["rear"]["obs"], default=0), max(lenses["front"]["obs"], default=0)) + 1
    imu = inputs.imu_video_clock
    imu_video_times_s = imu[:, 0]
    gyro, acc = imu[:, 1:4], imu[:, 4:7]
    video_times_s = input_times_s if input_video_times_s is not None else np.arange(frame_count) / frame_rate_hz
    static = np.linalg.norm(gyro, axis=1) < fusion_parameters.initial_static_gyro_rad_s
    initial_gyro_bias = gyro[static][:fusion_parameters.initial_bias_max_samples].mean(0) if static.sum() > fusion_parameters.initial_bias_min_samples else np.zeros(3)
    split_rotation_deg = fusion_parameters.split_rotation_deg


    split_counts = {}
    for m in lenses:
        obs, mp, R_CI = lenses[m]["obs"], lenses[m]["mapid"], lenses[m]["R_CI"]
        new = []
        ns = {"map": 0, "rot": 0}
        for (k0, k1) in lenses[m]["chains"]:
            start = k0
            for k in range(k0 + 1, k1 + 1):
                cut = None
                if mp and mp.get(k) != mp.get(k - 1):
                    cut = "map"
                elif split_rotation_deg > 0:
                    dRp = (obs[k - 1][1] @ R_CI).T @ (obs[k][1] @ R_CI)
                    gyro_rotation = integrate_gyro_rotation(
                        imu_video_times_s, gyro, initial_gyro_bias, video_times_s[k - 1], video_times_s[k])
                    rotation_error_deg = np.degrees(np.linalg.norm(
                        Rot.from_matrix(dRp.T @ gyro_rotation).as_rotvec()))
                    if rotation_error_deg > split_rotation_deg:
                        cut = "rot"
                if cut:
                    new.append((start, k - 1))
                    start = k
                    ns[cut] += 1
            new.append((start, k1))
        lenses[m]["chains"] = new
        split_counts[m] = ns
    log("chain_splits", split_counts, flush=True)


    refinement_pairs = fusion_parameters.refine_chain_pairs
    refinements = {}
    individually_good_chains = {m: set() for m in lenses}
    individual_estimates = {m: {} for m in lenses}
    if refinement_pairs > 0:
        # Stage 1 is one joint solve per lens with a SHARED accelerometer bias, so one bad chain can drag every other chain of the
        # lens over the gate. A long chain (>= 2W pairs) that fails jointly is therefore judged ALONE (its own b_a); if it also fails
        # alone it is cut in two at its middle pair and the pieces are judged the same way (pieces down to 3 pairs), recursively.
        for m in lenses:
            if not (enabled_lenses[m] and lenses[m]["obs"]):
                continue
            pieces, nsplit = set(), 0
            while True:
                r = estimate_lens_alignment(lenses[m], video_times_s, inputs.imu_video_clock, initial_gyro_bias, fusion_parameters)
                if r is None:
                    break
                new, changed, alone = [], False, set()
                for c, (k0, k1) in enumerate(lenses[m]["chains"]):
                    ch = r["chains"][c]
                    eligible = ch["pairs"] >= 2 * refinement_pairs or ((k0, k1) in pieces and ch["pairs"] >= fusion_parameters.minimum_chain_pairs)
                    if ch["good"] or not eligible:
                        new.append((k0, k1))
                        continue
                    ra = estimate_lens_alignment(lenses[m], video_times_s, inputs.imu_video_clock, initial_gyro_bias, fusion_parameters, {c})
                    if ra is not None and ra["chains"][c]["good"]:
                        new.append((k0, k1))
                        alone.add((k0, k1))
                        individual_estimates[m][(k0, k1)] = ra["chains"][c]
                        continue
                    if ch["pairs"] >= 2 * refinement_pairs:
                        mid = k0 + (ch["pairs"] // 2) * alignment_step
                        new += [(k0, mid - 1), (mid, k1)]
                        pieces |= {(k0, mid - 1), (mid, k1)}
                        changed = True
                        nsplit += 1
                    else:
                        new.append((k0, k1))
                lenses[m]["chains"] = new
                if not changed:
                    individually_good_chains[m] = alone
                    break
            refinements[m] = {"splits": nsplit, "good_alone": sorted([list(x) for x in individually_good_chains[m]])}
        log("refined_chains", refinements, flush=True)


    alignment_estimates = {}
    for m in lenses:
        if not (enabled_lenses[m] and lenses[m]["obs"]):
            alignment_estimates[m] = None
            continue
        r0 = estimate_lens_alignment(lenses[m], video_times_s, inputs.imu_video_clock, initial_gyro_bias, fusion_parameters)
        if r0 is not None:  # re-solve on the good chains only so bad chains do not corrupt b_a / scales
            goodc = {i for i, c in enumerate(r0["chains"]) if c["good"]}
            goodc |= {i for i, ch in enumerate(lenses[m]["chains"]) if tuple(ch) in individually_good_chains.get(m, ())}   # 3s: passed alone
            if goodc and len(goodc) < len(r0["chains"]):
                r1 = estimate_lens_alignment(lenses[m], video_times_s, inputs.imu_video_clock, initial_gyro_bias, fusion_parameters, goodc)
                if r1 is not None:
                    for i in range(len(r1["chains"])):
                        if i not in goodc:
                            r1["chains"][i] = dict(r0["chains"][i], good=False)
                    r0 = r1
        if r0 is not None:
            for i, ch in enumerate(lenses[m]["chains"]):   # 3s: a chain that is IMU-consistent on its OWN keeps its own stage 1
                if tuple(ch) in individually_good_chains.get(m, ()) and not r0["chains"][i]["good"]:
                    r0["chains"][i] = dict(individual_estimates[m][tuple(ch)], good=True)
            r0["sane"] = bool(any(c["good"] for c in r0["chains"]) and np.max(np.abs(r0["ba"])) <= fusion_parameters.maximum_acc_bias_m_s2)
        alignment_estimates[m] = r0
    cand = [m for m in ("rear", "front") if alignment_estimates[m] is not None and alignment_estimates[m]["sane"]]
    if not cand:
        cand = [m for m in ("rear", "front") if alignment_estimates[m] is not None and any(c["good"] for c in alignment_estimates[m]["chains"])]
    if not cand:
        raise ValueError("NO_USABLE_LENS")
    initial_lens = min(cand, key=lambda m: alignment_estimates[m]["pos_rms"])
    for m in lenses:
        log("stage1", m, json.dumps(summarize_alignment(alignment_estimates[m], lenses[m]["chains"])), flush=True)
    log("init_lens", initial_lens, flush=True)


    return AlignmentResult(lenses, video_times_s, imu_video_times_s, gyro, acc, initial_gyro_bias, alignment_estimates, initial_lens,
        dict(chain_splits=split_counts, refined=refinements))
