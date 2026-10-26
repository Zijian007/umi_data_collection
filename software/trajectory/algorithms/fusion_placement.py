"""将连续跟踪段放入独立录像的米制世界，并划分优化段。"""
from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from software import config
from software.camera.x3.frame import FPS_USED
from software.trajectory.algorithms.fusion_alignment import AlignmentResult


@dataclass(frozen=True)
class MapPlacements:
    chains: dict
    groups: dict
    segments: dict
    initial_imu_poses: dict
    gauge_chain: int
    dropped: list
    gates: list


def place_maps(aligned: AlignmentResult, parameters: config.TrajectoryParameters, *, log=print) -> MapPlacements:
    """冻结初始化镜头的参考段，再放置其它跟踪段和镜头。"""
    fusion_parameters = parameters.fusion
    gravity_m_s2, frame_rate_hz = fusion_parameters.gravity_m_s2, FPS_USED
    segment_duration_s = fusion_parameters.segment_s
    other_lens_rms_m, other_lens_rotation_deg = fusion_parameters.other_lens_rms_m, fusion_parameters.other_lens_rotation_deg
    lenses, alignment_estimates, initial_lens = aligned.lenses, aligned.estimates, aligned.initial_lens
    frame_count = len(aligned.video_times_s)
    def rotation_from_gravity(gmap):
        zg = -gmap / gravity_m_s2
        ax = np.cross(zg, [0, 0, 1.0])
        sa = np.linalg.norm(ax)
        ca = float(np.dot(zg, [0, 0, 1.0]))
        return np.eye(3) if sa < 1e-9 else Rot.from_rotvec(ax / sa * np.arctan2(sa, ca)).as_matrix()


    def placed_imu_pose(m, k, placement):
        """相机观测 → 世界系 IMU 位姿；物理杆臂不随视觉尺度缩放。"""
        p, Rm = lenses[m]["obs"][k]
        return placement["R"] @ Rm @ lenses[m]["R_CI"], placement["s"] * (placement["R"] @ p) + placement["t"] + placement["R"] @ Rm @ lenses[m]["lever"]


    # ---------------- chain placements ----------------
    chain_placements = {m: {} for m in lenses}   # chain index -> {"s","R","t","kind": "full"|"rot"}
    dropped, gates = [], []
    ch0 = lenses[initial_lens]["chains"]
    st0 = alignment_estimates[initial_lens]["chains"]
    good0 = [i for i, c in enumerate(st0) if c["good"]]
    gauge = max(good0, key=lambda i: ch0[i][1] - ch0[i][0])
    chain_placements[initial_lens][gauge] = {"s": st0[gauge]["scale"], "R": rotation_from_gravity(st0[gauge]["g_map"]), "t": np.zeros(3), "kind": "full"}
    todo = sorted([i for i in good0 if i != gauge], key=lambda i: abs(i - gauge))
    for i in todo:
        j = min(chain_placements[initial_lens], key=lambda jj: abs(jj - i))
        placement = {"s": st0[i]["scale"], "R": rotation_from_gravity(st0[i]["g_map"]), "t": np.zeros(3), "kind": "full"}
        ka, kc = (ch0[j][1], ch0[i][0]) if j < i else (ch0[j][0], ch0[i][1])
        pa = placed_imu_pose(initial_lens, ka, chain_placements[initial_lens][j])[1]
        pc = placed_imu_pose(initial_lens, kc, placement)[1]
        placement["t"] = pa - pc
        chain_placements[initial_lens][i] = placement
    group_scale_ratio = fusion_parameters.group_scale_ratio
    chain_groups = {m: {} for m in lenses}   # member chain -> leader chain


    def chain_map(m, i):
        return lenses[m]["mapid"].get(lenses[m]["chains"][i][0])


    def group_full_chains(m, scale_of):
        full = sorted([i for i, placement in chain_placements[m].items() if placement["kind"] == "full"], key=lambda i: -(lenses[m]["chains"][i][1] - lenses[m]["chains"][i][0]))
        leaders = []
        for i in full:
            for j in leaders:
                if chain_map(m, i) == chain_map(m, j) and abs(np.log(scale_of(i) / scale_of(j))) < np.log(group_scale_ratio):
                    chain_groups[m][i] = j
                    break
            else:
                leaders.append(i)
        for i, j in chain_groups[m].items():   # members use the leader's placement (same map frame)
            chain_placements[m][i] = dict(chain_placements[m][j], leader=j)


    if group_scale_ratio > 1.0:
        group_full_chains(initial_lens, lambda i: st0[i]["scale"])
    for i in range(len(ch0)):
        if i not in chain_placements[initial_lens]:
            j = min(good0, key=lambda jj: abs(jj - i))
            chain_placements[initial_lens][i] = {"R": chain_placements[initial_lens][j]["R"], "kind": "rot"}
    cover = {}
    for i, placement in chain_placements[initial_lens].items():
        if placement["kind"] == "full":
            for k in range(ch0[i][0], ch0[i][1] + 1):
                cover[k] = placed_imu_pose(initial_lens, k, placement)
    tk = np.array(sorted(cover))
    init_pose = {k: cover[int(tk[np.argmin(np.abs(tk - k))])] for k in range(frame_count)}
    for m in lenses:
        if m == initial_lens or alignment_estimates[m] is None:
            continue
        for i, (k0, k1) in enumerate(lenses[m]["chains"]):
            common = [k for k in range(k0, k1 + 1) if k in cover]
            if len(common) < fusion_parameters.minimum_common_frames:
                dropped.append({"lens": m, "chain": [k0, k1], "why": "fewer than 10 frames shared with the init lens"})
                continue
            Rs = [cover[k][0] @ (lenses[m]["obs"][k][1] @ lenses[m]["R_CI"]).T for k in common]
            Rm_W = Rot.from_matrix(np.array(Rs)).mean().as_matrix()
            st = alignment_estimates[m]["chains"][i]
            ext = np.linalg.norm(np.ptp(np.array([cover[k][1] for k in common]), axis=0))
            rot_med = float(np.median([np.degrees(np.linalg.norm(Rot.from_matrix(Rm_W.T @ Rr).as_rotvec())) for Rr in Rs]))
            gate = {"lens": m, "chain": [k0, k1], "n_common": len(common), "good_stage1": st["good"], "ext_mm": round(1e3 * ext, 1),
                    "rot_med_deg": round(rot_med, 3)}
            if rot_med > other_lens_rotation_deg:  # map-quality gate (rotation): this chain disagrees with the init lens -> not used at all
                dropped.append(dict(gate, why="rotation median %.2f deg > %.2f" % (rot_med, other_lens_rotation_deg)))
                gates.append(dict(gate, kind="dropped"))
                continue
            if st["good"]:
                A = np.zeros((3 * len(common), 4))
                y = np.zeros(3 * len(common))
                for n, k in enumerate(common):
                    p, Rk = lenses[m]["obs"][k]
                    A[3 * n:3 * n + 3, 0] = Rm_W @ p
                    A[3 * n:3 * n + 3, 1:4] = np.eye(3)
                    y[3 * n:3 * n + 3] = cover[k][1] - Rm_W @ Rk @ lenses[m]["lever"]
                sol = np.linalg.lstsq(A, y, rcond=None)[0]
                rms = float(np.sqrt(((A @ sol - y) ** 2).reshape(-1, 3).sum(1).mean()))
                gate["ls_scale"] = round(float(sol[0]), 5)
                gate["ls_rms_mm"] = round(1e3 * rms, 2)
                if st["good"] and sol[0] > 0 and rms <= other_lens_rms_m:  # gate: chain is IMU-consistent on its own and not grossly off
                    chain_placements[m][i] = {"s": float(sol[0]), "R": Rm_W, "t": sol[1:4], "kind": "full"}
                    gates.append(dict(gate, kind="full"))
                    continue
            chain_placements[m][i] = {"R": Rm_W, "kind": "rot"}
            gates.append(dict(gate, kind="rot"))

    if group_scale_ratio > 1.0:
        for m in lenses:
            if m != initial_lens:
                group_full_chains(m, lambda i, m=m: chain_placements[m][i]["s"])
    log("chain_groups", json.dumps({m: {str(lenses[m]["chains"][i]): str(lenses[m]["chains"][j]) for i, j in g.items()} for m, g in chain_groups.items()}), flush=True)

    # ---------------- segments ----------------
    segment_frames = int(round(segment_duration_s * frame_rate_hz)) if segment_duration_s > 0 else 10 ** 9
    segments = {m: [] for m in lenses}  # list of dicts: chain, k0, k1, sid
    for m in lenses:
        for i, placement in sorted(chain_placements[m].items()):
            if placement["kind"] != "full" or "leader" in placement:
                continue
            k0, k1 = lenses[m]["chains"][i]
            cuts = list(range(k0, k1 + 1, segment_frames))
            if len(cuts) > 1 and (k1 + 1 - cuts[-1]) < segment_frames // 2:
                cuts.pop()
            for n, c0 in enumerate(cuts):
                c1 = (cuts[n + 1] - 1) if n + 1 < len(cuts) else k1
                segments[m].append({"chain": i, "k0": c0, "k1": c1, "sid": len(segments[m])})
    log("chains", json.dumps({m: [[list(lenses[m]["chains"][i]), placement["kind"]] for i, placement in sorted(chain_placements[m].items())] for m in lenses}),
          "segments", {m: len(segments[m]) for m in lenses}, "gates", json.dumps(gates), flush=True)

    return MapPlacements(chain_placements, chain_groups, segments, init_pose, gauge, dropped, gates)
