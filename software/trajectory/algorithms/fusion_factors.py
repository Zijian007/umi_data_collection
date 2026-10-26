"""视觉因子、连续性因子及解析 Jacobian 自检；不读取命令行或数据文件。"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import gtsam
from gtsam.symbol_shorthand import X

def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def jr_inv(phi):
    th = np.linalg.norm(phi); K = skew(phi)
    if th < 1e-6:
        return np.eye(3) + 0.5 * K
    return np.eye(3) + 0.5 * K + (1.0 / th ** 2 - (1 + np.cos(th)) / (2 * th * np.sin(th))) * K @ K


def logR(Rm):
    return gtsam.Rot3.Logmap(gtsam.Rot3(Rm))


# ---------------- custom factors (analytic Jacobians; checked by --selftest) ----------------
def make_lens_factor(noise, kx, kl, kr, kt, klev, p_m, M, R_m, lever_fixed):
    """full lens observation; klev=None -> lever fixed"""
    keys = [kx, kl, kr, kt] + ([klev] if klev is not None else [])

    def f(this, v, H):
        ks = this.keys()
        T = v.atPose3(ks[0]); sc = np.exp(v.atVector(ks[1])[0]); Rwm = v.atRot3(ks[2]).matrix(); tt = v.atVector(ks[3])
        lv = v.atVector(ks[4]) if klev is not None else lever_fixed
        Rx, px = T.rotation().matrix(), T.translation()
        e_rot = logR((Rwm @ M).T @ Rx)
        w = sc * p_m + R_m @ lv
        e_pos = px - Rwm @ w - tt
        if H is not None:
            Ji = jr_inv(e_rot); C = (Rwm @ M).T @ Rx
            HX = np.zeros((6, 6)); HX[:3, :3] = Ji; HX[3:, 3:] = Rx
            HS = np.zeros((6, 1)); HS[3:, 0] = -Rwm @ p_m * sc
            HR = np.zeros((6, 3)); HR[:3, :] = -Ji @ C.T @ M.T; HR[3:, :] = Rwm @ skew(w)
            HT = np.zeros((6, 3)); HT[3:, :] = -np.eye(3)
            H[0], H[1], H[2], H[3] = HX, HS, HR, HT
            if klev is not None:
                HL = np.zeros((6, 3)); HL[3:, :] = -Rwm @ R_m; H[4] = HL
        return np.concatenate([e_rot, e_pos])
    return gtsam.CustomFactor(noise, keys, f)


def make_lens_factor_map(noise, kx, kl, kr, kt, klev, p_m, M, R_m, lever_fixed):
    """full lens observation, position residual in map units: e_pos = (R^T (p_X - t) - R_m l)/s - p_m"""
    keys = [kx, kl, kr, kt] + ([klev] if klev is not None else [])

    def f(this, v, H):
        ks = this.keys()
        T = v.atPose3(ks[0]); sc = np.exp(v.atVector(ks[1])[0]); Rwm = v.atRot3(ks[2]).matrix(); tt = v.atVector(ks[3])
        lv = v.atVector(ks[4]) if klev is not None else lever_fixed
        Rx, px = T.rotation().matrix(), T.translation()
        e_rot = logR((Rwm @ M).T @ Rx)
        wv = Rwm.T @ (px - tt)
        u = wv - R_m @ lv
        e_pos = u / sc - p_m
        if H is not None:
            Ji = jr_inv(e_rot); C = (Rwm @ M).T @ Rx
            HX = np.zeros((6, 6)); HX[:3, :3] = Ji; HX[3:, 3:] = Rwm.T @ Rx / sc
            HS = np.zeros((6, 1)); HS[3:, 0] = -u / sc
            HR = np.zeros((6, 3)); HR[:3, :] = -Ji @ C.T @ M.T; HR[3:, :] = skew(wv) / sc
            HT = np.zeros((6, 3)); HT[3:, :] = -Rwm.T / sc
            H[0], H[1], H[2], H[3] = HX, HS, HR, HT
            if klev is not None:
                HL = np.zeros((6, 3)); HL[3:, :] = -R_m / sc; H[4] = HL
        return np.concatenate([e_rot, e_pos])
    return gtsam.CustomFactor(noise, keys, f)


def make_rot_factor(noise, kx, kr, M):
    def f(this, v, H):
        ks = this.keys()
        Rx = v.atPose3(ks[0]).rotation().matrix(); Rwm = v.atRot3(ks[1]).matrix()
        e = logR((Rwm @ M).T @ Rx)
        if H is not None:
            Ji = jr_inv(e); C = (Rwm @ M).T @ Rx
            HX = np.zeros((3, 6)); HX[:, :3] = Ji
            H[0], H[1] = HX, -Ji @ C.T @ M.T
        return e
    return gtsam.CustomFactor(noise, [kx, kr], f)


def make_cont_factor(noise, ka, kb, p_b):
    """ka, kb = (lam, R, T) keys of consecutive segments"""
    def f(this, v, H):
        ks = this.keys()
        la, Ra, ta = v.atVector(ks[0])[0], v.atRot3(ks[1]).matrix(), v.atVector(ks[2])
        lb, Rb, tb = v.atVector(ks[3])[0], v.atRot3(ks[4]).matrix(), v.atVector(ks[5])
        sa, sb = np.exp(la), np.exp(lb)
        eR = logR(Ra.T @ Rb)
        ep = sb * Rb @ p_b + tb - sa * Ra @ p_b - ta
        if H is not None:
            Ji = jr_inv(eR)
            Hla = np.zeros((7, 1)); Hla[0, 0] = -1; Hla[4:, 0] = -sa * Ra @ p_b
            HRa = np.zeros((7, 3)); HRa[1:4, :] = -Ji @ Rb.T @ Ra; HRa[4:, :] = sa * Ra @ skew(p_b)
            Hta = np.zeros((7, 3)); Hta[4:, :] = -np.eye(3)
            Hlb = np.zeros((7, 1)); Hlb[0, 0] = 1; Hlb[4:, 0] = sb * Rb @ p_b
            HRb = np.zeros((7, 3)); HRb[1:4, :] = Ji; HRb[4:, :] = -sb * Rb @ skew(p_b)
            Htb = np.zeros((7, 3)); Htb[4:, :] = np.eye(3)
            H[0], H[1], H[2], H[3], H[4], H[5] = Hla, HRa, Hta, Hlb, HRb, Htb
        return np.concatenate([[lb - la], eR, ep])
    return gtsam.CustomFactor(noise, list(ka) + list(kb), f)


def selftest():
    rng = np.random.default_rng(0)
    vals = gtsam.Values()
    kx, kl, kr, kt, klev, kl2, kr2, kt2 = X(0), gtsam.symbol("p", 0), gtsam.symbol("q", 0), gtsam.symbol("r", 0), \
        gtsam.symbol("l", 0), gtsam.symbol("p", 1), gtsam.symbol("q", 1), gtsam.symbol("r", 1)
    vals.insert(kx, gtsam.Pose3(gtsam.Rot3.Expmap(rng.normal(size=3)), gtsam.Point3(*rng.normal(size=3))))
    vals.insert(kl, np.array([0.3])); vals.insert(kr, gtsam.Rot3.Expmap(rng.normal(size=3))); vals.insert(kt, rng.normal(size=3))
    vals.insert(klev, rng.normal(size=3) * 0.02)
    vals.insert(kl2, np.array([0.35])); vals.insert(kr2, gtsam.Rot3.Expmap(rng.normal(size=3) * 0.05) * vals.atRot3(kr))
    vals.insert(kt2, rng.normal(size=3))
    R_m = Rot.from_rotvec(rng.normal(size=3)).as_matrix(); M = R_m @ Rot.from_rotvec(rng.normal(size=3)).as_matrix()
    p_m = rng.normal(size=3)
    n6, n3, n7 = (gtsam.noiseModel.Unit.Create(d) for d in (6, 3, 7))
    facs = {"lens_rear": make_lens_factor(n6, kx, kl, kr, kt, klev, p_m, M, R_m, None),
            "lens_front": make_lens_factor(n6, kx, kl, kr, kt, None, p_m, M, R_m, rng.normal(size=3) * 0.02),
            "lens_map_rear": make_lens_factor_map(n6, kx, kl, kr, kt, klev, p_m, M, R_m, None),
            "lens_map_front": make_lens_factor_map(n6, kx, kl, kr, kt, None, p_m, M, R_m, rng.normal(size=3) * 0.02),
            "rot": make_rot_factor(n3, kx, kr, M),
            "cont": make_cont_factor(n7, (kl, kr, kt), (kl2, kr2, kt2), p_m)}
    worst = 0.0
    for name, fac in facs.items():
        Hs = fac.linearize(vals).jacobian()[0]  # unit noise -> raw Jacobian
        e0 = fac.unwhitenedError(vals); col = 0
        for key in fac.keys():
            x0 = vals.atPose3(key) if gtsam.Symbol(key).chr() == ord("x") else None
            dim = 6 if x0 is not None else (3 if gtsam.Symbol(key).chr() in (ord("q"),) else len(np.atleast_1d(
                vals.atVector(key))) if gtsam.Symbol(key).chr() != ord("q") else 3)
            for i in range(dim):
                d = np.zeros(dim); d[i] = 1e-6; v2 = gtsam.Values(vals)
                c = gtsam.Symbol(key).chr()
                if c == ord("x"):
                    v2.update(key, vals.atPose3(key).retract(d))
                elif c == ord("q"):
                    v2.update(key, vals.atRot3(key).retract(d))
                else:
                    v2.update(key, vals.atVector(key) + d)
                num = (fac.unwhitenedError(v2) - e0) / 1e-6
                err = np.abs(num - Hs[:, col]).max(); worst = max(worst, err); col += 1
        print("selftest", name, "max |analytic - numeric| so far", "%.2e" % worst)
    print("SELFTEST", "PASS" if worst < 1e-4 else "FAIL", "%.2e" % worst)
    return worst < 1e-4


