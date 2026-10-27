"""轨迹预览图；绘图依赖在调用时加载，失败不改变轨迹求解结果。"""
from pathlib import Path
from .quality import runs


def plot(path: Path, t, p_tcp, kind, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"full": "#1f77b4", "rot": "#ff7f0e", "imu": "#d62728"}
    lab = {"full": "camera map: position", "rot": "camera map: orientation only", "imu": "IMU bridge"}
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])
    for idx, (a, b, name) in enumerate(((0, 1, "top view (x-y)"), (0, 2, "side view (x-z)"))):
        ax = fig.add_subplot(gs[0, idx])
        ax.plot(p_tcp[:, a], p_tcp[:, b], color="0.75", lw=0.8, zorder=1)
        for kd in ("full", "rot", "imu"):
            m = kind == kd
            if m.any():
                ax.scatter(p_tcp[m, a], p_tcp[m, b], s=3, color=col[kd], label=lab[kd], zorder=2)
        ax.plot(p_tcp[0, a], p_tcp[0, b], "k^", ms=8, label="start", zorder=3)
        ax.plot(p_tcp[-1, a], p_tcp[-1, b], "ks", ms=7, label="end", zorder=3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("xyz"[a] + " [m]")
        ax.set_ylabel("xyz"[b] + " [m]")
        ax.set_title("TCP " + name)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")
    ax = fig.add_subplot(gs[1, :])
    for j, c in enumerate("xyz"):
        ax.plot(t, p_tcp[:, j] - p_tcp[0, j], lw=1, label=c + " - " + c + "0")
    for k0, k1 in runs(kind != "full"):
        ax.axvspan(t[k0], t[k1], color=col["imu"] if (kind[k0:k1 + 1] == "imu").all() else col["rot"], alpha=0.12, lw=0)
    ax.set_xlabel("video time [s]")
    ax.set_ylabel("TCP position [m]")
    ax.set_title("TCP position over time (shaded: no position from the camera map)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
