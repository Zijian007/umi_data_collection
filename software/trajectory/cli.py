"""umi-traj -- Insta360 X3 recordings in, gripper trajectories out.

    umi-traj <video.insv | folder> [more ...] [--out DIR] [--mode handheld|dual] [--jobs N] [--keep-work] [--force]
                                         [--device SERIAL] [--gripper ID]
    umi-traj check                         # is everything installed?

A folder means every *.insv directly inside it.  Results go to <out>/<video name>/ (default <out> = umi_traj_out/ next to
the input).  A video whose result folder already holds a successful summary.json is skipped, so new recordings can simply
be dropped into the same folder and the command run again (--force reprocesses).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from software.registry import RegistryError, list_devices, resolve_insv  # noqa: E402
from software.trajectory import calibration  # noqa: E402
from software.trajectory.x3_extract import Refused, extract  # noqa: E402

RUNNER = HERE / "run_kiwi_segment.sh"
_print_lock = threading.Lock()


@dataclass(frozen=True)
class RunRequest:
    """One recording through the trajectory tool. The command line fills this in; other callers can too."""

    mode: str = "handheld"
    max_tries: int = 3
    cores: str = ""
    keep_work: bool = False
    force: bool = False
    allow_irregular: bool = False
    device: str = ""
    gripper: str = ""


def say(msg: str) -> None:
    with _print_lock:
        print(time.strftime("%H:%M:%S ") + msg, flush=True)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or os.environ.get("UMI_COMMIT", "unknown")
    except Exception:
        return os.environ.get("UMI_COMMIT", "unknown")


def find_inputs(paths: list[str]) -> list[Path]:
    vids: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            vids += sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".insv")
        elif p.is_file():
            vids.append(p)
        else:
            raise SystemExit(f"not found: {p}")
    seen, out = set(), []
    for v in vids:
        r = v.resolve()
        if r not in seen:
            seen.add(r)
            out.append(v)
    return out


def default_out(paths: list[str]) -> Path:
    p = Path(paths[0])
    return (p if p.is_dir() else p.parent) / "umi_traj_out"


def already_done(odir: Path) -> bool:
    try:
        return json.loads((odir / "summary.json").read_text(encoding="utf-8")).get("status") == "ok"
    except Exception:
        return False


def write_times_csv(mav0: Path, path: Path) -> None:
    rows = [l.split(",")[0] for l in (mav0 / "cam0" / "data.csv").read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]
    path.write_text("t_canonical_s\n" + "".join("%.9f\n" % (int(r) / 1e9) for r in rows), encoding="utf-8")


# A run is suspicious when the fusion placed the camera map more than once or the map gave a position to fewer than
# 95% of the frames. ORB-SLAM3 mapping is not deterministic. Among the tries, the one with the fewest placements and
# then the largest map-position share is kept.
SUSPECT_FULL_FRACTION = 0.95


def assess_try(kd: Path) -> dict | None:
    try:
        info = json.loads((kd / "fuse" / "fuse_info.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    lens = info.get("init_lens") or "front"
    chains = (info.get("chains") or {}).get(lens, [])
    full = [c for c in chains if c[1] == "full"]
    grouped = (info.get("params", {}).get("chain_groups") or {}).get(lens, {})
    placements = max(1, len(full) - len(grouped))
    n = max(1, int(info.get("frames") or 1))
    full_fraction = sum(int(c[0][1]) - int(c[0][0]) + 1 for c in full) / n
    return {"placements": placements, "full_fraction": round(full_fraction, 4),
            "suspicious": bool(placements > 1 or full_fraction < SUSPECT_FULL_FRACTION)}


def try_key(q: dict) -> tuple:
    return (q["placements"], -q["full_fraction"])


def run_recording(video: Path, out_root: Path, req: RunRequest) -> dict:
    from software.trajectory.postprocess import postprocess

    name = video.stem
    odir = out_root / name
    try:
        cal = resolve_insv(video, device=req.device or None, gripper=req.gripper or None)
    except RegistryError as e:
        say(f"[{name}] FAILED: {e}")
        return {"video": video.name, "status": "failed", "out": str(odir), "reason": str(e)}
    if not req.force and already_done(odir):
        say(f"[{name}] already done -> {odir} (use --force to redo)")
        return {"video": video.name, "status": "skipped", "out": str(odir)}
    if odir.exists():
        shutil.rmtree(odir)
    work = odir / "work"
    logs = odir / "logs"
    logs.mkdir(parents=True)
    t0 = time.time()
    summary = {"video": video.name, "status": "failed", "out": str(odir), "tool_commit": git_commit()}
    try:
        summary["calibration_used"] = {k: cal.get(k) for k in
                                       ("source", "id", "dir", "serial", "gripper", "status", "selection")}
        run_env = dict(os.environ, **calibration.runner_env(cal))
        who = cal.get("serial") or cal.get("id")
        if cal.get("selection") == "UMI_TRAJ_CALIB":
            say(f"[{name}] UMI_TRAJ_CALIB override {cal['id']} ({cal['dir']})")
        else:
            say(f"[{name}] camera {who}  calib {cal['id']}  gripper {cal.get('gripper')}  ({cal['dir']})")
        say(f"[{name}] 1/3 reading timestamps + IMU")
        try:
            man = extract(video, work / "dataset")
        except (Refused, RegistryError) as e:
            summary["reason"] = f"refused: {e}"
            raise
        fr = man["frames"]
        times = []
        if not fr["regular"]:
            if not req.allow_irregular:
                summary["reason"] = ("frame timing is irregular (the camera dropped frames: max deviation %.1f ms, %d gaps); "
                                     "not processed by default -- rerun with --allow-irregular (unvalidated path)"
                                     % (fr["max_deviation_from_29p97_grid_ms"], fr["gaps_over_1p5x_median"]))
                raise RuntimeError(summary["reason"])
            write_times_csv(work / "dataset" / "mav0", work / "times.csv")
            times = ["--times", str(work / "times.csv")]
        say(f"[{name}]     {fr['n_frames_used']} frames, {fr['duration_s']:.1f} s")
        say(f"[{name}] 2/3 lens video -> map -> relocalization -> IMU fusion (several minutes)")
        tries, best = [], None
        for i in range(1, max(1, req.max_tries) + 1):
            kd = work / ("kiwi" if i == 1 else "kiwi_try%d" % i)
            cmd = ["bash", str(RUNNER), "--insv", str(video), "--euroc", str(work / "dataset" / "mav0"), "--out", str(kd),
                   "--mode", req.mode] + (["--cores", req.cores] if req.cores else []) + times
            if i > 1:
                cmd += ["--lens-video-from", str(work / "kiwi")]
            runner_log = logs / ("runner_try%d.log" % i)
            with open(runner_log, "w", encoding="utf-8") as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=run_env).returncode
            q = assess_try(kd) if rc == 0 else None
            tries.append({"try": i, "rc": rc, **(q or {})})
            if q is not None and (best is None or try_key(q) < try_key(tries[best - 1])):
                best = i
            if q is not None and not q["suspicious"]:
                break
            if i < req.max_tries:
                why = ("failed (rc %d)" % rc) if q is None else (
                    "%d separate map placements, map position on %.0f%% of frames" % (q["placements"], 100 * q["full_fraction"]))
                say(f"[{name}]     try {i}: {why} -> mapping again (ORB-SLAM3 maps differently every run)")
        summary["tries"] = tries
        if best is None:
            last = logs / ("runner_try%d.log" % len(tries))
            tail = last.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
            summary["reason"] = "trajectory step failed in all %d tries (last rc %d): %s" % (len(tries), tries[-1]["rc"], " | ".join(tail))
            raise RuntimeError(summary["reason"])
        summary["chosen_try"] = best
        kd = work / ("kiwi" if best == 1 else "kiwi_try%d" % best)
        # Unique artifacts only. TRACKED counts, extract timing, and fuse JSON also live in summary.json /
        # run_settings.json; runner stdout and fuse.log are kept for failed tries (work/ is deleted on success).
        for src, dest in (
            (kd / "logs" / "run.txt", logs / "run.txt"),
            (kd / "fuse" / "fuse_info.json", logs / "fuse_info.json"),
            (kd / "front" / "map" / "log.txt", logs / "front_map_log.txt"),
            (kd / "front" / "loc" / "frames.csv", logs / "front_loc_frames.csv"),
        ):
            if src.is_file():
                shutil.copy(src, dest)
        (logs / ("runner_try%d.log" % best)).unlink(missing_ok=True)
        say(f"[{name}] 3/3 writing results")
        s = postprocess(kd, man, odir, req.mode, video.name, cal, command={
            "max_tries": req.max_tries, "chosen_try": best, "tries": tries,
            "allow_irregular": req.allow_irregular, "keep_work": req.keep_work,
            "cores": req.cores or None,
            "UMI_TRAJ_CALIB": os.environ.get("UMI_TRAJ_CALIB") or None,
            "device": req.device or None, "gripper": req.gripper or None,
            "serial": cal.get("serial"),
        })
        summary.update(s)
        summary["status"] = "ok"
    except Exception as e:
        summary.setdefault("reason", repr(e))
    finally:
        summary["wall_s"] = round(time.time() - t0, 1)
        summary["tool_commit"] = summary.get("tool_commit") or git_commit()
        (odir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
        if summary["status"] == "ok" and not req.keep_work:
            shutil.rmtree(work, ignore_errors=True)
    if summary["status"] == "ok":
        say(f"[{name}] done in {summary['wall_s']:.0f} s: TCP path {summary['tcp_path_length_m']:.2f} m, map position on "
            f"{100 * summary['constraint_fraction']['full']:.0f}% of frames -> {odir}")
        if len(summary.get("tries", [])) > 1:
            say(f"[{name}]     mapped {len(summary['tries'])} times, kept try {summary['chosen_try']}")
        for n in summary.get("notes", []):
            say(f"[{name}]     note: {n}")
    else:
        say(f"[{name}] FAILED after {summary['wall_s']:.0f} s: {summary.get('reason')}  (logs: {logs})")
    return summary


def check() -> int:
    ok = True

    def line(good: bool, what: str, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(("  OK   " if good else "  FAIL ") + what + (("  " + detail) if detail else ""))

    print("umi-traj environment check (commit %s)" % git_commit())
    for mod in ("numpy", "scipy", "gtsam", "av", "matplotlib"):
        try:
            m = __import__(mod)
            line(True, "python module " + mod, getattr(m, "__version__", ""))
        except Exception as e:
            line(False, "python module " + mod, repr(e))
    for tool in ("ffmpeg", "ffprobe"):
        p = shutil.which(tool)
        line(p is not None, tool, p or "not on PATH")
    try:
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
        line("libx264" in enc, "ffmpeg encoder libx264")
    except Exception as e:
        line(False, "ffmpeg encoder libx264", repr(e))
    orb = Path(os.environ.get("ORB3_DIR", ""))
    exe = orb / "Examples" / "Monocular-Inertial" / "x3_mono_slam"
    voc = Path(os.environ.get("ORB3_VOCAB", orb / "Vocabulary" / "ORBvoc.txt"))
    line(exe.is_file(), "x3_mono_slam", str(exe))
    line(voc.is_file(), "ORB vocabulary", str(voc))
    if exe.is_file():
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join([str(orb / "lib"), str(orb / "Thirdparty/DBoW2/lib"), str(orb / "Thirdparty/g2o/lib"),
                                           os.environ.get("ORB3_EXTRA_LD", ""), env.get("LD_LIBRARY_PATH", "")])
        r = subprocess.run([str(exe), "--help"], capture_output=True, text=True, env=env, timeout=60)
        line(r.returncode == 0, "x3_mono_slam runs", "" if r.returncode == 0 else (r.stderr or r.stdout)[-300:])
    for f in (HERE / "calib" / "calibration_r1440.json", HERE / "calib" / "t_tcp_camera_nominal_cad.json",
              HERE / "config" / "front_x3_r1440_720.yaml", HERE / "config" / "front_hand_mask.png"):
        line(f.is_file(), "file " + str(f.relative_to(REPO)))
    try:
        rows = list_devices()
        line(bool(rows), "device registry", "%d camera(s)" % len(rows))
        for d in rows:
            line(bool(d.get("ok")), "  " + str(d.get("serial")),
                 ("%s  calib %s  gripper %s" % (d.get("status"), d.get("calib_id"), d.get("gripper")))
                 if d.get("ok") else d.get("error", ""))
    except Exception as e:
        line(False, "device registry", repr(e))
    print("ALL OK" if ok else "SOMETHING IS MISSING -- rerun install.sh (see its log in <deps>/logs)")
    return 0 if ok else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "check":
        return check()
    if argv and argv[0] == "run":
        argv = argv[1:]
    ap = argparse.ArgumentParser(prog="umi-traj", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".insv files and/or folders containing them")
    ap.add_argument("--out", type=Path, default=None, help="result folder (default: umi_traj_out/ next to the first input)")
    ap.add_argument("--mode", choices=["handheld", "dual"], default="handheld",
                    help="handheld (default): front lens + IMU; dual: also the rear lens (when it sees the room)")
    ap.add_argument("--jobs", type=int, default=1, help="videos processed at the same time (each uses ~4 CPU cores)")
    ap.add_argument("--cores", default="", help="optional CPU list for taskset, e.g. 0-3 (only sensible with --jobs 1)")
    ap.add_argument("--max-tries", type=int, default=3,
                    help="map again (up to this many runs) when a run looks unreliable; 1 = never (default 3)")
    ap.add_argument("--keep-work", action="store_true", help="keep intermediate files (lens video, maps) in <result>/work")
    ap.add_argument("--force", action="store_true", help="reprocess videos that already have a successful result")
    ap.add_argument("--allow-irregular", action="store_true",
                    help="also process recordings with dropped frames (uses real frame times; unvalidated)")
    ap.add_argument("--device", default="", help="registry serial (default: read from the .insv trailer)")
    ap.add_argument("--gripper", default="", help="registry gripper id (default: the device's recorded gripper)")
    a = ap.parse_args(argv)
    req = RunRequest(mode=a.mode, max_tries=a.max_tries, cores=a.cores, keep_work=a.keep_work,
                     force=a.force, allow_irregular=a.allow_irregular, device=a.device, gripper=a.gripper)
    videos = find_inputs(a.inputs)
    if not videos:
        print("no .insv found in: " + " ".join(a.inputs))
        return 1
    out_root = (a.out or default_out(a.inputs)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    say(f"{len(videos)} video(s) -> {out_root}  (mode {a.mode}, {a.jobs} at a time)")
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        results = list(ex.map(lambda v: run_recording(v, out_root, req), videos))
    print()
    print("%-40s %-8s %s" % ("video", "status", "result / reason"))
    for r in results:
        print("%-40s %-8s %s" % (r["video"], r["status"], r["out"] if r["status"] != "failed" else r.get("reason")))
    return 0 if all(r["status"] in ("ok", "skipped") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
