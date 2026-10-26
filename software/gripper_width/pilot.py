#!/usr/bin/env python3
"""pilot.py -- marker / grasp / release cuts from gripper width in millimetres.  (from paper/cvpr candy_pilot/pilot.py)

Brought onto dev_Zijian with software/gripper_width/.  Width rows come from apply_width.py or width_4k60.py.

Recording protocol (负责人 2026-10-15): one bag of yellow candy per recording.  Rest 3-5 s, look around 10-15 s with the
gripper CLOSED the whole time, go to the start pose, open and close the gripper once (the marker), then pick the front
bag off the hook, turn, put it in the paper bag on the table, rest 2-3 s, stop.  Training data = from the moment the
marker's close completes to the end, keeping about 1 s of the final rest.

Per recording WORK/<tail>/:
  runs/u1..u3/<stem>/   three runs of the trajectory tool (umi-traj, dev tree with kiwi_fuse3t --gap_init_gyro)
  ../width/<tail>_rows.csv   gripper opening per trajectory row (software/gripper_width/apply_width.py)
What is checked:
  * every run status ok; share of rows pinned by the camera map; gaps without a map position;
  * run-to-run: each pair of runs aligned rigidly on the rows both support; disagreement over the whole recording and
    inside the training part;
  * the marker: the first excursion of the opening above MARK_OPEN_MM, followed by a return to closed (<= CLOSED_MM for
    >= 3 rows); before it the gripper must stay closed (<= SCAN_OPEN_MAX_MM); anything else is flagged;
  * the training part: start = the marker's close, end = last row with TCP speed > MOVE_MPS (0.2 s smoothing) + 1 s;
    its duration, how many times the gripper opens past MARK_OPEN_MM inside it, the trajectory support inside it.

    python pilot.py analyse WORK --tails 006 007 ...     -> WORK/pilot_analysis.json + WORK/<tail>/timeline.png
    python pilot.py sheet WORK --raw RAWDIR --tails ...  -> WORK/<tail>/keyframes.jpg (needs WORK/pilot_analysis.json)
    python pilot.py release_sheet WORK --raw RAWDIR --tails ...  -> WORK/release_check_N.jpg (grasp / release / after, 6 clips per image)

Flags per recording (x["flags"]): extra_open_after_release_cut (training part ends 0.3 s before it), regrasp_N_opens_before_release,
no_open_before_release, release_near_start_pose (< 0.3 m), gripper_opened_during_scan, not_closed_at_end, no_open_in_training_part.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("gripper_run_io", Path(__file__).resolve().parent / "run_io.py")
run_io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_io)
FPS = run_io.FPS            # trajectory rows per second (every 2nd frame of 59.94 fps)
MARK_OPEN_MM = 30.0
EVENT_OPEN_MM = 10.0        # openings inside the training part (029 let the candy go with a 13 mm opening over the bag)
CLOSED_MM = 3.0
SCAN_OPEN_MAX_MM = 8.0
MOVE_MPS = 0.08            # a hand holding the gripper still at the start pose still drifts at 0.03-0.06 m/s (006, 2026-10-15)
KEEP_END_S = 1.0


def width_rows(p: Path) -> np.ndarray:
    return np.array([float(r["width_mm"]) if r["width_mm"] != "" else np.nan for r in csv.DictReader(open(p))])


def marker(w: np.ndarray) -> dict:
    above = np.flatnonzero(w > MARK_OPEN_MM)
    if not len(above):
        return {"found": False, "why": f"the opening never exceeds {MARK_OPEN_MM} mm"}
    i_open = int(above[0])
    # start of that excursion: last closed row before it
    before = np.flatnonzero(w[:i_open] <= CLOSED_MM)
    i_lift = int(before[-1]) + 1 if len(before) else 0
    closed = w <= CLOSED_MM
    i_close = None
    for k in range(i_open, len(w) - 2):
        if closed[k] and closed[k + 1] and closed[k + 2]:
            i_close = k
            break
    pre = w[:i_lift]
    out = {"found": i_close is not None, "lift_row": i_lift, "open_row": i_open, "close_row": i_close,
           "peak_mm": round(float(np.nanmax(w[i_lift:i_close if i_close else len(w)])), 1),
           "marker_s": [round(i_lift / FPS, 2), round((i_close or len(w)) / FPS, 2)],
           "scan_max_open_mm": round(float(np.nanmax(pre)), 1) if len(pre) else None,
           "scan_rows_without_width": int(np.isnan(pre).sum()) if len(pre) else 0}
    out["scan_closed_ok"] = bool(len(pre) == 0 or np.nanmax(pre) <= SCAN_OPEN_MAX_MM)
    if i_close is None:
        out["why"] = "the gripper never closes again after the first opening"
    return out


AGREE_MM = 5.0              # two runs within this (max, training part, aligned there) count as agreeing
MERGE_GAP_S = 0.25          # two openings closer than this are one (the opening wobbling around MARK_OPEN_MM: 007, 026)


def open_spans(w: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """[start, end) row spans where the opening is above thr; spans separated by less than MERGE_GAP_S are merged."""
    a = np.r_[False, np.nan_to_num(w, nan=0.0) > thr, False]
    e = np.flatnonzero(np.diff(a.astype(int)))
    spans = [[int(s), int(t)] for s, t in zip(e[::2], e[1::2])]
    out = []
    for sp in spans:
        if out and sp[0] - out[-1][1] < MERGE_GAP_S * FPS:
            out[-1][1] = sp[1]
        else:
            out.append(sp)
    return [tuple(x) for x in out]


def excursions(w: np.ndarray, thr: float) -> list[list[float]]:
    return [[round(s / FPS, 2), round(t / FPS, 2), round(float(np.nanmax(w[s:t])), 1)] for s, t in open_spans(w, thr)]


LONG = False


def cmd_analyse(a) -> int:
    global LONG
    LONG = bool(getattr(a, "long", False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    work = Path(a.work)
    res = {"protocol": __doc__.split("\n\n")[1], "thresholds": {"MARK_OPEN_MM": MARK_OPEN_MM, "EVENT_OPEN_MM": EVENT_OPEN_MM, "MERGE_GAP_S": MERGE_GAP_S, "CLOSED_MM": CLOSED_MM, "SCAN_OPEN_MAX_MM": SCAN_OPEN_MAX_MM,
                                                              "MOVE_MPS": MOVE_MPS, "KEEP_END_S": KEEP_END_S}, "recordings": {}}
    for tail in a.tails:
        d = work / tail
        runs = {}
        for rd in sorted((d / "runs").iterdir()):
            sub = [p for p in rd.iterdir() if p.is_dir()]
            runs[rd.name] = run_io.load_run(sub[0]) if sub else None
        x = {"runs": {}}
        ok = {}
        for t, r in runs.items():
            if r is None or r["status"] != "ok":
                x["runs"][t] = {"status": None if r is None else r["status"], "reason": None if r is None else r.get("reason")}
                continue
            s, sup = r["summary"], r["support"]
            gaps = [g for g in run_io.gaps_of(sup) if g["dur_s"] >= 0.3]
            x["runs"][t] = {"status": "ok", "chosen_try": s.get("chosen_try"), "tries": len(s.get("tries", [])),
                            "support_fraction": round(float(sup.mean()), 4), "gaps_over_0p3s": gaps,
                            "duration_s": s.get("duration_s"), "suspicious_flags": [tr.get("suspicious") for tr in s.get("tries", [])]}
            ok[t] = r
        w = width_rows(work / "width" / f"{tail}_rows.csv")
        m = marker(w)
        x["marker"] = m
        ref = ok[sorted(ok)[0]] if ok else None
        if ref is not None:
            p = ref["tcp"][:, 1:4]
            n = min(len(p), len(w))
            v = np.r_[0, np.linalg.norm(np.diff(p[:n], axis=0), axis=1)] * FPS
            k = max(1, int(round(0.2 * FPS)))
            v = np.convolve(v, np.ones(k) / k, mode="same")
            moving = np.flatnonzero(v > MOVE_MPS)
            last = int(moving[-1]) if len(moving) else n - 1
            e_end = min(n, last + int(round(KEEP_END_S * FPS)) + 1)
            x["recording_s"] = round(n / FPS, 2)
            x["final_rest_s"] = round((n - 1 - last) / FPS, 2)
            if m.get("found"):
                s0 = m["close_row"]
                # every opening past MARK_OPEN_MM after the marker; the one farthest from the start pose is the release over
                # the paper bag, openings before it are the pre-grasp (more than one = re-grasp), openings after it (back
                # near the shelf) are extra and the training part is cut 0.3 s before the first of them
                ex = []
                for es, ee in open_spans(w[s0:e_end], EVENT_OPEN_MM):
                    r0, r1 = s0 + es, s0 + ee
                    ex.append({"rows": [r0, r1], "s": [round(r0 / FPS, 2), round(r1 / FPS, 2)], "peak_mm": round(float(np.nanmax(w[r0:r1])), 1),
                               "dist_from_start_m": round(float(np.linalg.norm(p[(r0 + r1) // 2] - p[s0])), 3)})
                flags = []
                if ex:
                    rel = int(np.argmax([e["dist_from_start_m"] for e in ex]))
                    for i, e in enumerate(ex):
                        e["role"] = "release" if i == rel else ("before_release" if i < rel else "after_release")
                    post = [e for e in ex if e["role"] == "after_release"]
                    if post and not LONG:   # long videos: split_tasks.py cuts the tasks, the whole working part stays here
                        e_end = max(ex[rel]["rows"][1] + 1, post[0]["rows"][0] - int(round(0.3 * FPS)))
                        flags.append("extra_open_after_release_cut")
                    if rel == 0:
                        flags.append("no_open_before_release")
                    if rel >= 2:
                        flags.append(f"regrasp_{rel}_opens_before_release")
                    if ex[rel]["dist_from_start_m"] < 0.3:
                        flags.append("release_near_start_pose")
                    x["events"] = {"opens": ex, "n_before_release": rel, "n_after_release": len(post), "release_dist_m": ex[rel]["dist_from_start_m"]}
                else:
                    flags.append("no_open_in_training_part")
                if not m.get("scan_closed_ok"):
                    flags.append("gripper_opened_during_scan")
                if np.nanmax(w[max(s0, e_end - 5):e_end]) > CLOSED_MM:
                    flags.append("not_closed_at_end")
                seg = slice(s0, e_end)
                ws = w[seg]
                x["flags"] = flags
                x["training_part"] = {"rows": [s0, e_end], "s": [round(s0 / FPS, 2), round(e_end / FPS, 2)], "duration_s": round((e_end - s0) / FPS, 2),
                                      "opens_over_mark_s_from_start": excursions(np.nan_to_num(ws, nan=0.0), EVENT_OPEN_MM),
                                      "rows_without_width": int(np.isnan(ws).sum()),
                                      "tcp_path_m": round(float(np.linalg.norm(np.diff(p[seg], axis=0), axis=1).sum()), 3),
                                      "support_fraction_ref_run": round(float(ref["support"][seg].mean()), 4),
                                      "gaps_over_0p3s_ref_run": [g for g in run_io.gaps_of(ref["support"][seg]) if g["dur_s"] >= 0.3]}
            # run-to-run
            names = sorted(ok)
            pairs = {}
            for i, ti in enumerate(names):
                for tj in names[i + 1:]:
                    pi, pj = ok[ti]["tcp"][:, 1:4], ok[tj]["tcp"][:, 1:4]
                    nn = min(len(pi), len(pj))
                    both = ok[ti]["support"][:nn] & ok[tj]["support"][:nn]
                    R, tt = run_io.umeyama_rigid(pj[:nn][both], pi[:nn][both])
                    dd = np.linalg.norm((R @ pj[:nn].T).T + tt - pi[:nn], axis=1)
                    q = {"rms_supported_mm": round(1e3 * float(np.sqrt(np.mean(dd[both] ** 2))), 2), "max_all_mm": round(1e3 * float(dd.max()), 2),
                         "t_at_max_s": round(int(np.argmax(dd)) / FPS, 2)}
                    if "training_part" in x:
                        s0, s1 = x["training_part"]["rows"]
                        s1 = min(s1, nn)
                        q["max_in_training_part_mm"] = round(1e3 * float(dd[s0:s1].max()), 2)
                        # the training part only: aligned on its own supported rows (an episode is relative to its first
                        # frame, so a wrong map placement during the scan before it does not matter)
                        bt = both[s0:s1]
                        if bt.sum() >= 30:
                            R2, t2 = run_io.umeyama_rigid(pj[s0:s1][bt], pi[s0:s1][bt])
                            d2 = np.linalg.norm((R2 @ pj[s0:s1].T).T + t2 - pi[s0:s1], axis=1)
                            q["training_part_aligned_there_rms_mm"] = round(1e3 * float(np.sqrt(np.mean(d2[bt] ** 2))), 2)
                            q["training_part_aligned_there_max_mm"] = round(1e3 * float(d2.max()), 2)
                    pairs[f"{ti}|{tj}"] = q
            x["run_to_run"] = pairs
            # which run to use: the one that agrees best with the others inside the training part (runs of the tool can
            # land in a wrong map placement without any warning: 014 u1, 015 u3, 016 u2 were 1-4 cm off the other two)
            if "training_part" in x and len(names) >= 2:
                def dmax(a_, b_):
                    q = pairs.get(f"{a_}|{b_}") or pairs.get(f"{b_}|{a_}") or {}
                    return q.get("training_part_aligned_there_max_mm", float("inf"))
                score = {t_: sorted(dmax(t_, u_) for u_ in names if u_ != t_)[0] for t_ in names}
                best = min(names, key=lambda t_: (score[t_], t_))
                x["use_run"] = {"run": best, "best_agreement_mm": score[best], "n_runs": len(names)}
                if score[best] > AGREE_MM:
                    x.setdefault("flags", []).append(f"runs_disagree_best_{score[best]:.1f}mm")
                elif len(names) == 2 and score[best] > AGREE_MM / 2:
                    x.setdefault("flags", []).append("two_runs_only_check")
            # timeline figure
            fig, ax = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
            t = np.arange(n) / FPS
            ax[0].plot(t, w[:n], lw=1); ax[0].set_ylabel("opening mm"); ax[0].axhline(MARK_OPEN_MM, ls=":", c="gray")
            ax[1].plot(t, v[:n], lw=1); ax[1].set_ylabel("TCP speed m/s"); ax[1].axhline(MOVE_MPS, ls=":", c="gray")
            for tn in names:
                ax[2].plot(t[:len(ok[tn]["support"])], ok[tn]["support"][:n] + 0.05 * names.index(tn), lw=1, label=tn)
            ax[2].set_ylabel("map position"); ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(["no", "yes"]); ax[2].legend(loc="lower right")
            if "training_part" in x:
                for axx in ax:
                    axx.axvspan(*x["training_part"]["s"], color="C2", alpha=0.12)
            if m.get("lift_row") is not None:
                for axx in ax:
                    axx.axvspan(m["marker_s"][0], m["marker_s"][1], color="C1", alpha=0.25)
            ax[2].set_xlabel("time s (orange = marker open-close, green = training part)")
            fig.suptitle(f"{tail}: gripper opening, speed, map support")
            fig.tight_layout(); fig.savefig(d / "timeline.png", dpi=80); plt.close(fig)
        res["recordings"][tail] = x
        tp = x.get("training_part", {})
        print(tail, json.dumps({"runs_ok": sum(1 for r in x["runs"].values() if r["status"] == "ok"), "support": [r.get("support_fraction") for r in x["runs"].values()],
                                "marker": {k: m.get(k) for k in ("found", "marker_s", "peak_mm", "scan_max_open_mm", "scan_closed_ok")},
                                "train_s": tp.get("s"), "train_dur": tp.get("duration_s"), "opens_from_start": tp.get("opens_over_mark_s_from_start"),
                                "r2r_train_aligned_max_mm": [q.get("training_part_aligned_there_max_mm") for q in x.get("run_to_run", {}).values()],
                                "final_rest_s": x.get("final_rest_s"), "use_run": x.get("use_run"), "flags": x.get("flags")}))
    (work / "pilot_analysis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


def find_insv(raw_dirs: list, tail: str) -> Path:
    """The recording with this tail number in one of the given folders (not recursive: tails repeat across days)."""
    hits = [f for d in raw_dirs for f in Path(d).glob(f"VID_*_{tail}.insv")]
    if len(hits) != 1:
        raise SystemExit(f"tail {tail}: {len(hits)} recordings in {raw_dirs}")
    return hits[0]


def cmd_sheet(a) -> int:
    import av
    import cv2

    work = Path(a.work)
    res = json.loads((work / "pilot_analysis.json").read_text())
    for tail in a.tails:
        x = res["recordings"][tail]
        insv = find_insv(a.raw, tail)
        m, tp = x["marker"], x.get("training_part")
        marks = {}
        if m.get("lift_row") is not None:
            marks[2 * ((m["lift_row"] + (m["close_row"] or m["open_row"])) // 2)] = "marker open"
            marks[max(0, 2 * m["lift_row"] - 120)] = "scan (2 s before marker)"
        if tp:
            s0, s1 = tp["rows"]
            for f, lab in ((0, "train start"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1.0, "train end")):
                marks[2 * int(round(s0 + f * (s1 - 1 - s0)))] = lab
            for e in x.get("events", {}).get("opens", []):
                marks[2 * ((e["rows"][0] + e["rows"][1]) // 2)] = f"{e['role']} {e['peak_mm']:.0f} mm"
        tiles = {}
        with av.open(str(insv)) as c:
            st = c.streams.video[0]; st.thread_type = "AUTO"
            last = max(marks)
            for k, fr in enumerate(c.decode(st)):
                if k in marks:
                    img = fr.to_ndarray(format="bgr24")[:, 1920:][240:1680, 240:1680]
                    img = cv2.resize(img, (480, 480))
                    cv2.putText(img, f"{k / 59.94:.1f}s {marks[k]}", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    tiles[k] = img
                if k >= last:
                    break
        ts = [tiles[k] for k in sorted(tiles)]
        while len(ts) % 4:
            ts.append(np.zeros_like(ts[0]))
        cv2.imwrite(str(work / tail / "keyframes.jpg"), np.vstack([np.hstack(ts[i:i + 4]) for i in range(0, len(ts), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(tail, "keyframes", len(tiles))
    return 0


def cmd_release_sheet(a) -> int:
    """One row per recording: the grasp (0.3 s after the last pre-grasp opening closes), the release (middle of the
    opening over the bag) and 0.4 s after the release closes -- to see by eye whether the candy went into the bag."""
    import av
    import cv2

    work = Path(a.work)
    res = json.loads((work / "pilot_analysis.json").read_text())
    rows = []
    for tail in a.tails:
        x = res["recordings"][tail]
        ops = x.get("events", {}).get("opens", [])
        rel = [e for e in ops if e["role"] == "release"]
        pre = [e for e in ops if e["role"] == "before_release"]
        want = {}
        if pre:
            want[2 * (pre[-1]["rows"][1] + int(0.3 * FPS))] = "grasp+0.3s"
        if rel:
            want[2 * ((rel[0]["rows"][0] + rel[0]["rows"][1]) // 2)] = f"release {rel[0]['peak_mm']:.0f}mm"
            want[2 * (rel[0]["rows"][1] + int(0.4 * FPS))] = "release+0.4s"
        insv = find_insv(a.raw, tail)
        tiles = {}
        with av.open(str(insv)) as c:
            st = c.streams.video[0]; st.thread_type = "AUTO"
            for k, fr in enumerate(c.decode(st)):
                if k in want:
                    img = cv2.resize(fr.to_ndarray(format="bgr24")[:, 1920:][240:1680, 240:1680], (360, 360))
                    cv2.putText(img, f"{tail} {k / 59.94:.1f}s {want[k]}", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    tiles[k] = img
                if want and k >= max(want):
                    break
        row = [tiles[k] for k in sorted(tiles)]
        while len(row) < 3:
            row.append(np.zeros((360, 360, 3), np.uint8))
        rows.append(np.hstack(row[:3]))
    for i in range(0, len(rows), 6):
        cv2.imwrite(str(work / f"release_check_{i // 6 + 1}.jpg"), np.vstack(rows[i:i + 6]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("release sheets", (len(rows) + 5) // 6)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyse"); p.add_argument("work"); p.add_argument("--tails", nargs="+", required=True)
    p.add_argument("--long", action="store_true", help="several tasks per video (from 2026-10-16): keep the whole working part after the first marker; split_tasks.py cuts it")
    p = sub.add_parser("sheet"); p.add_argument("work"); p.add_argument("--raw", nargs="+", required=True); p.add_argument("--tails", nargs="+", required=True)
    p = sub.add_parser("release_sheet"); p.add_argument("work"); p.add_argument("--raw", nargs="+", required=True); p.add_argument("--tails", nargs="+", required=True)
    a = ap.parse_args(argv)
    return {"analyse": cmd_analyse, "sheet": cmd_sheet, "release_sheet": cmd_release_sheet}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
