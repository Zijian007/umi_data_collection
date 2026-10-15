"""Official Media SDK export wrapper: the only legitimate producer of equirect frames.

Responsibility: call ``MediaSDKTest.exe`` with EXPLICIT fixed parameters to export the requested
frame indices as equirect PNGs, and write ``sdk_export_manifest.json``.  The projector consumes only
that manifest -- hand-written parameters are refused:

  * SDK identity: sha256 of ``MediaSDKTest.exe`` and ``MediaSDK.dll``, plus the SDK version dir name;
  * source identity: chunked sha256 and byte size of every input INSV;
  * the full command line, cwd, exit code, start/end UTC;
  * requested frame indices vs frames actually produced (name, sha256, size) -- a mismatch is a FAIL;
  * the output directory must be empty (no mixing in stale frames); a frame-count guard (default 64)
    keeps this wrapper to sampling / rehearsal work -- full-length production must stream instead of
    writing thousands of panorama PNGs to disk.

Usage::

    python -m software.camera.x3.cli sdk-export --insv <_00.insv> <_10.insv> --out DIR --frames 0 300
    python -m software.camera.x3.sdk_export --insv ... --out DIR --frames 0 300   (same arguments)

Provenance: a logic-level COPY of ``software/postprocess/x3_sdk_export.py`` (not sha-pinned, so that
path is now a thin deprecated wrapper around this module).  Behaviour, manifest keys and the exact
error message strings are unchanged; the existing suite matches on those strings.  The only edit is
the repo-root depth, which changed with the directory level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]   # software/camera/x3/ -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_SDK_BIN = _REPO_ROOT / "data/vendor_sdk/insta360/MediaSDK-3.1.3/MediaSDK-3.1.3-20260128-win64/MediaSDK/bin"
MANIFEST_NAME = "sdk_export_manifest.json"
MANIFEST_VERSION = "1"


class ExportError(RuntimeError):
    """Export / verification failure (explicit exception; never ``assert``)."""


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _png_size(path: Path) -> tuple[int, int]:
    """Read the PNG IHDR for width/height (no pixel decoding)."""
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ExportError(f"非 PNG 或损坏: {path}")
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def run_export(insv: list[Path], out: Path, frames: list[int], stitch_type: str,
               output_size: str, sdk_bin: Path, max_frames: int,
               extra_flags: list[str] | None = None) -> Path:
    if not frames:
        raise ExportError("未指定帧索引")
    if len(frames) > max_frames:
        raise ExportError(f"请求 {len(frames)} 帧 > 上限 {max_frames}：本 wrapper 只服务抽帧/预演，"
                          "整段生产须走流式设计（不落盘数千张全景 PNG）")
    if len(set(frames)) != len(frames) or any(f < 0 for f in frames):
        raise ExportError(f"帧索引重复或为负: {frames}")
    for p in insv:
        if not p.is_file():
            raise ExportError(f"输入不存在: {p}")
    exe = sdk_bin / "MediaSDKTest.exe"
    dll = sdk_bin / "MediaSDK.dll"
    for p in (exe, dll):
        if not p.is_file():
            raise ExportError(f"SDK 二进制缺失: {p}")
    out.mkdir(parents=True, exist_ok=True)
    stale = [p.name for p in out.iterdir() if p.suffix.lower() in (".png", ".jpg") or p.name == MANIFEST_NAME]
    if stale:
        raise ExportError(f"输出目录非空（拒绝混入旧帧）: {out} 含 {stale[:5]}...")

    cmd = [str(exe), "-inputs", *[str(p.resolve()) for p in insv],
           "-image_sequence_dir", str(out.resolve()), "-image_type", "png",
           "-export_frame_index", "-".join(str(f) for f in frames),
           "-output_size", output_size, "-stitch_type", stitch_type,
           "-model_root_dir", "./models/", *(extra_flags or [])]
    t0 = datetime.now(timezone.utc)
    proc = subprocess.run(cmd, cwd=str(sdk_bin), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=3600)
    t1 = datetime.now(timezone.utc)

    produced = sorted((p for p in out.glob("*.png")), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    produced_idx = [int(p.stem) for p in produced if p.stem.isdigit()]
    frames_out = []
    for p in produced:
        w, h = _png_size(p)
        frames_out.append({"name": p.name, "frame_index": int(p.stem) if p.stem.isdigit() else None,
                           "sha256": sha256_file(p), "width": w, "height": h, "bytes": p.stat().st_size})
    exp_w, exp_h = (int(x) for x in output_size.lower().split("x"))
    problems = []
    if proc.returncode != 0:
        problems.append(f"sdk_rc={proc.returncode}")
    if sorted(produced_idx) != sorted(frames):
        problems.append(f"frames_mismatch: requested={sorted(frames)} produced={sorted(produced_idx)}")
    for fo in frames_out:
        if (fo["width"], fo["height"]) != (exp_w, exp_h):
            problems.append(f"size_mismatch:{fo['name']}:{fo['width']}x{fo['height']}!={output_size}")

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "status": "OK" if not problems else "FAIL",
        "problems": problems,
        "sdk": {"bin_dir": str(sdk_bin.resolve()), "sdk_dir_name": sdk_bin.resolve().parents[1].name,
                "MediaSDKTest.exe_sha256": sha256_file(exe), "MediaSDK.dll_sha256": sha256_file(dll)},
        "inputs": [{"path": str(p.resolve()), "name": p.name, "bytes": p.stat().st_size,
                    "sha256": sha256_file(p)} for p in insv],
        "params": {"stitch_type": stitch_type, "output_size": output_size, "image_type": "png",
                   "extra_flags": extra_flags or [], "requested_frames": sorted(frames)},
        "command": cmd, "cwd": str(sdk_bin.resolve()), "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:],
        "t_start_utc": t0.isoformat(), "t_end_utc": t1.isoformat(),
        "host": {"node": platform.node(), "platform": platform.platform()},
        "frames": frames_out,
        "determinism_scope": ("same host + same SDK binaries + listed frames only; "
                              "cross-host or full-sequence byte-identity NOT claimed"),
    }
    mp = out / MANIFEST_NAME
    mp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    if problems:
        raise ExportError(f"导出未通过校验: {problems}（manifest 已写 {mp}）")
    return mp


def load_manifest(out_dir: Path) -> dict:
    mp = out_dir / MANIFEST_NAME
    if not mp.is_file():
        raise ExportError(f"缺 {MANIFEST_NAME}: {out_dir}——投影器只消费 wrapper 产物")
    m = json.loads(mp.read_text(encoding="utf-8"))
    if m.get("status") != "OK":
        raise ExportError(f"manifest status={m.get('status')} problems={m.get('problems')}")
    for fo in m["frames"]:  # re-check every frame hash before consuming (tamper / stale-frame guard)
        p = out_dir / fo["name"]
        if not p.is_file() or sha256_file(p) != fo["sha256"]:
            raise ExportError(f"帧与 manifest 不符或缺失: {p}")
    extra = [p.name for p in out_dir.glob("*.png") if p.name not in {f["name"] for f in m["frames"]}]
    if extra:
        raise ExportError(f"目录含 manifest 之外的帧（混入旧帧？）: {extra[:5]}")
    return m


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m software.camera.x3.sdk_export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insv", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--stitch-type", default="optflow")
    ap.add_argument("--output-size", default="3840x1920")
    ap.add_argument("--sdk-bin", type=Path, default=DEFAULT_SDK_BIN)
    ap.add_argument("--max-frames", type=int, default=64)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mp = run_export(args.insv, args.out, args.frames, args.stitch_type, args.output_size,
                    args.sdk_bin, args.max_frames)
    print("manifest:", mp)
    return 0


__all__ = ["DEFAULT_SDK_BIN", "MANIFEST_NAME", "MANIFEST_VERSION", "ExportError",
           "sha256_file", "run_export", "load_manifest", "build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
