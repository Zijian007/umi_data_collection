"""Square video of one X3 lens, frame-aligned with the EuRoC dataset from camera.x3.extraction.

The crop comes from software.camera.x3.frame (right = front, left = rear). Frame k of the output
is dataset frame k, so the two lenses pair by index.

usage: make_lens_video.py <insv> <euroc_mav0_dir> <out_mp4> <left|right> [step]
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from software import config
from software.camera.x3.frame import FPS_SRC_RATE, FPS_USED_RATE, STEP, crop_xywh, validate_video_size  # noqa: E402


def render(insv: str, mav0: str, out: str, half: str, step: int = STEP, *,
           size_px: int | None = None, crf: int | None = None, preset: str | None = None) -> None:
    if size_px is None or crf is None or preset is None:
        defaults = config.load("trajectory").values.video
        size_px = defaults.size_px if size_px is None else size_px
        crf = defaults.crf if crf is None else crf
        preset = defaults.preset if preset is None else preset
    validate_video_size(size_px)
    if step not in (1, STEP):
        raise ValueError(f'取帧步长必须为 1 或 {STEP}')
    x, y, w, h = crop_xywh(half)
    with (Path(mav0) / 'cam0/data.csv').open(encoding='utf-8') as source:
        cam = [int(r[0]) for r in csv.reader(source) if r and not r[0].startswith('#')]
    n = len(cam)
    nv = n if step == STEP else STEP * n - 1
    rate = FPS_USED_RATE if step == STEP else FPS_SRC_RATE
    sel = f"select='not(mod(n\\,{STEP}))'," if step == STEP else ''
    vf = sel + f'setpts=N/({rate})/TB,crop={w}:{h}:{x}:{y},scale={size_px}:{size_px}:flags=area'
    subprocess.run(["nice", "-n", "10", "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", insv, "-map", "0:v:0",
                    "-vf", vf, "-frames:v", str(nv), "-r", rate, "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
                    "-pix_fmt", "yuv420p", "-an", out], check=True)
    pr = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                         "stream=nb_read_frames", "-of", "json", out], check=True, capture_output=True, text=True)
    got = int(json.loads(pr.stdout)["streams"][0]["nb_read_frames"])
    if got != nv:
        raise RuntimeError(f"{out}: wrote {got} frames, expected {nv}")
    with Path(out).open('rb') as video:
        digest = hashlib.file_digest(video, 'sha256').hexdigest()
    manifest = {"insv": os.path.basename(insv), "half": half, "crop_xywh": [x, y, w, h], "size": size_px,
                "step": step, "fps": rate, "n_frames": nv, "sha256": digest, "vf": vf}
    Path(out).with_name(Path(out).stem + '_manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(half, os.path.basename(out), nv, digest[:12])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('insv', 'mav0', 'out', 'half'):
        parser.add_argument(name)
    parser.add_argument('step', nargs='?', type=int, default=STEP)
    parser.add_argument('--size-px', type=int)
    parser.add_argument('--crf', type=int)
    parser.add_argument('--preset')
    args = parser.parse_args(argv)
    render(args.insv, args.mav0, args.out, args.half, args.step,
           size_px=args.size_px, crf=args.crf, preset=args.preset)


if __name__ == "__main__":
    main()
