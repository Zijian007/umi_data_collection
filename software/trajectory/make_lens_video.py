"""720x720 video of one X3 lens, frame-aligned with the EuRoC dataset from x3_extract.

The crop comes from software.trajectory.frame (right = front, left = rear). Frame k of the output
is dataset frame k, so the two lenses pair by index.

usage: make_lens_video.py <insv> <euroc_mav0_dir> <out_mp4> <left|right> [step]
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from software.trajectory.frame import STEP, crop_xywh  # noqa: E402


def render(insv: str, mav0: str, out: str, half: str, step: int = STEP) -> None:
    x, y, w, h = crop_xywh(half)
    cam = [int(r[0]) for r in csv.reader(open(os.path.join(mav0, "cam0", "data.csv"))) if r and not r[0].startswith("#")]
    n = len(cam)
    nv = n if step == STEP else 2 * n - 1
    sel = "select='not(mod(n\\,2))',setpts=N/(30000/1001)/TB," if step == STEP else "setpts=N/(60000/1001)/TB,"
    vf = sel + "crop=%d:%d:%d:%d,scale=720:720:flags=area" % (w, h, x, y)
    rate = "30000/1001" if step == STEP else "60000/1001"
    subprocess.run(["nice", "-n", "10", "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", insv, "-map", "0:v:0",
                    "-vf", vf, "-frames:v", str(nv), "-r", rate, "-c:v", "libx264", "-crf", "10", "-preset", "medium",
                    "-pix_fmt", "yuv420p", "-an", out], check=True)
    pr = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                         "stream=nb_read_frames", "-of", "json", out], check=True, capture_output=True, text=True)
    got = int(json.loads(pr.stdout)["streams"][0]["nb_read_frames"])
    if got != nv:
        raise RuntimeError(f"{out}: wrote {got} frames, expected {nv}")
    digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
    json.dump({"insv": os.path.basename(insv), "half": half, "crop_xywh": [x, y, w, h], "size": 720,
               "step": step, "fps": rate, "n_frames": nv, "sha256": digest, "vf": vf},
              open(out.replace(".mp4", "_manifest.json"), "w"), indent=1)
    print(half, os.path.basename(out), nv, digest[:12])


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    insv, mav0, out, half = args[:4]
    step = int(args[4]) if len(args) > 4 else STEP
    render(insv, mav0, out, half, step)


if __name__ == "__main__":
    main()
