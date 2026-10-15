"""720x720 video of one X3 lens for the KIWI reproduction, frame-aligned with the canonical r1440 dataset.

Same selection as make_gopro_inputs.py (every 2nd frame from frame 0 -> 29.97 fps, or every frame -> 59.94 fps);
only the crop differs: right half = front lens (2160, 240, 1440, 1440), left half = rear lens (240, 240, 1440, 1440).
Frame k of the output == canonical frame k (step 2), so front and rear rows pair by index.

usage: make_lens_video.py <insv> <euroc_mav0_dir> <out_mp4> <left|right> [step=2]
"""
import csv, json, os, subprocess, sys, hashlib

insv, mav0, out, half = sys.argv[1:5]
step = int(sys.argv[5]) if len(sys.argv) > 5 else 2
x0 = 2160 if half == "right" else 240
cam = [int(r[0]) for r in csv.reader(open(os.path.join(mav0, "cam0", "data.csv"))) if r and not r[0].startswith("#")]
n = len(cam)
nv = n if step == 2 else 2 * n - 1
sel = "select='not(mod(n\\,2))',setpts=N/(30000/1001)/TB," if step == 2 else "setpts=N/(60000/1001)/TB,"
vf = sel + "crop=1440:1440:%d:240,scale=720:720:flags=area" % x0
rate = "30000/1001" if step == 2 else "60000/1001"
subprocess.run(["nice", "-n", "10", "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", insv, "-map", "0:v:0",
                "-vf", vf, "-frames:v", str(nv), "-r", rate, "-c:v", "libx264", "-crf", "10", "-preset", "medium",
                "-pix_fmt", "yuv420p", "-an", out], check=True)
pr = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                     "stream=nb_read_frames", "-of", "json", out], check=True, capture_output=True, text=True)
got = int(json.loads(pr.stdout)["streams"][0]["nb_read_frames"])
assert got == nv, (got, nv)
h = hashlib.sha256(open(out, "rb").read()).hexdigest()
json.dump({"insv": os.path.basename(insv), "half": half, "crop_xywh": [x0, 240, 1440, 1440], "size": 720,
           "step": step, "fps": rate, "n_frames": nv, "sha256": h, "vf": vf},
          open(out.replace(".mp4", "_manifest.json"), "w"), indent=1)
print(half, os.path.basename(out), nv, h[:12])
