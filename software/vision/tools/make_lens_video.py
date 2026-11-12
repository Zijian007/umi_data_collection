"""Square video of one X3 lens, frame-aligned with the EuRoC dataset from camera.x3.extraction.

The crop comes from software.camera.x3.frame (right = front, left = rear). Frame k of the output
is dataset frame k, so the two lenses pair by index.

usage: make_lens_video.py <insv> <euroc_mav0_dir> <out_mp4> <left|right> --config trajectory.yaml
"""

from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from software import config  # noqa: E402
from software import parameters as parameter_models  # noqa: E402
from software.camera.x3.frame import crop_xywh, output_frame_rate, validate_video_parameters  # noqa: E402


def render(insv: str, mav0: str, out: str, half: str, *, video: parameter_models.VideoParameters) -> None:
    """使用父流程解析的全部视频参数；不再隐式读取默认值。"""
    validate_video_parameters(video)
    x, y, w, h = crop_xywh(half, video.crop)
    with (Path(mav0) / 'cam0/data.csv').open(encoding='utf-8') as source:
        n = sum(1 for row in csv.reader(source) if row and not row[0].startswith('#'))
    if n < 1:
        raise ValueError('镜头视频需要至少一个已登记的视频帧')
    rate = output_frame_rate(video.frame_step)
    select = f"select='not(mod(n\\,{video.frame_step}))'," if video.frame_step > 1 else ''
    vf = (select + f'setpts=N/({rate})/TB,crop={w}:{h}:{x}:{y}:exact=1,'
          f'scale={video.size_px}:{video.size_px}:flags={video.resize_filter}')
    command = ['nice', '-n', '10', 'ffmpeg', '-nostdin', '-loglevel', 'error', '-y',
               '-i', insv, '-map', '0:v:0', '-vf', vf, '-frames:v', str(n), '-r', str(rate),
               '-c:v', video.codec, '-crf', str(video.crf), '-preset', video.preset,
               '-pix_fmt', video.pixel_format]
    if video.codec == 'libx265':
        command += ['-tag:v', 'hvc1']
    if video.keep_audio:
        command += ['-map', '0:a?', '-c:a', 'aac', '-b:a', f'{video.audio_bitrate_kbps}k',
                    '-af', f'atrim=duration={float(n / rate):.12f},asetpts=PTS-STARTPTS']
    else:
        command += ['-an']
    command += [out]
    subprocess.run(command, check=True)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                            '-show_entries', 'stream=nb_read_frames', '-of', 'json', out],
                           check=True, capture_output=True, text=True)
    got = int(json.loads(probe.stdout)['streams'][0]['nb_read_frames'])
    if got != n:
        raise RuntimeError(f'{out}: wrote {got} frames, expected {n}')
    with Path(out).open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    manifest = dict(insv=os.path.basename(insv), half=half, crop_xywh=[x, y, w, h],
                    size=video.size_px, step=video.frame_step, fps=str(rate), n_frames=n,
                    sha256=digest, vf=vf, parameters=asdict(video))
    Path(out).with_name(Path(out).stem + '_manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(half, os.path.basename(out), n, digest[:12])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('insv', 'mav0', 'out', 'half'):
        parser.add_argument(name)
    parser.add_argument('--config', type=Path, required=True, help='父流程写出的完整 trajectory YAML')
    args = parser.parse_args(argv)
    parameters = config.load('trajectory', args.config).values
    render(args.insv, args.mav0, args.out, args.half, video=parameters.video)


if __name__ == "__main__":
    main()
