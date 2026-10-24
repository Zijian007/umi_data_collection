"""Resolve recordings without guessing when a stage has multiple candidates."""

import json
import math
from argparse import Namespace
from pathlib import Path

from software import registry
from software.calibration.models import (
    CalibrationInputs, CalibrationRequest, SegmentPaths, VerificationInputs, VerificationRequest,
)
from software.calibration import runtime


def discover_recordings(folder: Path) -> dict[str, Path]:
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise ValueError(f"标定目录不存在或不是文件夹：{folder}")

    def videos(directory: Path) -> list[Path]:
        return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".insv")

    direct = videos(folder)
    stages = {k: folder / k for k in "ABCD"}
    if any(p.exists() for p in stages.values()):
        if direct:
            raise ValueError("目录同时包含 .insv 和 A/B/C/D 子目录，请整理为一种布局，或用 --a --b --c --d 明确指定。")
        result = {}
        for stage, directory in stages.items():
            if not directory.is_dir():
                raise ValueError(f"缺少 {stage} 段文件夹：{directory}")
            candidates = videos(directory)
            if len(candidates) != 1:
                raise ValueError(f"{directory} 中有 {len(candidates)} 个 .insv，{stage} 段需要正好 1 个。")
            result[stage] = candidates[0]
        return result
    if len(direct) != 4:
        raise ValueError(f"{folder} 中有 {len(direct)} 个 .insv，需要正好 4 个，或建立 A/B/C/D 子目录，每个放 1 个 .insv。")
    return dict(zip("ABCD", direct))


def prepare_inputs(request: CalibrationRequest) -> CalibrationInputs:
    """Find A/B/C/D, read identity metadata only, and load the board dimensions."""
    if request.folder:
        found = discover_recordings(request.folder)
        recordings = SegmentPaths(**{key.lower(): path for key, path in found.items()})
    else:
        recordings = request.recordings
    if recordings is None:
        raise ValueError('需要 A/B/C/D 四段录像。')
    for name, path in recordings.items():
        runtime.console.write(f"{name} = {path.name}")
    runtime.console.write("分段用途：A 内参；B 相机–IMU；C 轨迹验证；D 多朝向静止。--a --b --c --d 可显式指定录像。")
    serials = {name: registry.read_insv_serial(path) for name, path in recordings.items()}
    if len(set(serials.values())) != 1:
        raise ValueError("四段录像不是同一台相机：" + "，".join(f"{name}={camera_serial}" for name, camera_serial in serials.items()))
    camera_serial = serials['C']
    runtime.console.write(f"相机序列号 {camera_serial}")
    grid = load_board(request.tag_size_mm)
    return CalibrationInputs(recordings=recordings, camera_serial=camera_serial, grid=grid)


def parse_run_args(args: Namespace) -> CalibrationRequest:
    """Convert CLI values into a request; reject mixed or incomplete input modes."""
    explicit = [args.a, args.b, args.c, args.d]
    if args.folder and any(explicit):
        raise ValueError('文件夹和 --a --b --c --d 不能同时指定，请选择一种输入方式。')
    if not args.folder and not all(explicit):
        raise ValueError('给一个放着 4 段录像的文件夹，或者用 --a --b --c --d 分别指定。')
    recordings = SegmentPaths(*(Path(path).expanduser() for path in explicit)) if all(explicit) else None
    return CalibrationRequest(
        folder=Path(args.folder).expanduser() if args.folder else None,
        recordings=recordings, output=Path(args.out).expanduser() if args.out else None,
        tag_size_mm=args.tag_size_mm, cores=args.cores, keep_work=args.keep_work,
        force=args.force, install=args.install, gripper=args.gripper, compare_sample=args.compare_sample,
        config=args.config, dry_run=args.dry_run,
    )


def parse_verify_args(args: Namespace) -> VerificationRequest:
    return VerificationRequest(
        video=Path(args.video).expanduser(), output=Path(args.out).expanduser() if args.out else None,
        tag_size_mm=args.tag_size_mm, cores=args.cores, keep_work=args.keep_work, force=args.force,
        config=args.config, dry_run=args.dry_run,
    )


def load_board(tag_size_mm: float) -> dict:
    if not math.isfinite(tag_size_mm) or tag_size_mm <= 0:
        raise ValueError("标定板 tag 边长必须是有限正数（mm）")
    grid = json.loads(runtime.BOARD_JSON.read_text(encoding='utf-8'))
    grid['tagSize'] = tag_size_mm / 1000.0
    return grid


def prepare_verification(request: VerificationRequest) -> VerificationInputs:
    """Resolve the video and calibration before touching any existing output."""
    video = request.video
    if not video.is_file():
        raise ValueError(f'找不到录像 {video}')
    camera_serial = registry.read_insv_serial(video)
    try:
        selected = registry.resolve_insv(video)
        name = f"档案 {selected['serial']} / {selected['id']}"
    except registry.UnknownCameraError:
        selected = registry.sample()
        name = f"样例标定 {selected['id']}（序列号 {camera_serial} 还没有档案）"
    return VerificationInputs(video, camera_serial, load_board(request.tag_size_mm), selected, name)
