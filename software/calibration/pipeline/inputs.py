"""Resolve recordings without guessing when a stage has multiple candidates."""

from argparse import Namespace
from pathlib import Path
import json
import math

from software import registry
from software.calibration.models import CalibrationInputs, CalibrationRequest, SegmentPaths, VerificationInputs, VerificationRequest
from software.calibration.pipeline import runtime


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
            raise ValueError("目录同时包含 .insv 和 A/B/C/D 子目录，请整理为一种布局。")
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
    runtime.console.write(f"本次标定：{'工作镜头 front' if request.lens == 'front' else '环境镜头 rear'}")
    runtime.console.write("分段用途：A 内参求解/验证；B 相机–IMU；C 轨迹验证；D 多朝向静止。")
    serials = {name: registry.read_insv_serial(path) for name, path in recordings.items()}
    if len(set(serials.values())) != 1:
        raise ValueError("四段录像不是同一台相机：" + "，".join(f"{name}={camera_serial}" for name, camera_serial in serials.items()))
    camera_serial = serials['C']
    runtime.console.write(f"相机序列号 {camera_serial}")
    grid = load_board(request.tag_size_mm)
    return CalibrationInputs(recordings=recordings, camera_serial=camera_serial, grid=grid)


def recording_lens(folder: Path) -> str:
    """镜头身份由传入目录的末级名称明确指定，不搜索父目录或猜测。"""
    lens = folder.expanduser().name
    if lens not in ('front', 'rear'):
        raise ValueError('标定目录必须命名为 front（工作镜头）或 rear（环境镜头），并直接传入该目录')
    return lens


def parse_run_args(args: Namespace) -> CalibrationRequest:
    """从目录约定生成请求；此处不读取录像，供执行和配置预览共用。"""
    folder = Path(args.folder).expanduser()
    lens = recording_lens(folder)
    if lens == 'rear' and args.compare_sample:
        raise ValueError('--compare-sample 仅适用于 front 工作镜头')
    return CalibrationRequest(
        folder=folder, recordings=None, output=Path(args.out).expanduser() if args.out else None,
        tag_size_mm=args.tag_size_mm, cores=args.cores, keep_work=args.keep_work,
        force=args.force, install=args.install, compare_sample=args.compare_sample,
        config=args.config, dry_run=args.dry_run,
        resume=getattr(args, 'resume', False), workers=getattr(args, 'workers', None),
        lens=lens,
        intrinsics_source=getattr(args, 'intrinsics_source', 'measured'),
    )


def parse_verify_args(args: Namespace) -> VerificationRequest:
    return VerificationRequest(
        video=Path(args.video).expanduser(), output=Path(args.out).expanduser() if args.out else None,
        tag_size_mm=args.tag_size_mm, cores=args.cores, keep_work=args.keep_work, force=args.force,
        config=args.config, dry_run=args.dry_run,
        resume=getattr(args, 'resume', False), workers=getattr(args, 'workers', None),
    )


def load_board(tag_size_mm: float) -> dict:
    if not math.isfinite(tag_size_mm) or tag_size_mm <= 0:
        raise ValueError("标定板 tag 边长必须是有限正数（mm）")
    layout = _read_board_layout()
    # Basalt consumes only geometry; physical size belongs to this run, not the print preset.
    return {key: layout[key] for key in ('tagCols', 'tagRows', 'tagSpacing')} | {'tagSize': tag_size_mm / 1000.0}


def prepare_verification(request: VerificationRequest) -> VerificationInputs:
    """Resolve the video and calibration before touching any existing output."""
    video = request.video
    if not video.is_file():
        raise ValueError(f'找不到录像 {video}')
    camera_serial = registry.read_insv_serial(video)
    selected = registry.resolve_insv(video)
    name = f"档案 {selected['serial']} / {selected.get('calibration_version', selected['id'])}"
    return VerificationInputs(video, camera_serial, load_board(request.tag_size_mm), selected, name)


BOARD_JSON = runtime.REPO / "hardware/calibration_targets/aprilgrid.json"


def _read_board_layout() -> dict:
    layout = json.loads(BOARD_JSON.read_text(encoding='utf-8'))
    if not isinstance(layout, dict) or set(layout) != {'family', 'black_border_bits', 'tagCols', 'tagRows', 'tagSpacing'}:
        raise ValueError('板布局必须包含 family、black_border_bits、tagCols、tagRows、tagSpacing；尺寸由 --tag-size-mm 指定')
    # These are compatibility constraints of the shipped Basalt and OpenCV detectors.
    if layout['family'] != 'tag36h11' or layout['black_border_bits'] != 2:
        raise ValueError('当前检测器仅支持 tag36h11、2 格黑边的 AprilGrid')
    if any(type(layout[key]) is not int or layout[key] <= 0 for key in ('tagCols', 'tagRows')):
        raise ValueError('标定板行列数必须是正整数')
    spacing = layout['tagSpacing']
    if type(spacing) not in (int, float) or not math.isfinite(spacing) or spacing <= 0:
        raise ValueError('tagSpacing 必须是有限正数（标签间白色间隔 / 标签外黑框边长）')
    return layout


def board_layout_problems() -> list[str]:
    """Check the supported layout and the geometry recorded for each print asset."""
    try:
        reference = _read_board_layout()
    except (OSError, ValueError) as error:
        return [f"无法读取 {BOARD_JSON.name}：{error}"]
    problems = []
    presets = sorted((BOARD_JSON.parent / 'prints').glob('*_meta.json'))
    if not presets:
        return ['prints/ 中没有标定板打印记录']
    for path in presets:
        try:
            metadata = json.loads(path.read_text(encoding='utf-8'))
            layout = metadata['layout']
            if (metadata['family'] != reference['family'] or
                    metadata['black_border_bits'] != reference['black_border_bits'] or
                    layout['rows'] != reference['tagRows'] or layout['cols'] != reference['tagCols'] or
                    layout['spacing'] != reference['tagSpacing']):
                problems.append(f'{path.name} 与 {BOARD_JSON.name} 的标签编码、黑边、行列或间距不同，--tag-size-mm 无法覆盖')
            for name in metadata['files']:
                if Path(name).name != name or not (path.parent / name).is_file():
                    problems.append(f'{path.name} 引用的打印资产不存在或路径无效：{name}')
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(f"无法读取 {path.name}：{error}")
    return problems
