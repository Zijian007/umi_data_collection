"""发布结果、兼容旧布局；成功后可单独清理 debug 中的临时数据。"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from software.trajectory.models import RecordingPaths

AUXILIARY_RESULTS = ('traj_camera.txt', 'traj_imu.txt', 'frames.csv', 'run_settings.json', 'summary.json', 'map.json')


def result_path(output: Path, name: str) -> Path:
    """优先读取 details，兼容 diagnostics 与最早的平铺结果。"""
    if name in AUXILIARY_RESULTS:
        for directory in ('details', 'diagnostics'):
            path = output / directory / name
            if path.is_file():
                return path
    return output / name


def load_summary(output: Path) -> dict:
    return json.loads(result_path(output, 'summary.json').read_text(encoding='utf-8'))


def attempt_paths(output: Path, summary: dict) -> tuple[Path, ...]:
    """选中重建轮次的当前及历史路径，按读取优先级排列。"""
    chosen = summary.get('chosen_try', 1)
    if type(chosen) is not int or chosen < 1:
        raise ValueError('摘要中的重建轮次无效')
    legacy = 'kiwi' if chosen == 1 else f'kiwi_try{chosen}'
    return (output / 'debug/reconstruction' / f'try_{chosen}',
            output / 'diagnostics/work' / legacy, output / 'work' / legacy)


def front_video_path(output: Path, summary: dict) -> Path | None:
    """优先发布的视频，再查选中轮次；不混入其他轮次的视频。"""
    published = output / 'front.mp4'
    if published.is_file():
        return published
    return next((path / 'front/front.mp4' for path in attempt_paths(output, summary)
                 if (path / 'front/front.mp4').is_file()), None)


def _legacy_moves(output: Path) -> list[tuple[Path, Path]]:
    """只规划搬迁；所有目标冲突检查通过后才能开始移动。"""
    moves = []
    for base in (output, output / 'diagnostics'):
        for name in AUXILIARY_RESULTS:
            if (base / name).exists() or (base / name).is_symlink():
                moves.append((base / name, output / 'details' / name))
        if (base / 'logs').exists():
            moves.append((base / 'logs', output / 'debug/logs'))
        work = base / 'work'
        if work.is_dir():
            if work.is_symlink():
                raise ValueError(f'产物布局冲突：工作目录是符号链接：{work}')
            for source in work.iterdir():
                match = re.fullmatch(r'kiwi(?:_try(\d+))?', source.name)
                target = (output / 'debug/reconstruction' / f'try_{match.group(1) or 1}'
                          if match else output / 'debug' / source.name)
                moves.append((source, target))
    legacy = output / 'diagnostics'
    if legacy.is_dir():
        for source in legacy.iterdir():
            if source.name not in (*AUXILIARY_RESULTS, 'logs', 'work'):
                moves.append((source, output / 'debug/legacy' / source.name))
    targets = set()
    for source, target in moves:
        if target.exists() or target.is_symlink() or target in targets:
            raise ValueError(f'产物布局冲突：{source} -> {target}')
        if not source.resolve().is_relative_to(output) or not target.resolve().is_relative_to(output):
            raise ValueError(f'产物路径指向输出目录之外：{source} -> {target}')
        targets.add(target)
    return moves


def _move_legacy(output: Path, moves: list[tuple[Path, Path]]) -> None:
    # 重试会用绝对软链接引用第一轮视频；搬目录时也要修复这些引用。
    links = []
    for source, target in moves:
        children = source.rglob('*') if source.is_dir() and not source.is_symlink() else [source]
        for link in children:
            if link.is_symlink():
                resolved = link.resolve()
                relocated = resolved
                for old, new in moves:
                    if resolved.is_relative_to(old):
                        relocated = new / resolved.relative_to(old)
                        break
                links.append((target / link.relative_to(source), relocated))
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), target)
    for link, target in links:
        link.unlink()
        link.symlink_to(os.path.relpath(target, link.parent))
    for directory in (output / 'diagnostics/work', output / 'work', output / 'diagnostics'):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


def _publish_video(output: Path, summary: dict) -> None:
    video = front_video_path(output, summary)
    if video is None:
        return  # 旧结果可能没有保留视频，轨迹仍可读取。
    target = output / 'front.mp4'
    if target.is_file() and not target.is_symlink():
        return
    source = video.resolve()
    if not source.is_relative_to(output):
        raise ValueError(f'视频路径指向结果目录之外：{video}')
    # 发布为独立文件；同一文件系统优先硬链接，避免复制视频占用额外空间。
    descriptor, name = tempfile.mkstemp(prefix='.front-', suffix='.mp4', dir=output)
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_result(output: Path, summary: dict) -> None:
    """发布视频、details 和报告。迁移旧结果时保留所有独有中间数据。"""
    from software.trajectory import reports

    output = output.resolve()
    for name in ('details', 'debug', 'diagnostics'):
        if not (output / name).resolve().is_relative_to(output):
            raise ValueError(f'结果目录指向输出目录之外：{output / name}')
    moves = _legacy_moves(output)
    _publish_video(output, summary)  # 先发布视频，避免旧重试链接在搬迁中暂时失效。
    _move_legacy(output, moves)
    details = output / 'details'
    details.mkdir(parents=True, exist_ok=True)
    if summary.get('run_settings'):
        summary['run_settings'] = 'details/run_settings.json'
    summary['result_layout'] = 'details-v2'
    summary.setdefault('debug_retained', (output / 'debug/dataset').exists() or (output / 'debug/reconstruction').exists())
    (details / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    reports.write_result_report(output, summary)


def cleanup_debug(paths: RecordingPaths) -> None:
    """仅用于新运行成功且报告发布后；保留根视频和 debug/logs。"""
    video = paths.output / 'front.mp4'
    if not video.is_file() or video.is_symlink():
        raise OSError('front.mp4 尚未发布为独立文件；保留 debug 数据')
    for directory in (paths.dataset, paths.reconstruction, paths.lens_videos, paths.debug / 'settings'):
        if directory.exists():
            shutil.rmtree(directory)
    (paths.debug / 'times.csv').unlink(missing_ok=True)


def publish_logs(reconstruction_dir: Path, logs: Path) -> None:
    for src, dest in (
        (reconstruction_dir / 'logs/run.txt', logs / 'run.txt'),
        (reconstruction_dir / 'logs/run.json', logs / 'run.json'),
        (reconstruction_dir / 'logs/tracking.json', logs / 'tracking.json'),
        (reconstruction_dir / 'logs/fuse.log', logs / 'fuse.log'),
        (reconstruction_dir / 'fuse/fuse_info.json', logs / 'fuse_info.json'),
        (reconstruction_dir / 'front/map/log.txt', logs / 'front_map_log.txt'),
        (reconstruction_dir / 'front/loc/frames.csv', logs / 'front_loc_frames.csv'),
    ):
        if src.is_file():
            shutil.copy(src, dest)
