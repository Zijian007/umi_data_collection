"""发布轨迹结果并整理调试数据；成功后可单独清理 debug 中的临时数据。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import json
import os
import shutil
import tempfile


if TYPE_CHECKING:
    from software.trajectory.models import RecordingPaths

AUXILIARY_RESULTS = ('traj_camera.txt', 'traj_imu.txt', 'frames.csv', 'run_settings.json', 'summary.json', 'map.json')


def result_path(output: Path, name: str) -> Path:
    """Resolve a published result in the current layout."""
    return output / 'details' / name if name in AUXILIARY_RESULTS else output / name


def load_summary(output: Path) -> dict:
    return json.loads(result_path(output, 'summary.json').read_text(encoding='utf-8'))


def attempt_paths(output: Path, summary: dict) -> tuple[Path, ...]:
    """Return the selected reconstruction attempt in the current layout."""
    chosen = summary.get('chosen_try', 1)
    if type(chosen) is not int or chosen < 1:
        raise ValueError('摘要中的重建轮次无效')
    return (output / 'debug/reconstruction' / f'try_{chosen}',)


def front_video_path(output: Path, summary: dict) -> Path | None:
    """优先发布的视频，再查选中轮次；不混入其他轮次的视频。"""
    published = output / 'front.mp4'
    if published.is_file():
        return published
    return next((path / 'front/front.mp4' for path in attempt_paths(output, summary)
                 if (path / 'front/front.mp4').is_file()), None)


def _publish_video(output: Path, summary: dict) -> None:
    video = front_video_path(output, summary)
    if video is None:
        return  # 视频未保留时，轨迹结果仍可读取。
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
    """Publish the current result layout and report."""
    from . import reports

    output = output.resolve()
    for name in ('details', 'debug'):
        if not (output / name).resolve().is_relative_to(output):
            raise ValueError(f'结果目录指向输出目录之外：{output / name}')
    if (output / 'summary.json').exists():
        raise ValueError('结果摘要必须写入 details/summary.json')
    _publish_video(output, summary)
    details = output / 'details'
    details.mkdir(parents=True, exist_ok=True)
    for name in AUXILIARY_RESULTS:
        if name == 'summary.json':
            continue
        source = output / name
        target = details / name
        if source.exists() or source.is_symlink():
            if target.exists() or target.is_symlink():
                raise ValueError(f'结果布局冲突：{source} -> {target}')
            shutil.move(str(source), target)
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
        (reconstruction_dir / 'fuse/seg_placement.csv', logs / 'seg_placement.csv'),
        (reconstruction_dir / 'front/map/log.txt', logs / 'front_map_log.txt'),
        (reconstruction_dir / 'front/loc/frames.csv', logs / 'front_loc_frames.csv'),
    ):
        if src.is_file():
            shutil.copy(src, dest)
