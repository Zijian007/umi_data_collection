"""Read UMI output layouts into a small, explicit browser data contract."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

from software.trajectory.results import front_video_path, load_summary, result_path


class ViewerDataError(ValueError):
    pass


def read_trajectory(path: Path) -> dict:
    times, positions, quaternions = [], [], []
    with path.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            try:
                values = list(map(float, line.split()))
                if len(values) != 8 or not all(math.isfinite(v) for v in values):
                    raise ValueError('需要 8 个有限数值')
                if times and values[0] <= times[-1]:
                    raise ValueError('时间戳必须严格递增')
                q = values[4:]
                norm = math.sqrt(sum(v * v for v in q))
                if not 0.9 < norm < 1.1:
                    raise ValueError('姿态四元数不是单位四元数')
            except ValueError as error:
                raise ViewerDataError(f'{path.name} 第 {line_number} 行：{error}') from error
            times.append(values[0])
            positions.append(values[1:4])
            quaternions.append([v / norm for v in q])
    if len(times) < 2:
        raise ViewerDataError(f'{path.name} 至少需要两个轨迹点')
    distances = [math.dist(a, b) for a, b in zip(positions, positions[1:])]
    speed = [0.0] + [d / (b - a) for d, a, b in zip(distances, times, times[1:])]
    return dict(times=times, position=positions, quaternion=quaternions, speed=speed,
                path_length=sum(distances),
                extent=[max(p[k] for p in positions) - min(p[k] for p in positions) for k in range(3)])


def read_frames(path: Path, times: list[float]) -> tuple[list[dict], list[str]]:
    frames = [dict(constraint='unknown', tracked=None, video_frame=i * 2) for i in range(len(times))]
    if not path.is_file():
        return frames, ['缺少逐帧信息，约束类型显示为未知。']
    invalid = 0
    seen = set()
    with path.open(encoding='utf-8', newline='') as source:
        for row in csv.DictReader(source):
            try:
                index = int(row['row'])
                if index < 0 or index >= len(times) or index in seen:
                    raise ValueError('row')
                time = float(row.get('t_scene_s') or row['t_video_s'])
                if not math.isfinite(time) or abs(time - times[index]) > 1e-4:
                    raise ValueError('time')
                constraint = row['constraint']
                if constraint not in ('full', 'rot', 'imu'):
                    raise ValueError('constraint')
                video_frame = int(row['video_frame'])
                if video_frame < 0:
                    raise ValueError('video_frame')
                tracked = int(row['front_lens_tracked'])
                if tracked not in (0, 1):
                    raise ValueError('tracked')
                frames[index] = dict(constraint=constraint, tracked=bool(tracked), video_frame=video_frame)
                seen.add(index)
            except (ValueError, KeyError, TypeError):
                invalid += 1
    notes = []
    if invalid or len(seen) != len(times):
        notes.append(f'逐帧信息有 {invalid} 行无效，{len(times) - len(seen)} 帧缺失；缺失帧约束显示为未知。')
    return frames, notes


class RecordingCatalog:
    """Only expose result directories and videos within the selected local root."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise ViewerDataError(f'数据目录不存在：{self.root}')
        self.directories: dict[str, Path] = {}

    def contains(self, path: Path) -> bool:
        return path.resolve().is_relative_to(self.root)

    def summary(self, directory: Path) -> dict:
        path = result_path(directory, 'summary.json')
        if not self.contains(path):
            raise ViewerDataError('摘要文件超出当前数据目录。')
        try:
            value = load_summary(directory)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def scan(self) -> dict:
        directories = {}
        entries = []
        for current, children, files in os.walk(self.root, followlinks=False):
            directory = Path(current)
            children[:] = sorted(name for name in children if name not in {
                '.git', '.deps', 'node_modules', 'details', 'debug', 'diagnostics', 'work', 'bundle', 'umi_calib_out', '__pycache__',
                'runs', 'versions',
            } and not name.startswith('.'))
            # 任务的 results 是原子发布的版本指针；只展示当前版本，不遍历历史 runs。
            episode = directory / 'results' if 'episode.yaml' in files else directory
            if (episode / 'episode.json').is_file():
                from software.viewer.episodes import read_episode, recording_key
                children[:] = []
                try:
                    metadata, arms = read_episode(episode, self.contains)
                    key = recording_key(episode)
                    directories[key] = episode.resolve()
                    for path in arms.values():
                        directories[recording_key(path)] = path
                    interval = metadata['common_interval_s']
                    entries.append(dict(id=key, name=metadata.get('name', directory.name), path=str(episode.relative_to(self.root)),
                        serial='双手场景' if len(arms) == 2 else '场景任务', status='ok', duration=interval[1]-interval[0],
                        rows=metadata.get('rows'), calibration=metadata['scene']['version']))
                except (ViewerDataError, OSError, ValueError, KeyError):
                    continue
                continue
            if 'traj_tcp.txt' not in files:
                continue
            children[:] = []
            if not self.contains(directory) or not self.contains(directory / 'traj_tcp.txt'):
                continue
            key = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:20]
            directories[key] = directory
            try:
                summary = self.summary(directory)
            except ViewerDataError:
                continue
            selection = summary.get('calibration_used') or {}
            entries.append(dict(id=key, name=directory.name, path=str(directory.relative_to(self.root)),
                                serial=selection.get('serial'), status=summary.get('status', 'unknown'),
                                duration=summary.get('duration_s'), rows=summary.get('rows'),
                                calibration=selection.get('id')))
        self.directories = directories
        entries.sort(key=lambda entry: entry['name'], reverse=True)
        return dict(root=str(self.root), recordings=entries)

    def directory(self, key: str) -> Path:
        directory = self.directories.get(key)
        if directory is None or not self.contains(directory):
            raise ViewerDataError('录像不存在，请刷新列表。')
        return directory

    def video_path(self, key: str) -> Path | None:
        directory = self.directory(key)
        if (directory / 'episode.json').is_file():
            from software.viewer.episodes import read_episode
            _, arms = read_episode(directory, self.contains)
            directory = arms.get('left', next(iter(arms.values())))
        try:
            summary = self.summary(directory)
        except (OSError, ValueError, TypeError):
            summary = {}
        path = front_video_path(directory, summary)
        return path if path is not None and self.contains(path) else None

    def map_data(self, key: str) -> dict:
        from software.viewer import maps
        directory = self.directory(key)
        if (directory / 'episode.json').is_file():
            from software.viewer.episodes import read_episode
            _, arms = read_episode(directory, self.contains)
            directory = arms.get('left', next(iter(arms.values())))
        return maps.load(directory, self.summary(directory))

    def load(self, key: str) -> dict:
        directory = self.directory(key)
        if (directory / 'episode.json').is_file():
            from software.viewer.episodes import load_episode
            return load_episode(self, key, directory)
        summary = self.summary(directory)
        tcp_path = directory / 'traj_tcp.txt'
        if not self.contains(tcp_path):
            raise ViewerDataError('轨迹文件超出当前数据目录。')
        tcp = read_trajectory(tcp_path)
        times = tcp.pop('times')
        tracks = dict(tcp=tcp)
        frames_path = result_path(directory, 'frames.csv')
        if not self.contains(frames_path):
            raise ViewerDataError('逐帧文件超出当前数据目录。')
        frames, notes = read_frames(frames_path, times)
        for name in ('camera', 'imu'):
            path = result_path(directory, f'traj_{name}.txt')
            if not path.is_file():
                continue
            if not self.contains(path):
                notes.append(f'{name} 轨迹超出当前数据目录，已跳过。')
                continue
            try:
                track = read_trajectory(path)
                if len(track['times']) != len(times) or any(abs(a - b) > 1e-4 for a, b in zip(times, track.pop('times'))):
                    raise ViewerDataError('时间轴与 TCP 不一致')
                tracks[name] = track
            except (OSError, ViewerDataError) as error:
                notes.append(f'{name} 轨迹未加载：{error}')
        video = self.video_path(key)
        from software.viewer.media import cached_preview, video_version
        version = video_version(video) if video else None
        preview = cached_preview(video) if video else None
        return dict(id=key, name=directory.name, path=str(directory), summary=summary, times=times,
                    tracks=tracks, frames=frames, notes=notes + list(summary.get('notes') or []),
                    video_url=f'/api/recordings/{key}/video?v={version}' if video else None,
                    poster_url=f'/api/recordings/{key}/frame?t=0&v={version}' if video else None,
                    preview_url=f'/api/recordings/{key}/preview' if video else None,
                    preview_video_url=f'/api/recordings/{key}/preview/video?v={preview.stem}' if preview else None,
                    frame_url=f'/api/recordings/{key}/frame' if video else None,
                    video_time_offset_s=summary.get('video_time_offset_s', times[0]),
                    video_note='前镜头裁切视频 · 与轨迹使用同一视频时钟' if video else '未保留前镜头视频，仍可独立回放三维轨迹。')
