"""独立轨迹 CLI 的录像发现与默认输出目录；实际录像准备位于 vision.recordings。"""

from __future__ import annotations

from pathlib import Path


def find_inputs(paths: list[str]) -> list[Path]:
    videos: list[Path] = []
    for path in map(Path, paths):
        if path.is_dir():
            videos += sorted(file for file in path.iterdir() if file.is_file() and file.suffix.lower() == ".insv")
        elif path.is_file():
            videos.append(path)
        else:
            raise FileNotFoundError(f"录像路径不存在：{path}")
    seen, unique = set(), []
    for video in videos:
        resolved = video.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(video)
    return unique


def default_output(paths: list[str]) -> Path:
    first = Path(paths[0])
    return (first if first.is_dir() else first.parent) / "umi_traj_out"
