"""共享场景任务的只读浏览器适配；不拟合坐标，只按已同步时间插值另一只手。"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from pathlib import Path

from software.viewer.data import ViewerDataError


def recording_key(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20]


def read_episode(directory: Path, contains) -> tuple[dict, dict[str, Path]]:
    path = directory / 'episode.json'
    if not contains(path):
        raise ViewerDataError('任务清单超出查看目录')
    data = json.loads(path.read_text())
    if (data.get('kind') != 'umi_scene_episode' or data.get('status') != 'ok'
            or not isinstance(data.get('arms'), dict) or not data['arms']
            or data['arms'].keys() - {'left', 'right'}):
        raise ViewerDataError('共享场景任务清单无效')
    arms = {}
    for arm, relative in data['arms'].items():
        if not isinstance(relative, str):
            raise ViewerDataError('共享场景手路径无效')
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or not contains(path / 'traj_tcp.txt') or not (path / 'traj_tcp.txt').is_file():
            raise ViewerDataError('共享场景手路径缺失或越界')
        arms[arm] = path
    return data, arms


def trajectory(times, positions, quaternions):
    distance = [math.dist(a, b) for a, b in zip(positions, positions[1:])]
    return dict(position=positions, quaternion=quaternions,
        speed=[0.]+[d/(b-a) for d, a, b in zip(distance, times, times[1:])], path_length=sum(distance),
        extent=[max(p[k] for p in positions)-min(p[k] for p in positions) for k in range(3)])


def sample_track(times, track, query):
    positions, quaternions = [], []
    for time in query:
        right = min(len(times)-1, max(1, bisect.bisect_left(times, time)))
        left = right-1
        if not times[0]-1e-8 <= time <= times[-1]+1e-8:
            raise ViewerDataError('双手轨迹不能在共同时间区间外插值')
        fraction = max(0., min(1., (time-times[left])/(times[right]-times[left])))
        positions.append([(1-fraction)*a+fraction*b for a, b in zip(track['position'][left], track['position'][right])])
        a, b = track['quaternion'][left], track['quaternion'][right]
        dot = sum(x*y for x, y in zip(a, b))
        if dot < 0:
            b = [-v for v in b]; dot = -dot
        if dot > .9995:
            q = [(1-fraction)*x+fraction*y for x, y in zip(a, b)]
        else:
            angle = math.acos(max(-1., min(1., dot)))
            q = [(math.sin((1-fraction)*angle)*x+math.sin(fraction*angle)*y)/math.sin(angle) for x, y in zip(a, b)]
        norm = math.sqrt(sum(v*v for v in q))
        quaternions.append([v/norm for v in q])
    return trajectory(query, positions, quaternions)


def load_episode(catalog, key: str, directory: Path) -> dict:
    metadata, paths = read_episode(directory, catalog.contains)
    loaded = {arm: catalog.load(recording_key(path)) for arm, path in paths.items()}
    for arm, data in loaded.items():
        scene = data['summary'].get('scene', {})
        if scene.get('frame_id') != metadata.get('frame_id') or scene.get('version') != metadata.get('scene', {}).get('version'):
            raise ViewerDataError(f'{arm} 轨迹不属于本任务声明的场景坐标/地图版本')
    primary_arm = 'left' if 'left' in loaded else 'right'
    primary = loaded[primary_arm]
    start = max(data['times'][0] for data in loaded.values())
    end = min(data['times'][-1] for data in loaded.values())
    selected = [i for i, time in enumerate(primary['times']) if start <= time <= end]
    if len(selected) < 2:
        raise ViewerDataError('左右手没有共同有效时间')
    query = [primary['times'][i] for i in selected]
    primary['tracks'] = {name: sample_track(primary['times'], track, query) for name, track in primary['tracks'].items()}
    primary['frames'] = [primary['frames'][i] for i in selected]
    primary['times'] = query
    primary.update(id=key, name=metadata.get('name', directory.parent.parent.name), path=str(directory),
        primary_arm=primary_arm, episode=metadata, companion=None, companion_tracks={})
    for arm, other in loaded.items():
        if arm != primary_arm:
            primary['companion'] = dict(other, arm=arm)
            primary['companion_tracks'] = {name: sample_track(other['times'], track, query) for name, track in other['tracks'].items()}
    return primary
