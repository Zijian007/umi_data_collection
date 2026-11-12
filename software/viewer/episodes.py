"""共享场景任务的只读浏览器适配；不拟合坐标，只按已同步时间插值另一只手。"""

from __future__ import annotations

from pathlib import Path
import bisect
import hashlib
import json
import math

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from software.scene import layout
from software.scene.result_io import scene_reference, trajectory_path
from software.viewer.errors import ViewerDataError
from software.viewer.readers import trajectory_from_poses


def recording_key(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20]


def capture_status(directory: Path) -> dict:
    """从采集清单读取身份和完整性；状态只保存尝试与验收，查看无需算法环境。"""
    def read_yaml(path):
        try:
            with path.open() as stream:
                yaml_document = YAML(typ='safe', pure=True).load(stream)
        except YAMLError as error:
            raise ViewerDataError(f'采集清单格式无效：{path}') from error
        if not isinstance(yaml_document, dict):
            raise ViewerDataError('采集清单需要映射')
        return yaml_document

    session_dir = directory.parent
    episode_path = layout.manifest(directory, 'episode.yaml')
    session_path = layout.manifest(session_dir, 'manifest.yaml')
    episode = read_yaml(episode_path)
    episode.update(layout.task_configuration(directory))
    session = read_yaml(session_path)
    scene_path = session_dir.parent / 'scene.yaml'
    scene = read_yaml(scene_path) if scene_path.is_file() else {}
    if scene.get('name', scene_path.parent.name) != scene_path.parent.name:
        raise ViewerDataError('场景目录与配置名称不一致')
    if ('scene' in session and (session_dir / session['scene']).resolve() != scene_path
            or 'session' in episode and layout.manifest((directory / episode['session']).resolve(), 'manifest.yaml') != session_path):
        raise ViewerDataError('采集清单与所属目录不一致')
    batch_arms = session.get('expected_arms', [arm for arm in ('left', 'right') if arm in session['kit']['arms']])
    expected_arms = episode.get('expected_arms', batch_arms)
    state = layout.read_state(directory)
    attempt = state.get('attempt', {})
    fresh = all(hashlib.sha256(path.read_bytes()).hexdigest() == attempt.get(key)
                for path, key in ((episode_path, 'manifest_sha256'), (session_path, 'session_sha256')))
    missing = set(expected_arms) - episode['recordings'].keys()
    status = 'pending'
    if missing:
        status = 'incomplete'
    elif fresh and attempt.get('task_configuration_sha256', hashlib.sha256(b'{}').hexdigest()) == layout.task_configuration_hash(directory):
        status = attempt.get('status', 'pending')
    return dict(scene=scene_path.parent.name, kit=session['kit']['id'], session=session_dir.name, name=directory.name,
        expected_arms=expected_arms, status=status, attempt=attempt,
        review=state.get('review', {}) if status in ('ok', 'skipped') else {})


def read_episode(directory: Path, contains) -> tuple[dict, dict[str, Path]]:
    path = directory / 'episode.json'
    if not contains(path):
        raise ViewerDataError('任务清单超出查看目录')
    episode_manifest = json.loads(path.read_text())
    local = episode_manifest.get('kind') == 'umi_local_episode'
    if (episode_manifest.get('kind') not in ('umi_scene_episode', 'umi_local_episode')
            or episode_manifest.get('status') != ('local_ok' if local else 'ok')
            or not isinstance(episode_manifest.get('arms'), dict) or not episode_manifest['arms']
            or episode_manifest['arms'].keys() - {'left', 'right'}):
        raise ViewerDataError('共享场景任务清单无效')
    if local:
        scene_reference(episode_manifest)
    arms = {}
    for arm, relative in episode_manifest['arms'].items():
        if not isinstance(relative, str):
            raise ViewerDataError('共享场景手路径无效')
        path = (directory / relative).resolve()
        tcp_path = trajectory_path(path)
        if not path.is_relative_to(directory.resolve()) or not contains(tcp_path) or not tcp_path.is_file():
            raise ViewerDataError('共享场景手路径缺失或越界')
        arms[arm] = path
    return episode_manifest, arms


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
        dot = sum(first_coordinate*second_coordinate for first_coordinate, second_coordinate in zip(a, b))
        if dot < 0:
            b = [-coordinate for coordinate in b]; dot = -dot
        if dot > .9995:
            quaternion = [(1-fraction)*first_coordinate+fraction*second_coordinate for first_coordinate, second_coordinate in zip(a, b)]
        else:
            angle = math.acos(max(-1., min(1., dot)))
            quaternion = [(math.sin((1-fraction)*angle)*first_coordinate+math.sin(fraction*angle)*second_coordinate)/math.sin(angle) for first_coordinate, second_coordinate in zip(a, b)]
        norm = math.sqrt(sum(coordinate*coordinate for coordinate in quaternion))
        quaternions.append([coordinate/norm for coordinate in quaternion])
    return trajectory_from_poses(query, positions, quaternions)


def load_episode(catalog, key: str, directory: Path) -> dict:
    metadata, paths = read_episode(directory, catalog.contains)
    loaded = {arm: catalog.load(recording_key(path)) for arm, path in paths.items()}
    scene_record = scene_reference(metadata)
    for arm, recording in loaded.items():
        scene = recording['summary'].get('scene', {})
        if scene.get('frame_id') != scene_record['frame_id'] or scene.get('version') != scene_record['version']:
            raise ViewerDataError(f'{arm} 轨迹不属于本任务声明的场景坐标/地图版本')
    primary_arm = 'left' if 'left' in loaded else 'right'
    primary = loaded[primary_arm]
    start = max(recording['times'][0] for recording in loaded.values())
    end = min(recording['times'][-1] for recording in loaded.values())
    selected = [frame_index for frame_index, time in enumerate(primary['times']) if start <= time <= end]
    if len(selected) < 2:
        raise ViewerDataError('左右手没有共同有效时间')
    query = [primary['times'][frame_index] for frame_index in selected]
    primary['tracks'] = {name: sample_track(primary['times'], track, query) for name, track in primary['tracks'].items()}
    primary['frames'] = [primary['frames'][frame_index] for frame_index in selected]
    if primary.get('tag_distance') is not None:
        primary['tag_distance'] = {name: [values[frame_index] for frame_index in selected]
                                   for name, values in primary['tag_distance'].items()}
    primary['times'] = query
    primary.update(id=key, name=metadata.get('name', directory.parent.parent.name), path=str(directory),
        primary_arm=primary_arm, episode=metadata, companion=None, companion_tracks={})
    for arm, other in loaded.items():
        if arm != primary_arm:
            primary['companion'] = dict(other, arm=arm)
            primary['companion_tracks'] = {name: sample_track(other['times'], track, query) for name, track in other['tracks'].items()}
    return primary
