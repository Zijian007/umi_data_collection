"""Read UMI output layouts into a small, explicit browser data contract."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import bisect
import hashlib
import json
import math
import os

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from umi.store import layout
from umi.store.result_io import read_arm_settings, read_arm_summary, scene_reference, trajectory_path
from umi.workflows.segment.reporting.artifacts import front_video_path, load_summary, result_path
from umi.apps.viewer import ViewerDataError
from umi.apps.viewer.resources.cache import file_identity
from umi.apps.viewer.results.trajectory_reader import derive_body_tracks, read_frames, read_tag_distance, read_trajectory, trajectory_from_poses
from umi.apps.viewer.resources.video import rear_source


@lru_cache(maxsize=32)
def _read_manifest(identity: tuple) -> dict:
    return json.loads(Path(identity[0]).read_text(encoding='utf-8'))


def read_manifest(path: Path) -> dict:
    """Reuse parsed manifests while returning independent metadata to each consumer."""
    return deepcopy(_read_manifest(file_identity(path)))


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
    collection_path = session_dir / 'collection.yaml'
    collection = read_yaml(collection_path) if collection_path.is_file() else {}
    scene = collection.get('scene', {}) if isinstance(collection, dict) else {}
    scene_path = collection_path
    if not collection_path.is_file() or not isinstance(scene, dict):
        raise ViewerDataError('批次缺少有效 collection.yaml')
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
    return dict(scene=session_dir.parent.name, kit=session['kit']['id'], session=session_dir.name, name=directory.name,
        expected_arms=expected_arms, status=status, attempt=attempt,
        review=state.get('review', {}) if status in ('ok', 'skipped') else {})


def read_episode(directory: Path, contains) -> tuple[dict, dict[str, Path]]:
    path = directory / 'episode.json'
    if not contains(path):
        raise ViewerDataError('任务清单超出查看目录')
    episode_manifest = read_manifest(path)
    local = episode_manifest.get('kind') == 'umi_local_episode'
    if (episode_manifest.get('kind') not in ('umi_scene_episode', 'umi_local_episode', 'umi_poster_episode')
            or episode_manifest.get('status') != ('local_ok' if local else 'ok')
            or not isinstance(episode_manifest.get('arms'), dict) or not episode_manifest['arms']
            or episode_manifest['arms'].keys() - {'left', 'right'}):
        raise ViewerDataError('共享场景任务清单无效')
    if local or episode_manifest.get('kind') == 'umi_poster_episode':
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


def video_urls(base: str, video: Path | None) -> dict:
    from umi.apps.viewer.resources.video import cached_preview, video_version

    if video is None:
        return dict(video_url=None, poster_url=None, preview_url=None, preview_video_url=None, frame_url=None)
    version = video_version(video)
    preview = cached_preview(video)
    return dict(video_url=f'{base}/video?v={version}', poster_url=f'{base}/frame?t=0&v={version}',
                preview_url=f'{base}/preview', frame_url=f'{base}/frame',
                preview_video_url=f'{base}/preview/video?v={preview.stem}' if preview else None)


class RecordingCatalog:
    """Only expose result directories and videos within the selected local root."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise ViewerDataError(f'数据目录不存在：{self.root}')
        self.directories: dict[str, Path] = {}

    def contains(self, path: Path) -> bool:
        resolved = path.resolve()
        if resolved.is_relative_to(self.root):
            return True
        # 单独打开任务时，允许读取该任务在批次内的集中产物。
        if layout.is_batch(self.root.parent):
            internal = self.root.parent / layout.INTERNAL / 'tasks' / self.root.name
            if internal.resolve().is_relative_to(self.root.parent):
                return resolved.is_relative_to(internal.resolve())
        return False

    def summary(self, directory: Path) -> dict:
        manifest = directory.parent / 'episode.json'
        if manifest.is_file() and self.contains(manifest):
            try:
                metadata = read_manifest(manifest)
                if metadata.get('schema_version', 1) >= 3:
                    return read_arm_summary(directory, metadata)
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ViewerDataError(f'场景运行元数据无效：{error}') from error
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
                '.git', '.deps', 'node_modules', 'details', 'debug', 'diagnostics', 'work', 'bundle', 'umi_calib_out', 'calibration_result', 'run', '__pycache__',
                'runs', 'versions', 'archive', 'cache', 'imports', 'mapping', 'videos', 'pending',
            } and not name.startswith('.'))
            capture = {}
            capture_path = layout.manifest(directory, 'episode.yaml')
            has_capture = capture_path.is_file() and self.contains(capture_path)
            if has_capture:
                try:
                    state_path = layout.manifest(directory, 'state.json')
                    if state_path.exists() and not self.contains(state_path):
                        raise ViewerDataError('任务状态超出查看目录')
                    capture = capture_status(directory)
                except (OSError, ValueError, TypeError, KeyError, IndexError):
                    capture = dict(status='pending')
                result_name = 'local_results' if capture.get('status') in ('local_ok', 'local_skipped') else 'results'
                if 'scene' in capture and not (directory / result_name / 'episode.json').is_file():
                    key = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:20]
                    label = '/'.join(str(capture.get(name, directory.name)) for name in ('scene', 'kit', 'session', 'name'))
                    entries.append(dict(id=key, name=label, path=str(directory.relative_to(self.root)),
                        status=capture['status'], review=capture.get('review', {}).get('decision', 'unreviewed'),
                        available=False, serial=', '.join(capture.get('expected_arms', [])),
                        reason=capture.get('attempt', {}).get('reason', '')))
                    children[:] = []
                    continue
            # 任务的 results 是原子发布的版本指针；只展示当前版本，不遍历历史 runs。
            episode = directory / result_name if has_capture else directory
            if (episode / 'episode.json').is_file():
                children[:] = []
                try:
                    metadata, arms = read_episode(episode, self.contains)
                    key = recording_key(episode)
                    directories[key] = episode.resolve()
                    for path in arms.values():
                        directories[recording_key(path)] = path
                    interval = metadata['common_interval_s']
                    session = metadata.get('session')
                    kit = capture.get('kit') or metadata.get('kit', {}).get('id')
                    label = '/'.join(filter(None, (metadata.get('scene', {}).get('name'), kit, session, metadata.get('name', directory.name))))
                    entries.append(dict(id=key, name=label, path=str(episode.relative_to(self.root)),
                        serial=('任务海报轨迹' if metadata.get('kind') == 'umi_poster_episode' else
                                '独立局部轨迹' if metadata.get('kind') == 'umi_local_episode' else
                                '双手场景' if len(arms) == 2 else '场景任务'), status=capture.get('status', metadata['status']),
                        review=capture.get('review', {}).get('decision', 'unreviewed'), available=True, duration=interval[1]-interval[0],
                        rows=metadata.get('rows'), calibration=metadata['scene']['version']))
                except (ViewerDataError, OSError, ValueError, KeyError):
                    continue
                continue
            if not {'traj_tcp.txt', 'trajectory.csv'} & set(files):
                continue
            children[:] = []
            if not self.contains(directory) or not self.contains(trajectory_path(directory)):
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
        entries.sort(key=lambda entry: (entry.get('available', True), entry['name']), reverse=True)
        return dict(root=str(self.root), recordings=entries)

    def directory(self, key: str) -> Path:
        directory = self.directories.get(key)
        if directory is None or not self.contains(directory):
            raise ViewerDataError('录像不存在，请刷新列表。')
        return directory

    def _media_directory(self, key: str) -> Path:
        directory = self.directory(key)
        if (directory / 'episode.json').is_file():
            _, arms = read_episode(directory, self.contains)
            return arms['left'] if 'left' in arms else arms['right']
        return directory

    def _front_video(self, directory: Path, summary: dict) -> Path | None:
        path = front_video_path(directory, summary)
        return path if path is not None and self.contains(path) else None

    def video_path(self, key: str) -> Path | None:
        directory = self._media_directory(key)
        try:
            summary = self.summary(directory)
        except (OSError, ValueError, TypeError):
            summary = {}
        return self._front_video(directory, summary)

    def rear_source(self, key: str):
        directory = self.directory(key)

        def contains_raw(path: Path) -> bool:
            if self.contains(path):
                return True
            # Opening take_001/results resolves to its published run directory.
            # In that mode only the matching task and arm may supply an INSV.
            run = directory.parent
            task = run.parent.parent
            if (self.root != run or run.parent.name != 'runs' or task.parent.name != 'tasks'
                    or task.parent.parent.name != layout.INTERNAL):
                return False
            arm_inputs = task.parent.parent.parent / task.name / directory.name
            return path.resolve().is_relative_to(arm_inputs)

        return rear_source(directory, self.contains, contains_raw=contains_raw)

    def map_data(self, key: str) -> dict:
        from umi.apps.viewer.resources import maps
        directory = self._media_directory(key)
        return maps.load(directory, self.summary(directory))

    def details(self, key: str) -> dict:
        directory = self.directory(key)
        if (directory / 'episode.json').is_file():
            episode, _ = read_episode(directory, self.contains)
            return dict(directory=str(directory), episode=episode)
        return dict(directory=str(directory), summary=self.summary(directory))

    def playback(self, key: str) -> dict:
        """Return the trajectory and media data needed before opening diagnostics."""
        recording = self.load(key)
        recording.pop('episode', None)
        fields = ('scene', 'status', 'mode', 'reason', 'calibration_used', 'constraint_fraction')
        for arm in (recording, recording.get('companion')):
            if arm is None:
                continue
            media_key = arm.get('video_recording_id', arm['id'])
            source = self.rear_source(media_key)
            if source is not None:
                path, clip = source
                base = f'/api/recordings/{media_key}/rear'
                arm['rear'] = (video_urls(base, path) if clip is None else
                               dict(preview_url=f'{base}/preview', frame_url=f'{base}/frame'))
            summary = arm['summary']
            arm['summary'] = {field: summary[field] for field in fields if field in summary}
            fusion = summary.get('fusion') or {}
            for field in ('reconstruction_backend', 'maximum_position_support_gap_s', 'maximum_visual_gap_s'):
                if field in fusion:
                    arm['summary'][field] = fusion[field]
        if recording.get('companion'):
            recording['companion'].pop('tracks', None)
        return recording

    def load(self, key: str) -> dict:
        directory = self.directory(key)
        if (directory / 'episode.json').is_file():
            return load_episode(self, key, directory)
        summary = self.summary(directory)
        tcp_path = trajectory_path(directory)
        if not self.contains(tcp_path):
            raise ViewerDataError('轨迹文件超出当前数据目录。')
        tcp = read_trajectory(tcp_path)
        times = tcp.pop('times')
        tag_distance = read_tag_distance(tcp_path, times)
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
        if summary.get('mode') in ('scene', 'local', 'poster') and {'camera', 'imu'} - tracks.keys():
            try:
                derived = derive_body_tracks(times, tcp, read_arm_settings(directory))
                for name in ('camera', 'imu'):
                    tracks.setdefault(name, derived[name])
            except (OSError, ValueError, KeyError, TypeError) as error:
                notes.append(f'相机/IMU 显示轨迹未重建：{error}')
        video = self._front_video(directory, summary)
        urls = video_urls(f'/api/recordings/{key}', video)
        return dict(id=key, name=directory.name, path=str(directory), summary=summary, times=times,
                    tag_distance=tag_distance,
                    tracks=tracks, frames=frames, notes=notes + list(summary.get('notes') or []),
                    video_recording_id=key, **urls,
                    video_time_offset_s=summary.get('video_time_offset_s', times[0]),
                    video_note='工作镜头裁切视频 · 与轨迹使用同一视频时钟' if video else '未保留工作镜头视频，仍可独立回放三维轨迹。')
