"""只读场景结果适配；当前结果统一从任务清单取值。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import os
from umi.infra import provenance as artifact_identity, provenance


TCP_COLUMNS = ('t_episode_s', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')
SCENE_RESULT_VERSIONS = (4, 5, 6)


@dataclass(frozen=True)
class ResultPaths:
    run: Path
    published: Path
    data: Path

    @property
    def manifest(self) -> Path:
        return self.data / 'episode.json'


def _internal_publication(directory: Path) -> bool:
    run = directory.parent
    return (directory.name == 'published' and run.parent.name == 'runs'
            and run.parent.parent.parent.name == 'tasks'
            and run.parent.parent.parent.parent.name == '.pipeline')


def result_paths(directory: Path) -> ResultPaths:
    """Resolve a legacy run, an internal run or its published result package."""
    directory = directory.resolve()
    if _internal_publication(directory):
        return ResultPaths(directory.parent, directory, directory / 'data')
    published = directory / 'published'
    if (published / 'data/episode.json').is_file():
        return ResultPaths(directory, published, published / 'data')
    data = directory / 'data'
    if (data / 'episode.json').is_file():
        return ResultPaths(directory, directory, data)
    if directory.name == 'data' and _internal_publication(directory.parent) and (directory / 'episode.json').is_file():
        return ResultPaths(directory.parent.parent, directory.parent, directory)
    return ResultPaths(directory, directory, directory)


def published_arm_run(directory: Path) -> Path:
    """Find the run containing an arm's published data or legacy files."""
    data = directory.resolve().parent
    if data.name == 'data' and (data / 'episode.json').is_file():
        return result_paths(data.parent).run
    return data


def arm_evidence(directory: Path, name: str) -> Path:
    """Read a per-arm diagnostic from its run, or beside a legacy arm."""
    return published_arm_run(directory) / directory.name / 'details' / name


def read_result(paths: ResultPaths) -> dict:
    published = paths.published.resolve()
    if not published.is_relative_to(paths.run) or not paths.manifest.resolve().is_relative_to(published):
        raise ValueError('结果清单超出发布目录')
    return json.loads(paths.manifest.read_text(encoding='utf-8'))


def validate_result(paths: ResultPaths, manifest: dict) -> None:
    validate_assets(paths.published, manifest['assets'])
    if manifest.get('schema_version') in (5, 6):
        validate_assets(paths.run, manifest['evidence_assets'])


def is_local_scene(coordinate_record: dict) -> bool:
    """任务坐标记录是否为单手独立回退的局部坐标（不与场景共享）。"""
    return coordinate_record.get('scope') == 'local'


def is_poster_scene(coordinate_record: dict) -> bool:
    return coordinate_record.get('scope') == 'task_poster'


def uses_scene_map(scene: dict) -> bool:
    return not is_local_scene(scene) and (not is_poster_scene(scene) or 'T_board_scene' in scene)


def _validate_poster_placement(matrix) -> None:
    if (not isinstance(matrix, list) or len(matrix) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
            or any(type(x) not in (float, int) or not math.isfinite(x) for row in matrix for x in row)
            or matrix[3] != [0, 0, 0, 1]):
        raise ValueError('海报坐标变换无效')
    x, y, z = [row[:3] for row in matrix[:3]]
    cross = [x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0]]
    if (any(abs(math.hypot(*axis)-1) > 1e-6 for axis in (x, y, z))
            or abs(sum(a*b for a, b in zip(x, y))) > 1e-6 or math.dist(cross, z) > 1e-6):
        raise ValueError('海报坐标旋转必须为右手正交矩阵')


def scene_map_identity(map_manifest: dict) -> str:
    """固定地图的坐标身份和全部资产共同参与任务复用。"""
    return artifact_identity.canonical_hash({key: map_manifest[key]
        for key in ('version', 'frame_id', 'mode', 'assets')})


def scene_reference(manifest: dict) -> dict:
    """读取唯一的场景坐标声明，并核对它与输入摘要中的地图身份一致。"""
    scene = manifest['scene']
    if 'T_board_scene' in scene and (manifest.get('kind') != 'umi_scene_episode' or not is_poster_scene(scene)):
        raise ValueError('地图到海报变换与结果坐标声明不一致')
    if manifest.get('kind') == 'umi_poster_episode':
        if (manifest.get('schema_version') not in SCENE_RESULT_VERSIONS or not is_poster_scene(scene)
                or not scene.get('frame_id', '').startswith('poster:')
                or scene.get('version') is not None or manifest.get('map') is not None):
            raise ValueError('任务海报坐标声明无效')
        for arm in manifest['arms']:
            evidence = manifest['arm_results'][arm]['fusion']
            if (evidence.get('scene_map_priors_used') is not False
                    or evidence.get('environment_poster', {}).get('status') != 'fused'):
                raise ValueError('任务海报结果缺少实际融合或含场景地图先验')
        return scene
    if manifest.get('kind') == 'umi_local_episode':
        if (not is_local_scene(scene) or not scene.get('frame_id', '').startswith('local:') or
                scene.get('version') is not None or len(manifest['arms']) != 1 or manifest.get('map') is not None):
            raise ValueError('独立局部结果的坐标声明无效')
        return scene
    if manifest.get('schema_version') not in SCENE_RESULT_VERSIONS:
        raise ValueError('任务清单需要受支持的结果版本')
    reference = scene
    if is_poster_scene(scene):
        reference = manifest['inputs'].get('scene_map_reference')
        if (manifest.get('schema_version') != 6 or not uses_scene_map(scene)
                or not scene.get('frame_id', '').startswith('poster:')
                or scene.get('axes') != 'board_layout' or scene.get('units') != 'metres'
                or scene.get('reconstruction_backend') != 'scene_map' or not manifest.get('map')
                or not reference or reference['version'] != scene['version']
                or not any(item['fusion'].get('environment_poster', {}).get('status') == 'fused'
                           for item in manifest['arm_results'].values())):
            raise ValueError('地图定位的任务海报坐标声明无效')
        _validate_poster_placement(scene['T_board_scene'])
    identity = scene_map_identity(dict(version=reference['version'], frame_id=reference['frame_id'],
        mode=manifest['inputs']['map_mode'], assets=manifest['inputs']['scene_assets']))
    if identity != manifest['inputs']['scene_map_sha256']:
        raise ValueError('任务声明的场景坐标或地图资产与输入摘要不一致')
    return scene


def _manifest(directory: Path, manifest: dict | None) -> dict:
    if manifest is None:
        path = directory.parent / 'episode.json'
        if not path.is_file():
            raise ValueError(f'缺少当前格式任务清单：{path}')
        manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('kind') not in ('umi_scene_episode', 'umi_local_episode', 'umi_poster_episode')
            or manifest.get('schema_version') not in SCENE_RESULT_VERSIONS):
        raise ValueError('任务清单需要受支持的场景结果版本')
    arm = directory.name
    if manifest['arms'].get(arm) != arm or arm not in manifest['arm_results']:
        raise ValueError('任务结果缺少对应的手或运行元数据')
    return manifest


def read_arm_summary(directory: Path, manifest: dict | None = None) -> dict:
    """返回查看器使用的摘要，不在磁盘生成重复副本。"""
    directory = directory.resolve()
    manifest = _manifest(directory, manifest)
    arm = directory.name
    arm_metadata = manifest['arm_results'][arm]
    summary = {key: arm_metadata[key] for key in ('rows', 'duration_s', 'path_length_m',
        'constraint_fraction', 'calibration_used', 'notes', 'fusion')}
    summary.update(status=manifest['status'], arm=arm, scene=scene_reference(manifest),
        video=manifest['inputs']['cameras'][arm]['serial'], tcp_frame=manifest['tcp_frames'][arm],
        video_time_offset_s=arm_metadata.get('video_time_offset_s', manifest['synchronization']['offsets_s'][arm]))
    if manifest.get('kind') == 'umi_local_episode':
        summary.update(mode='local', notes=[*summary['notes'], '独立回退：局部坐标，不与场景共享。'])
        return summary
    if not uses_scene_map(summary['scene']):
        summary.update(mode='poster', notes=[*summary['notes'], '任务海报坐标；无预建场景地图，不跨任务共享。'])
        return summary
    reference = manifest['map']
    if not isinstance(reference.get('path'), str) or Path(reference['path']).is_absolute():
        raise ValueError('场景地图引用必须为相对路径')
    # 清单中的地图路径相对于任务运行目录。
    map_frame = manifest['inputs'].get('scene_map_reference') or summary['scene']
    summary.update(mode='scene', map=dict(path=os.path.relpath(published_arm_run(directory) / reference['path'], directory),
                                          sha256=reference['sha256'], frame_id=map_frame['frame_id']))
    return summary


def read_arm_settings(directory: Path, manifest: dict | None = None) -> dict:
    """读取本次实际测量与视频配置。"""
    directory = directory.resolve()
    manifest = _manifest(directory, manifest)
    arm = directory.name
    arm_metadata = manifest['arm_results'][arm]
    return dict(scene=scene_reference(manifest), camera=manifest['inputs']['cameras'][arm],
        measurements=arm_metadata['measurements'], cad_status=arm_metadata['cad_status'],
        T_camera_tcp=arm_metadata['T_camera_tcp'], tcp_frame=manifest['tcp_frames'][arm],
        video_time_offset_s=arm_metadata.get('video_time_offset_s', manifest['synchronization']['offsets_s'][arm]),
        video_parameters=manifest['inputs']['trajectory_parameters']['video'], frame_geometry=arm_metadata['frame_geometry'])


def trajectory_path(directory: Path) -> Path:
    """返回场景统一轨迹表；独立轨迹结果使用 TUM 文件。"""
    combined = directory / 'trajectory.csv'
    return combined if combined.is_file() else directory / 'traj_tcp.txt'


def pose_rows(path: Path) -> Iterator[list[str]]:
    """按相同顺序提供 TCP 数值列；无需为查看器引入数值计算依赖。"""
    with path.open(encoding='utf-8', newline='') as stream:
        if path.suffix == '.csv':
            for row in csv.DictReader(stream):
                yield [row[column] for column in TCP_COLUMNS]
        else:
            for line in stream:
                if line.strip() and not line.lstrip().startswith('#'):
                    yield line.split()


def quality_summary(arm_results: dict, coordinate_record: dict, common_interval_s: list[float]) -> dict:
    """Return the compact, raw quality evidence used by terminal and JSON output.

    These values describe what the pipeline verified; they are not an independent
    accuracy score or a substitute for reviewing the trajectory against video.
    """
    arms = {}
    for arm, metadata in arm_results.items():
        constraints = metadata.get('constraint_fraction', {})
        fusion = metadata.get('fusion', {})
        poster = fusion.get('environment_poster')
        arms[arm] = dict(
            rows=metadata.get('rows'),
            duration_s=metadata.get('duration_s'),
            path_length_m=metadata.get('path_length_m'),
            visual_constraint_fraction=constraints.get('full'),
            imu_constraint_fraction=constraints.get('imu'),
            longest_visual_gap_s=fusion.get('maximum_visual_gap_s'),
            longest_position_support_gap_s=fusion.get('independent', {}).get('maximum_position_support_gap_s'),
            position_support=metadata.get('position_support'),
            poster_geometry_support=({key: value for key, value in poster.get('position_support', {}).items()
                                      if key != 'frames'} if isinstance(poster, dict) else None),
            visual_consistency_rms_m=fusion.get('visual_consistency_rms_m'),
            environment_poster_status=(poster.get('status')
                                        if isinstance(poster, dict) else None),
        )
    return dict(
        schema_version=1,
        scope='task_poster' if is_poster_scene(coordinate_record) else 'local' if is_local_scene(coordinate_record) else 'shared_scene',
        frame_id=coordinate_record.get('frame_id'),
        common_interval_s=list(common_interval_s),
        arms=arms,
        accuracy='not_measured',
    )


def recording_identity(path: Path) -> dict:
    """不读内容的原片身份（路径、大小、修改时间），用于运行期间的变化检测。"""
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def artifact_hashes(directory: Path, roots: tuple[str, ...]) -> dict:
    """directory 下指定文件或子树内文件的相对路径 → SHA256。"""
    return {str(path.relative_to(directory)): provenance.file_hash(path)
            for root in (directory / name for name in roots)
            for path in ([root] if root.is_file() else sorted(root.rglob('*'))) if path.is_file()}


def validate_assets(directory: Path, hashes: dict) -> None:
    """核对资产清单：非空、不越界、文件存在且内容未变。"""
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('场景/任务资产清单为空')
    for name, digest in hashes.items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory.resolve()) or not target.is_file() or provenance.file_hash(target) != digest:
            raise ValueError(f'场景/任务资产缺失、越界或内容已变：{name}')
