"""阅读入口：build_scene() 固定场景地图；run_episode() 复用地图并分别融合双手。

所有输入先预检；新版本全部成功后才发布指针。失败不替换已有成功地图或任务结果。
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from software import config
from software.console import Console
from software.scene import engine, fusion, mapping, results
from software.scene.geometry import camera_to_imu
from software.scene.models import Camera, Scene, read_episode, read_scene, rear_files, select_camera
from software.storage import locked, publish_link, write_json
from software.trajectory import runtime
from software.trajectory.models import ReconstructionError

console = Console('场景')


def cleanup_work(path: Path, keep: bool) -> None:
    if keep or not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError as error:
        console.write(f'结果已发布，中间数据未清理：{error}', state='提醒')


def recording_identity(path: Path) -> dict:
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def select_cameras(videos: dict, scene: Scene, expected: dict | None = None) -> dict[str, Camera]:
    cameras = {arm: select_camera(video, scene) for arm, video in videos.items()}
    if expected is not None and any(camera.serial != expected[arm] for arm, camera in cameras.items()):
        raise ValueError('INSV 内部序列号与任务清单不一致')
    if len({camera.serial for camera in cameras.values()}) != len(cameras):
        raise ValueError('左右手必须来自两台不同序列号的相机')
    if scene.mode == 'dual' and set(videos) - scene.recordings.keys():
        raise ValueError('双镜头任务的每只手都需要先在该场景建立对应前地图')
    return cameras


def artifact_hashes(directory: Path, roots: tuple[str, ...]) -> dict:
    return {str(path.relative_to(directory)): config.file_hash(path)
            for name in roots for path in sorted((directory / name).rglob('*')) if path.is_file()}


def validate_assets(directory: Path, hashes: dict) -> None:
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('场景/任务资产清单为空')
    for name, digest in hashes.items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory.resolve()) or not target.is_file() or config.file_hash(target) != digest:
            raise ValueError(f'场景/任务资产缺失、越界或内容已变：{name}')


def load_published(scene: Scene, *, version: str | None = None, manifest_sha256: str | None = None) -> tuple[Path, dict]:
    pinned = version is not None
    if not pinned:
        if not (scene.root / 'maps/current.json').is_file():
            raise ValueError('场景尚未发布地图，请先运行 umi-scene build <scene.yaml>')
        pointer = json.loads((scene.root / 'maps/current.json').read_text())
        version, manifest_sha256 = pointer['version'], pointer['manifest_sha256']
    if not isinstance(version, str) or not version or any(c not in '0123456789abcdef-' for c in version):
        raise ValueError('场景版本指针无效')
    directory = (scene.root / 'maps/versions' / version).resolve()
    if not directory.is_relative_to((scene.root / 'maps/versions').resolve()):
        raise ValueError('场景版本指针越界')
    manifest = directory / 'manifest.json'
    if manifest_sha256 is not None and config.file_hash(manifest) != manifest_sha256:
        raise ValueError('场景版本清单哈希不匹配')
    data = json.loads(manifest.read_text())
    if not pinned and data['definition'] != mapping.definition(scene):
        raise ValueError('场景建图配置已变化；请先运行 umi-scene build，任务不会自动重建地图')
    if data['name'] != scene.name or data['mode'] != scene.mode:
        raise ValueError('固定地图的场景或镜头模式与当前配置不一致')
    validate_assets(directory, data['assets'])
    current_engine = config.engine_identity('trajectory')
    if data['inputs']['engines'].get('orb_library') != current_engine.get('orb_library'):
        raise ValueError('ORB 地图库版本已变化，请重新建立场景地图')
    return directory, data


def build_scene(path: Path, *, force=False, keep_work=False, cores='', dry_run=False) -> dict:
    scene = read_scene(path)
    cameras = select_cameras(scene.recordings, scene, scene.recording_serials)
    runtime.validate_engines(dict(os.environ))
    inputs = dict(definition=mapping.definition(scene), recordings={arm: recording_identity(video) for arm, video in scene.recordings.items()},
        cameras={arm: camera.identity() for arm, camera in cameras.items()}, code=config.code_identity('scene'),
        trajectory_code=config.code_identity('trajectory'), engines=config.engine_identity('trajectory'))
    if dry_run:
        return dict(status='planned', scene=scene.name, inputs=inputs)
    with locked(scene.root / 'maps'):
        if not force and (scene.root / 'maps/current.json').is_file():
            try:
                _, current = load_published(scene)
                if current['inputs'] == inputs:
                    return dict(status='skipped', version=current['version'], frame_id=current['frame_id'])
            except (OSError, ValueError, KeyError) as error:
                console.write(f'准备新的场景版本：{error}')
        version = config.canonical_hash(inputs)[:12]+'-'+uuid.uuid4().hex[:8]
        directory = scene.root / 'maps/versions' / version
        directory.mkdir(parents=True)
        work = directory / 'work'
        started = time.monotonic()
        try:
            console.write('准备建图录像与实测相机参数', step=(1, 3))
            prepared = {arm: engine.prepare(video, cameras[arm], scene, work / arm, cores=cores)
                        for arm, video in scene.recordings.items()}
            console.write('纯视觉建图 → 米制尺度与重力 → 场景放置', step=(2, 3))
            errors = []
            for attempt in range(1, scene.trajectory.values.retry.max_tries+1):
                try:
                    maps = mapping.build_maps(scene, prepared, work / f'try_{attempt}')
                    break
                except (ValueError, ReconstructionError) as error:
                    errors.append(str(error))
                    console.write(str(error), attempt=attempt, state='重试' if attempt < scene.trajectory.values.retry.max_tries else '失败')
            else:
                raise ValueError('场景地图未通过验收：'+'；'.join(errors))
            console.write('固定并发布场景地图', step=(3, 3))
            frozen, display = mapping.freeze_maps(scene, maps, cameras, directory)
            write_json(directory / 'map.json', display)
            assets = artifact_hashes(directory, ('maps', 'devices'))
            assets['map.json'] = config.file_hash(directory / 'map.json')
            manifest = dict(kind='umi_scene', schema_version=1, name=scene.name, mode=scene.mode, version=version,
                frame_id=f'{scene.name}/{version}', units='metres', gravity_axis='-Z', definition=mapping.definition(scene),
                maps=frozen, inputs=inputs, assets=assets, elapsed_s=time.monotonic()-started,
                parameters=asdict(scene.parameters), trajectory_parameters=scene.trajectory.snapshot(), actual_error='not_measured')
            # 最后再核对身份，防止原片或测量在长任务期间被替换。
            if inputs['recordings'] != {arm: recording_identity(video) for arm, video in scene.recordings.items()}:
                raise ValueError('建图期间原始录像发生变化，拒绝发布')
            if inputs['cameras'] != {arm: select_camera(video, scene).identity() for arm, video in scene.recordings.items()}:
                raise ValueError('建图期间设备标定发生变化，拒绝发布')
            write_json(directory / 'manifest.json', manifest)
            (directory / 'report.txt').write_text(f'场景：{scene.name}\n版本：{version}\n坐标：{manifest["frame_id"]}\n'
                f'地图：{", ".join(frozen)}\n单位：米；Z 朝上。\n实际误差：未测量；对齐验收仅检查内部一致性。\n', encoding='utf-8')
            write_json(scene.root / 'maps/current.json', dict(version=version, manifest_sha256=config.file_hash(directory / 'manifest.json')))
        except Exception as error:
            write_json(directory / 'failure.json', dict(status='failed', reason=str(error)))
            runtime.write_error_log(directory / 'error.log', error)
            raise
        cleanup_work(work, keep_work)
        return dict(status='ok', version=version, frame_id=manifest['frame_id'], output=str(directory))


def episode_inputs(episode, cameras, published) -> dict:
    return dict(scene_version=published['version'], frame_id=published['frame_id'], scene_assets=published['assets'],
        session=episode.session.name, session_manifest_sha256=config.file_hash(episode.session.path),
        episode_manifest_sha256=config.file_hash(episode.path), scene_manifest_sha256=config.file_hash(episode.scene.path),
        recordings={arm: recording_identity(video) for arm, video in episode.recordings.items()},
        cameras={arm: camera.identity() for arm, camera in cameras.items()}, synchronization=episode.synchronization,
        parameters=asdict(episode.scene.parameters), trajectory_parameters=episode.scene.trajectory.snapshot()['values'],
        code=config.code_identity('scene'), trajectory_code=config.code_identity('trajectory'), engines=config.engine_identity('trajectory'))


def run_episode(path: Path, *, force=False, keep_work=False, cores='', dry_run=False) -> dict:
    episode = read_episode(path)
    offsets = episode.offsets()
    scene = episode.scene
    device_scene = replace(scene, devices=episode.session.devices, rear_calibrations=rear_files(episode.session.devices))
    cameras = select_cameras(episode.recordings, device_scene, episode.serials)
    runtime.validate_engines(dict(os.environ))
    directory, published = load_published(scene, version=episode.session.map_version,
                                          manifest_sha256=episode.session.map_manifest_sha256)
    # 已建图相机的镜头测量被固定；新标定不能静默套用在旧地图的场景放置上。
    pinned = {item['serial']: item['assets'] for item in published['inputs']['cameras'].values()}
    if scene.mode == 'dual':
        for arm, camera in cameras.items():
            if published['maps'][f'{arm}_front']['camera_serial'] != camera.serial:
                raise ValueError(f'{arm} 设备与固定前地图不匹配；更换设备或左右手需要新地图版本')
    for camera in cameras.values():
        if camera.serial in pinned and any(pinned[camera.serial].get(lens) != config.file_hash(camera.files[lens]) for lens in camera.measurements):
            raise ValueError(f'相机 {camera.serial} 的镜头标定已变化，请重建场景地图')
    inputs = episode_inputs(episode, cameras, published)
    if dry_run:
        return dict(status='planned', inputs=inputs, output=str(episode.output))
    with locked(episode.path.parent):
        previous = episode.output / 'episode.json'
        if not force and previous.is_file():
            old = json.loads(previous.read_text())
            if old['inputs'] == inputs:
                validate_assets(episode.output.resolve(), old['assets'])
                return dict(status='skipped', output=str(episode.output), frame_id=published['frame_id'])
        target = episode.path.parent / 'runs' / uuid.uuid4().hex
        target.mkdir(parents=True)
        work = target / 'work'
        started = time.monotonic()
        try:
            console.write('准备任务录像与时间轴', step=(1, 3))
            prepared = {arm: engine.prepare(video, cameras[arm], scene, work / arm, cores=cores)
                        for arm, video in episode.recordings.items()}
            if max(prepared[arm].times[0]+offsets[arm] for arm in prepared) >= min(prepared[arm].times[-1]+offsets[arm] for arm in prepared):
                raise ValueError('左右手录像在任务时间轴上没有重叠')
            display = json.loads((directory / 'map.json').read_text())
            scene_record = dict(name=scene.name, version=published['version'], frame_id=published['frame_id'], session=episode.session.name)
            tracks = {}
            for arm, recording in prepared.items():
                console.write(f'{arm}：已有场景地图定位 → 固定场景系融合', step=(2, 3))
                observations = {}
                for lens in cameras[arm].measurements:
                    role = scene.role(arm, lens)
                    item = published['maps'][role]
                    raw = engine.localize(recording, lens, directory / item['archive'], item['index'], work / arm / f'loc_{role}')
                    observations[lens] = raw.transformed(item['placement'])
                T_cam_imu = {lens: np.linalg.inv(camera_to_imu(measured)) for lens, measured in cameras[arm].measurements.items()}
                front_time = recording.times+cameras[arm].measurements['front']['cam_time_offset_ns']/1e9
                result = fusion.fuse(front_time, recording.imu, observations, T_cam_imu, scene.trajectory.values, scene.parameters)
                tracks[arm] = results.write_arm(arm, recording, result, offsets[arm], scene_record, display, target / arm)
            console.write('对齐双手时间并发布结果', step=(3, 3))
            common = results.write_bimanual(tracks, target)
            if inputs != episode_inputs(episode, select_cameras(episode.recordings, device_scene, episode.serials), published):
                raise ValueError('任务期间录像或标定发生变化，拒绝发布')
            (target / 'report.txt').write_text(f'场景任务：{episode.path.parent.name}\n采集批次：{episode.session.name}\n坐标：{published["frame_id"]}\n'
                f'手：{", ".join(tracks)}\n共同有效时间：{common[0]:.6f}–{common[1]:.6f} s\n'
                f'同步：{episode.synchronization["method"]}；offsets_s={offsets}\n'
                '轨迹共享米制坐标；左右手分别融合。\n实际误差：未测量；TCP 使用 CAD 名义几何。\n'
                '各手指标见 left/report.txt、right/report.txt（以实际存在的手为准）。\n', encoding='utf-8')
            assets = artifact_hashes(target, tuple(tracks))
            for name in ('report.txt', 'bimanual.csv'):
                if (target / name).is_file():
                    assets[name] = config.file_hash(target / name)
            primary_times = tracks.get('left', next(iter(tracks.values())))[0]
            metadata = dict(kind='umi_scene_episode', schema_version=1, status='ok', name=episode.path.parent.name, session=episode.session.name, scene=scene_record,
                frame_id=published['frame_id'], arms={arm: arm for arm in tracks}, common_interval_s=common,
                rows=int(np.count_nonzero((primary_times >= common[0]) & (primary_times <= common[1]))),
                inputs=inputs, assets=assets, elapsed_s=time.monotonic()-started, actual_error='not_measured')
            write_json(target / 'episode.json', metadata)
            publish_link(target, episode.output)
        except Exception as error:
            write_json(target / 'failure.json', dict(status='failed', reason=str(error)))
            runtime.write_error_log(target / 'error.log', error)
            raise
        cleanup_work(work, keep_work)
        return dict(status='ok', output=str(episode.output), frame_id=published['frame_id'], common_interval_s=common)
