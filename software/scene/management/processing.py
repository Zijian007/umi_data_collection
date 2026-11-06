"""从批次声明和原片启动处理；设备与地图快照沿用现有登记和处理实现。"""
from __future__ import annotations

from software.scene import layout

import os
import json
from pathlib import Path

from ruamel.yaml.error import YAMLError

from software import config, registry
from software.scene import workflow
from . import capture_config, episodes, ingest, inputs
from software.scene.pipeline import recordings
from software.storage import write_json, write_yaml


def configured_batches(paths: list[Path]) -> list[Path]:
    """发现真实 batch.yaml；示例、缓存和运行结果不参与发现。"""
    found = set()
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_file():
            if path.name not in {'batch.yaml', 'manifest.yaml', 'scene.yaml'}:
                continue
            path = layout.owner(path)
        if not path.is_dir():
            raise ValueError(f'处理路径不存在：{path}')
        for current, children, files in os.walk(path):
            children[:] = sorted(name for name in children if not name.startswith('.')
                                 and name not in inputs.INTERNAL_DIRECTORIES)
            if 'batch.yaml' in files:
                found.add(Path(current))
                children[:] = []
    return sorted(found)


def batch_request(batch: Path) -> dict:
    """每次读取当前 batch.yaml、kit.yaml 和 mapping/，不由缓存覆盖配置。"""
    inputs.validate_batch_name(batch.name)
    inputs.validate_name(batch.parent.name)
    data = inputs.read_yaml(batch / 'batch.yaml')
    inputs.require_keys(data, {'kit', 'arms', 'map_version'}, {'kit'}, 'batch')
    inputs.validate_name(data['kit'])
    manifest = layout.manifest(batch, 'manifest.yaml')
    previous = inputs.read_yaml(manifest) if manifest.is_file() else {}
    if not previous:
        capture_config.validate_new_batch(batch)
    kit = inputs.read_kit(data['kit'])
    capture_config.validate_kit_assets(kit)
    expected = inputs.read_arms(data.get('arms', list(kit['arms'])), kit['arms'])
    scene_path = batch.parent / 'scene.yaml'
    scene = inputs.read_scene(scene_path, load_mapping=False)
    mapping = capture_config.mapping_paths(batch, scene.mode)
    if mapping and 'map_version' in data:
        raise ValueError('本轮建图原片与 map_version 不能同时提供')
    if not mapping and 'map_version' not in data:
        raise ValueError(f'请将本轮建图原片放入 {batch / "mapping"}，或在 batch.yaml 明确指定 map_version')
    kit_serials = {item['serial'] for item in kit['arms'].values()}
    for video in mapping.values():
        serial = recordings.inspect(video)['serial']
        if serial not in kit_serials:
            registry.pin(serial)
    if 'map_version' in data:
        registry.validate_id(data['map_version'])
        workflow.load_published(scene, version=data['map_version'])
    return dict(data=data, previous=previous, kit=kit, expected=expected, mapping=mapping,
                parameters=scene.parameters)


def prepare_batch(batch: Path, *, dry_run=False) -> dict:
    """自动登记和导入；预览不创建场景、快照、锁或处理目录。"""
    if not dry_run and (batch / 'manifest.yaml').is_file():
        from .migration import migrate_scene
        migrate_scene(batch.parent)
    request = batch_request(batch)
    if dry_run:
        found, errors = ingest.scan_raw_videos(inputs.batch_video_sources(batch))
        allowed = {arm: request['kit']['arms'][arm]['serial'] for arm in request['expected']}
        mapping_hashes = {registry.sha256_file(path) for path in request['mapping'].values()}
        ignored = [dict(item.summary(), reason='不属于本批次预期使用的设备')
                   for item in found if item.serial not in allowed.values()]
        selected = [item for item in found if item.serial in allowed.values() and item.sha256 not in mapping_hashes]
        existing = {item['sha256']: path for path in episodes.episode_paths([batch])
                    for item in inputs.read_yaml(path).get('recordings', {}).values()}
        plans, pending = ingest.plan_tasks(selected, allowed, request['parameters'], batch, existing)
        return dict(episodes=[dict(path=str(path), name=layout.owner(path).name, arms=list(plan)) for path, plan in plans],
                    pending=pending, errors=errors, ignored=ignored)
    manifest = refresh_record(batch, request)
    return ingest.ingest([], manifest)


def refresh_record(batch: Path, request: dict) -> Path:
    """保存当前输入的解析结果；旧 map 只作为待校验的复用候选。"""
    kit = {key: value for key, value in request['kit'].items() if key != 'description'}
    data = dict(schema_version=5, kit=kit)
    if request['mapping']:
        data['mapping'] = capture_config.snapshot_mapping(batch, request['mapping'], kit)
        previous = request['previous']
        if (previous.get('map') and inputs.mapping_records(previous.get('mapping', {}))
                == inputs.mapping_records(data['mapping'])):
            data['map'] = previous['map']
    elif 'map_version' in request['data']:
        scene = inputs.read_scene(batch.parent, load_mapping=False)
        directory, selected = workflow.load_published(scene, version=request['data']['map_version'])
        data['map'] = dict(version=selected['version'], manifest_sha256=config.file_hash(directory / 'manifest.json'))
    if request['expected'] != inputs.read_arms(list(kit['arms'])):
        data['expected_arms'] = list(request['expected'])
    path = layout.metadata(batch, 'manifest.yaml')
    with layout.locked(batch, blocking=True):
        if inputs.read_yaml(batch / 'batch.yaml') != request['data'] or inputs.read_kit(request['data']['kit']) != request['kit']:
            raise ValueError('batch.yaml 或 kit.yaml 在准备期间发生变化，请重新运行')
        previous = inputs.read_yaml(path) if path.is_file() else {}
        if previous != request['previous']:
            raise ValueError('批次记录在准备期间发生变化，请重新运行')
        if previous != data:
            # 为旧格式失败记录补上当时的设备/来源，随后更新最新解析记录。
            for task in episodes.episode_paths([batch]):
                root = layout.owner(task)
                state = layout.read_state(root).get('attempt', {})
                if state.get('run') and state.get('session_sha256') == config.file_hash(path):
                    error_path = root / state['run'] / 'error.json'
                    if error_path.is_file():
                        error = json.loads(error_path.read_text())
                        if 'batch' not in error:
                            write_json(error_path, dict(error, batch=previous))
            write_yaml(path, data)
    return path


def run(paths: list[Path], *, jobs=1, force=False, cores='', dry_run=False,
        check_recordings=True, map_version=None) -> list[dict]:
    if jobs < 1:
        raise ValueError('jobs 必须至少为 1')
    batches = configured_batches(paths)
    summaries, ready = [], []
    for batch in batches:
        try:
            imported = prepare_batch(batch, dry_run=dry_run)
            problems = imported['errors'] + imported['pending'] + imported['ignored']
            if problems:
                detail = '; '.join(f'{item["path"]}: {item.get("reason") or item.get("status", "待确认")}'
                                   for item in problems)
                summaries.append(dict(path=str(batch), name=batch.name, status='failed',
                    error='存在无法读取、设备不匹配或待补齐的原片：' + detail, details=problems))
            registered = episodes.episode_paths([batch])
            if dry_run:
                for item in imported['episodes']:
                    summaries.append(dict(item, status='planned', stages=['登记原片', '固定或构建批次地图', '抽取轨迹']))
                # 已登记任务可能全部被入库复用，仍列出处理计划。
                planned = {item['path'] for item in imported['episodes']}
                for path in registered:
                    if str(path) not in planned:
                        summaries.append(dict(path=str(path), name=layout.owner(path).name, status='planned'))
            else:
                ready.extend(registered)
            if not imported['episodes'] and not registered and not problems:
                raise ValueError(f'没有任务原片，请放入 {batch}/take_001/videos/')
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, YAMLError) as error:
            summaries.append(dict(path=str(batch), name=batch.name, status='failed', error=str(error)))
    # 已登记的旧批次和显式任务路径继续使用相同的处理引擎。
    direct = episodes.episode_paths(paths, exclude=batches)
    if not dry_run:
        from .migration import migrate_scene
        legacy_scenes = {layout.owner(path).parent.parent for path in direct if path.parent.name != layout.INTERNAL}
        for scene in sorted(legacy_scenes):
            migrate_scene(scene)
        direct = [layout.manifest(path, 'episode.yaml') for path in direct]
    for batch in sorted({layout.owner(path).parent for path in direct
                         if (layout.owner(path).parent / 'batch.yaml').is_file()}):
        selected = [path for path in direct if layout.owner(path).parent == batch]
        try:
            imported = prepare_batch(batch, dry_run=dry_run)
            if imported['errors'] or imported['pending'] or imported['ignored']:
                raise ValueError('当前批次存在无法读取、设备不匹配或待补齐的原片，请先核对')
            if dry_run:
                summaries.extend(dict(path=str(path), name=layout.owner(path).name, status='planned') for path in selected)
                direct = [path for path in direct if path not in selected]
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, YAMLError) as error:
            summaries.extend(dict(path=str(path), name=layout.owner(path).name, status='failed', error=str(error)) for path in selected)
            direct = [path for path in direct if path not in selected]
    ready.extend(direct)
    ready = list(dict.fromkeys(ready))
    if ready:
        summaries.extend(workflow.run_episodes(ready, jobs=jobs, force=force, cores=cores, dry_run=dry_run,
            check_recordings=check_recordings, map_version=map_version))
    if not summaries:
        raise ValueError('没有找到批次配置或任务清单；请准备 batch.yaml 和 take_001/videos/')
    return summaries
