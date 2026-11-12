"""旧目录的一次性原位整理；原片、标定和地图清单内容保持原值。"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import fcntl
import json
import os
import shutil

from . import inputs
from software import provenance as artifact_identity
from software.scene import layout
from software.storage import publish_link, write_json, write_yaml


def _move(source: Path, target: Path) -> None:
    if not source.exists() and not source.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() and not target.is_symlink():
        source.rename(target)
    elif source.is_symlink() or target.is_symlink():
        if source.is_symlink() and target.is_symlink() and source.readlink() == target.readlink():
            source.unlink()
        else:
            raise ValueError(f'内部链接迁移冲突，已保留：{source} → {target}')
    elif source.is_dir() and target.is_dir():
        for child in source.iterdir():
            _move(child, target / child.name)
        source.rmdir()
    elif source.is_file() and target.is_file() and artifact_identity.file_hash(source) == artifact_identity.file_hash(target):
        source.unlink()
    else:
        raise ValueError(f'内部目录迁移冲突，已保留原文件：{source} → {target}')


def _relocate(root: Path, names: tuple[str, ...]) -> None:
    for name in names:
        target = layout.metadata(root, name)
        for source in (root / name, root / '.umi' / name):
            if source != target:
                _move(source, target)


def _task(root: Path) -> None:
    previous_targets = {root / name: Path(os.path.abspath(root / (root / name).readlink()))
                        for name in ('results', 'local_results') if (root / name).is_symlink()}
    old_manifest = layout.manifest(root, 'episode.yaml')
    previous_hash = artifact_identity.file_hash(old_manifest)
    configuration = layout.task_configuration(root)
    _relocate(root, ('episode.yaml', 'state.json', 'cache', 'runs'))
    capture_path = layout.metadata(root, 'episode.yaml')
    capture = inputs.read_yaml(capture_path)
    if capture.keys() & layout.TASK_FIELDS:
        write_yaml(root / 'task.yaml', configuration)
        write_yaml(capture_path, {key: value for key, value in capture.items() if key not in layout.TASK_FIELDS})
    current_hash = artifact_identity.file_hash(capture_path)
    runs = layout.metadata(root, 'runs')
    def moved_run(path):
        for previous in (root / 'runs', root / '.umi/runs'):
            if path.is_relative_to(previous):
                return runs / path.relative_to(previous)
        return path
    for result, previous_target in previous_targets.items():
        relocated = moved_run(previous_target)
        if relocated != previous_target:
            publish_link(relocated, result)
    state_path = layout.metadata(root, 'state.json')
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    for record in (state.get('attempt', {}), state.get('review', {})):
        if record.get('run'):
            record['run'] = layout.run_reference(root, moved_run(Path(os.path.abspath(root / record['run']))))
    if state.get('attempt', {}).get('manifest_sha256') == previous_hash:
        state['attempt']['manifest_sha256'] = current_hash
        state['attempt'].setdefault('task_configuration_sha256', artifact_identity.canonical_hash(configuration))
    for run in sorted(runs.iterdir()) if runs.is_dir() else ():
        if not run.is_dir():
            continue
        failure, trace = run / 'failure.json', run / 'error.log'
        if failure.is_file():
            error = json.loads(failure.read_text())
            if 'reason' not in error:
                error['reason'] = '；'.join(error.pop('reasons', []))
            error['traceback'] = trace.read_text() if trace.is_file() else ''
            write_json(run / 'error.json', error)
            failure.unlink()
            trace.unlink(missing_ok=True)
        changed = {}
        for summary in run.glob('*/details/summary.json'):
            manifest_record = json.loads(summary.read_text())
            if 'map' in manifest_record and manifest_record.get('scene', {}).get('version'):
                target = layout.maps(root.parent) / 'versions' / manifest_record['scene']['version'] / 'map.json'
                relative = os.path.relpath(target, summary.parent.parent)
                if manifest_record['map']['path'] != relative:
                    manifest_record['map']['path'] = relative
                    write_json(summary, manifest_record)
                    changed[str(summary.relative_to(run))] = artifact_identity.file_hash(summary)
        manifest = run / 'episode.json'
        if manifest.is_file():
            manifest_record = json.loads(manifest.read_text())
            map_changed = False
            if manifest_record.get('map') and manifest_record.get('scene', {}).get('version'):
                map_path = layout.maps(root.parent) / 'versions' / manifest_record['scene']['version'] / 'map.json'
                relative = os.path.relpath(map_path, run)
                map_changed = manifest_record['map']['path'] != relative
                manifest_record['map']['path'] = relative
            if not (changed or previous_hash != current_hash or map_changed):
                continue
            manifest_record['assets'].update(changed)
            if manifest_record.get('inputs', {}).get('episode_manifest_sha256') == previous_hash:
                manifest_record['inputs']['episode_manifest_sha256'] = current_hash
            write_json(manifest, manifest_record)
            review = state.get('review', {})
            if review.get('run') == layout.run_reference(root, run):
                review['result_sha256'] = artifact_identity.file_hash(manifest)
    # 最新失败记录优先保留；仅合并原因和其余产物完全相同的失败现场。
    seen = {}
    current = state.get('attempt', {}).get('run')
    candidates = sorted(runs.iterdir(), key=lambda p: layout.run_reference(root, p) != current) if runs.is_dir() else ()
    for run in candidates:
        error_path = run / 'error.json'
        if not error_path.is_file() or (run / 'episode.json').is_file():
            continue
        error = json.loads(error_path.read_text())
        assets = {str(p.relative_to(run)): artifact_identity.file_hash(p) for p in run.rglob('*')
                  if p.is_file() and p != error_path}
        key = artifact_identity.canonical_hash(dict(status=error.get('status'), reason=error.get('reason'),
                                        input_hash=error.get('input_hash'), assets=assets))
        if key in seen:
            shutil.rmtree(run)
        else:
            seen[key] = run
    if state_path.is_file():
        if not state:
            state_path.unlink()
            return
        attempt = state.get('attempt', {})
        if attempt.get('run') and (root / attempt['run'] / 'error.json').is_file():
            attempt.pop('reason', None)
        attempt.pop('preflight_report', None)
        write_json(state_path, state)


def _split_maps(scene: Path, batches: list[Path]) -> bool:
    """把旧的场景级地图按建图原片和批次引用归还各批次；丢弃可重建旧缓存。"""
    source = scene / '.umi/maps'
    if not source.is_dir():
        return False
    assignments = []
    for version in sorted((source / 'versions').iterdir()) if (source / 'versions').is_dir() else ():
        manifest_path = version / 'manifest.json'
        error_path = version / 'error.json'
        identity_path = manifest_path if manifest_path.is_file() else error_path
        manifest = json.loads(identity_path.read_text()) if identity_path.is_file() else {}
        recordings = manifest.get('inputs', {}).get('recordings', {})
        source_hashes = {item['sha256'] for item in recordings.values() if item.get('sha256')}
        if not manifest_path.is_file():
            # 中断的旧建图没有发布清单，可按提取记录的原片哈希识别所属批次。
            source_hashes.update(json.loads(path.read_text()).get('source', {}).get('sha256')
                                 for path in version.glob('work/*/debug/dataset/extract_manifest.json'))
            source_hashes.discard(None)
        owners = []
        for batch in batches:
            record_path = layout.manifest(batch, 'manifest.yaml')
            record = inputs.read_yaml(record_path) if record_path.is_file() else {}
            belongs = any(Path(item['path']).is_relative_to(batch) for item in recordings.values() if item.get('path'))
            mapping_hashes = {item.get('sha256') for item in inputs.mapping_records(record.get('mapping', {})).values()}
            matches_unpublished_source = not manifest_path.is_file() and bool(source_hashes & mapping_hashes)
            if belongs or matches_unpublished_source or record.get('map', {}).get('version') == version.name:
                owners.append(batch)
        if not owners:
            # 历史任务也可能仍引用已不是批次当前指针的版本。
            for batch in batches:
                histories = (path for pattern in ('.umi/tasks/*/runs/*/episode.json',
                             '*/.umi/runs/*/episode.json', '*/runs/*/episode.json') for path in batch.glob(pattern))
                if any(json.loads(path.read_text()).get('scene', {}).get('version') == version.name
                       for path in histories):
                    owners.append(batch)
        if not owners and not manifest_path.is_file():
            continue  # 无法识别的中断现场保留，不伪装成可用地图。
        if not owners:
            raise ValueError(f'旧地图 {version.name} 无法确定所属批次，已保留，请核对建图原片和批次引用')
        assignments.append((version, owners))
    for version, owners in assignments:
        first = layout.maps(owners[0]) / 'versions' / version.name
        _move(version, first)
        for batch in owners[1:]:
            target = layout.maps(batch) / 'versions' / version.name
            if not target.exists():
                shutil.copytree(first, target, copy_function=os.link, symlinks=True)
        for batch in owners:
            target = layout.maps(batch) / 'versions' / version.name
            _repair_links(target, {version: target})
    for batch in batches:
        versions = layout.maps(batch) / 'versions'
        if not versions.is_dir():
            continue
        record_path = layout.manifest(batch, 'manifest.yaml')
        record = inputs.read_yaml(record_path) if record_path.is_file() else {}
        selected = record.get('map', {}).get('version')
        manifests = sorted(versions.glob('*/manifest.json'))
        if selected and (versions / selected / 'manifest.json').is_file():
            manifest_path = versions / selected / 'manifest.json'
        elif manifests:
            manifest_path = manifests[-1]
        else:
            continue
        write_json(layout.maps(batch) / 'current.json', dict(version=manifest_path.parent.name,
                   manifest_sha256=artifact_identity.file_hash(manifest_path)))
    if (source / 'cache').is_dir():
        shutil.rmtree(source / 'cache')
    (source / 'current.json').unlink(missing_ok=True)
    if (source / 'versions').is_dir() and not any((source / 'versions').iterdir()):
        (source / 'versions').rmdir()
    if not any(source.iterdir()):
        source.rmdir()
    return bool(assignments)


def _migrate_scene(scene: Path) -> None:
    """重复调用安全；只整理已识别的程序文件，不扫描或改写原片。"""
    scene = scene.expanduser().resolve()
    _relocate(scene, ('maps',))
    scene_file = scene / 'scene.yaml'
    if scene_file.is_file() and inputs.read_yaml(scene_file) == dict(schema_version=2):
        scene_file.unlink()
    batches = [batch for batch in sorted(scene.iterdir()) if batch.is_dir() and not batch.name.startswith('.')
               and batch.name not in inputs.INTERNAL_DIRECTORIES and layout.is_batch(batch)]
    maps_moved = _split_maps(scene, batches)
    for batch in batches:
        _relocate(batch, ('manifest.yaml', 'snapshots', 'import.json', 'imports'))
        imports = layout.metadata(batch, 'imports')
        if imports.is_dir():
            reports = sorted(imports.glob('*.json'), key=lambda p: p.stat().st_mtime_ns)
            values = [json.loads(p.read_text()) for p in reports]
            meaningful = [manifest_record for manifest_record in values if any(manifest_record.get(k) for k in ('episodes', 'pending', 'ignored', 'errors'))]
            if meaningful:
                write_json(layout.metadata(batch, 'import.json'), meaningful[-1])
            shutil.rmtree(imports)
        report = layout.metadata(batch, 'import.json')
        if report.is_file():
            manifest_record = json.loads(report.read_text())
            before = json.dumps(manifest_record, sort_keys=True)
            manifest_record['session'] = str(layout.manifest(batch, 'manifest.yaml'))
            for item in manifest_record.get('episodes', []):
                if item.get('path'):
                    item['path'] = str(layout.metadata(layout.owner(Path(item['path'])), 'episode.yaml'))
            if json.dumps(manifest_record, sort_keys=True) != before:
                write_json(report, manifest_record)
        for task in inputs.batch_task_directories(batch):
            record = layout.manifest(task, 'episode.yaml')
            legacy = any((task / prefix / name).exists() for prefix in ('', '.umi')
                         for name in ('episode.yaml', 'state.json', 'runs', 'cache'))
            if record.is_file() and (legacy or maps_moved or inputs.read_yaml(record).keys() & layout.TASK_FIELDS):
                _task(task)


def _needs_migration(scene: Path) -> bool:
    if any((scene / path).exists() for path in ('maps', '.umi', '.scene.lock')):
        return True
    scene_path = scene / 'scene.yaml'
    if scene_path.is_file() and inputs.read_yaml(scene_path) == dict(schema_version=2):
        return True
    for batch in scene.iterdir():
        if not batch.is_dir() or batch.name.startswith('.') or not layout.is_batch(batch):
            continue
        if any((batch / name).exists() for name in ('manifest.yaml', 'snapshots', 'import.json', 'imports', '.umi/imports', '.scene.lock')):
            return True
        for task in inputs.batch_task_directories(batch):
            if any((task / name).exists() for name in ('episode.yaml', 'state.json', 'runs', 'cache', '.umi', '.scene.lock')):
                return True
            record = layout.metadata(task, 'episode.yaml')
            if record.is_file() and inputs.read_yaml(record).keys() & layout.TASK_FIELDS:
                return True
    return False


def _repair_links(root: Path, relocations: dict[Path, Path]) -> None:
    for link in root.rglob('*'):
        if not link.is_symlink():
            continue
        previous = Path(os.path.abspath(link.parent / link.readlink()))
        for source, target in relocations.items():
            if previous.is_relative_to(source) and source != target:
                publish_link(target / previous.relative_to(source), link)
                break


def migrate_scene(scene: Path) -> None:
    """迁移前锁定场景、批次与任务；活跃处理中的目录不会被移动。"""
    scene = scene.expanduser().resolve()
    if not _needs_migration(scene):
        return
    scopes = [scene]
    for batch in sorted(scene.iterdir()):
        if (not batch.is_dir() or batch.name.startswith('.') or batch.name in inputs.INTERNAL_DIRECTORIES
                or not ((batch / 'batch.yaml').is_file() or layout.manifest(batch, 'manifest.yaml').is_file())):
            continue
        scopes.append(batch)
        scopes.extend(task for task in inputs.batch_task_directories(batch)
                      if layout.manifest(task, 'episode.yaml').is_file())
    obsolete_locks = []
    relocations = {root / prefix / name: layout.metadata(root, name)
                   for root in scopes for prefix in ('', '.umi') for name in ('runs', 'cache')
                   if root != scene}
    with ExitStack() as locks:
        for root in scopes:
            if root != scene:
                locks.enter_context(layout.locked(root))
            for legacy in (root / '.scene.lock', root / '.umi/.scene.lock'):
                if (root != scene and legacy == layout.metadata(root, '.scene.lock')) or not legacy.is_file():
                    continue
                stream = locks.enter_context(legacy.open('a'))
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError(f'目录仍在处理，请完成后重试整理：{root}') from error
                obsolete_locks.append(legacy)
        _migrate_scene(scene)
        for batch in (root for root in scopes if layout.is_batch(root)):
            _repair_links(batch / '.umi', relocations)
    for lock in obsolete_locks:
        lock.unlink(missing_ok=True)
    for root in reversed(scopes):
        if not layout.is_batch(root) and (root / '.umi').is_dir() and not any((root / '.umi').iterdir()):
            (root / '.umi').rmdir()
