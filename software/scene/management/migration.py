"""旧目录的一次性原位整理；原片、标定和地图清单内容保持原值。"""
from __future__ import annotations

from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil

from software import config
from software.scene import layout
from software.storage import publish_link, write_json
from . import inputs


def _move(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        source.rename(target)
    elif source.is_dir() and target.is_dir():
        for child in source.iterdir():
            _move(child, target / child.name)
        source.rmdir()
    elif source.is_file() and target.is_file() and config.file_hash(source) == config.file_hash(target):
        source.unlink()
    else:
        raise ValueError(f'内部目录迁移冲突，已保留原文件：{source} → {target}')


def _relocate(root: Path, names: tuple[str, ...]) -> None:
    for name in names:
        _move(root / name, layout.metadata(root, name))
    (root / '.scene.lock').unlink(missing_ok=True)


def _run_reference(value: str) -> str:
    return str(Path('.umi') / value) if value.startswith('runs/') else value


def _task(root: Path, scene: Path) -> None:
    result = root / 'results'
    previous_target = Path(os.path.abspath(root / result.readlink())) if result.is_symlink() else None
    _relocate(root, ('episode.yaml', 'state.json', 'cache', 'runs'))
    if previous_target and previous_target.is_relative_to(root / 'runs'):
        publish_link(layout.metadata(root, 'runs') / previous_target.relative_to(root / 'runs'), result)
    runs = layout.metadata(root, 'runs')
    state_path = layout.metadata(root, 'state.json')
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    for record in (state.get('attempt', {}), state.get('review', {})):
        if record.get('run'):
            record['run'] = _run_reference(record['run'])
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
            data = json.loads(summary.read_text())
            if 'map' in data and data.get('scene', {}).get('version'):
                target = layout.maps(scene) / 'versions' / data['scene']['version'] / 'map.json'
                relative = os.path.relpath(target, summary.parent.parent)
                if data['map']['path'] != relative:
                    data['map']['path'] = relative
                    write_json(summary, data)
                    changed[str(summary.relative_to(run))] = config.file_hash(summary)
        manifest = run / 'episode.json'
        if changed and manifest.is_file():
            data = json.loads(manifest.read_text())
            data['assets'].update(changed)
            write_json(manifest, data)
            review = state.get('review', {})
            if review.get('run') == str(run.relative_to(root)):
                review['result_sha256'] = config.file_hash(manifest)
    # 最新失败记录优先保留；仅合并原因和其余产物完全相同的失败现场。
    seen = {}
    current = state.get('attempt', {}).get('run')
    candidates = sorted(runs.iterdir(), key=lambda p: str(p.relative_to(root)) != current) if runs.is_dir() else ()
    for run in candidates:
        error_path = run / 'error.json'
        if not error_path.is_file() or (run / 'episode.json').is_file():
            continue
        error = json.loads(error_path.read_text())
        assets = {str(p.relative_to(run)): config.file_hash(p) for p in run.rglob('*')
                  if p.is_file() and p != error_path}
        key = config.canonical_hash(dict(status=error.get('status'), reason=error.get('reason'),
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


def _migrate_scene(scene: Path) -> None:
    """重复调用安全；只整理已识别的程序文件，不扫描或改写原片。"""
    scene = scene.expanduser().resolve()
    _relocate(scene, ('maps',))
    scene_file = scene / 'scene.yaml'
    if scene_file.is_file() and inputs.read_yaml(scene_file) == dict(schema_version=2):
        scene_file.unlink()
    for batch in sorted(scene.iterdir()):
        if not batch.is_dir() or batch.name.startswith('.') or batch.name in inputs.INTERNAL_DIRECTORIES:
            continue
        if not (batch / 'batch.yaml').is_file() and not layout.manifest(batch, 'manifest.yaml').is_file():
            continue
        _relocate(batch, ('manifest.yaml', 'imports'))
        imports = layout.metadata(batch, 'imports')
        if imports.is_dir():
            reports = sorted(imports.glob('*.json'), key=lambda p: p.stat().st_mtime_ns)
            values = [json.loads(p.read_text()) for p in reports]
            meaningful = [data for data in values if any(data.get(k) for k in ('episodes', 'pending', 'ignored', 'errors'))]
            if meaningful:
                write_json(layout.metadata(batch, 'import.json'), meaningful[-1])
            shutil.rmtree(imports)
        report = layout.metadata(batch, 'import.json')
        if report.is_file():
            data = json.loads(report.read_text())
            before = json.dumps(data, sort_keys=True)
            data['session'] = str(layout.manifest(batch, 'manifest.yaml'))
            for item in data.get('episodes', []):
                if item.get('path'):
                    item['path'] = str(layout.metadata(layout.owner(Path(item['path'])), 'episode.yaml'))
            if json.dumps(data, sort_keys=True) != before:
                write_json(report, data)
        for task in sorted(batch.iterdir()):
            if task.is_dir() and not task.name.startswith('.') and layout.manifest(task, 'episode.yaml').is_file():
                _task(task, scene)


def migrate_scene(scene: Path) -> None:
    """迁移前锁定场景、批次与任务；活跃处理中的目录不会被移动。"""
    scene = scene.expanduser().resolve()
    scopes = [scene]
    for batch in sorted(scene.iterdir()):
        if (not batch.is_dir() or batch.name.startswith('.') or batch.name in inputs.INTERNAL_DIRECTORIES
                or not ((batch / 'batch.yaml').is_file() or layout.manifest(batch, 'manifest.yaml').is_file())):
            continue
        scopes.append(batch)
        scopes.extend(task for task in sorted(batch.iterdir()) if task.is_dir() and not task.name.startswith('.')
                      and layout.manifest(task, 'episode.yaml').is_file())
    with ExitStack() as locks:
        for root in scopes:
            locks.enter_context(layout.locked(root))
            legacy = root / '.scene.lock'
            if legacy.is_file():
                stream = locks.enter_context(legacy.open('a'))
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError(f'目录仍在处理，请完成后重试整理：{root}') from error
        _migrate_scene(scene)
