"""任务状态、人工验收与合格数据导出；state.json 持久保存处理状态和人工验收记录。"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

from software import config
from software.scene import inputs
from software.storage import locked, write_json


def episode_paths(paths: list[Path]) -> list[Path]:
    """只发现任务清单，跳过缓存、运行产物和归档；不启动处理。"""
    found = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_file():
            if path.name not in ('scene.yaml', 'session.yaml', 'episode.yaml'):
                raise ValueError(f'请指定场景、套件、批次、任务目录或对应清单：{path}')
            if path.name == 'episode.yaml':
                found.append(path)
                continue
            path = path.parent
        if not path.is_dir():
            raise ValueError(f'任务路径不存在：{path}')
        for current, children, files in os.walk(path):
            children[:] = sorted(name for name in children if not name.startswith('.')
                                 and name not in inputs.INTERNAL_DIRECTORIES)
            if 'episode.yaml' in files:
                found.append(Path(current) / 'episode.yaml')
                children[:] = []
    return list(dict.fromkeys(found))


def read_state(directory: Path) -> dict:
    path = directory / 'state.json'
    return json.loads(path.read_text()) if path.exists() else {}


def record_capture(path: Path) -> None:
    """清单变化使处理和验收失效；身份和使用手始终从清单读取。"""
    inputs.read_episode(path, verify=False)
    write_json(path.parent / 'state.json', {})


def record_attempt(episode, status: str, target: Path | None, *, reason='') -> None:
    state = read_state(episode.path.parent)
    previous = state.get('attempt', {})
    manifest_sha256 = config.file_hash(episode.path)
    session_sha256 = config.file_hash(episode.session.path)
    if previous.get('manifest_sha256') != manifest_sha256 or previous.get('session_sha256') != session_sha256:
        state = {}
    report = target / 'preflight_report.json' if target else None
    state['attempt'] = dict(status=status, run=str(target.relative_to(episode.path.parent)) if target else None,
        reason=reason, manifest_sha256=manifest_sha256, session_sha256=session_sha256,
        preflight_report=str(report.relative_to(episode.path.parent)) if report and report.is_file() else None)
    if status not in ('ok', 'skipped') or state.get('review', {}).get('run') != state['attempt']['run']:
        state.pop('review', None)
    write_json(episode.path.parent / 'state.json', state)


def task_status(path: Path) -> dict:
    episode = inputs.read_episode(path, verify=False)
    state = read_state(episode.path.parent)
    attempt = state.get('attempt', {})
    status = 'pending'
    if set(episode.expected_arms)-episode.recordings.keys():
        status = 'incomplete'
    elif (attempt.get('manifest_sha256') == config.file_hash(episode.path)
          and attempt.get('session_sha256') == config.file_hash(episode.session.path)):
        status = attempt.get('status', 'pending')
    review = state.get('review', {}) if status in ('ok', 'skipped') else {}
    return dict(scene=episode.scene.name, session=episode.session.name, name=episode.path.parent.name,
        path=str(episode.path), kit=episode.session.kit['id'],
        arms=list(episode.expected_arms), status=status, review=review.get('decision', 'unreviewed'),
        reason=attempt.get('reason', ''), output=str(episode.output) if episode.output.exists() else None)


def successful_run(episode) -> tuple[Path, dict]:
    """成功入口、最新尝试、原片和资产必须相符，避免导出旧的已失效结果。"""
    status = task_status(episode.path)
    if status['status'] not in ('ok', 'skipped'):
        raise ValueError(f'任务当前尚无可验收的成功结果：{status["status"]}')
    run_dir = episode.output.resolve()
    state = read_state(episode.path.parent)
    if (episode.path.parent / state['attempt']['run']).resolve() != run_dir:
        raise ValueError('最近成功尝试与 results 入口不一致')
    run_manifest = json.loads((run_dir / 'episode.json').read_text())
    if run_manifest['status'] != 'ok':
        raise ValueError('任务结果未完成')
    inputs.validate_assets(run_dir, run_manifest['assets'])
    cameras = inputs.select_cameras(episode.recordings, replace(episode.scene, mode=run_manifest['inputs']['map_mode'],
        devices=episode.session.devices, rear_calibrations=inputs.rear_files(episode.session.devices)), episode.serials)
    map_manifest = dict(version=run_manifest['scene']['version'], frame_id=run_manifest['frame_id'], assets=run_manifest['inputs']['scene_assets'], mode=run_manifest['inputs']['map_mode'])
    if inputs.episode_inputs(episode, cameras, map_manifest) != run_manifest['inputs']:
        raise ValueError('输入或算法配置已变化，请重新处理后验收')
    return run_dir, run_manifest


def review(path: Path, *, accept: bool, reviewer: str, reason='') -> dict:
    episode = inputs.read_episode(path)
    with locked(episode.path.parent):
        run_dir, _ = successful_run(episode)
        state = read_state(episode.path.parent)
        decision = dict(decision='accepted' if accept else 'rejected', reviewer=inputs.validate_name(reviewer), reason=reason,
            run=str(run_dir.relative_to(episode.path.parent)), result_sha256=config.file_hash(run_dir / 'episode.json'),
            checked_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
        state['review'] = decision
        write_json(episode.path.parent / 'state.json', state)
    return decision


def accepted_run(episode) -> tuple[Path, dict, dict]:
    """成果打包和训练导出共用的只读入口，验证当前结果及其人工验收。"""
    run_dir, manifest = successful_run(episode)
    record = read_state(episode.path.parent).get('review', {})
    if record.get('decision') != 'accepted' or record.get('result_sha256') != config.file_hash(run_dir / 'episode.json'):
        raise ValueError(f'任务尚未人工接收或验收对应的结果已变化：{episode.path}')
    return run_dir, manifest, record


def export(paths: list[Path], export_dir: Path, *, with_video=False) -> dict:
    export_dir = export_dir.expanduser().resolve()
    if export_dir.exists():
        raise ValueError(f'导出目录已存在，拒绝覆盖：{export_dir}')
    accepted_episodes = []
    for path in paths:
        episode = inputs.read_episode(path)
        run_dir, run_manifest, review_record = accepted_run(episode)
        accepted_episodes.append((episode, run_dir, run_manifest, review_record))
    if not accepted_episodes:
        raise ValueError('没有可导出的任务')
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix='.export-', dir=export_dir.parent))
    try:
        exported_episodes = []
        for episode, run_dir, run_manifest, review_record in accepted_episodes:
            episode_relative_path = Path(episode.scene.name) / episode.session.kit['id'] / episode.session.name / episode.path.parent.name
            episode_dir = staging_dir / episode_relative_path
            episode_dir.mkdir(parents=True)
            files = ['report.txt', 'episode.json']
            if len(run_manifest['arms']) == 2:
                files.append('bimanual.csv')
            files += [f'{arm}/traj_tcp.txt' for arm in run_manifest['arms']]
            files += [f'{arm}/{name}' for arm in run_manifest.get('gripper_width', {})
                      for name in ('gripper_width.csv', 'details/gripper_width.json', 'details/gripper_width_frames.csv')]
            if with_video:
                files += [f'{arm}/front.mp4' for arm in run_manifest['arms']]
            exported_file_hashes = {}
            for name in files:
                export_path = episode_dir / name
                export_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(run_dir / name, export_path)
                exported_file_hashes[name] = config.file_hash(export_path)
                if exported_file_hashes[name] != config.file_hash(run_dir / name):
                    raise ValueError('导出期间源结果发生变化')
            exported_episodes.append(dict(path=str(episode_relative_path), scene=run_manifest['scene'], frame_id=run_manifest['frame_id'],
                kit=episode.session.kit, arms=run_manifest['arms'], instruction=episode.instruction,
                synchronization=episode.synchronization, review=review_record, files=exported_file_hashes))
        manifest = dict(kind='umi_dataset', schema_version=1, episodes=exported_episodes,
            frames=sorted({entry['frame_id'] for entry in exported_episodes}))
        write_json(staging_dir / 'manifest.json', manifest)
        staging_dir.rename(export_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return dict(output=str(export_dir), episodes=len(exported_episodes), frames=manifest['frames'])
