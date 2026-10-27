"""写出可独立读取的训练中间包；失败不发布部分数据集。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from software import config
from software.storage import locked, write_json
from software.dataset import gripper, sampling, source

FORMAT = 'umi_training_dataset'
SCHEMA_VERSION = 2


def build(paths: list[Path], output: Path, *, fps: float = 30., max_gap_s: float = .25) -> dict:
    if not math.isfinite(fps) or not 0 < fps <= 120:
        raise ValueError('fps 必须为 (0, 120] 内的有限数值')
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError('max_gap_s 必须为有限正秒数')
    selected = source.select(paths)
    output = output.expanduser().resolve()
    if any(output.is_relative_to(item.scene_root) or item.scene_root.is_relative_to(output) for item in selected):
        raise ValueError('数据集输出目录不能位于源场景内或覆盖源场景')
    output.parent.mkdir(parents=True, exist_ok=True)
    with locked(output.parent, name=f'.{output.name}.dataset.lock'):
        if output.exists():
            raise ValueError(f'导出目录已存在，拒绝覆盖：{output}')
        staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
        try:
            entries = []
            for item in selected:
                target = staging / 'episodes' / item.name
                target.mkdir(parents=True)
                offsets = item.manifest['synchronization']['offsets_s']
                arms = {arm: sampling.read_arm(item.run / arm, offsets[arm]) for arm in item.manifest['arms']}
                arrays, quality = sampling.sample(arms, fps, max_gap_s)
                width_arrays, readiness = gripper.sample(item.run, item.manifest, arms, arrays['t_episode_s'], max_gap_s)
                arrays.update(width_arrays)
                np.savez_compressed(target / 'samples.npz', **arrays)
                tcp_geometry = {}
                for arm in arms:
                    settings = json.loads((item.run / arm / 'details/run_settings.json').read_text())
                    tcp_geometry[arm] = dict(T_camera_tcp=settings['T_camera_tcp'], cad_status=settings['cad_status'])
                    video = target / 'videos' / f'{arm}.mp4'
                    video.parent.mkdir(exist_ok=True)
                    shutil.copy2(item.run / arm / 'front.mp4', video)
                    if config.file_hash(video) != item.manifest['assets'][f'{arm}/front.mp4']:
                        raise ValueError('导出期间源视频发生变化')
                metadata = dict(kind='umi_training_episode', schema_version=SCHEMA_VERSION,
                    instruction=item.manifest['instruction'], synchronization=item.manifest['synchronization'],
                    frame_id=item.manifest['frame_id'], scene=item.manifest['scene'], kit=item.manifest['kit'],
                    arms=list(arms), tcp_frames=item.manifest['tcp_frames'],
                    tcp_geometry=tcp_geometry, actual_error=item.manifest['actual_error'],
                    pose_representation='T_scene_tcp', position_unit='m', fps=fps,
                    rows=len(arrays['t_episode_s']), quality=quality,
                    training_ready=readiness['training_ready'], missing_channels=readiness['missing_channels'],
                    not_ready_reasons=readiness['not_ready_reasons'], gripper_width=readiness['measurements'],
                    review=item.review, source=dict(episode_id=str(item.name), run_id=item.run.name,
                        result_sha256=item.result_sha256, inputs=item.manifest['inputs']),
                    arrays={name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in arrays.items()},
                    videos={arm: f'videos/{arm}.mp4' for arm in arms})
                write_json(target / 'episode.json', metadata)
                hashes = {str(file_path.relative_to(target)): config.file_hash(file_path) for file_path in sorted(target.rglob('*')) if file_path.is_file()}
                entries.append(dict(path=str(target.relative_to(staging)), rows=metadata['rows'], files=hashes,
                                    training_ready=metadata['training_ready'], missing_channels=metadata['missing_channels'],
                                    not_ready_reasons=metadata['not_ready_reasons']))
            for item in selected:
                source.verify_current(item)
            missing_channels = sorted({name for entry in entries for name in entry['missing_channels']})
            ready = all(entry['training_ready'] for entry in entries)
            manifest = dict(kind=FORMAT, schema_version=SCHEMA_VERSION, fps=fps, max_gap_s=max_gap_s,
                training_ready=ready, missing_channels=missing_channels, episodes=entries)
            write_json(staging / 'export_report.json', dict(episodes=len(entries), rows=sum(entry['rows'] for entry in entries),
                training_ready=ready, missing_channels=missing_channels,
                not_ready=[dict(path=entry['path'], reasons=entry['not_ready_reasons']) for entry in entries if not entry['training_ready']],
                note='training_ready 仅表示基础观测通道齐全、宽度已核验且采样有效；模型动作切片与部署需另行适配。'))
            write_json(staging / 'manifest.json', manifest)
            if output.exists():
                raise ValueError('导出期间目标目录已被创建，拒绝覆盖')
            staging.rename(output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return dict(output=str(output), episodes=len(entries), rows=sum(e['rows'] for e in entries),
                training_ready=ready, missing_channels=missing_channels)


def load_episode(dataset: Path, episode_path: str, *, require_training_ready=False) -> tuple[dict, dict[str, np.ndarray]]:
    """仅使用包内文件读取已登记任务并验证哈希，无原片/注册表依赖。"""
    dataset = dataset.resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest.get('kind') != FORMAT or manifest.get('schema_version') not in (1, SCHEMA_VERSION):
        raise ValueError('不支持的数据集格式版本')
    matches = [entry for entry in manifest['episodes'] if entry['path'] == episode_path]
    if len(matches) != 1:
        raise ValueError('任务未唯一登记在数据集清单中')
    entry = matches[0]
    directory = (dataset / episode_path).resolve()
    if not directory.is_relative_to(dataset):
        raise ValueError('数据集任务路径越界')
    for name, digest in entry['files'].items():
        file = (directory / name).resolve()
        if not file.is_relative_to(directory) or not file.is_file() or config.file_hash(file) != digest:
            raise ValueError(f'数据集文件缺失、越界或哈希变化：{name}')
    metadata = json.loads((directory / 'episode.json').read_text())
    if require_training_ready and not metadata['training_ready']:
        raise ValueError('任务尚不满足训练数据条件：'+', '.join(metadata.get('not_ready_reasons', metadata['missing_channels'])))
    with np.load(directory / 'samples.npz', allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    if set(arrays) != set(metadata['arrays']):
        raise ValueError('样本字段与任务清单不一致')
    for name, value in arrays.items():
        spec = metadata['arrays'][name]
        finite = np.isfinite(value).all()
        for suffix in ('_gripper_width_m', '_gripper_close_01'):
            if name.endswith(suffix):
                valid = arrays[name[:-len(suffix)]+'_gripper_valid']
                finite = (valid.dtype == np.dtype(bool) and valid.shape == value.shape
                          and np.isfinite(value[valid]).all() and np.isnan(value[~valid]).all())
        if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype'] or not finite:
            raise ValueError(f'样本形状、类型或数值无效：{name}')
    return metadata, arrays
