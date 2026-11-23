"""Read a portable training package using only NumPy and standard-library dependencies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from umi.infra import provenance as artifact_identity
from umi.dataset.schema import read_pose_series
from umi.dataset.schema import ARMS, EPISODE_FORMAT, FORMAT, READABLE_VERSIONS, validate_timeline


def load_episode(dataset: Path, episode_path: str, *, require_training_ready=False) -> tuple[dict, dict[str, np.ndarray]]:
    """仅使用包内文件读取已登记任务并验证哈希，无原片/注册表依赖。"""
    dataset = dataset.resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest.get('kind') != FORMAT or manifest.get('schema_version') not in READABLE_VERSIONS:
        raise ValueError('不支持的数据集格式版本')
    matches = [entry for entry in manifest['episodes'] if entry['path'] == episode_path]
    if len(matches) != 1:
        raise ValueError('任务未唯一登记在数据集清单中')
    entry = matches[0]
    if not {'episode.json', 'samples.npz'} <= set(entry['files']):
        raise ValueError('数据集清单缺少元数据或样本文件的哈希')
    directory = (dataset / episode_path).resolve()
    if not directory.is_relative_to(dataset):
        raise ValueError('数据集任务路径越界')
    for name, digest in entry['files'].items():
        file = (directory / name).resolve()
        if not file.is_relative_to(directory) or not file.is_file() or artifact_identity.file_hash(file) != digest:
            raise ValueError(f'数据集文件缺失、越界或哈希变化：{name}')
    metadata = json.loads((directory / 'episode.json').read_text())
    if metadata.get('kind') != EPISODE_FORMAT or metadata.get('schema_version') not in READABLE_VERSIONS:
        raise ValueError('不支持的训练任务格式版本')
    arms = metadata.get('arms')
    if not isinstance(arms, list) or not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS):
        raise ValueError('训练任务使用手必须为不重复的 left/right 列表')
    if set(metadata.get('videos', {})) != set(arms):
        raise ValueError('训练任务视频必须与使用手一致')
    if not set(metadata['videos'].values()) <= set(entry['files']):
        raise ValueError('数据集清单缺少视频文件的哈希')
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
    for arm in metadata['arms']:
        read_pose_series(metadata, arrays, arm)
    validate_timeline(arrays, metadata['rows'])
    return metadata, arrays
