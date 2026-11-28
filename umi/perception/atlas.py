"""Convert a raw ORB atlas and prepare display geometry in declared coordinates."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from umi.config import operational_settings as settings
from umi.infra.orb_install import exporter_paths
from umi.perception import runtime

MAX_POINTS = 50000


class MapDataError(ValueError):
    pass


def read_atlas(archive: Path, output: Path, *, log: Path | None = None, environment=None) -> dict:
    """Convert an atlas read-only, optionally leaving a durable engine log."""
    environment = dict(os.environ if environment is None else environment)
    executable, vocabulary = exporter_paths(environment)
    timeout = settings.values('runtime')['map_export_timeout_s']
    command = [str(executable), str(archive), str(vocabulary), str(output)]
    if log is not None:
        if not executable.is_file():
            raise MapDataError('缺少 umi_export_atlas，请运行 install.sh 更新轨迹组件')
        runtime.run_logged(['timeout', str(timeout), *command], log,
                           stage='转换地图图集', environment=environment)
    else:
        for path in (archive, executable, vocabulary):
            if not path.is_file():
                raise MapDataError(f'地图读取缺少 {path.name}；图集转换工具可通过 install.sh 安装')
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=environment)
        except subprocess.TimeoutExpired as error:
            raise MapDataError('地图读取超时') from error
        if result.returncode:
            raise MapDataError('地图读取失败：' + (result.stderr or result.stdout)[-500:])
    return json.loads(output.read_text())


def prepare_display_map(raw: dict, alignment: dict) -> dict:
    import numpy as np
    from scipy.spatial.transform import Rotation

    points = np.asarray(raw['points'], float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    count = len(points)
    points = points[::max(1, (count+MAX_POINTS-1)//MAX_POINTS)]
    keyframes = sorted(raw['keyframes'], key=lambda frame: frame['time'])
    if alignment['status'] == 'aligned':
        rotation = np.asarray(alignment['rotation'])
        translation = np.asarray(alignment['translation'])
        scale = alignment['scale']
        points = scale*points @ rotation.T+translation
        keyframes = [dict(frame,
            position=(scale*rotation @ frame['position']+translation).tolist(),
            quaternion=(Rotation.from_matrix(rotation)*Rotation.from_quat(frame['quaternion'])).as_quat().tolist())
            for frame in keyframes]
    bounds = np.quantile(points, [.02, .98], axis=0).tolist() if len(points) else None
    return dict(index=raw['index'], id=raw['id'], lens='front', points=points.round(7).tolist(),
                point_count=count, keyframes=keyframes, bounds=bounds, alignment=alignment,
                coordinates='trajectory' if alignment['status'] == 'aligned' else 'map')
