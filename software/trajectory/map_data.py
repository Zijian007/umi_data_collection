"""ORB 图集 → 轻量地图展示数据；相似变换仅供展示，不改写轨迹或原始地图。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

from software.trajectory.results import load_summary, result_path

ROOT = Path(__file__).resolve().parents[2]
MAX_POINTS = 50000  # 每张子地图的浏览器绘制预算；完整点云仍在 .osa 中。


class MapDataError(ValueError):
    pass


def attempt_directory(output: Path, summary: dict) -> Path | None:
    """只选摘要采用的轮次，兼容旧布局；不会把其他轮次的地图混进结果。"""
    chosen = summary.get('chosen_try', 1)
    if type(chosen) is not int or chosen < 1:
        raise MapDataError('摘要中的重建轮次无效')
    old = 'kiwi' if chosen == 1 else f'kiwi_try{chosen}'
    for path in (output / f'debug/reconstruction/try_{chosen}', output / f'diagnostics/work/{old}', output / f'work/{old}'):
        if path.is_dir():
            require_local(path, output)
            return path
    return None


def require_local(path: Path, output: Path) -> None:
    if not path.resolve().is_relative_to(output.resolve()):
        raise MapDataError('地图数据路径超出当前结果目录')


def exporter_paths() -> tuple[Path, Path]:
    deps = Path(os.environ.get('UMI_DEPS') or ROOT / '.deps')
    orb = Path(os.environ.get('ORB3_DIR') or deps / 'src/ORB_SLAM3')
    return orb / 'Examples/Monocular-Inertial/umi_export_atlas', Path(os.environ.get('ORB3_VOCAB') or orb / 'Vocabulary/ORBvoc.txt')


def fit_similarity(source, target) -> dict:
    """稳健拟合 p_world = scale·R·p_map + t；退化/残差过大时拒绝叠加。"""
    import numpy as np

    x, y = np.asarray(source, float), np.asarray(target, float)
    if len(x) < 6 or x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        return dict(status='unaligned', reason='有效定位对应点不足 6 个')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return dict(status='unaligned', reason='定位对应点包含无效数值')
    keep = np.ones(len(x), dtype=bool)
    for iteration in range(5):
        a, b = x[keep], y[keep]
        xc, yc = a.mean(axis=0), b.mean(axis=0)
        a, b = a-xc, b-yc
        u, singular, vt = np.linalg.svd(b.T @ a / len(a))
        if singular[0] < 1e-12 or singular[1] < singular[0]*1e-5:
            return dict(status='unaligned', reason='运动接近静止或直线，无法可靠确定地图朝向')
        sign = np.diag([1., 1., np.linalg.det(u @ vt)])
        rotation = u @ sign @ vt
        scale = float(np.sum(singular*np.diag(sign)) / np.mean(np.sum(a*a, axis=1)))
        translation = yc - scale*rotation @ xc
        errors = np.linalg.norm(scale*x @ rotation.T + translation-y, axis=1)
        median = np.median(errors[keep])
        cutoff = max(.01, median + 3*1.4826*np.median(np.abs(errors[keep]-median)))
        updated = errors <= cutoff
        # 最后一轮只拟合、不再改变样本，使返回的变换与残差使用同一组点。
        if iteration == 4 or updated.sum() < 6 or np.array_equal(updated, keep):
            break
        keep = updated
    rmse = float(np.sqrt(np.mean(errors[keep]**2)))
    extent = float(np.linalg.norm(np.ptp(y[keep], axis=0)))
    if not scale > 0 or not np.isfinite(scale) or rmse > max(.05, .1*extent):
        return dict(status='unaligned', reason='单个相似变换无法可靠解释本地图与融合轨迹的差异', rmse_m=rmse)
    return dict(status='aligned', scale=scale, rotation=rotation.tolist(), translation=translation.tolist(),
                pairs=int(keep.sum()), total_pairs=len(x), rmse_m=rmse)


def _correspondences(csv_path: Path, camera_path: Path, index: int):
    import numpy as np

    trajectory = np.atleast_2d(np.loadtxt(camera_path, comments='#'))
    if trajectory.shape[1] != 8 or not np.isfinite(trajectory).all() or np.any(np.diff(trajectory[:, 0]) <= 0):
        raise MapDataError('相机轨迹格式或时间轴无效')
    source, target = [], []
    with csv_path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('ok') != '1' or row.get('map', f'm{index}') != f'm{index}':
                continue
            time = float(row['t'])
            if not np.isfinite(time):
                continue
            k = int(np.searchsorted(trajectory[:, 0], time))
            k = min((max(0, k-1), min(k, len(trajectory)-1)), key=lambda j: abs(trajectory[j, 0]-time))
            if abs(trajectory[k, 0]-time) > 1e-4:
                continue
            point = [float(row[key]) for key in ('tx', 'ty', 'tz')]
            if np.isfinite(point).all():
                source.append(point)
                target.append(trajectory[k, 1:4])
    return source, target


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


def export_attempt(attempt: Path, output: Path) -> dict:
    """仅导出前镜头图集；原片、地图、相机轨迹和设备档案均只读。"""
    archive = attempt / 'front/map/map.osa'
    require_local(archive, output)
    executable, vocabulary = exporter_paths()
    for path in (archive, executable, vocabulary):
        if not path.is_file():
            raise MapDataError(f'地图读取缺少 {path.name}；导出器可通过 install.sh 安装')
    with tempfile.TemporaryDirectory(prefix='umi-map-export-') as temp:
        raw_path = Path(temp) / 'atlas.json'
        try:
            result = subprocess.run([str(executable), str(archive), str(vocabulary), str(raw_path)],
                                    capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired as error:
            raise MapDataError('地图读取超时') from error
        if result.returncode:
            raise MapDataError('地图读取失败：'+(result.stderr or result.stdout)[-500:])
        raw = json.loads(raw_path.read_text())
    camera = result_path(output, 'traj_camera.txt')
    require_local(camera, output)
    maps = []
    for item in raw['maps']:
        index = item['index']
        locations = attempt / f'front/loc_m{index}/frames.csv'
        if not locations.is_file():
            locations = output / 'debug/logs/front_loc_frames.csv'
        require_local(locations, output)
        alignment = dict(status='unaligned', reason='缺少逐图定位或相机轨迹，使用原始地图坐标')
        if locations.is_file() and camera.is_file():
            try:
                alignment = fit_similarity(*_correspondences(locations, camera, index))
            except (ValueError, KeyError, OSError) as error:
                alignment = dict(status='unaligned', reason=f'地图对齐失败：{error}')
        maps.append(prepare_display_map(item, alignment))
    return dict(kind='umi_map_view', schema_version=1, maps=maps,
                source=dict(archive=str(archive.relative_to(output)), bytes=archive.stat().st_size),
                note='前镜头稀疏地图；相似变换拟合仅供展示，拟合残差不代表真实轨迹精度。')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output = args.result.resolve()
        attempt = attempt_directory(output, load_summary(output))
        if attempt is None:
            raise MapDataError('未保留原始地图；新的轨迹抽取会自动保留轻量地图展示数据')
        data = export_attempt(attempt, output)
        args.output.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    except (OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
