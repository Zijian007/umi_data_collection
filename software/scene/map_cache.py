"""建图阶段的内容身份、可校验缓存和计时；调用方持有场景写入锁。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator
import json
import shutil
import time
import uuid

from software import provenance as artifact_identity
from software.console import Console
from software.scene.models import CalibratedCamera, PreparedRecording, Scene
from software.storage import write_json


console = Console('场景')

# 显式列出各阶段的实现依赖。CLI、查看器和任务编排不改变地图。
# 新增算法依赖时应加入对应组；不以 Git 提交或文件时间代替内容哈希。
COMMON_SOURCES = (
    'software/config.py', 'software/parameters.py', 'software/provenance.py',
    'software/environment.py', 'software/storage.py', 'software/scene/models.py',
    'software/scene/map_cache.py', 'software/vision/models.py',
)
PREPARATION_SOURCES = (
    'software/registry.py', 'software/camera/*.py', 'software/camera/x3/*.py',
    'software/scene/pipeline/recordings.py', 'software/scene/management/camera_assets.py',
    'software/calibration/algorithms/factory_kb4.py',
    'software/trajectory/calibration/assets.py', 'software/trajectory/calibration/selection.py',
    'software/trajectory/assets/config/*.yaml',
    'software/vision/recordings.py', 'software/vision/runtime.py', 'software/vision/paths.py',
    'software/vision/capabilities.py', 'software/vision/tools/make_lens_video.py',
)
VISUAL_SOURCES = (
    'software/vision/slam.py', 'software/vision/runtime.py', 'software/vision/paths.py',
    'software/scene/spatial/slam.py',
)
PLACEMENT_SOURCES = (
    'software/scene/pipeline/map_lifecycle.py', 'software/scene/management/inputs.py',
    'software/scene/layout.py', 'software/scene/spatial/mapping.py',
    'software/scene/spatial/geometry.py', 'software/scene/spatial/fusion.py',
    'software/scene/spatial/poster_scale.py', 'software/vision/poster_pose.py',
    'software/vision/tag_board.py', 'software/vision/tags.py',
    'software/trajectory/algorithms/fusion_alignment.py',
    'software/trajectory/algorithms/fusion_alignment_solver.py',
    'software/trajectory/algorithms/inertial.py',
    'software/trajectory/algorithms/fusion_data.py', 'software/pose_io.py',
    'software/trajectory/reporting/map_data.py',
)


def source_identity(stage: str) -> dict[str, str]:
    dependencies = {
        'prepare': PREPARATION_SOURCES,
        'visual': VISUAL_SOURCES,
        'map': (*PREPARATION_SOURCES, *VISUAL_SOURCES, *PLACEMENT_SOURCES),
    }
    paths = set()
    for pattern in (*COMMON_SOURCES, *dependencies[stage]):
        matches = tuple(artifact_identity.ROOT.glob(pattern))
        if not matches:
            raise ValueError(f'建图代码依赖不存在：{pattern}')
        paths.update(matches)
    return {str(path.relative_to(artifact_identity.ROOT)): artifact_identity.file_hash(path)
            for path in sorted(paths)}


def camera_identity(camera: CalibratedCamera) -> dict:
    """地图只依赖镜头标定和遮手图；TCP、夹爪宽度和标签测距不参与建图。"""
    return dict(serial=camera.serial, intrinsics_source=camera.selected.intrinsics_source,
                calibration_version=camera.selected.calibration_id,
                assets={name: digest for name, digest in camera.asset_hashes.items()
                        if name in ('front', 'front_manifest', 'rear', 'mask')})


def reuse_key(input_identity: dict) -> str:
    """路径留在输入记录中供追溯，复用比较使用内容和影响计算的配置。"""
    definition = {name: value for name, value in input_identity['definition'].items()
                  if name not in ('mapping', 'devices')}
    definition['devices'] = {
        serial: {name: value for name, value in binding.items() if name != 'gripper'}
        for serial, binding in input_identity['definition']['devices'].items()}
    return artifact_identity.canonical_hash(dict(
        schema_version=1, definition=definition,
        recordings={arm: recording['sha256'] for arm, recording in input_identity['recordings'].items()},
        cameras=input_identity['cameras'], code=input_identity['code'], engines=input_identity['engines']))


def matches(previous: dict, current: dict) -> bool:
    # 老地图缺少原片内容身份，仍可显式引用；不猜测它是否匹配当前输入。
    return bool(previous.get('reuse_key')) and previous['reuse_key'] == current['reuse_key']


class MapBuildCache:
    def __init__(self, root: Path, engines: dict, *, force: bool = False):
        self.root = root
        self.engines = engines
        self.force = force
        self.timings: list[dict] = []

    def preparation_inputs(self, video: Path, camera: CalibratedCamera, scene: Scene) -> dict:
        return dict(recording_sha256=artifact_identity.file_hash(video), serial=camera.serial,
                    lenses=sorted(camera.measurements), video=asdict(scene.trajectory.values.video),
                    code=source_identity('prepare'),
                    engines={name: digest for name, digest in self.engines.items()
                             if not name.startswith(('orb_', 'Thirdparty/'))
                             and name != 'map_exporter' and 'gtsam' not in name})

    def visual_inputs(self, recording: PreparedRecording, lens: str) -> dict:
        video = recording.lens_video(lens)
        return dict(video=artifact_identity.file_hash(video.path),
                    settings=artifact_identity.file_hash(video.settings),
                    frames=recording.context.frame_count, lens=lens,
                    orb=asdict(recording.context.parameters.orb), code=source_identity('visual'),
                    engines={name: digest for name, digest in self.engines.items()
                             if name.startswith(('orb_', 'Thirdparty/'))})

    @contextmanager
    def measure(self, stage: str, label: str) -> Iterator[dict]:
        started = time.monotonic()
        record = dict(stage=stage, label=label, status='computed')
        console.write(label, state='开始')
        try:
            yield record
        except BaseException:
            record['status'] = 'failed'
            raise
        finally:
            record['elapsed_s'] = time.monotonic() - started
            self.timings.append(record)
            console.write(f'{label}；耗时 {record["elapsed_s"]:.1f} 秒',
                          state={'cached': '复用', 'computed': '完成', 'failed': '失败'}[record['status']])

    @contextmanager
    def stage(self, stage: str, label: str, identity: dict, directory: Path,
              *, roots: tuple[str, ...], refresh: bool = False) -> Iterator[bool]:
        """只发布完整阶段；缓存命中复制到本次工作目录，避免写穿缓存。"""
        key = artifact_identity.canonical_hash(identity)
        cache_dir = self.root / stage / key
        with self.measure(stage, label) as timing:
            timing['cache_key'] = key
            assets = None
            if not self.force and not refresh and (cache_dir / 'ready.json').is_file():
                try:
                    manifest = json.loads((cache_dir / 'ready.json').read_text())
                    if manifest['inputs'] != identity:
                        raise ValueError('缓存输入身份不匹配')
                    assets = manifest['assets']
                    from software.scene.management.inputs import validate_assets
                    validate_assets(cache_dir, assets)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    console.write(f'{label}缓存无效，将重新计算：{error}', state='提醒')
                    assets = None
            if assets is not None:
                for name in assets:
                    target = directory / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(cache_dir / name, target)
                yield True
                timing['status'] = 'cached'
                return

            yield False
            from software.scene.management.inputs import artifact_hashes
            assets = artifact_hashes(directory, roots)
            if not assets:
                raise ValueError(f'{label}未生成可缓存的产物')
            staging = cache_dir.with_name(f'.{key}-{uuid.uuid4().hex}')
            try:
                for name in assets:
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(directory / name, target)
                write_json(staging / 'ready.json', dict(inputs=identity, assets=assets))
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                staging.rename(cache_dir)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
