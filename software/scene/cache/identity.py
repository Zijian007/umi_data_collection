"""准备、视觉、定位与建图的内容身份；不执行缓存读写或求解。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from software.infrastructure import provenance as artifact_identity
from software.scene.models import CalibratedCamera, PreparedRecording, Scene


# 显式列出各阶段的实现依赖。CLI、查看器和任务编排不改变地图。
# 新增算法依赖时应加入对应组；不以 Git 提交或文件时间代替内容哈希。
COMMON_SOURCES = (
    'software/configuration/loader.py', 'software/configuration/operational_settings.py',
    'software/configuration/parameter_models.py', 'software/infrastructure/provenance.py',
    'software/infrastructure/paths.py',
    'software/infrastructure/runtime_environment.py', 'software/infrastructure/file_store.py', 'software/scene/models.py',
    'software/scene/cache/storage.py', 'software/scene/cache/identity.py',
    'software/scene/pipeline/maps/identity.py', 'software/scene/asset_identity.py', 'software/vision/models.py',
)
# 录像准备只把 INSV 解码为 EuRoC 数据集和镜头视频；标定、ORB 和对齐参数
# 在之后才使用，不能让它们的代码改动使同一原片重新解码。
DECODE_SOURCES = (
    'software/configuration/loader.py', 'software/configuration/parameter_models.py', 'software/vision/models.py',
    'software/camera/x3/extraction.py', 'software/camera/x3/frame.py',
    'software/camera/x3/imu_reader.py', 'software/camera/x3/insv_reader.py',
    'software/vision/recordings.py', 'software/vision/tools/make_lens_video.py',
)
PREPARATION_SOURCES = (
    'software/camera/registry.py', 'software/camera/*.py', 'software/camera/x3/*.py',
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
    'software/scene/pipeline/maps/lifecycle.py',
    'software/scene/management/inputs/scene.py',
    'software/scene/management/inputs/batch.py',
    'software/scene/management/inputs/episode.py',
    'software/scene/layout.py', 'software/scene/spatial/mapping.py',
    'software/scene/spatial/geometry.py', 'software/scene/spatial/fusion.py',
    'software/scene/spatial/poster_scale.py', 'software/vision/poster/geometry.py',
    'software/vision/poster/quality.py',
    'software/vision/poster/layout.py', 'software/vision/poster/detection.py', 'software/vision/decoding.py',
    'software/trajectory/algorithms/fusion/alignment.py',
    'software/trajectory/algorithms/fusion/alignment_solver.py',
    'software/trajectory/algorithms/inertial.py',
    'software/trajectory/algorithms/fusion/data.py', 'software/trajectory/io/tum.py',
    'software/trajectory/reporting/map_data.py',
)


def source_identity(stage: str) -> dict[str, str]:
    dependencies = {
        'prepare': DECODE_SOURCES,
        'visual': (*COMMON_SOURCES, *VISUAL_SOURCES),
        'localize': (*COMMON_SOURCES, *DECODE_SOURCES, *PREPARATION_SOURCES, *VISUAL_SOURCES,
                     'software/scene/spatial/geometry.py', 'software/scene/spatial/relocalization.py',
                     'software/scene/pipeline/tasks/backends/scene_reconstruction.py'),
        'map': (*COMMON_SOURCES, *DECODE_SOURCES, *PREPARATION_SOURCES, *VISUAL_SOURCES, *PLACEMENT_SOURCES),
    }
    paths = set()
    for pattern in dependencies[stage]:
        matches = tuple(artifact_identity.ROOT.glob(pattern))
        if not matches:
            raise ValueError(f'建图代码依赖不存在：{pattern}')
        paths.update(matches)
    return {str(path.relative_to(artifact_identity.ROOT)): artifact_identity.source_hash(path)
            for path in sorted(paths)}


def preparation_inputs(video: Path, camera: CalibratedCamera, scene: Scene, *, engines: dict) -> dict:
    """建图和任务共用；原片内容已确定设备，标定只决定生成哪些镜头。"""
    return dict(recording_sha256=artifact_identity.file_hash(video),
                lenses=sorted(camera.measurements), video=asdict(scene.trajectory.values.video),
                code=source_identity('prepare'),
                engines={name: digest for name, digest in engines.items()
                         if name in ('ffmpeg', 'ffprobe') or '/numpy-' in name})


def visual_inputs(recording: PreparedRecording, lens: str, *, engines: dict) -> dict:
    video = recording.lens_video(lens)
    return dict(video=artifact_identity.file_hash(video.path),
                settings=artifact_identity.file_hash(video.settings),
                frames=recording.context.frame_count, lens=lens,
                orb=asdict(recording.context.parameters.orb), code=source_identity('visual'),
                engines={name: digest for name, digest in engines.items()
                         if name.startswith(('orb_', 'Thirdparty/'))})
