"""任务夹爪测量：读取冻结配置、执行宽度/标签测距并保留拒绝证据。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from software.console import Console
from software.gripper_width import pipeline as width_pipeline
from software.gripper_width import tag_distance
from software.gripper_width.pipeline import WidthResult
from software.gripper_width.tag_models import TagDistanceRequest, TagDistanceResult, TagPoseQuality
from software.scene import layout
from software.scene.models import Episode, PreparedRecording
from software.storage import write_json


console = Console('场景')


@dataclass(frozen=True)
class ArmMeasurements:
    width: WidthResult | None
    tag_distance: TagDistanceResult | None


def prepare_detections(episode: Episode, arm: str, recording: PreparedRecording) -> None:
    """多类检测共用一次首次解码，各自缓存供融合和测量阶段读取。"""
    camera, scene = recording.camera, episode.scene
    if sum(('width' in camera.files, 'tag_distance' in camera.files, scene.tag_layout is not None)) < 2:
        return
    from software.gripper_width import detection
    from software.gripper_width.sampled_detection import SampledDetectionCache
    from software.vision import tag_board, tags
    from software.vision.decoding import VideoDecode

    video, digest = episode.recordings[arm], episode.recording_refs[arm]['sha256']
    cache = layout.metadata(episode.root, 'cache')
    decoder = VideoDecode(video)
    progress = lambda message: console.write(f'{arm}：{message}', recording=episode.root.name)
    if 'width' in camera.files:
        detection.cached(video, cache / 'gripper_width', digest, decoder=decoder)
    if 'tag_distance' in camera.files:
        SampledDetectionCache(video, cache, digest, recording.video_times_s, progress,
            roi_px=tuple(scene.parameters.tag_distance_roi_px)).load(decoder=decoder)
    if scene.tag_layout is not None:
        board = tag_board.load(scene.tag_layout)
        tags.cached(video, cache / 'environment_tags' / arm,
            tags.DetectionSettings(border_bits=board.border_bits, ids=board.ids, dictionary=board.dictionary,
                                   minimum_side_px=scene.parameters.tag_minimum_side_px),
            layout=board, progress=progress, decoder=decoder)
    decoder.run()


def measure(episode: Episode, arm: str, recording: PreparedRecording,
            video_offset_s: float, output: Path) -> ArmMeasurements:
    """共享或局部重建共用同一验收；失败由任务编排层阻止发布。"""
    parameters = episode.scene.parameters
    camera = recording.camera
    width_result = None
    if 'width' in camera.files:
        console.write(f'{arm}：提取指尖标记开合宽度', recording=episode.root.name)
        width_result = width_pipeline.process(
            episode.recordings[arm], camera.width_measurement, recording.video_times_s,
            video_offset_s, output, layout.metadata(episode.root, 'cache/gripper_width'),
            episode.recording_refs[arm]['sha256'], calibration_sha256=camera.asset_hashes['width'],
            max_gap_s=parameters.gripper_width_max_gap_s,
            closed_threshold_mm=parameters.gripper_width_closed_threshold_mm,
            diagnostics=parameters.gripper_width_diagnostics)
    distance_result = None
    if 'tag_distance' in camera.files:
        console.write(f'{arm}：计算两标签中心三维间距', recording=episode.root.name)
        request = TagDistanceRequest(
            video=episode.recordings[arm], recording_sha256=episode.recording_refs[arm]['sha256'],
            measurement=camera.tag_distance_measurement, camera=camera.measurements['front'],
            video_times_s=recording.video_times_s, episode_offset_s=video_offset_s,
            measurement_sha256=camera.asset_hashes['tag_distance'], camera_sha256=camera.asset_hashes['front'],
            quality=TagPoseQuality(parameters.tag_distance_max_reprojection_error_px,
                                  parameters.tag_distance_max_center_ambiguity_mm),
            min_valid_fraction=parameters.tag_distance_min_valid_fraction,
            max_missing_frames=parameters.tag_distance_max_missing_frames,
            roi_px=tuple(parameters.tag_distance_roi_px))
        try:
            distance_result = tag_distance.process(request, layout.metadata(episode.root, 'cache'),
                progress=lambda message: console.write(f'{arm}：{message}', recording=episode.root.name))
        except tag_distance.TagDistanceAcceptanceError as error:
            write_json(output / 'tag_distance.json', error.report)
            raise ValueError(f'{arm}：{error}') from error
    return ArmMeasurements(width_result, distance_result)
