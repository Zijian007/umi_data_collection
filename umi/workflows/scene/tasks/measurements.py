"""任务环境标签与夹爪检测、双手检测进度、测量和拒绝证据。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
import time

from umi.infra.concurrency import daemon_map
from umi.infra.terminal import Console
from umi.perception.gripper import width_estimation
from umi.perception.gripper.distance import measurement as tag_distance
from umi.perception.gripper.width_estimation import WidthResult
from umi.perception.gripper.distance.pose_estimation import TagPoseQuality
from umi.perception.gripper.distance.measurement import TagDistanceRequest, TagDistanceResult
from umi.store import layout
from umi.store.models import Episode, PreparedRecording, TaskInputs
from umi.perception.decoding import ScanProgress
from umi.infra.file_store import write_json


console = Console('场景')


@dataclass(frozen=True)
class ArmMeasurements:
    width: WidthResult | None
    tag_distance: TagDistanceResult | None


def _counts(scan: ScanProgress) -> str:
    if scan.total is not None:
        return f'{scan.frames}/{scan.total} 帧'
    if scan.video_time_s is not None:
        return f'{scan.frames} 帧（原片 {scan.video_time_s:.1f}s）'
    return f'{scan.frames} 帧'


def _describe(gripper: ScanProgress | None, poster: ScanProgress | None) -> str:
    if (gripper is not None and poster is not None and gripper.cached == poster.cached
            and gripper.total is not None and gripper.total == poster.total):
        frames = min(gripper.frames, poster.frames)
        cache = '，缓存' if gripper.cached else ''
        return f'{frames}/{gripper.total} 帧（海报 {poster.observations} 次{cache}）'

    scans = []
    if gripper is not None:
        scans.append(f'夹爪 {_counts(gripper)}' + ('（缓存）' if gripper.cached else ''))
    if poster is not None:
        scans.append(f'海报 {_counts(poster)} / {poster.observations} 次' + ('（缓存）' if poster.cached else ''))
    return ' · '.join(scans)


class DetectionSummary:
    def __init__(self, console, recording: str, arms: tuple[str, ...], expected: dict[str, set[str]]):
        self.console, self.recording, self.arms, self.expected = console, recording, arms, expected
        self.lock = Lock()
        self.stop = Event()
        self.started = time.monotonic()
        self.arm_started: dict[str, float] = {}
        self.updates: dict[str, dict[str, ScanProgress]] = {arm: {} for arm in arms}
        self.finished: set[str] = set()
        self.worker = Thread(target=self._monitor, name='umi-detection-progress', daemon=True)

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, error_type, error, traceback):
        self.stop.set()
        self.worker.join()
        if error_type is None:
            self._write(final=True)

    def _monitor(self):
        while not self.stop.wait(3):
            self._write(final=False)

    def start_arm(self, arm):
        with self.lock:
            self.arm_started[arm] = time.monotonic()

    def update(self, arm, detector, scan):
        with self.lock:
            self.updates[arm][detector] = scan

    def finish_arm(self, arm):
        with self.lock:
            self.finished.add(arm)

    def _write(self, *, final):
        with self.lock:
            updates = {arm: scans.copy() for arm, scans in self.updates.items()}
            arm_started = self.arm_started.copy()
            finished = self.finished.copy()
        now = time.monotonic()
        fields, remaining = [], []
        can_estimate = True
        for arm in self.arms:
            hand = {'left': '左', 'right': '右'}.get(arm, arm)
            scans = updates[arm]
            details = _describe(scans.get('gripper'), scans.get('poster'))
            if arm in finished:
                if not final or not details:
                    details = f'{details} · 完成' if details else '完成'
            else:
                if not details:
                    details = '准备中' if arm in arm_started else '等待'
                if self.expected[arm] - scans.keys():
                    can_estimate = False
                for scan in scans.values():
                    if scan.cached:
                        continue
                    if scan.total is None or not scan.frames:
                        can_estimate = False
                    else:
                        elapsed = now - arm_started[arm]
                        remaining.append(max(0., elapsed * (scan.total - scan.frames) / scan.frames))
            fields.append(f'{hand} {details}')
        message = ('检测完成' if final else '检测进度') + ' · ' + ' ｜ '.join(fields)
        message += f' · 已用 {now - self.started:.1f}s'
        if not final and can_estimate and remaining and max(remaining) >= 1:
            message += f' · 预计剩余 {max(remaining):.0f}s'
        self.console.write(message, recording=self.recording, step=(2, 3))


def prepare_detection_group(inputs: TaskInputs, arms: tuple[str, ...], *, parallel: bool):
    """Precompute detections with one progress line for the whole group."""
    scene = inputs.episode.scene
    known = inputs.detections
    pending = tuple(arm for arm in arms if arm not in known or (known[arm] is None and scene.environment_tag_detection))
    if not pending:
        return [known[arm] for arm in arms]
    recording = inputs.episode.root.name
    stage = '环境标签与夹爪检测' if scene.environment_tag_detection else '夹爪检测'
    is_parallel = parallel and len(pending) > 1
    expected = {arm: ({'poster'} if scene.environment_tag_detection else set())
                | ({'gripper'} if 'tag_distance' in inputs.recordings[arm].camera.files else set())
                for arm in pending}
    if not scene.environment_tag_detection and not any(
            {'width', 'tag_distance'} & inputs.recordings[arm].camera.files.keys() for arm in pending):
        return [prepare_task_detections(inputs, arm) for arm in arms]
    inputs.console.write(stage + ('（双手并行）' if is_parallel else ''), recording=recording, step=(2, 3))
    with DetectionSummary(inputs.console, recording, pending, expected) as summary:
        def prepare(arm):
            summary.start_arm(arm)
            snapshot = prepare_task_detections(inputs, arm, summary=summary)
            summary.finish_arm(arm)
            return snapshot

        prepared = daemon_map(prepare, pending) if is_parallel else [prepare(arm) for arm in pending]
    snapshots = dict(zip(pending, prepared))
    return [snapshots[arm] if arm in snapshots else known[arm] for arm in arms]


def prepare_task_detections(inputs: TaskInputs, arm: str, *, summary: DetectionSummary | None = None):
    """Reuse each arm's completed detection across attempts and trimming."""
    if arm not in inputs.detections or (inputs.detections[arm] is None
                                        and inputs.episode.scene.environment_tag_detection):
        inputs.detections[arm] = prepare_detections(inputs.episode, arm, inputs.recordings[arm],
                                                   cache_audit=inputs.cache_audit, summary=summary)
    return inputs.detections[arm]


def prepare_detections(episode: Episode, arm: str, recording: PreparedRecording, *, cache_audit=None,
                       summary: DetectionSummary | None = None):
    """为后端准备海报观测，或提前完成两类夹爪检测的共用解码。"""
    camera, scene = recording.camera, episode.scene
    if not scene.environment_tag_detection and not {'width', 'tag_distance'} <= camera.files.keys():
        if cache_audit is not None:
            cache_audit.record(stage='gripper_detection', key=f'{arm}:none', path='', status='not_applicable',
                               reason='本手未配置夹爪宽度、夹爪标签测距或环境标签检测')
        return
    from umi.perception.gripper.detection import frames as detection
    from umi.perception.gripper.detection.sampling import SampledDetectionCache
    from umi.perception.poster import detection as environment_tags, layout as tag_board
    from umi.perception.decoding import VideoDecode

    video, digest = episode.recordings[arm], episode.recording_refs[arm]['sha256']
    cache = layout.metadata(episode.root, 'cache')
    decoder = VideoDecode(video)
    def write_progress(message):
        console.write(f'{arm}：{message}', recording=episode.root.name)

    def scan_progress(detector):
        if summary is not None:
            return lambda scan: summary.update(arm, detector, scan)
        return None

    progress = write_progress if summary is None else None
    if 'width' in camera.files:
        detection.cached_width(video, cache / 'gripper_width', digest, decoder=decoder)
    elif cache_audit is not None:
        cache_audit.record(stage='gripper_width', key=f'{arm}:none', path='', status='not_applicable',
                           reason='本手未配置夹爪宽度标定')
    if 'tag_distance' in camera.files:
        full_times = getattr(recording, 'full_video_times_s', None)
        times = recording.video_times_s if full_times is None else full_times
        SampledDetectionCache(video, cache, digest, times, progress,
            roi_px=tuple(scene.parameters.tag_distance_roi_px), on_progress=scan_progress('gripper')).load(decoder=decoder)
    elif cache_audit is not None:
        cache_audit.record(stage='tag_distance', key=f'{arm}:none', path='', status='not_applicable',
                           reason='本手未配置夹爪标签距离资产')
    snapshot = None
    if scene.environment_tag_detection:
        board = tag_board.load(scene.tag_layout)
        directory = cache / 'environment_tags' / arm

        def receive(observations, report):
            nonlocal snapshot
            snapshot = environment_tags.DetectionSnapshot(directory, board, observations, report)

        environment_tags.cached(video, directory,
            environment_tags.DetectionSettings(border_bits=board.border_bits, ids=board.ids, dictionary=board.dictionary,
                                   minimum_side_px=scene.parameters.tag_minimum_side_px),
            layout=board, progress=progress, decoder=decoder, cache_audit=cache_audit, on_ready=receive,
            on_progress=scan_progress('poster'))
    elif cache_audit is not None:
        cache_audit.record(stage='environment_tags', key=f'{arm}:none', path='', status='not_applicable',
                           reason='环境标签检测开关未开启或场景未配置布局')
    decoder.run()
    return snapshot


def measure(episode: Episode, arm: str, recording: PreparedRecording,
            video_offset_s: float, output: Path) -> ArmMeasurements:
    """共享或局部重建共用同一验收；失败由任务编排层阻止发布。"""
    parameters = episode.scene.parameters
    camera = recording.camera
    width_result = None
    if 'width' in camera.files:
        console.write(f'{arm}：提取指尖标记开合宽度', recording=episode.root.name)
        width_result = width_estimation.process(
            episode.recordings[arm], camera.width_measurement, recording.video_times_s,
            video_offset_s, output, layout.metadata(episode.root, 'cache/gripper_width'),
            episode.recording_refs[arm]['sha256'], calibration_sha256=camera.asset_hashes['width'],
            max_gap_s=parameters.gripper_width_max_gap_s,
            closed_threshold_mm=parameters.gripper_width_closed_threshold_mm,
            diagnostics=parameters.gripper_width_diagnostics)
    distance_result = None
    if 'tag_distance' in camera.files:
        console.write(f'{arm}：计算夹爪标签中心三维间距', recording=episode.root.name)
        request = TagDistanceRequest(
            video=episode.recordings[arm], recording_sha256=episode.recording_refs[arm]['sha256'],
            measurement=camera.tag_distance_measurement, camera=camera.measurements['front'],
            video_times_s=recording.video_times_s, episode_offset_s=video_offset_s,
            measurement_sha256=camera.asset_hashes['tag_distance'], camera_sha256=camera.asset_hashes['front'],
            quality=TagPoseQuality(parameters.tag_distance_max_reprojection_error_px,
                                  parameters.tag_distance_max_center_ambiguity_mm),
            min_valid_fraction=parameters.tag_distance_min_valid_fraction,
            max_missing_frames=parameters.tag_distance_max_missing_frames,
            roi_px=tuple(parameters.tag_distance_roi_px),
            detection_times_s=recording.full_video_times_s)
        try:
            distance_result = tag_distance.process(request, layout.metadata(episode.root, 'cache'),
                progress=lambda message: console.write(f'{arm}：{message}', recording=episode.root.name))
        except tag_distance.TagDistanceAcceptanceError as error:
            write_json(output / 'tag_distance.json', error.report)
            raise ValueError(f'{arm}：{error}') from error
    return ArmMeasurements(width_result, distance_result)
