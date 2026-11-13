"""环境海报的原始角点观测与可见性；不假定 ID 排版、不求位姿或融合轨迹。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable
import json
import time

import cv2
import numpy as np

from software import provenance
from software.camera.timing import video_to_imu_seconds
from software.camera.x3.frame import CROP, FPS_SRC, FPS_TOLERANCE, SRC_H, SRC_W
from software.storage import atomic_file, mark_used, write_json
from software.vision.decoding import VideoDecode

if TYPE_CHECKING:
    from software.vision.tag_board import TagBoardLayout


@dataclass(frozen=True)
class DetectionSettings:
    """现有海报的字典与 ID；黑边须由排版确认后显式指定。"""
    border_bits: int
    frame_step: int = 2
    minimum_side_px: float = 0.0
    ids: tuple[int, ...] = tuple(range(12, 89))
    dictionary: str = 'DICT_4X4_100'

    def __post_init__(self):
        if self.dictionary != 'DICT_4X4_100':
            raise ValueError('环境海报当前只支持 DICT_4X4_100')
        if type(self.border_bits) is not int or self.border_bits < 1:
            raise ValueError('border_bits 必须为正整数')
        if type(self.frame_step) is not int or self.frame_step < 1:
            raise ValueError('frame_step 必须为正整数')
        if not np.isfinite(self.minimum_side_px) or self.minimum_side_px < 0:
            raise ValueError('minimum_side_px 必须为有限非负数')
        if (not self.ids or any(type(i) is not int or not 0 <= i < 100 for i in self.ids)
                or len(set(self.ids)) != len(self.ids)):
            raise ValueError('海报 ID 必须为不重复的 0–99 整数')


@dataclass(frozen=True)
class TagObservations:
    """N 个采样时刻，M 次唯一 ID 观测；像素为原片左半幅坐标。

    sample_indices 将 M 次观测关联到 N 个采样帧，不将同帧的多个 ID 当成多个时刻。
    保留标定视场外角点供报告，但 usable=False，后续几何不得使用。
    """
    source_frame_indices: np.ndarray  # (N,) 解码源帧号
    video_times_s: np.ndarray         # (N,) 原片真实 PTS 秒，不含镜头–IMU 偏移
    sample_indices: np.ndarray        # (M,) 对上述 N 个时刻的索引
    tag_ids: np.ndarray               # (M,)
    corners_half_px: np.ndarray       # (M, 4, 2)
    side_px: np.ndarray               # (M,) 最短投影边长
    in_calibrated_crop: np.ndarray    # (M,)
    usable: np.ndarray               # (M,)
    duplicate_ids: np.ndarray         # (N,) 本帧重复的已配置 ID 个数，全部拒绝
    unknown_detections: np.ndarray    # (N,) 未配置 ID 的检测个数

    @property
    def corners_px(self) -> np.ndarray:
        """1440 标定裁切坐标，保留浮点像素中心，不作缩放。"""
        return self.corners_half_px - (SRC_H - CROP) / 2

    def imu_times_s(self, camera_offset_ns: int) -> np.ndarray:
        return video_to_imu_seconds(self.video_times_s[self.sample_indices], camera_offset_ns)

    def validate(self) -> None:
        n, m = len(self.video_times_s), len(self.tag_ids)
        if n == 0 or self.source_frame_indices.shape != (n,) or self.sample_indices.shape != (m,):
            raise ValueError('标签观测时间轴或索引维度无效')
        if (not np.isfinite(self.video_times_s).all() or not np.all(np.diff(self.video_times_s) > 0)
                or not np.all(np.diff(self.source_frame_indices) > 0)
                or np.any(self.source_frame_indices < 0)):
            raise ValueError('标签观测帧号或时间必须严格递增且有效')
        for indices in (self.source_frame_indices, self.sample_indices, self.tag_ids):
            if not np.issubdtype(indices.dtype, np.integer):
                raise ValueError('标签观测索引必须为整数')
        if np.any(self.sample_indices < 0) or np.any(self.sample_indices >= n):
            raise ValueError('标签观测索引越界')
        if self.corners_half_px.shape != (m, 4, 2) or not np.isfinite(self.corners_half_px).all():
            raise ValueError('标签角点无效')
        if self.side_px.shape != (m,) or not np.isfinite(self.side_px).all() or np.any(self.side_px <= 0):
            raise ValueError('标签投影边长无效')
        for mask in (self.in_calibrated_crop, self.usable):
            if mask.shape != (m,) or mask.dtype != np.bool_:
                raise ValueError('标签有效性数组无效')
        for counts in (self.duplicate_ids, self.unknown_detections):
            if counts.shape != (n,) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
                raise ValueError('标签诊断计数无效')
        if len(set(zip(self.sample_indices.tolist(), self.tag_ids.tolist()))) != m:
            raise ValueError('标签观测含同帧重复 ID')
        if np.any(self.usable & ~self.in_calibrated_crop):
            raise ValueError('标定裁切外角点不能用于几何')


class PosterDetector:
    def __init__(self, settings: DetectionSettings):
        self.settings = settings
        parameters = cv2.aruco.DetectorParameters()
        parameters.markerBorderBits = settings.border_bits
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect(self, gray: np.ndarray):
        """仅接收原始环境镜头半幅，返回唯一 ID 角点和拒绝计数。"""
        if gray.shape != (SRC_H, SRC_H) or gray.dtype != np.uint8:
            raise ValueError('环境标签检测需要原始 1920×1920 灰度左半幅')
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return [], 0, 0
        ids = ids.ravel()
        unique, counts = np.unique(ids, return_counts=True)
        repeated = set(unique[counts > 1].tolist()) & set(self.settings.ids)
        found, unknown = [], 0
        for tag_id, polygon in zip(ids.tolist(), corners):
            if tag_id not in self.settings.ids:
                unknown += 1
            elif tag_id not in repeated:
                points = np.asarray(polygon, float).reshape(4, 2)
                if np.isfinite(points).all():
                    found.append((tag_id, points))
        return sorted(found, key=lambda item: item[0]), len(repeated), unknown


class PosterDetection:
    """按源帧号采样环境半幅；完整源时间轴仍逐帧校验。"""

    def __init__(self, settings: DetectionSettings, progress: Callable[[str], None] | None = None):
        self.settings, self.progress = settings, progress
        self.detector = PosterDetector(settings)
        self.frames, self.times, self.samples, self.ids, self.corners = [], [], [], [], []
        self.duplicates, self.unknowns = [], []
        self.previous_time, self.updated = None, time.monotonic()

    def validate(self, streams) -> None:
        if len(streams) != 1:
            raise ValueError('环境标签检测需要单路 X3 4K60 原片')
        stream = streams[0]
        if (stream.codec_context.width, stream.codec_context.height) != (SRC_W, SRC_H):
            raise ValueError('环境标签检测需要 X3 原始 3840×1920 双镜头画面')
        if stream.average_rate is None or abs(float(stream.average_rate) - FPS_SRC) > FPS_TOLERANCE:
            raise ValueError('环境标签检测需要 X3 4K60 录制模式')

    def consume(self, source_index: int, frame) -> None:
        if frame.pts is None or frame.time_base is None:
            raise ValueError('环境标签原片缺少帧时间戳')
        timestamp = float(frame.pts * frame.time_base)
        if not np.isfinite(timestamp) or self.previous_time is not None and timestamp <= self.previous_time:
            raise ValueError('环境标签原片时间戳非递增或无效')
        self.previous_time = timestamp
        if source_index % self.settings.frame_step:
            return
        gray = np.ascontiguousarray(frame.to_ndarray(format='gray')[:, :SRC_H])
        found, duplicate, unknown = self.detector.detect(gray)
        sample_index = len(self.times)
        self.frames.append(source_index)
        self.times.append(timestamp)
        self.duplicates.append(duplicate)
        self.unknowns.append(unknown)
        for tag_id, points in found:
            self.samples.append(sample_index)
            self.ids.append(tag_id)
            self.corners.append(points)
        if self.progress is not None and time.monotonic() - self.updated >= 3:
            self.progress(f'环境标签：已采样 {len(self.times)} 帧，原片 {timestamp:.2f} s，唯一 ID 观测 {len(self.ids)} 次')
            self.updated = time.monotonic()

    def finish(self) -> TagObservations:
        points = np.asarray(self.corners, float).reshape(-1, 4, 2)
        sides = np.linalg.norm(points - np.roll(points, -1, axis=1), axis=2).min(axis=1)
        margin = (SRC_H - CROP) / 2
        inside = ((points >= margin) & (points < margin + CROP)).all(axis=(1, 2))
        result = TagObservations(np.asarray(self.frames, int), np.asarray(self.times, float), np.asarray(self.samples, int),
            np.asarray(self.ids, int), points, sides, inside, inside & (sides >= self.settings.minimum_side_px),
            np.asarray(self.duplicates, int), np.asarray(self.unknowns, int))
        result.validate()
        return result


def detect(video: Path, settings: DetectionSettings, *, progress: Callable[[str], None] | None = None) -> TagObservations:
    """顺序解码真实 PTS，按源帧号抽样；不读取 IMU，不插值或沿用角点。"""
    decoder = VideoDecode(video)
    decoder.add(PosterDetection(settings, progress))
    return decoder.run()[0]


def _visibility(observations: TagObservations, sample_indices: np.ndarray, frame_step: int) -> dict:
    visible = np.zeros(len(observations.video_times_s), bool)
    visible[sample_indices] = True
    # 每个采样点代表到下一采样点的区间，末帧使用名义采样周期。
    # 这是采样可见性，不宣称跳过的源帧也不可见。
    edges = np.r_[observations.video_times_s, observations.video_times_s[-1] + frame_step / FPS_SRC]
    transitions = np.diff(np.r_[False, ~visible, False].astype(int))
    longest = max((float(edges[end] - edges[start]) for start, end in
                   zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))), default=0.)
    return dict(frames=int(visible.sum()), fraction=float(visible.mean()), maximum_sampled_gap_s=longest)


def report(observations: TagObservations, settings: DetectionSettings) -> dict:
    observations.validate()
    n = len(observations.video_times_s)
    valid = observations.usable
    counts = np.bincount(observations.sample_indices[valid], minlength=n)
    per_id = {}
    for tag_id in settings.ids:
        selected = observations.tag_ids == tag_id
        sides = observations.side_px[selected]
        per_id[str(tag_id)] = dict(
            detected=_visibility(observations, observations.sample_indices[selected], settings.frame_step),
            usable=_visibility(observations, observations.sample_indices[selected & valid], settings.frame_step),
            side_px_quantiles=np.quantile(sides, [.1, .5, .9]).tolist() if len(sides) else None)
    total = len(observations.tag_ids)
    return dict(kind='umi_environment_tag_visibility', schema_version=1, lens='rear',
        dictionary=settings.dictionary, sampled_frames=n, observations=total,
        first_video_time_s=float(observations.video_times_s[0]), last_video_time_s=float(observations.video_times_s[-1]),
        at_least_one_usable=_visibility(observations, observations.sample_indices[valid], settings.frame_step),
        usable_ids_per_frame=counts.tolist(), usable_corners_per_frame=(counts * 4).tolist(),
        outside_calibrated_crop_fraction=float((~observations.in_calibrated_crop).mean()) if total else None,
        rejected_duplicate_ids=int(observations.duplicate_ids.sum()),
        unknown_detections=int(observations.unknown_detections.sum()), per_id=per_id,
        coordinates=dict(corners_half_px='original_rear_1920', corners_px='calibrated_crop_1440',
                         time='source_video_pts_seconds', imu_offset_applied=False),
        geometry_status='not_evaluated', actual_error='not_measured',
        note='采样可见性报告；usable 仅表示像素尺寸及标定裁切筛选通过，不代表位姿可观测或轨迹精度。')


def cached(video: Path, output: Path, settings: DetectionSettings, *, force: bool = False,
           progress: Callable[[str], None] | None = None, layout: TagBoardLayout | None = None,
           decoder: VideoDecode | None = None) -> dict | None:
    """角点缓存独立绑定检测身份；decoder 将缺失检测推迟到共享解码。"""
    import av

    video, output = video.expanduser().resolve(), output.expanduser().resolve()
    if not video.is_file():
        raise ValueError(f'原片不存在：{video}')
    if video.is_relative_to(output):
        raise ValueError('检测输出目录不能包含输入原片，请指定独立目录')
    if layout is not None and (settings.dictionary != layout.dictionary or settings.border_bits != layout.border_bits
                               or not set(settings.ids).issubset(layout.ids)):
        raise ValueError('检测字典、黑边或 ID 与海报布局不一致')
    source_files = (Path(__file__), Path(__file__).with_name('decoding.py'),
                    provenance.ROOT / 'software/camera/x3/frame.py',
                    provenance.ROOT / 'software/camera/timing.py')
    identity = dict(schema_version=1, recording_sha256=provenance.file_hash(video), settings=asdict(settings),
        implementation={str(p.relative_to(provenance.ROOT)): provenance.source_hash(p) for p in source_files},
        dependencies=dict(opencv=cv2.__version__, av=av.__version__, numpy=np.__version__))
    key = provenance.canonical_hash(identity)
    cache_file, manifest_file = output / 'observations.npz', output / 'detection.json'
    observations = None
    if not force and cache_file.is_file() and manifest_file.is_file():
        try:
            saved = json.loads(manifest_file.read_text())
            if saved['identity_hash'] == key and saved['observations_sha256'] == provenance.file_hash(cache_file):
                with np.load(cache_file, allow_pickle=False) as data:
                    observations = TagObservations(**{name: data[name] for name in TagObservations.__dataclass_fields__})
                observations.validate()
        except (ValueError, KeyError, OSError, TypeError):
            observations = None
        if observations is not None:
            mark_used(output)
    def publish(observations: TagObservations, reused: bool) -> dict:
        if not reused:
            if provenance.file_hash(video) != identity['recording_sha256']:
                raise ValueError('原片在标签检测期间发生变化，未保存检测结果')
            with atomic_file(cache_file) as temporary, temporary.open('wb') as stream:
                np.savez_compressed(stream, **{name: getattr(observations, name) for name in TagObservations.__dataclass_fields__})
            write_json(manifest_file, dict(identity=identity, identity_hash=key,
                                          observations_sha256=provenance.file_hash(cache_file)))
        result = dict(report(observations, settings), identity_hash=key, recording=str(video),
                      settings=asdict(settings), cache_reused=reused, observations_file=str(cache_file))
        if layout is not None:
            result['board_layout'] = layout.record()
            result['geometry_status'] = 'layout_known_pose_not_estimated'
            # 角点缓存与物理布局分开：更正打印尺寸复用像素，但报告携带当前布局身份。
            result['report_identity_hash'] = provenance.canonical_hash(dict(detection=key, layout=layout.asset_hash))
        write_json(output / 'report.json', result)
        return result

    if observations is not None:
        return publish(observations, True)
    if decoder is not None:
        decoder.add(PosterDetection(settings, progress), lambda found: publish(found, False))
        return None
    return publish(detect(video, settings, progress=progress), False)
