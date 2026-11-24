"""原片归档与算法数据准备：身份、时间戳、IMU 和镜头视频。"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import json
import math
import os
import shutil
import tempfile
import numpy as np
from umi.config import operational_settings as settings
from umi.infra import provenance as artifact_identity
from umi.infra.concurrency import daemon_map
from umi.devices import registry
from umi.core.timing import euroc_ns_to_seconds
from umi.store import layout
from umi.store.models import CalibratedCamera, Episode, PreparedRecording, Scene, SceneParameters
from umi.store.cache import StageCache, preparation_inputs
from umi.devices import selection as calibration_assets
from umi.perception import recordings as recording_data, runtime
from umi.perception.models import LensVideo, RecordingData, RecordingPaths, VisualContext
from dataclasses import asdict, dataclass, field
import struct
from umi.core import frame
from umi.devices.x3 import imu_reader, insv_reader
from umi.workflows.scene.inputs.scene import load_scene_parameters


def insv_files(directory: Path) -> list[Path]:
    """所有目录入口共用原片发现规则，包括大写扩展名。"""
    return sorted(path for path in directory.rglob('*') if path.is_file() and path.suffix.lower() == '.insv')


def inspect(video: Path) -> dict:
    video = video.expanduser().resolve()
    if not video.is_file() or video.suffix.lower() != '.insv':
        raise ValueError(f'需要原始 INSV 录像：{video}')
    serial = registry.validate_id(registry.read_insv_serial(video))
    return dict(serial=serial, sha256=artifact_identity.file_hash(video))


def video_duration_s(video: Path) -> float:
    """Read the recording duration from container metadata without decoding frames."""
    import av

    with av.open(str(video)) as container:
        duration = float(container.duration or 0) / 1e6
        if duration <= 0:
            stream = next(iter(container.streams.video), None)
            if stream is not None and stream.duration is not None:
                duration = float(stream.duration * stream.time_base)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'无法确定视频时长：{video}')
    return duration


def import_video(video: Path, directory: Path, *, metadata: dict, link: bool = False,
                 owned_root: Path | None = None) -> Path:
    """本场景指定原片目录内直接引用；外部原片复制或显式硬链接到目标目录。"""
    video = video.expanduser().resolve()
    directory = directory.expanduser().resolve()
    if video.is_relative_to((owned_root or directory).resolve()):
        if artifact_identity.file_hash(video) != metadata['sha256']:
            raise ValueError(f'原片在导入前发生变化：{video}')
        return video
    target = directory / video.name
    if target.exists():
        if artifact_identity.file_hash(target) != metadata['sha256']:
            raise ValueError(f'同名录像内容不同，拒绝覆盖：{target}')
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(video, target)
            return target
        except OSError:
            pass  # 跨文件系统或无硬链接权限时平滑回退至安全复制
    fd, name = tempfile.mkstemp(prefix='.import-', dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(video, temporary)
        if artifact_identity.file_hash(temporary) != metadata['sha256']:
            raise ValueError(f'录像复制校验失败：{video}')
        # link 的排他创建语义保证并发导入也不能覆盖已有原片。
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def prepare_recording(recording_path: Path, camera: CalibratedCamera, scene: Scene, cache_dir: Path, *, cores: str = '',
                      cache: StageCache | None = None, defer_lens_videos: bool = False) -> PreparedRecording:
    identity = preparation_inputs(recording_path, camera, scene) if cache else {}
    roots = ('debug/dataset',) if defer_lens_videos else ('debug/dataset', 'debug/lens_videos')
    preparation_identity = dict(identity, products='timestamps_and_imu') if defer_lens_videos else identity

    def render_lens(lens: str, context: VisualContext, source_frame_start: int) -> LensVideo:
        path = context.paths.lens_videos / f'{lens}.mp4'
        manifest = path.with_name(f'{lens}_manifest.json')
        settings = context.settings
        if not path.is_file():
            lens_identity = dict(identity, lens=lens, source_frame_start=source_frame_start,
                                 frame_count=context.frame_count)
            try:
                with (cache.stage('lens_video', f'{camera.serial} {lens} 镜头视频', lens_identity,
                                  context.paths.output, roots=(str(path.relative_to(context.paths.output)),
                                  str(manifest.relative_to(context.paths.output))), link=True)
                      if cache else nullcontext(False)) as reuse:
                    if not reuse:
                        recording_data.prepare_lens_videos(recording_path, context, lenses=(lens,),
                                                          source_frame_start=source_frame_start)
                        if cache and artifact_identity.file_hash(recording_path) != identity['recording_sha256']:
                            raise ValueError('原片在生成镜头视频期间发生变化')
            except Exception:
                path.unlink(missing_ok=True)
                manifest.unlink(missing_ok=True)
                raise
        return LensVideo(lens, path, settings.front_settings if lens == 'front' else settings.rear_settings,
                         settings.front_mask if lens == 'front' else None)

    # 无缓存时（rear 标定等一次性用途）总是重新准备。
    with (cache.stage('prepare', f'{camera.serial} 录像准备', preparation_identity, cache_dir,
                      roots=roots, link=True) if cache else nullcontext(False)) as reuse:
        paths = RecordingPaths(cache_dir)
        paths.logs.mkdir(parents=True, exist_ok=True)
        if reuse:
            extracted_data = RecordingData(json.loads((paths.dataset / 'extract_manifest.json').read_text()), paths.dataset / 'mav0/imu0/data.csv')
        else:
            extracted_data = recording_data.prepare(recording_path, paths, allow_irregular=False,
                frame_step=scene.trajectory.values.video.frame_step)
        if extracted_data.manifest['source']['serial'] != camera.serial:
            raise ValueError('录像设备身份在处理期间发生变化')
        settings = runtime.visual_settings(dict(os.environ),
            lens_settings={lens: calibration_assets.slam_settings_text(measured, scene.trajectory.values)
                           for lens, measured in camera.measurements.items()},
            front_mask=camera.selected.assets.front_mask, output=cache_dir / 'settings',
            parameters=scene.trajectory.values,
            **(dict(require_slam=False) if scene.reconstruction_mode == 'poster_vio' else {}))
        context = VisualContext(paths, settings, scene.trajectory.values, extracted_data.manifest['frames']['n_frames_used'], cores)
        if defer_lens_videos:
            lens_videos = ()
        elif reuse:
            lens_videos = tuple(render_lens(lens, context, 0) for lens in camera.measurements)
        else:
            lens_videos = recording_data.prepare_lens_videos(recording_path, context, lenses=tuple(camera.measurements))
        frame_times_s = euroc_ns_to_seconds(np.loadtxt(paths.dataset / 'mav0/cam0/data.csv', delimiter=',', comments='#', usecols=0))
        imu = np.loadtxt(extracted_data.imu_csv, delimiter=',', comments='#')
        imu[:, 0] = euroc_ns_to_seconds(imu[:, 0])
        if len(imu) < 2 or not np.isfinite(imu).all() or not np.all(np.diff(imu[:, 0]) > 0):
            raise ValueError('IMU 数据无效、重复或非递增')
        return PreparedRecording(camera, context, lens_videos, frame_times_s, imu, extracted_data, render_lens=render_lens)


def arm_cores(arms: tuple[str, ...], cores: str, *, parallel: bool) -> dict[str, str] | None:
    if not parallel or len(arms) < 2:
        return None
    available = settings.available_cpus(cores)
    if len(available) < settings.values('runtime')['arm_min_cpus'] * len(arms):
        return None
    ordered = sorted(available)
    return {arm: ','.join(map(str, ordered[index::len(arms)])) for index, arm in enumerate(arms)}


def shared_interval(recordings: dict[str, PreparedRecording], offsets_s: dict[str, float]) -> tuple[float, float]:
    start = max(recording.video_times_s[0] + offsets_s[arm] for arm, recording in recordings.items())
    end = min(recording.video_times_s[-1] + offsets_s[arm] for arm, recording in recordings.items())
    return start, end


def prepare_with_cache(episode, arm, camera, scene, cores, *, work_dir: Path, cache_audit=None, force: bool = False):
    """与建图共用阶段缓存；本次设置和日志写在运行工作目录，缓存只存解码产物。"""
    cache = StageCache(layout.metadata(episode.root, 'cache'), force=force, audit=cache_audit)
    return prepare_recording(episode.recordings[arm], camera, scene, work_dir / arm, cores=cores,
                             cache=cache, defer_lens_videos=True)


def prepare_episode_recordings(episode: Episode, cameras: dict[str, CalibratedCamera],
                               video_offsets_s: dict[str, float], *, work_dir: Path, cores: str = '',
                               arm_cores: dict[str, str] | None = None, cache_audit=None) -> dict[str, PreparedRecording]:
    """准备各手录像并检查共同时间区间；不估计跨相机时间偏移。"""
    def prepare(arm: str) -> tuple[str, PreparedRecording]:
        return arm, prepare_with_cache(
            episode, arm, cameras[arm], episode.scene,
            arm_cores[arm] if arm_cores else cores,
            work_dir=work_dir, cache_audit=cache_audit)

    arms = tuple(episode.recordings)
    if arm_cores and len(arms) > 1:
        # Each arm has independent source/cache directories and an already
        # partitioned CPU set. Parallelize timestamp and IMU preparation
        # without creating nested task-level parallelism.
        prepared = dict(daemon_map(prepare, arms))
    else:
        prepared = dict(prepare(arm) for arm in arms)
    start, end = shared_interval(prepared, video_offsets_s)
    if start >= end:
        raise ValueError('左右手录像在任务时间轴上没有重叠')
    return prepared


class RecordingCheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    message: str
    metrics: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status != 'FAIL'


@dataclass
class RecordingCheckReport:
    path: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class EpisodeCheckReport:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    videos: dict[str, RecordingCheckReport] = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)


def check_stream(path: Path) -> CheckResult:
    try:
        import av
        with av.open(str(path)) as container:
            streams = list(container.streams.video)
            geometry = [dict(w=s.codec_context.width, h=s.codec_context.height) for s in streams]
            rate = float(streams[0].average_rate or 0) if streams else 0.
            good = frame.detect_profile_from_streams(geometry) == frame.PROFILE and abs(rate - frame.FPS_SRC) <= frame.FPS_TOLERANCE
            return CheckResult('video_stream', 'PASS' if good else 'FAIL',
                '视频流符合 X3 360 4K60' if good else '视频流不符合 X3 360 4K60', dict(streams=geometry, fps=rate))
    except (OSError, ValueError, RuntimeError) as error:
        return CheckResult('video_stream', 'FAIL', f'无法读取视频容器：{error}')


def check_imu(path: Path, parameters: SceneParameters) -> tuple[CheckResult, dict | None]:
    try:
        imu = imu_reader.read_imu_and_meta(path)
    except (OSError, ValueError, RuntimeError) as error:
        return CheckResult('imu_integrity', 'FAIL', f'无法读取 IMU：{error}'), None
    times = np.asarray(imu['t_s'])
    status = imu.get('integrity', {}).get('status')
    if (len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0)
            or not all(np.isfinite(imu[name]).all() for name in ('acc', 'gyr')) or status != 'OK'):
        return CheckResult('imu_integrity', 'FAIL', 'IMU 数据或时间戳无效', dict(integrity=imu.get('integrity'))), None
    duration = float(times[-1] - max(0., times[0]))
    metrics = dict(samples=len(times), duration_s=duration, integrity=imu.get('integrity'))
    if duration < parameters.recording_min_duration_s:
        return CheckResult('imu_integrity', 'FAIL', '录像有效时长过短', metrics), None
    return CheckResult('imu_integrity', 'PASS', 'IMU 完整性正常', metrics), imu


def check_initial_static(imu: dict, parameters: SceneParameters) -> CheckResult:
    times = np.asarray(imu['t_s'])
    mask = (times >= 0) & (times <= parameters.recording_static_window_s)
    if np.count_nonzero(mask) < 20:
        return CheckResult('initial_static', 'UNKNOWN', '开头静止窗口采样不足')
    acc = np.linalg.norm(imu['acc'][mask], axis=1)
    gyro = np.linalg.norm(imu['gyr'][mask], axis=1)
    mean, std, p95 = float(acc.mean()), float(acc.std()), float(np.percentile(gyro, 95))
    good = (abs(mean - imu_reader.G0) <= parameters.recording_max_static_acc_dev_m_s2
            and std <= parameters.recording_max_static_acc_std_m_s2 and p95 <= parameters.recording_max_static_gyro_rad_s)
    return CheckResult('initial_static', 'PASS' if good else 'WARN',
        '开头静止质量正常' if good else '开头静止质量不足；以固定地图定位和融合质量继续判断',
        dict(acc_mean_m_s2=mean, acc_std_m_s2=std, gyro_p95_rad_s=p95))


def check_exposure(path: Path, parameters: SceneParameters) -> CheckResult:
    try:
        records = insv_reader.records_by_id(insv_reader.parse_trailer(path, gyro_parser=False))
        if 4 not in records:
            return CheckResult('exposure', 'UNKNOWN', '缺少曝光记录')
        exposure_records = insv_reader.read_record_bytes(path, records[4])
        if not exposure_records or len(exposure_records) % 16:
            return CheckResult('exposure', 'UNKNOWN', '曝光记录格式无法识别')
        shutters = np.array([item[1] for item in struct.iter_unpack('<Qd', exposure_records)])
        if not np.isfinite(shutters).all() or np.any(shutters <= 0):
            return CheckResult('exposure', 'UNKNOWN', '曝光记录含无效数值')
        median, cv = float(np.median(shutters)), float(shutters.std() / shutters.mean())
        good = median <= parameters.recording_max_shutter_s and cv <= parameters.recording_max_exposure_cv
        return CheckResult('exposure', 'PASS' if good else 'FAIL',
            '快门和曝光稳定性正常' if good else '快门过慢或曝光变化过大', dict(median_shutter_s=median, exposure_cv=cv))
    except (OSError, ValueError, RuntimeError, struct.error) as error:
        return CheckResult('exposure', 'UNKNOWN', f'无法评估曝光：{error}')


def check_video(path: Path, parameters: SceneParameters | None = None) -> RecordingCheckReport:
    parameters = parameters or load_scene_parameters()
    checks = [check_stream(path)]
    result, imu = check_imu(path, parameters)
    checks.append(result)
    if imu is not None:
        checks.append(check_initial_static(imu, parameters))
    # 已有米制场景地图，不用任务加速度变化重新判断建图尺度可观测性。
    checks.append(check_exposure(path, parameters))
    reasons = [item.message for item in checks if not item.passed]
    return RecordingCheckReport(str(path), not reasons, checks, reasons)


def check_episode(episode, parameters: SceneParameters | None = None) -> EpisodeCheckReport:
    parameters = parameters or episode.scene.parameters
    def check(item):
        arm, video = item
        return arm, check_video(video, parameters)

    items = tuple(episode.recordings.items())
    recording_reports = dict(daemon_map(check, items)) if len(items) > 1 else dict(check(item) for item in items)
    reasons = [f'[{arm}] {message}' for arm, report in recording_reports.items() for message in report.reasons]
    missing = set(episode.expected_arms) - set(episode.recordings.keys())
    if missing:
        reasons.append('录像不完整，缺少：' + ', '.join(sorted(missing)))
    parameters = {key: value for key, value in asdict(parameters).items() if key.startswith('recording_')}
    return EpisodeCheckReport(not reasons, reasons, recording_reports, parameters)
