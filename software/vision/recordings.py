"""原片 → 时间戳、IMU 和显式选择的镜头视频。"""
from __future__ import annotations
from pathlib import Path
from software.camera.x3 import extraction
from software.vision import runtime
from software.vision.models import RecordingData, RecordingPaths, LensVideo, VisualContext, ProcessingError

def prepare(video: Path, paths: RecordingPaths, *, allow_irregular: bool = False) -> RecordingData:
    manifest = extraction.extract(video, paths.dataset)
    imu_csv = paths.dataset / 'mav0/imu0/data.csv'
    frames = manifest["frames"]
    if frames["regular"]:
        return RecordingData(manifest, imu_csv)
    if not allow_irregular:
        raise extraction.Refused(
            "帧时间不规则（最大偏差 %.1f ms，%d 处间隔异常）；默认拒绝处理。"
            "如需继续，可加 --allow-irregular（该处理路径尚未验证）"
            % (frames["max_deviation_from_29p97_grid_ms"], frames["gaps_over_1p5x_median"]))
    times_csv = paths.debug / "times.csv"
    write_times_csv(paths.dataset / "mav0", times_csv)
    return RecordingData(manifest, imu_csv, times_csv)


def prepare_lens_videos(video: Path, context: VisualContext, *, lenses: tuple[str, ...]) -> tuple[LensVideo, ...]:
    """解码和裁切原片一次。所有建图/定位重试共用这些视频，不重复处理 INSV。"""
    paths, settings = context.paths, context.settings
    paths.lens_videos.mkdir(parents=True, exist_ok=True)
    if not lenses or len(set(lenses)) != len(lenses) or set(lenses) - {'front', 'rear'}:
        raise ProcessingError('准备镜头视频', '镜头列表必须由不重复的 front/rear 组成')
    prepared = []
    for lens in lenses:
        stage = f'准备镜头视频（{lens}）'
        output = paths.lens_videos / f'{lens}.mp4'
        log = paths.logs / f'{lens}_video.log'
        runtime.run_logged(
            [settings.tools_python, str(runtime.TOOLS / 'make_lens_video.py'), str(video),
             str(paths.dataset / 'mav0'), str(output), 'right' if lens == 'front' else 'left',
             '--size-px', str(context.parameters.video.size_px),
             '--crf', str(context.parameters.video.crf), '--preset', context.parameters.video.preset],
            log, stage=stage, environment=settings.environment, failure_code=3,
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise ProcessingError(stage, f'未生成镜头视频；日志：{log}', return_code=3)
        prepared.append(LensVideo(
            lens, output, settings.front_settings if lens == 'front' else settings.rear_settings,
            settings.front_mask if lens == 'front' else None,
        ))
    return tuple(prepared)


def write_times_csv(mav0: Path, path: Path) -> None:
    rows = [line.split(",")[0] for line in (mav0 / "cam0" / "data.csv").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]
    path.write_text("t_canonical_s\n" + "".join("%.9f\n" % (int(row) / 1e9) for row in rows), encoding="utf-8")

