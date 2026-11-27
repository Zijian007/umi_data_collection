"""视觉运行环境、第三方进程与日志边界。"""

from __future__ import annotations

import subprocess
import traceback
from pathlib import Path

from umi.config import operational_settings as settings
from umi.config import parameter_models
from umi.core.frame import CROP, calibrated_crop_xywh
from umi.infra.orb_install import orb_directory, orb_executable
from umi.infra.paths import PROJECT_ROOT, python_executable
from umi.perception.models import ProcessingError, VisualContext, VisualSettings

REPO = PROJECT_ROOT


def write_error_log(path: Path, error: BaseException) -> None:
    path.write_text(''.join(traceback.format_exception(error)), encoding='utf-8')


def prepare_mask(source: Path, output: Path, size_px: int, *, crop=None) -> Path:
    """按同一视野缩放遮手图；最近邻保留遮挡类别，原始设备资产只读。"""
    import cv2

    mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape[0] != mask.shape[1]:
        raise ValueError(f'遮手图必须是可读取的正方形图像：{source}')
    x, y, size, _ = calibrated_crop_xywh(crop)
    cropped = (x, y, size) != (0, 0, CROP)
    if cropped:
        native = cv2.resize(mask, (CROP, CROP), interpolation=cv2.INTER_NEAREST)
        mask = native[y:y+size, x:x+size]
    if not cropped and mask.shape == (size_px, size_px):
        return source
    resized = cv2.resize(mask, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
    if not cv2.imwrite(str(output), resized):
        raise ValueError(f'无法写入本次运行的遮手图：{output}')
    return output


def visual_settings(environment: dict[str, str], *, lens_settings: dict[str, str], front_mask: Path,
                    output: Path, parameters: parameter_models.TrajectoryParameters,
                    require_slam: bool = True) -> VisualSettings:
    """写显式指定的镜头设置与遮手图；不选择或回退任何镜头标定。"""
    if 'front' not in lens_settings or set(lens_settings)-{'front', 'rear'}:
        raise ValueError('视觉设置需要 front，且只允许 front/rear 镜头')
    if require_slam:
        orb, vocabulary = validate_engines(environment)
    else:
        # Decoding and direct pixel BA do not consume the ORB executable/vocabulary.
        orb = orb_directory(environment)
        vocabulary = Path(environment.get('ORB3_VOCAB') or orb / 'Vocabulary/ORBvoc.txt')
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for lens, text in lens_settings.items():
        paths[lens] = output / f'{lens}.yaml'
        paths[lens].write_text(text, encoding='utf-8')
    mask = prepare_mask(front_mask, output / 'front_mask.png', parameters.video.size_px,
                        crop=parameters.video.crop)
    return VisualSettings(environment=dict(environment),
        tools_python=python_executable(environment, tools=True),
        orb_directory=orb, vocabulary=vocabulary, front_settings=paths['front'],
        rear_settings=paths.get('rear', paths['front']), front_mask=mask)


def run_logged(command: list[str], log: Path, *, stage: str, environment: dict[str, str],
               cwd: Path | None = None, stderr: Path | None = None, check: bool = True,
               failure_code: int | None = None) -> int:
    """所有外部进程走同一日志边界；保留原始 stdout/stderr 和阶段归因。"""
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open('w', encoding='utf-8') as stdout:
            if stderr is None:
                result = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT, env=settings.child_environment(environment), cwd=cwd)
            else:
                with stderr.open('w', encoding='utf-8') as errors:
                    result = subprocess.run(command, stdout=stdout, stderr=errors, env=settings.child_environment(environment), cwd=cwd)
    except OSError as error:
        raise ProcessingError(stage, f'{error}；日志：{log}') from error
    if check and result.returncode:
        raise ProcessingError(stage, f'外部程序退出码 {result.returncode}；日志：{log}',
                                  return_code=failure_code or result.returncode)
    return result.returncode


def engine_command(context: VisualContext, command: list[str], *, timeout_s: int | None = None) -> list[str]:
    """沿用 nice、可选 taskset，以及 SLAM 的外部 timeout。"""
    prefix = ['nice', '-n', '10']
    if context.cores:
        prefix += ['taskset', '-c', context.cores]
    if timeout_s is not None:
        prefix += ['timeout', str(timeout_s)]
    return prefix + command


def validate_engines(environment: dict[str, str]) -> tuple[Path, Path]:
    """Check engine inputs read-only before a workflow can replace old output."""
    try:
        python_executable(environment)
    except ValueError as error:
        raise ProcessingError('准备运行环境', str(error)) from error
    if not environment.get('ORB3_DIR'):
        raise ProcessingError('准备运行环境', '缺少环境变量 ORB3_DIR；请通过 ./umi-scene 启动')
    orb = Path(environment['ORB3_DIR']).resolve()
    vocabulary = Path(environment.get('ORB3_VOCAB') or orb / 'Vocabulary/ORBvoc.txt').resolve()
    for path in (orb_executable(orb), vocabulary):
        if not path.is_file():
            raise ProcessingError('准备运行环境', f'ORB-SLAM3 文件不存在：{path}')
    return orb, vocabulary
