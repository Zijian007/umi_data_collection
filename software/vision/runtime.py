"""视觉运行环境、第三方进程与日志边界。"""
from __future__ import annotations
import subprocess
import traceback
from pathlib import Path
from software import config
from software.environment import python_executable
from software.camera.images import prepare_mask
from software.vision.paths import orb_executable
from software.vision.models import VisualContext, VisualSettings, ProcessingError

HERE = Path(__file__).resolve().parent
TOOLS = HERE / 'tools'


def write_error_log(path: Path, error: BaseException) -> None:
    path.write_text(''.join(traceback.format_exception(error)), encoding='utf-8')

def visual_settings(environment: dict[str, str], *, lens_settings: dict[str, str], front_mask: Path,
                    output: Path, parameters: config.TrajectoryParameters) -> VisualSettings:
    """写显式指定的镜头设置与遮手图；不选择或回退任何镜头标定。"""
    if 'front' not in lens_settings or set(lens_settings)-{'front', 'rear'}:
        raise ValueError('视觉设置需要 front，且只允许 front/rear 镜头')
    orb, vocabulary = validate_engines(environment)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for lens, text in lens_settings.items():
        paths[lens] = output / f'{lens}.yaml'
        paths[lens].write_text(text, encoding='utf-8')
    mask = prepare_mask(front_mask, output / 'front_mask.png', parameters.video.size_px)
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
                result = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT, env=environment, cwd=cwd)
            else:
                with stderr.open('w', encoding='utf-8') as errors:
                    result = subprocess.run(command, stdout=stdout, stderr=errors, env=environment, cwd=cwd)
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
    for name in ('ORB3_DIR',):
        if not environment.get(name):
            raise ProcessingError('准备运行环境', f'缺少环境变量 {name}；请通过 ./umi-traj 启动')
    orb = Path(environment['ORB3_DIR']).resolve()
    vocabulary = Path(environment.get('ORB3_VOCAB') or orb / 'Vocabulary/ORBvoc.txt').resolve()
    for path in (orb_executable(orb), vocabulary):
        if not path.is_file():
            raise ProcessingError('准备运行环境', f'ORB-SLAM3 文件不存在：{path}')
    return orb, vocabulary

