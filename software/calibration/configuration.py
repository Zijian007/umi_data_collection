"""标定配置预览；不解析录像或访问设备档案。"""
from __future__ import annotations

from software import settings

from argparse import Namespace
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from software import config
from software.calibration.models import CalibrationRequest

if TYPE_CHECKING:
    from software.scene.models import SceneParameters


def rear_validation_parameters() -> SceneParameters:
    """环境镜头 C 独立验证，不采用任务海报或建图尺度修正。"""
    from software.scene.management.inputs import load_scene_parameters
    return replace(load_scene_parameters(), tag_fusion_enabled=False,
                   tag_mapping_enabled=False, tag_mapping_scale_enabled=False)


def effective_report(arguments: Namespace, request: CalibrationRequest | None = None) -> dict:
    calibration = config.load('calibration', arguments.config)
    trajectory = calibration.load_trajectory_validation(tag_size_m=arguments.tag_size_mm / 1000.)
    effective = dict(calibration=calibration.snapshot(), trajectory_validation=trajectory.snapshot(),
                     tag_size_mm=arguments.tag_size_mm)
    if request is not None and request.lens == 'rear':
        effective['scene_parameters'] = asdict(rear_validation_parameters())
    if request is not None:
        effective['output'] = str(request.output_directory)
    return dict(program='umi-calib', operation=arguments.cmd, scope='configuration_only',
        precedence=['project_defaults', '--config', 'C_validation_overrides'],
        inputs={key: str(value) if isinstance(value, Path) else value
                for key, value in vars(arguments).items() if key in ('folder', 'video', 'out')},
        execution={key: value for key, value in vars(arguments).items()
                   if key in ('cores', 'workers', 'force', 'resume', 'keep_work', 'install', 'compare_sample')},
        selection=dict(lens=request.lens if request is not None else 'front',
                       intrinsics_source=getattr(arguments, 'intrinsics_source', 'installed_version')),
        effective=effective, operational=settings.execution_report(cores=arguments.cores or '', workers=arguments.workers, calibration=True))
