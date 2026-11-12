"""Probe the actual binary, rather than inferring geometry from source files."""

from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess

from software.vision.paths import orb_executable


def require_forward_kb4(environment=None):
    environment = dict(os.environ if environment is None else environment)
    directory = environment.get('ORB3_DIR')
    if not directory:
        raise ValueError('工厂内参需要 ORB3_DIR 指向具有前向 KB4 能力的引擎')
    try:
        result = subprocess.run([str(orb_executable(Path(directory))), '--capabilities'], env=environment,
            capture_output=True, text=True, timeout=10, check=True)
        value = json.loads(result.stdout)
        if (not isinstance(value, dict) or value.get('schema_version') != 1 or value.get('kb4_geometry') != 'forward_v1'
                or value.get('maximum_ray_angle_degrees') != 85):
            raise ValueError('前向 KB4 能力不匹配')
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ValueError('所选 ORB 引擎不支持已验证的 <=85° 前向 KB4；'
                         '请用 UMI_ORB_KB4_FORWARD=1 重新构建项目引擎') from error
    return value
