"""第三方 ORB 安装布局，供执行、查看与依赖指纹共同使用。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def orb_executable(orb: Path, name: str = 'x3_mono_slam') -> Path:
    return orb / 'Examples/Monocular-Inertial' / name


def orb_directory(environment) -> Path:
    deps = Path(environment.get('UMI_DEPS') or ROOT / '.deps')
    return Path(environment.get('ORB3_DIR') or deps / 'src/ORB_SLAM3')


def exporter_paths() -> tuple[Path, Path]:
    orb = orb_directory(os.environ)
    return orb_executable(orb, 'umi_export_atlas'), Path(os.environ.get('ORB3_VOCAB') or orb / 'Vocabulary/ORBvoc.txt')
