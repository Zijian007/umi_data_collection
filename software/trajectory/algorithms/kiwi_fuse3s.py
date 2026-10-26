"""旧脚本路径的兼容入口；新调用使用 fusion_cli.py，求解器位于 fusion.py。"""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from software.trajectory.algorithms.fusion_cli import main

if __name__ == '__main__':
    raise SystemExit(main())
