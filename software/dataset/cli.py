"""umi-dataset：将已验收任务导出为训练数据中间包。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ruamel.yaml.error import YAMLError

from software.dataset.export import build


def main(argv=None):
    parser = argparse.ArgumentParser(prog='umi-dataset', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('build', help='导出指令、同步视频与实测 TCP；当前缺夹爪通道')
    export.add_argument('paths', nargs='+', type=Path, help='场景、套件、批次或任务目录')
    export.add_argument('--out', required=True, type=Path)
    export.add_argument('--fps', type=float, default=30.)
    export.add_argument('--max-gap-s', type=float, default=.25, help='最大轨迹插值间隔，超出时分段，默认 0.25 s')
    args = parser.parse_args(argv)
    try:
        result = build(args.paths, args.out, fps=args.fps, max_gap_s=args.max_gap_s)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, YAMLError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
