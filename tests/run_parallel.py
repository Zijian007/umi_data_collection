#!/usr/bin/env python3
"""Run independent unittest modules in parallel and preserve useful failures."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _run(module: str, targets: list[str]) -> tuple[str, int, str]:
    environment = dict(os.environ)
    tests_dir = str(Path(__file__).resolve().parent)
    python_path = environment.get('PYTHONPATH', '')
    environment['PYTHONPATH'] = tests_dir if not python_path else tests_dir + os.pathsep + python_path
    result = subprocess.run(
        [sys.executable, '-m', 'unittest', '-q', *targets],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return module, result.returncode, result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, required=True)
    parser.add_argument('targets', nargs='+')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs 必须为正整数')

    by_module = defaultdict(list)
    for target in args.targets:
        module = Path(target).stem if target.endswith('.py') else target.partition('.')[0]
        by_module[module].append(target)

    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(_run, module, targets) for module, targets in by_module.items()]
        for future in as_completed(futures):
            module, code, output = future.result()
            if code:
                failures.append((module, output))

    if failures:
        print(f'并行测试失败：{len(failures)} 个模块', file=sys.stderr)
        for module, output in sorted(failures):
            print(f'\n===== {module} =====', file=sys.stderr)
            print(output, file=sys.stderr, end='')
        return 1
    print(f'并行测试通过：{len(by_module)} 个模块，worker={args.jobs}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
