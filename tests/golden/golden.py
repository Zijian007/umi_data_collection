"""Golden regression harness for the umi/ refactor.

Runs the retained batch ``scenes/test_shelf/20261120_002`` through every
trajectory backend and compares published trajectories between two code
versions.  Raw recordings are hard-linked (not symlinked) so ingest path
checks see each variant as an independent batch.

    python tests/golden/golden.py prepare
    python tests/golden/golden.py run      [--backend B ...]   # uses ./umi-scene
    python tests/golden/golden.py snapshot OUT [--backend B ...]
    python tests/golden/golden.py compare  BASE NEW [--position-tol M] [--rotation-tol DEG]

Snapshots are written outside the repository (default ``~/umi_golden``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / 'scenes/test_shelf/20261120_002'
BACKENDS = ('poster_vio', 'poster_independent', 'scene_map')
SNAPSHOT_FILES = ('episode.json', 'bimanual.csv', 'poster_vio/quality.json', 'reconstruction.json')
ARMS = ('left', 'right')


def variant(backend: str) -> Path:
    return PROJECT / f'scenes/golden_{backend}/{SOURCE.name}'


def _collection_for(backend: str) -> str:
    text = (SOURCE / 'collection.yaml').read_text(encoding='utf-8')
    marker = '  backend: poster_vio'
    if marker not in text:
        raise SystemExit(f'{SOURCE}/collection.yaml 中找不到 "{marker}"，无法生成后端变体')
    return text.replace(marker, f'  backend: {backend}', 1)


def prepare(backends: list[str]) -> None:
    for backend in backends:
        target = variant(backend)
        if target.exists():
            shutil.rmtree(target.parent)
        for raw in sorted(SOURCE.rglob('*.insv')):
            relative = raw.relative_to(SOURCE)
            if relative.parts[0].startswith('.'):
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(raw, destination)
        (target / 'collection.yaml').write_text(_collection_for(backend), encoding='utf-8')
        print(f'prepared {target.relative_to(PROJECT)}')


def run(backends: list[str], extra: list[str]) -> int:
    failures = 0
    for backend in backends:
        command = [str(PROJECT / 'umi-scene'), 'run', str(variant(backend)), '--force', *extra]
        print('$', ' '.join(command), flush=True)
        failures += subprocess.call(command, cwd=PROJECT) != 0
    return failures


def _published_run(batch: Path) -> Path:
    results = batch / 'take_001/results'
    if not results.exists():
        raise SystemExit(f'{results} 不存在；先运行 run')
    return results.resolve()


def snapshot(out: Path, backends: list[str], source: Path | None) -> None:
    for backend in backends:
        batch = source if source is not None else variant(backend)
        run_dir = _published_run(batch)
        target = out / backend
        if target.exists():
            shutil.rmtree(target)
        for name in SNAPSHOT_FILES:
            if (run_dir / name).exists():
                (target / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(run_dir / name, target / name)
        for arm in ARMS:
            for name in ('trajectory.csv', 'details/frames.csv'):
                if (run_dir / arm / name).exists():
                    (target / arm / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(run_dir / arm / name, target / arm / name)
        (target / 'SOURCE').write_text(f'{run_dir}\n', encoding='utf-8')
        print(f'snapshot {backend} <- {run_dir}')


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _angle_deg(a: list[float], b: list[float]) -> float:
    # Chord form stays accurate for nearly identical rotations, unlike acos(dot).
    if sum(x * y for x, y in zip(a, b)) < 0:
        b = [-y for y in b]
    chord = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return math.degrees(4 * math.asin(min(1.0, chord / 2)))


def _compare_trajectory(base: Path, new: Path) -> dict:
    a, b = _rows(base), _rows(new)
    report = dict(rows=(len(a), len(b)))
    count = min(len(a), len(b))
    position, rotation = [], []
    for left, right in zip(a[:count], b[:count]):
        if abs(float(left['t_episode_s']) - float(right['t_episode_s'])) > 1e-6:
            report['time_mismatch'] = True
            break
        position.append(math.dist([float(left[k]) for k in 'xyz'], [float(right[k]) for k in 'xyz']))
        rotation.append(_angle_deg([float(left[k]) for k in ('qx', 'qy', 'qz', 'qw')],
                                   [float(right[k]) for k in ('qx', 'qy', 'qz', 'qw')]))
    if position:
        report.update(position_max_m=max(position),
                      position_rms_m=math.sqrt(sum(p * p for p in position) / len(position)),
                      rotation_max_deg=max(rotation))
    return report


def _summary(episode: dict) -> dict:
    scene = episode.get('scene') or {}
    quality = scene.get('quality') or {}
    return dict(status=episode.get('status'), backend=scene.get('reconstruction_backend'),
                scope=scene.get('scope'), rows=episode.get('rows'),
                common_interval_s=episode.get('common_interval_s'),
                quality_status=quality.get('status'),
                reprojection_rms_px=quality.get('reprojection_rms_px'),
                arms={arm: {k: result.get(k) for k in ('rows', 'path_length_m', 'constraint_fraction')}
                      for arm, result in (episode.get('arm_results') or {}).items()})


def compare(base: Path, new: Path, position_tol: float, rotation_tol: float) -> int:
    failures = 0
    for backend in sorted(p.name for p in base.iterdir() if p.is_dir()):
        print(f'== {backend}')
        if not (new / backend).is_dir():
            print('  MISSING in new snapshot'); failures += 1
            continue
        summary_a = _summary(json.loads((base / backend / 'episode.json').read_text()))
        summary_b = _summary(json.loads((new / backend / 'episode.json').read_text()))
        for key in summary_a:
            if summary_a[key] != summary_b[key]:
                print(f'  {key}: {summary_a[key]!r} -> {summary_b[key]!r}')
        if (summary_a['status'], summary_a['backend'], summary_a['rows']) != \
           (summary_b['status'], summary_b['backend'], summary_b['rows']):
            failures += 1
        for arm in ARMS:
            path_a, path_b = base / backend / arm / 'trajectory.csv', new / backend / arm / 'trajectory.csv'
            if not path_a.exists():
                continue
            if not path_b.exists():
                print(f'  {arm}: trajectory missing'); failures += 1
                continue
            report = _compare_trajectory(path_a, path_b)
            bad = (report['rows'][0] != report['rows'][1] or report.get('time_mismatch')
                   or report.get('position_max_m', 0) > position_tol
                   or report.get('rotation_max_deg', 0) > rotation_tol)
            failures += bool(bad)
            print(f'  {arm}: {"FAIL" if bad else "ok"} ' + json.dumps(report))
    print('RESULT:', 'FAIL' if failures else 'PASS')
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'snapshot'):
        command = sub.add_parser(name)
        command.add_argument('--backend', action='append', choices=BACKENDS)
        if name == 'snapshot':
            command.add_argument('out', type=Path)
            command.add_argument('--source', type=Path, help='直接对某个批次目录做快照（单后端）')
        if name == 'run':
            command.add_argument('extra', nargs=argparse.REMAINDER)
    command = sub.add_parser('compare')
    command.add_argument('base', type=Path)
    command.add_argument('new', type=Path)
    command.add_argument('--position-tol', type=float, default=1e-4, help='米；默认 0.1 mm')
    command.add_argument('--rotation-tol', type=float, default=0.01, help='度')
    args = parser.parse_args()
    backends = list(getattr(args, 'backend', None) or BACKENDS)
    if args.command == 'prepare':
        prepare(backends)
    elif args.command == 'run':
        return run(backends, args.extra)
    elif args.command == 'snapshot':
        snapshot(args.out.expanduser(), backends, args.source)
    else:
        return compare(args.base.expanduser(), args.new.expanduser(), args.position_tol, args.rotation_tol)
    return 0


if __name__ == '__main__':
    sys.exit(main())
