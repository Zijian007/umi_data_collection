"""独立融合的命令行边界；参数解析、文件读取、求解、结果写入分开。"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['--selftest']:
        from software.trajectory.algorithms.fusion_factors import selftest
        return 0 if selftest() else 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rear_frames', type=Path)
    parser.add_argument('front_frames', type=Path)
    parser.add_argument('imu_csv', type=Path)
    parser.add_argument('offset_ns', type=int)
    parser.add_argument('rear_quaternion', help='xyzw，以逗号分隔')
    parser.add_argument('output', type=Path)
    parser.add_argument('--lever_prior', required=True, help='环境镜头杆臂 xyz，以逗号分隔，单位米')
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--parameters', required=True, type=Path)
    parser.add_argument('--times', type=Path)
    parser.add_argument('--poster', type=Path, help='冻结的海报观测输入清单')
    parser.add_argument('--rear-calibration', type=Path, help='冻结的正式 rear 标定清单')
    parser.add_argument('--refine-initialization', action='store_true',
                        help='实验：固定重力模长联合细化；保留原始门限与采样复核')
    # argparse treats a CSV vector beginning with '-' as an option. Prefix that
    # positional value while parsing, and use --option=value for a signed lever.
    if len(argv) >= 6 and ',' in argv[4]:
        argv[4] = 'vector:'+argv[4]
    if '--lever_prior' in argv:
        index = argv.index('--lever_prior')
        if index+1 < len(argv) and ',' in argv[index+1]:
            argv[index:index+2] = ['--lever_prior='+argv[index+1]]
    args = parser.parse_args(argv)
    from software import config
    from software.trajectory.algorithms.fusion import solve
    from software.trajectory.algorithms.fusion_alignment import AlignmentError
    from software.trajectory.algorithms.fusion_data import load_inputs, write_solution

    try:
        parameters = config.decode_parameters('trajectory', json.loads(args.parameters.read_text()))
        inputs = load_inputs(front_frames=args.front_frames, rear_frames=args.rear_frames, imu_csv=args.imu_csv,
            camera_json=args.calibration, offset_ns=args.offset_ns,
            rear_quaternion=[float(x) for x in args.rear_quaternion.removeprefix('vector:').split(',')],
            rear_lever=[float(x) for x in args.lever_prior.split(',')], use_rear=parameters.input.mode == 'dual', times_csv=args.times)
        if args.poster is not None:
            from dataclasses import replace
            from software.trajectory.pipeline.poster_assets import load
            inputs = replace(inputs,poster=load(args.poster))
        if args.rear_calibration is not None:
            from dataclasses import replace
            from software.trajectory.pipeline.calibration_assets import load_frozen_rear
            inputs = replace(inputs, rear_camera=load_frozen_rear(args.rear_calibration))
        result = solve(inputs, parameters, refine_initialization=args.refine_initialization)
        write_solution(result, args.output)
        print(json.dumps(result.evidence))
    except AlignmentError as error:
        args.output.mkdir(parents=True, exist_ok=True)
        diagnostic = args.output / 'alignment_failure.json'
        diagnostic.write_text(json.dumps(dict(status='failed', code='NO_USABLE_LENS',
            **error.diagnostics), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        parser.exit(2, f'{error}；诊断：{diagnostic}\n')
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, str(error)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
