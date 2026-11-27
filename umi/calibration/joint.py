"""Offline two-lens spatial calibration from the lens-specific B clips and a dual-board clip."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from umi.calibration import acceptance
from umi.calibration.pipeline.inputs import discover_recordings
from umi.calibration.pipeline.joint_observations import observe
from umi.calibration.pipeline.recordings import n_workers
from umi.calibration.solve.board_pnp import read_kb4
from umi.calibration.solve.joint_rig import InertialWindows, aligned_pairs, fit, motion_noise_report, predict_pixels
from umi.calibration.validation.joint_trajectory import compare_c
from umi.config import loader
from umi.core.geometry import imu_from_camera
from umi.devices.calibration_assets import read_basalt
from umi.devices.x3.factory_geometry import Kb4Model
from umi.devices.x3.imu_reader import read_imu_and_meta
from umi.infra.file_store import write_json
from umi.infra.provenance import file_hash
from umi.perception.poster import layout as posters
from umi.scene.inputs.recordings import inspect


def _sources(directory, dual):
    identity = inspect(dual)
    paths, videos, signatures = {}, {}, {'dual': identity}
    grid = None
    for lens in ('front', 'rear'):
        folder = directory / lens
        paths[lens] = folder / 'calibration_result/calibration.json'
        manifest = json.loads(paths[lens].with_name('manifest.json').read_text())
        prov = manifest['provenance']
        if (
            manifest['camera_serial'] != identity['serial']
            or manifest['lens'] != lens
            or manifest['camera_model'] != 'kb4'
            or prov.get('intrinsics_source', 'measured') != 'measured'
            or file_hash(paths[lens]) != manifest['files']['calibration.json']
        ):
            raise ValueError(f'{lens} 初始参数必须为同设备的完整实测 KB4 标定结果')
        if grid is not None and prov['board'] != grid:
            raise ValueError('两镜头的 AprilGrid 定义不同')
        grid = prov['board']
        clips = discover_recordings(folder)
        for stage in ('B', 'C'):
            key = (lens, stage)
            videos[key] = clips[stage]
            signature = inspect(clips[stage])
            if signature['serial'] != identity['serial'] or signature['sha256'] != prov['recordings'][stage]['sha256']:
                raise ValueError(f'{lens} {stage} 原片与标定身份不一致')
            signatures[lens + '_' + stage] = signature
    train = {signatures[k]['sha256'] for k in ('front_B', 'rear_B', 'dual')}
    if train & {signatures[k]['sha256'] for k in ('front_C', 'rear_C')} or identity['sha256'] in {
        signatures[k]['sha256'] for k in ('front_B', 'rear_B')
    }:
        raise ValueError('双板、B 训练集与独立验证 C 原片重复')
    return paths, videos, signatures, grid


def _candidate_document(path, camera_from_imu):
    document = json.loads(path.read_text())
    pose = np.linalg.inv(camera_from_imu)
    x, y, z = pose[:3, 3].tolist()
    qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat().tolist()
    (document.get('value0') or document)['T_imu_cam'][0] = dict(px=x, py=y, pz=z, qx=qx, qy=qy, qz=qz, qw=qw)
    return document


def _pooled_optical_scores(folds):
    n_pairs = sum(f['n_test'] for f in folds)
    pixels, centers = {}, {}
    for name in ('measured', 'candidate'):
        centers[name] = float(
            np.sqrt(sum(f['n_test'] * f['scores'][name]['center_rmse_mm'] ** 2 for f in folds) / n_pairs)
        )
        pixels[name] = {}
        for lens in ('front', 'rear'):
            scores = [f['scores'][name][lens] for f in folds]
            pixels[name][lens] = float(np.sqrt(sum(s['sum_sq_px'] for s in scores) / sum(s['corners'] for s in scores)))
    return dict(cross_pixel_rmse=pixels, cross_front_center_rmse_mm=centers, n_pairs=n_pairs)


def _optical_checks(initial, inertial, dual, offsets, models):
    ids, rear_at_front, front_at_rear = aligned_pairs(dual, offsets)
    front = dual['poses'][ids]
    times = dual['t'][ids]
    folds = []
    edges = np.linspace(times[0], times[-1], 4)
    labels = np.digitize(times, edges[1:-1])
    for k in range(3):
        test = labels == k
        train = (times < edges[k] - 1.0) | (times > edges[k + 1] + 1.0)
        if test.sum() < 3 or train.sum() < 12:
            raise ValueError('双板录像不足以按时间三折验证')
        scores = {}
        for name in ('measured', 'candidate'):
            cams, board, _ = fit(initial, inertial, front[train], rear_at_front[train], optimize=name == 'candidate')
            scores[name] = predict_pixels(
                dual, ids[test], rear_at_front[test], front_at_rear[test], cams, board, models
            )
        folds.append(
            dict(
                test_range_s=[float(times[test].min()), float(times[test].max())],
                n_test=int(test.sum()),
                n_train=int(train.sum()),
                scores=scores,
            )
        )
    cams, board, diagnostic = fit(initial, inertial, front, rear_at_front)
    return cams, board, dict(folds=folds, fit=diagnostic, **_pooled_optical_scores(folds))


def decide(report, checks):
    """Apply the same C safety limits used by umi-calib and paired non-regression checks."""
    failures = []
    scores = report['optical']['cross_pixel_rmse']
    for lens in ('front', 'rear'):
        if scores['candidate'][lens] >= scores['measured'][lens]:
            failures.append(f'{lens} 双板留出角点没有改善')
        c = report['C'][lens]
        if 'error' in c:
            failures.append(f'{lens} C 回放失败')
            continue
        for key in ('ate_se3_rmse_mm', 'chord_1s_rmse_mm'):
            if c['candidate'][key] > c['measured'][key]:
                failures.append(f'{lens} C 的 {key} 变差')
        absolute = acceptance.assess_verification(c['candidate'], '候选', checks)
        c['checks'] = absolute
        if not acceptance.decision(absolute).installable:
            failures.append(f'{lens} C 未通过既有标定验收门槛')
    report['failures'] = failures
    report['status'] = 'rejected' if failures else 'passed_offline'


def run(args):
    directory = args.session.expanduser().resolve()
    dual_video = args.dual_board.expanduser().resolve()
    paths, videos, signatures, grid = _sources(directory, dual_video)
    output = args.out.expanduser().resolve()
    if args.workers is not None and args.workers < 1:
        raise ValueError('--workers 需要正整数')
    workers = n_workers(args.workers)
    layout = posters.load(args.poster_layout) if args.poster_layout else posters.load()
    config = loader.load('calibration')
    board = config.values.board
    output.mkdir(parents=True, exist_ok=False)
    cache = args.cache.expanduser().resolve() if args.cache else output / 'observations'
    cv2.setNumThreads(1)

    def extract(lens, video, signature):
        print(f'提取或读取观测：{lens} {video.name}', flush=True)
        return observe(video, signature['sha256'], lens, paths, grid, layout, board, cache, workers)

    dual = extract('dual', dual_video, signatures['dual'])
    initial, models, offsets, inertial = {}, {}, {}, {}
    for lens, path in paths.items():
        measured = read_basalt(path)
        initial[lens] = np.linalg.inv(imu_from_camera(measured))
        models[lens] = Kb4Model(*read_kb4(path))
        offsets[lens] = int(measured['cam_time_offset_ns'])
        bvideo = videos[lens, 'B']
        data = extract(lens, bvideo, signatures[lens + '_B'])
        inertial[lens] = InertialWindows(data, read_imu_and_meta(bvideo), offsets[lens])

    cams, T_grid_poster, optical = _optical_checks(initial, inertial, dual, offsets, models)
    delta = {}
    for lens in paths:
        candidate = output / (lens + '_candidate.json')
        write_json(candidate, _candidate_document(paths[lens], cams[lens]))
        delta[lens] = dict(
            translation_change_mm=(
                1000 * (np.linalg.inv(cams[lens])[:3, 3] - np.linalg.inv(initial[lens])[:3, 3])
            ).tolist(),
            rotation_change_deg=float(
                np.degrees(Rotation.from_matrix(cams[lens][:3, :3] @ initial[lens][:3, :3].T).magnitude())
            ),
            initial_sha256=file_hash(paths[lens]),
            candidate_sha256=file_hash(candidate),
        )
    report = dict(
        schema_version=1,
        status='pending_C',
        serial=signatures['dual']['serial'],
        sources=signatures,
        grid=grid,
        poster_layout_sha256=layout.asset_hash,
        calibrations=delta,
        B_motion_noise=motion_noise_report(inertial, initial),
        T_grid_poster=T_grid_poster.tolist(),
        T_rear_front=(cams['rear'] @ np.linalg.inv(cams['front'])).tolist(),
        optical=optical,
        C={},
        failures=[],
    )
    write_json(output / 'report.json', report)
    trajectory = config.load_trajectory_validation(tag_size_m=grid['tagSize'])

    def validate(lens):
        video = videos[lens, 'C']
        ref = extract(lens, video, signatures[lens + '_C'])
        return compare_c(
            lens,
            video,
            paths,
            report['serial'],
            trajectory,
            ref,
            cams[lens],
            output / ('C_' + lens),
            motion_noise=report['B_motion_noise']['pooled'] if args.compare_map_fusion else None,
        )

    with ThreadPoolExecutor(2) as pool:
        jobs = {lens: pool.submit(validate, lens) for lens in paths}
        for lens, job in jobs.items():
            try:
                report['C'][lens] = job.result()
            except (ValueError, OSError, RuntimeError) as error:
                report['C'][lens] = dict(error=str(error))
            write_json(output / 'report.json', report)
    decide(report, config.values.checks)
    write_json(output / 'report.json', report)
    print(f'{report["status"]} · {output / "report.json"}', flush=True)
    return 1 if report['failures'] else 0
