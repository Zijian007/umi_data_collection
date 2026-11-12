"""Automatic factory-MEI comparison; diagnostic evidence, never acceptance policy.

Metadata lens1 is the working lens; lens0 is the environment lens. Both lenses
get an optical KB4 surrogate. Board comparisons use the selected lens and label
the current calibration as measured or factory. The workflow runs this for the
working lens in both modes and for the environment lens in factory mode.
1440 is the calibration crop; 1920 is labelled KB4 extrapolation. Native lens/IMU extrinsics
are preserved as metadata and are not used for this optical comparison.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from scipy.optimize import least_squares
import av
import cv2
import numpy as np

from software import provenance as artifact_identity
from software.calibration.algorithms import aprilgrid_detect, board_pnp, factory_kb4
from software.calibration.pipeline.recordings import LUMA_FORMATS
from software.camera.x3.factory import Kb4Model, WORKING_LENS_INDEX, ENVIRONMENT_LENS_INDEX, LENS_ASSIGNMENT, lens_model, read_factory
from software.camera.x3.frame import SRC_H, SRC_W
from software.storage import atomic_file, write_json


METHOD_VERSION = 4
MAX_BOARD_FRAMES = 24
GRID_SIDE = 65


def statistics(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, rms=None, median=None, p95=None, p99=None, maximum=None)
    return dict(count=len(values), rms=float(np.sqrt(np.mean(values**2))),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)), p99=float(np.percentile(values, 99)),
                maximum=float(values.max()))


def sources(recordings, serial):
    result = {role: read_factory(path) for role, path in recordings.items()}
    if any(item['serial'] != serial for item in result.values()):
        raise ValueError('工厂参数来源的设备身份与本次标定不一致')
    return dict(recordings=result, consistent=len({v['raw'] for v in result.values()}) == 1)


def reusable(output: Path, result: dict) -> bool:
    if (result.get('method_version') != METHOD_VERSION
            or result.get('status') not in ('ok', 'partial') or not result.get('artifacts')):
        return False
    if result.get('board_decode_error') or any(fit['status'] == 'failed' for fit in result.get('fits', {}).values()):
        return False
    try:
        for relative, digest in result['artifacts'].items():
            path = (output/relative).resolve()
            if not path.is_relative_to(output.resolve()) or artifact_identity.file_hash(path) != digest:
                return False
    except (OSError, ValueError):
        return False
    return True


def _image(path, pixels):
    with atomic_file(path) as temporary:
        ok, encoded = cv2.imencode('.png', pixels)
        if not ok:
            raise RuntimeError('工厂对照诊断图编码失败')
        temporary.write_bytes(encoded.tobytes())


def _heatmap(path, angles, size):
    """Shared 0–5 degree colour scale; gray is outside the common valid domain."""
    field = np.asarray(angles).reshape(GRID_SIDE, GRID_SIDE)
    colour = cv2.applyColorMap(np.uint8(np.nan_to_num(np.clip(field/5, 0, 1))*255), cv2.COLORMAP_TURBO)
    colour[~np.isfinite(field)] = [80, 80, 80]
    canvas = cv2.resize(colour, (520, 520), interpolation=cv2.INTER_NEAREST)
    if size == 1920:
        cv2.rectangle(canvas, (65, 65), (455, 455), (255, 255, 255), 1)
    cv2.putText(canvas, 'Ray difference: 0..5 deg; gray=invalid', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    _image(path, canvas)


def ray_comparison(kb4, mei, size, image_path):
    axis = np.linspace(0, size-1, GRID_SIDE)
    x, y = np.meshgrid(axis, axis)
    pixels = np.column_stack((x.ravel(), y.ravel()))
    # Nominal circular sensor region, not a claim of measured peripheral accuracy.
    nominal_disc = np.linalg.norm(pixels-(size-1)/2, axis=1) <= SRC_H/2
    measured = np.max(np.abs(pixels-(size-1)/2), axis=1) <= 719.5
    a, b = kb4.unproject(pixels), mei.unproject(pixels)
    valid = nominal_disc & np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    angles = np.full(len(pixels), np.nan)
    angles[valid] = np.degrees(np.arccos(np.clip(np.sum(a[valid]*b[valid], axis=1), -1, 1)))
    delta_px = np.linalg.norm(kb4.project(b)-pixels, axis=1)
    _heatmap(image_path, angles, size)
    regions = {'measured_crop': nominal_disc & measured}
    if size == 1920:
        regions['outside_measured_crop'] = nominal_disc & ~measured
    return dict(nominal_disc_points=int(nominal_disc.sum()), common_valid_points=int(valid.sum()),
                invalid_mei_points=int((nominal_disc & ~np.isfinite(b).all(axis=1)).sum()),
                invalid_kb4_points=int((nominal_disc & ~np.isfinite(a).all(axis=1)).sum()),
                regions={name: dict(sampled_points=int(mask.sum()),
                    angular_degrees=statistics(angles[mask & valid]),
                    kb4_reprojection_pixels=statistics(delta_px[mask & valid])) for name, mask in regions.items()})


def fit_board(model, detections, grid):
    """Fit on 2/3 of visible tags; evaluate the other tags without refitting."""
    ids = sorted(set(detections) & set(grid))
    held_ids = ids[::3]
    fitted_ids = [i for i in ids if i not in held_ids]
    if len(fitted_ids) < 4 or len(held_ids) < 2:
        raise ValueError('留出验证至少需要 6 个可见标签（拟合至少 4 个）')
    board_object_points = np.concatenate([grid[i] for i in fitted_ids])
    pixels = np.concatenate([detections[i] for i in fitted_ids])
    rays = model.unproject(pixels)
    if not np.isfinite(rays).all() or np.any(rays[:, 2] <= 1e-3):
        raise ValueError('拟合角点不在模型的前向有效区域')
    normal = rays[:, :2]/rays[:, 2:3]
    _, rotations, translations, _ = cv2.solvePnPGeneric(board_object_points, normal, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    if not len(rotations):
        raise ValueError('无法初始化标定板位姿')

    def project(parameters, points):
        rotation = cv2.Rodrigues(parameters[:3])[0]
        return model.project(points@rotation.T+parameters[3:])

    def residual(parameters):
        return np.nan_to_num(project(parameters, board_object_points)-pixels, nan=1e6, posinf=1e6, neginf=-1e6).ravel()

    candidates = [np.r_[r.ravel(), t.ravel()] for r, t in zip(rotations, translations)]
    initial = min(candidates, key=lambda p: float(np.linalg.norm(residual(p))))
    fit = least_squares(residual, initial, loss='soft_l1', f_scale=1., max_nfev=100)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('标定板位姿拟合未收敛')
    held_obj = np.concatenate([grid[i] for i in held_ids])
    held_pixels = np.concatenate([detections[i] for i in held_ids])
    predicted = project(fit.x, held_obj)
    held_rays = model.unproject(held_pixels)
    camera_points = held_obj@cv2.Rodrigues(fit.x[:3])[0].T+fit.x[3:]
    if (not np.isfinite(predicted).all() or not np.isfinite(held_rays).all()
            or np.any(np.sum(camera_points*held_rays, axis=1) <= 0)):
        raise ValueError('留出角点不在模型的有效投影区域')
    return dict(errors=np.linalg.norm(predicted-held_pixels, axis=1),
                pixels=held_pixels, predicted=predicted,
                fit_errors=np.linalg.norm(project(fit.x, board_object_points)-pixels, axis=1))


def board_comparison(observations, models, grid, size):
    results = {name: {} for name in models}
    failures = {name: {} for name in models}
    for index, detected in observations.items():
        for name, model in models.items():
            try:
                results[name][index] = fit_board(model, detected, grid)
            except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
                reason = str(error)
                failures[name][reason] = failures[name].get(reason, 0)+1
    common = sorted(set.intersection(*(set(r) for r in results.values())))
    summary = dict(status='ok' if common else 'insufficient', sampled_frames=len(observations),
                   common_frames=len(common), holdout='every third visible tag, sorted by id', models={})
    for name, frames in results.items():
        errors = np.concatenate([frames[k]['errors'] for k in common]) if common else np.array([])
        pixels = np.concatenate([frames[k]['pixels'] for k in common]) if common else np.empty((0, 2))
        inside = np.max(np.abs(pixels-(size-1)/2), axis=1) <= 719.5
        regions = dict(measured_crop=statistics(errors[inside]))
        if size == 1920:
            regions['outside_measured_crop'] = statistics(errors[~inside])
        summary['models'][name] = dict(successful_frames=len(frames), failures=failures[name],
            heldout_pixels=statistics(errors), regions=regions)
    return summary, results, common


def _observations(video, found, board, lens='front'):
    eligible = [k for k in sorted(found['dets']) if len(found['dets'][k]) >= 6]
    selected = [eligible[i] for i in np.unique(np.linspace(0, len(eligible)-1, min(MAX_BOARD_FRAMES, len(eligible))).astype(int))] if eligible else []
    observations = {1440: {}, 1920: {}}
    images, times, skipped = {}, {}, []
    detector = aprilgrid_detect.GridDetector()
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height) != (SRC_W, SRC_H):
            raise ValueError('工厂对照仅支持单路 3840×1920 原片')
        for index in selected:
            target = float(found['video_time_s'][index])
            container.seek(max(0, int(target/float(stream.time_base))), stream=stream, backward=True)
            frame = next((f for f in container.decode(stream) if float(f.pts*f.time_base) >= target-1e-6), None)
            if frame is None or abs(float(frame.pts*frame.time_base)-target) > 1e-5:
                skipped.append(dict(frame=index, reason='解码帧与板检测的捕获时间不一致'))
                continue
            if frame.format.name not in LUMA_FORMATS:
                raise ValueError('工厂对照需要与标定提取一致的 8-bit Y 平面')
            plane = frame.planes[0]
            full = np.frombuffer(plane, np.uint8).reshape(plane.height, plane.line_size)[:1920, (1920 if lens == 'front' else 0):(3840 if lens == 'front' else 1920)].copy()
            observations[1440][index] = found['dets'][index]
            observations[1920][index] = aprilgrid_detect.detect_up(full, detector, board.detection_upscale)
            if len(images) < 4:
                images[index] = full
            times[index] = target
    return observations, images, times, skipped


def _overlay(path, gray, results, index, size):
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colours = ((0, 220, 0), (255, 120, 0), (0, 80, 255))
    for row, ((name, by_frame), colour) in enumerate(zip(results.items(), colours)):
        if index in by_frame:
            for measured, predicted in zip(by_frame[index]['pixels'], by_frame[index]['predicted']):
                a, b = tuple(np.rint(measured).astype(int)), tuple(np.rint(predicted).astype(int))
                cv2.circle(image, a, 3, (255, 255, 255), 1)
                cv2.drawMarker(image, b, colour, cv2.MARKER_CROSS, 8, 1)
        cv2.putText(image, name, (12, 28+row*28), cv2.FONT_HERSHEY_SIMPLEX, .7, colour, 2)
    if size == 1920:
        cv2.rectangle(image, (240, 240), (1679, 1679), (255, 255, 255), 2)
    _image(path, cv2.resize(image, (960, 960), interpolation=cv2.INTER_AREA))


def _fit_heatmap(path, mei, kb4, size):
    side = 129
    pixels = factory_kb4.grid(size, side, .37)
    error = np.linalg.norm(kb4.project(mei.unproject(pixels))-pixels, axis=1)
    valid = factory_kb4.in_disc(pixels, size) & np.isfinite(error)
    colour = cv2.applyColorMap(np.uint8(np.nan_to_num(np.clip(error/3, 0, 1))*255), cv2.COLORMAP_TURBO)
    colour = colour.reshape(-1, 3)
    colour[~valid] = [80, 80, 80]
    canvas = cv2.resize(colour.reshape(side, side, 3), (640, 640), interpolation=cv2.INTER_NEAREST)
    if size == 1920:
        cv2.rectangle(canvas, (80, 80), (560, 560), (255, 255, 255), 1)
    cv2.putText(canvas, f'MEI -> fitted KB4: 0..3 px at {size}', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    cv2.putText(canvas, 'gray=outside nominal disc / invalid', (10, 44), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    _image(path, canvas)


def _fit_lenses(native, output, artifacts):
    """Persist both optical candidates even if subsequent video decoding fails."""
    source = native['recordings']['C']
    fits = {}
    for lens in source['lenses']:
        name = f"lens{lens['index']}"
        try:
            fit = factory_kb4.fit(lens_model(lens, 1440))
            fitted = Kb4Model(**fit['kb4'])
            fit['views']['1920'] = factory_kb4.validate(lens_model(lens, 1920), fitted.full_view(), 1920)
            fit['views']['1920']['role'] = 'extrapolation_only_not_fitted'
            for size in (1440, 1920):
                path = output/'factory'/f'{size}_{name}_kb4_fit_pixels.png'
                _fit_heatmap(path, lens_model(lens, size), fitted if size == 1440 else fitted.full_view(), size)
                artifacts[str(path.relative_to(output))] = artifact_identity.file_hash(path)
        except (ValueError, RuntimeError, np.linalg.LinAlgError, cv2.error) as error:
            fit = dict(status='failed', reason=str(error), diagnostic_only=True, acceptance='not_evaluated')
            for size in (1440, 1920):
                path = output/'factory'/f'{size}_{name}_kb4_fit_pixels.png'
                path.unlink(missing_ok=True)
                artifacts.pop(str(path.relative_to(output)), None)
        fit.update(lens_index=lens['index'], lens_role=lens['role'])
        fits[name] = fit
    path = output/'factory/kb4_fit.json'
    write_json(path, dict(kind='umi_factory_kb4_fits', method_version=METHOD_VERSION,
        fit_version=factory_kb4.FIT_VERSION, native=source,
        recordings_consistent=native['consistent'], parameters_source_recording='C',
        coordinate_mapping='factory continuous scale to 1920, then centred 1440 crop; no optical-frame rotation',
        acceptance='not_evaluated', diagnostic_only=True, fits=fits))
    artifacts[str(path.relative_to(output))] = artifact_identity.file_hash(path)
    return fits


def compare(video, native, calibration, found, grid, board, output, *, lens='front', intrinsics_source='measured'):
    output = Path(output)
    directory = output/'factory'
    directory.mkdir(parents=True, exist_ok=True)
    kb4 = Kb4Model(*board_pnp.read_kb4(calibration))
    factory = native['recordings']['C']
    lens_index = WORKING_LENS_INDEX if lens == 'front' else ENVIRONMENT_LENS_INDEX
    lens_role = 'working' if lens == 'front' else 'environment'
    working_lens = next(item for item in factory['lenses'] if item['index'] == lens_index)
    artifacts = {}
    fits = _fit_lenses(native, output, artifacts)
    decode_error = None
    try:
        observations, images, times, skipped = _observations(video, found, board, lens)
    except (OSError, ValueError, av.FFmpegError) as error:
        decode_error = str(error)
        observations, images, times = {1440: {}, 1920: {}}, {}, {}
        skipped = [dict(reason=f'C 实拍对照未完成：{error}')]
    points = board_pnp.grid_points_3d(grid)
    report = dict(kind='umi_insv_factory_comparison', method_version=METHOD_VERSION,
        status='ok', source='insv_factory', native=native,
        calibration_sha256=artifact_identity.file_hash(calibration),
        mapping_status='fixed', lens_assignment=dict(LENS_ASSIGNMENT), compared_lens=lens_role, intrinsics_source=intrinsics_source,
        assignment_basis=factory['assignment_basis'],
        scope='optical diagnostics only; no acceptance or camera-IMU calibration',
        mapping=f'factory continuous scaling to 1920; centred 1440 crop; lens{lens_index} {lens_role}',
        board_sample_limit=MAX_BOARD_FRAMES, sampled_video_times_s={str(k): v for k, v in times.items()}, skipped_frames=skipped,
        fits=fits, board_decode_error=decode_error, views={}, artifacts=artifacts)
    for size in (1440, 1920):
        measured = kb4 if size == 1440 else kb4.full_view()
        models = {('kb4' if intrinsics_source == 'measured' else 'selected_factory_kb4'): measured}
        view = dict(role='measured_crop' if size == 1440 else 'exploratory_with_kb4_extrapolation',
                    kb4=asdict(measured), candidates={})
        name = f'mei_lens{lens_index}'
        model = lens_model(working_lens, size)
        models[name] = model
        working_fit = fits[f'lens{lens_index}']
        if working_fit['status'] == 'ok':
            fitted = Kb4Model(**working_fit['kb4'])
            models['factory_fitted_kb4'] = fitted if size == 1440 else fitted.full_view()
        path = directory/f'{size}_{name}_rays.png'
        view['candidates'][name] = dict(lens_index=lens_index, lens_role=lens_role,
            parameters=asdict(model), rays=ray_comparison(measured, model, size, path))
        report['artifacts'][str(path.relative_to(output))] = artifact_identity.file_hash(path)
        view['board'], results, common = board_comparison(observations[size], models, points, size)
        if not common:
            report['status'] = 'partial'
        example = next((k for k in common if k in images), None)
        if example is not None:
            path = directory/f'{size}_heldout_overlay.png'
            gray = images[example] if size == 1920 else images[example][240:1680, 240:1680]
            _overlay(path, gray, results, example, size)
            view['overlay_time_s'] = times[example]
            report['artifacts'][str(path.relative_to(output))] = artifact_identity.file_hash(path)
        report['views'][str(size)] = view
    if not native['consistent'] or skipped or any(fit['status'] != 'ok' for fit in fits.values()):
        report['status'] = 'partial'
    path = directory/'comparison.json'
    write_json(path, report)
    report['artifacts'][str(path.relative_to(output))] = artifact_identity.file_hash(path)
    # Remove the obsolete v1 alternative-lens plots after successfully writing v2.
    for size in (1440, 1920):
        if lens_index != 0:
            (directory/f'{size}_mei_lens0_rays.png').unlink(missing_ok=True)
    return report
