"""分派固定后端与自动回退。"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from av.error import InvalidDataError

from umi.config.collection_schema import apply_scene_map_poster_policy, uses_edge_trim
from umi.infra.file_store import write_json
from umi.reconstruction.backends import PosterIndependentQualityError, PosterVioQualityError
from umi.reconstruction.backends.poster_independent import run as poster_independent
from umi.reconstruction.backends.scene_map import run as scene_map
from umi.reconstruction.estimation.fixed_map.fusion import SceneFusionQualityError
from umi.scene import measurements, publication
from umi.scene.errors import MapUnavailableError, fallback_record
from umi.scene.inputs import recording_intervals as tag_trimming
from umi.scene.maps.binding import MapSelection
from umi.store.models import ArmExecutionResult, BackendServices, TaskInputs


def backend_services() -> BackendServices:
    return BackendServices(
        prepare_detection_group=measurements.prepare_detection_group,
        prepare_task_detections=measurements.prepare_task_detections,
        write_task_results=publication.write_task_results,
    )


def _run_scene_map(inputs: TaskInputs, selection: MapSelection) -> ArmExecutionResult:
    manifest = selection.manifest
    if manifest is None:
        raise ValueError('scene_map 估计器需要已验证的场景地图')
    episode = inputs.episode
    coordinate_record = dict(name=episode.scene.name, session=episode.session.name,
                      version=manifest['version'], frame_id=manifest['frame_id'])
    # Explicit scene_map keeps the legacy single-arm local fallback.  Auto
    # mode disables it so a weak map retries the complete task coherently.
    allow_local_fallback = episode.scene.reconstruction_mode == 'scene_map'
    scene = apply_scene_map_poster_policy(episode.scene)
    if scene is not episode.scene:
        inputs = replace(inputs, episode=replace(episode, session=replace(episode.session, scene=scene)))
    return scene_map.execute_task(
        inputs, map_dir=selection.directory, map_manifest=manifest, coordinate_record=coordinate_record,
        allow_local_fallback=allow_local_fallback)


def _run_poster_independent(inputs: TaskInputs, selection: MapSelection) -> ArmExecutionResult:
    return poster_independent.execute_task(inputs, fallback=selection.failure)


def _run_poster_vio(inputs: TaskInputs, selection: MapSelection) -> ArmExecutionResult:
    if selection.manifest is not None:
        raise ValueError('poster_vio 不接受场景地图输入')
    from umi.reconstruction.backends.poster_vio import run as poster_vio

    return poster_vio.execute_task(inputs, fallback=selection.failure)


# These adapters return complete task tracks, unlike poster.FUSION_METHODS,
# whose callbacks only produce initialization poses within scene_map.
ESTIMATORS = {'scene_map': _run_scene_map, 'poster_independent': _run_poster_independent,
              'poster_vio': _run_poster_vio}


def execute(inputs: TaskInputs, selection: MapSelection, *, resolve_map=None, prepare_independent=None) -> ArmExecutionResult:
    """Run one explicit backend or the ordered whole-task auto chain.

    Explicit modes execute exactly one backend. Auto follows the configured
    order and resolves a map when the scene_map candidate is reached.
    """
    scene = inputs.episode.scene
    mode = scene.reconstruction_mode
    if mode != 'auto':
        trimmed, report = tag_trimming.prepare(inputs) if uses_edge_trim(scene, mode) else (inputs, None)
        result = ESTIMATORS[mode](trimmed, selection)
        if report is not None:
            result.coordinate_record['recording_trim'] = report
        return result
    return _AutoRoute(inputs, selection, resolve_map=resolve_map, prepare_independent=prepare_independent).run()


# 允许 auto 换下一个后端的失败类型：(异常类型, 记录类别, 只对哪个后端生效)。
# 输入、标定、引擎和程序错误不在此列，不能作为改用下一种重建策略的依据。
_CANDIDATE_FAILURES = (
    # The map was present, but its usable visual support was too weak.
    (SceneFusionQualityError, 'scene_quality', None),
    (PosterVioQualityError, 'poster_vio_quality', None),
    (PosterIndependentQualityError, 'poster_independent_quality', 'poster_independent'),
    # A candidate can be unavailable because the prepared recording cannot be
    # decoded by its feature frontend; the independent poster backend may
    # still have a valid engine path.
    (InvalidDataError, 'poster_vio_unavailable', 'poster_vio'),
)


class _AutoRoute:
    """auto：按顺序尝试候选后端，每个候选在独立目录运行，成功者的产物移入任务目录。"""

    def __init__(self, inputs: TaskInputs, selection: MapSelection, *, resolve_map, prepare_independent):
        self.inputs, self.selection = inputs, selection
        self.resolve_map = resolve_map
        self.prepare_independent = prepare_independent
        self.candidates = tuple(backend for backend in inputs.episode.scene.auto_order
                                if backend != 'scene_map' or resolve_map is not None or selection.manifest is not None)
        self.attempts_root = inputs.directory / 'reconstruction_attempts'
        self.attempts: list[dict] = []
        self.failure = dict(selection.failure) if selection.failure else None
        self._trimmed = None

    def message(self, text: str) -> None:
        console = getattr(self.inputs, 'console', None)
        if console is not None:
            recording = getattr(getattr(self.inputs, 'episode', None), 'root', None)
            console.write(text, recording=getattr(recording, 'name', None))

    def run(self) -> ArmExecutionResult:
        self.message('自动后端路由：' + ' → '.join(self.candidates))
        self.attempts_root.mkdir(exist_ok=True)
        for index, backend in enumerate(self.candidates):
            self.message(f'后端尝试 {index + 1}/{len(self.candidates)}：{backend}')
            attempt_dir = self.attempts_root / f'{index:02d}_{backend}'
            attempt_dir.mkdir()
            next_backend = self.candidates[index + 1] if index + 1 < len(self.candidates) else None
            try:
                result = self._run_candidate(backend, attempt_dir)
            except Exception as error:
                category = (error.category if backend == 'scene_map' and isinstance(error, MapUnavailableError)
                            else next((category for kind, category, only in _CANDIDATE_FAILURES
                                       if isinstance(error, kind) and only in (None, backend)), None))
                if category is None:
                    self._record_fatal(backend, error)
                    raise
                self._record_failure(backend, error, category=category, next_backend=next_backend)
                if next_backend is None:
                    raise
                continue
            if result is None:
                if (not self.attempts or self.attempts[-1]['backend'] != backend
                        or self.attempts[-1]['status'] != 'skipped'):
                    raise RuntimeError(f'{backend} 未返回重建结果')
                details = self.attempts[-1]
                self.message(f'跳过 {backend}，{details["reason"]}')
                if next_backend is None:
                    raise MapUnavailableError(details['reason'])
                continue
            return self._adopt(backend, attempt_dir, result)
        raise MapUnavailableError('自动重建没有可尝试的后端')

    def _run_candidate(self, backend: str, attempt_dir: Path) -> ArmExecutionResult | None:
        inputs, trim_report = self.inputs, None
        selection = self.selection
        if backend == 'scene_map' and self.resolve_map is not None:
            inputs, selection = self.resolve_map(inputs)
            if selection.manifest is None:
                self._record_details(backend, selection.failure, status='skipped')
                return None
        if backend == 'poster_independent' and self.prepare_independent is not None:
            self.prepare_independent()
        if uses_edge_trim(inputs.episode.scene, backend):
            if self._trimmed is None:
                self._trimmed = tag_trimming.prepare(inputs)
            trimmed, trim_report = self._trimmed
            inputs = replace(inputs, recordings=trimmed.recordings, detections=trimmed.detections,
                             poster_pnp_candidates=trimmed.poster_pnp_candidates)
        if backend != 'scene_map':
            selection = MapSelection.without_map(self.failure)
        result = ESTIMATORS[backend](replace(inputs, directory=attempt_dir), selection)
        if trim_report is not None:
            result.coordinate_record['recording_trim'] = trim_report
        return result

    def _write_attempts(self, selected_backend: str | None) -> None:
        write_json(self.attempts_root / 'attempts.json', dict(
            requested_mode='auto', selected_backend=selected_backend, attempts=self.attempts))

    def _record_failure(self, backend, error, *, category, next_backend=None) -> None:
        details = fallback_record(error, category=category)
        self._record_details(backend, details, status='failed')
        if next_backend is None:
            self.message(f'后端失败：{backend}；类别={category}；原因={details["reason"]}')
        else:
            self.message(f'后端切换：{backend} → {next_backend}；类别={category}；原因={details["reason"]}')

    def _record_details(self, backend, details, *, status):
        self.attempts.append(dict(backend=backend, status=status, **details))
        self.failure = dict(self.failure or details, attempts=list(self.attempts))
        self._write_attempts(None)

    def _record_fatal(self, backend, error) -> None:
        # Still record the terminal candidate for failed-run diagnostics.
        self.attempts.append(dict(backend=backend, status='failed', category='fatal',
                                  reason=str(error), error_type=type(error).__name__))
        self._write_attempts(None)
        self.message(f'后端终止：{backend}；类别=fatal；原因={error}')

    def _adopt(self, backend: str, attempt_dir: Path, result: ArmExecutionResult) -> ArmExecutionResult:
        self.attempts.append(dict(backend=backend, status='succeeded'))
        for entry in attempt_dir.iterdir():
            target = self.inputs.directory / entry.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            entry.rename(target)
        attempt_dir.rmdir()
        self._write_attempts(backend)
        result.coordinate_record['attempts'] = self.attempts
        if self.failure:
            result.coordinate_record['fallback'] = dict(self.failure, attempts=self.attempts)
        self.message(f'后端选定：{backend}；尝试次数={len(self.attempts)}')
        return result
