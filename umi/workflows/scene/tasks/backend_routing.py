"""Task estimator adapters and whole-task routing within the scene pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import shutil

from av.error import InvalidDataError

from umi.infra.file_store import write_json
from umi.store.models import TaskInputs, ArmExecutionResult
from umi.workflows.scene.tasks import scene_map, poster_independent
from umi.workflows.scene.maps import repository as map_repository
from umi.estimation.fixed_map.fusion import SceneFusionQualityError
from umi.workflows.scene.errors import MapUnavailableError, PosterVioQualityError, fallback_record
from umi.workflows.scene.tasks import recording_intervals as tag_trimming
from umi.config.collection_schema import apply_scene_map_poster_policy, uses_edge_trim


AUTO_CANDIDATES_WITH_MAP = ('scene_map', 'poster_vio', 'poster_independent')
AUTO_CANDIDATES_WITHOUT_MAP = ('poster_vio', 'poster_independent')


@dataclass(frozen=True)
class MapSelection:
    directory: Path | None = None
    manifest: dict | None = None
    failure: dict | None = None

    def __post_init__(self):
        if (self.directory is None) != (self.manifest is None):
            raise ValueError('地图目录与清单必须同时提供')
        if self.manifest is not None and self.failure is not None:
            raise ValueError('已选地图不能同时声明地图失败')

    @classmethod
    def available(cls, directory: Path, manifest: dict) -> MapSelection:
        if directory is None or manifest is None:
            raise ValueError('可用地图需要目录与清单')
        return cls(directory=directory, manifest=manifest)

    @classmethod
    def without_map(cls, failure: dict | None = None) -> MapSelection:
        return cls(failure=failure)


def select_map(episode, requested_map_version, *, batch_map=None, dry_run=False, cores=''):
    mode = episode.scene.reconstruction_mode
    if mode in ('poster_independent', 'poster_vio'):
        return episode, MapSelection.without_map()
    # Bind the preferred map before trying mapless backends when auto has mapping clips.
    auto_builds_map = (mode == 'auto' and not requested_map_version and episode.session.mapping
                      and episode.session.map_version is None)
    if mode == 'scene_map' or auto_builds_map:
        episode, directory, manifest = (batch_map.bind(episode) if batch_map is not None else
            map_repository.bind_map(episode, requested_map_version, dry_run, cores=cores))
        return episode, MapSelection.available(directory, manifest)
    # Auto without mapping recordings only consumes an existing map.
    try:
        if (episode.map_selection or {}).get('status') in ('unavailable', 'pending'):
            raise MapUnavailableError(episode.map_selection['reason'])
        version = requested_map_version or episode.map_version
        manifest_sha256 = (episode.session.map_manifest_sha256
                  if version == episode.session.map_version else None)
        directory, manifest = map_repository.load_published(episode.scene, version=version,
                                                         manifest_sha256=manifest_sha256, allow_import=True)
        episode = replace(episode, session=replace(episode.session,
            scene=replace(episode.scene, mode=manifest['mode'])))
        return episode, MapSelection.available(directory, manifest)
    except MapUnavailableError as error:
        # This boundary only reads map assets; camera/recording validation is later.
        episode = replace(episode, session=replace(episode.session,
            scene=replace(episode.scene, mode='shared_front')))
        fallback_details = fallback_record(error)
        fallback_details['reason'] = f'场景地图不可用：{error}'
        return episode, MapSelection.without_map(fallback_details)


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
    from umi.workflows.scene.tasks import poster_vio

    return poster_vio.execute_task(inputs, fallback=selection.failure)


# These adapters return complete task tracks, unlike poster.FUSION_METHODS,
# whose callbacks only produce initialization poses within scene_map.
ESTIMATORS = {'scene_map': _run_scene_map, 'poster_independent': _run_poster_independent,
              'poster_vio': _run_poster_vio}


def execute(inputs: TaskInputs, selection: MapSelection) -> ArmExecutionResult:
    """Run one explicit backend or the ordered whole-task auto chain.

    Explicit modes execute exactly one backend.  Auto mode uses the map only
    when selection succeeded, then tries poster_vio and poster_independent in
    order; only structured candidate-quality failures continue the chain.
    """
    scene = inputs.episode.scene
    mode = scene.reconstruction_mode
    if mode != 'auto':
        trimmed, report = tag_trimming.prepare(inputs) if uses_edge_trim(scene, mode) else (inputs, None)
        result = ESTIMATORS[mode](trimmed, selection)
        if report is not None:
            result.coordinate_record['recording_trim'] = report
        return result
    return _AutoRoute(inputs, selection).run()


# 允许 auto 换下一个后端的失败类型：(异常类型, 记录类别, 只对哪个后端生效)。
# 输入、标定、引擎和程序错误不在此列，不能作为改用下一种重建策略的依据。
_CANDIDATE_FAILURES = (
    # The map was present, but its usable visual support was too weak.
    (SceneFusionQualityError, 'scene_quality', None),
    (PosterVioQualityError, 'poster_vio_quality', None),
    # A candidate can be unavailable because the prepared recording cannot be
    # decoded by its feature frontend; the independent poster backend may
    # still have a valid engine path.
    (InvalidDataError, 'poster_vio_unavailable', 'poster_vio'),
)


class _AutoRoute:
    """auto：按顺序尝试候选后端，每个候选在独立目录运行，成功者的产物移入任务目录。"""

    def __init__(self, inputs: TaskInputs, selection: MapSelection):
        self.inputs, self.selection = inputs, selection
        # A published map is still the preferred source.  It is skipped only when
        # map selection failed before reconstruction; a low-quality map attempt is
        # represented by SceneFusionQualityError and falls through the same chain.
        self.candidates = (AUTO_CANDIDATES_WITH_MAP if selection.manifest is not None
                           else AUTO_CANDIDATES_WITHOUT_MAP)
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
        self.message('自动后端路由：' + ' → '.join(self.candidates) +
                     ('；场景地图可用' if self.selection.manifest is not None else '；无可用场景地图'))
        self.attempts_root.mkdir(exist_ok=True)
        for index, backend in enumerate(self.candidates):
            self.message(f'后端尝试 {index + 1}/{len(self.candidates)}：{backend}')
            attempt_dir = self.attempts_root / f'{index:02d}_{backend}'
            attempt_dir.mkdir()
            next_backend = self.candidates[index + 1] if index + 1 < len(self.candidates) else None
            try:
                result = self._run_candidate(backend, attempt_dir)
            except Exception as error:
                category = next((category for kind, category, only in _CANDIDATE_FAILURES
                                 if isinstance(error, kind) and only in (None, backend)), None)
                if category is None:
                    self._record_fatal(backend, error)
                    raise
                self._record_failure(backend, error, category=category, next_backend=next_backend)
                continue
            return self._adopt(backend, attempt_dir, result)
        # The final candidate is expected to raise its own actionable error.  This
        # guard protects the invariant if a future candidate list is edited.
        raise RuntimeError('自动重建没有可尝试的后端')

    def _run_candidate(self, backend: str, attempt_dir: Path) -> ArmExecutionResult:
        inputs, trim_report = self.inputs, None
        if uses_edge_trim(inputs.episode.scene, backend):
            if self._trimmed is None:
                self._trimmed = tag_trimming.prepare(inputs)
            inputs, trim_report = self._trimmed
        selection = self.selection if backend == 'scene_map' else MapSelection.without_map(self.failure)
        result = ESTIMATORS[backend](replace(inputs, directory=attempt_dir), selection)
        if trim_report is not None:
            result.coordinate_record['recording_trim'] = trim_report
        return result

    def _write_attempts(self, selected_backend: str | None) -> None:
        write_json(self.attempts_root / 'attempts.json', dict(
            requested_mode='auto', selected_backend=selected_backend, attempts=self.attempts))

    def _record_failure(self, backend, error, *, category, next_backend=None) -> None:
        details = fallback_record(error, category=category)
        self.attempts.append(dict(backend=backend, status='failed', **details))
        self.failure = dict(self.failure or details)
        # Keep the selection passed to the next candidate immutable with
        # respect to later successes; the root manifest records the full list.
        self.failure['attempts'] = list(self.attempts)
        self._write_attempts(None)
        if next_backend is None:
            self.message(f'后端失败：{backend}；类别={category}；原因={details["reason"]}')
        else:
            self.message(f'后端切换：{backend} → {next_backend}；类别={category}；原因={details["reason"]}')

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
