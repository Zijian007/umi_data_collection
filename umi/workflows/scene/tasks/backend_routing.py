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

    # A published map is still the preferred source.  It is skipped only when
    # map selection failed before reconstruction; a low-quality map attempt is
    # represented by SceneFusionQualityError and falls through the same chain.
    candidates = (AUTO_CANDIDATES_WITH_MAP if selection.manifest is not None
                  else AUTO_CANDIDATES_WITHOUT_MAP)
    console = getattr(inputs, 'console', None)
    recording = getattr(getattr(inputs, 'episode', None), 'root', None)
    recording = getattr(recording, 'name', None)

    def route_message(message):
        if console is not None:
            console.write(message, recording=recording)

    route_message('自动后端路由：' + ' → '.join(candidates) +
                  ('；场景地图可用' if selection.manifest is not None else '；无可用场景地图'))
    attempts_root = inputs.directory / 'reconstruction_attempts'
    attempts_root.mkdir(exist_ok=True)
    attempts = []
    failure = dict(selection.failure) if selection.failure else None

    def record_failure(backend, error, *, category, next_backend=None):
        nonlocal failure
        details = fallback_record(error, category=category)
        attempts.append(dict(backend=backend, status='failed', **details))
        failure = dict(failure or details)
        # Keep the selection passed to the next candidate immutable with
        # respect to later successes; the root manifest records the full list.
        failure['attempts'] = list(attempts)
        write_json(attempts_root / 'attempts.json', dict(
            requested_mode='auto', selected_backend=None, attempts=attempts))
        if next_backend is None:
            route_message(f'后端失败：{backend}；类别={category}；原因={details["reason"]}')
        else:
            route_message(f'后端切换：{backend} → {next_backend}；类别={category}；原因={details["reason"]}')

    trimmed_inputs, trim_report = None, None
    for index, backend in enumerate(candidates):
        route_message(f'后端尝试 {index + 1}/{len(candidates)}：{backend}')
        attempt_dir = attempts_root / f'{index:02d}_{backend}'
        attempt_dir.mkdir()
        candidate_selection = (selection if backend == 'scene_map' else
                               MapSelection.without_map(failure))
        try:
            candidate_inputs, candidate_trim_report = inputs, None
            if uses_edge_trim(scene, backend):
                if trimmed_inputs is None:
                    trimmed_inputs, trim_report = tag_trimming.prepare(inputs)
                candidate_inputs, candidate_trim_report = trimmed_inputs, trim_report
            result = ESTIMATORS[backend](replace(candidate_inputs, directory=attempt_dir), candidate_selection)
            if candidate_trim_report is not None:
                result.coordinate_record['recording_trim'] = candidate_trim_report
        except SceneFusionQualityError as error:
            # The map was present, but its usable visual support was too weak.
            record_failure(backend, error, category='scene_quality',
                           next_backend=candidates[index + 1] if index + 1 < len(candidates) else None)
            continue
        except PosterVioQualityError as error:
            record_failure(backend, error, category='poster_vio_quality',
                           next_backend=candidates[index + 1] if index + 1 < len(candidates) else None)
            continue
        except InvalidDataError as error:
            # A candidate can be unavailable because the prepared recording
            # cannot be decoded by its feature frontend; the independent
            # poster backend may still have a valid engine path.
            if backend != 'poster_vio':
                raise
            record_failure(backend, error, category='poster_vio_unavailable',
                           next_backend=candidates[index + 1] if index + 1 < len(candidates) else None)
            continue
        except Exception as error:
            # Input, calibration, engine and programming errors are not
            # evidence that the next reconstruction strategy should be tried.
            # Still record the terminal candidate for failed-run diagnostics.
            attempts.append(dict(backend=backend, status='failed', category='fatal',
                                 reason=str(error), error_type=type(error).__name__))
            write_json(attempts_root / 'attempts.json', dict(
                requested_mode='auto', selected_backend=None, attempts=attempts))
            route_message(f'后端终止：{backend}；类别=fatal；原因={error}')
            raise

        attempts.append(dict(backend=backend, status='succeeded'))
        for entry in attempt_dir.iterdir():
            target = inputs.directory / entry.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            entry.rename(target)
        attempt_dir.rmdir()
        write_json(attempts_root / 'attempts.json', dict(
            requested_mode='auto', selected_backend=backend, attempts=attempts))
        result.coordinate_record['attempts'] = attempts
        if failure:
            result.coordinate_record['fallback'] = dict(failure, attempts=attempts)
        route_message(f'后端选定：{backend}；尝试次数={len(attempts)}')
        return result

    # The final candidate is expected to raise its own actionable error.  This
    # guard protects the invariant if a future candidate list is edited.
    raise RuntimeError('自动重建没有可尝试的后端')
