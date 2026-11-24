"""Select calibration results without overwriting another run's evidence."""

from contextlib import contextmanager
from pathlib import Path

from umi.infra import provenance
from umi.infra.file_store import locked
from umi.workflows.calibration import result_io
from umi.workflows.calibration.models import Decision, RunPaths, Step
from umi.workflows.calibration.pipeline import runtime


def _directories(base: Path) -> tuple[Path | None, Path]:
    prefix = f'{base.name}_v'
    existing = {}
    if base.exists() or base.is_symlink():
        existing[1] = base
    for path in base.parent.iterdir() if base.parent.is_dir() else ():
        suffix = path.name[len(prefix):]
        if path.name.startswith(prefix) and suffix.isascii() and suffix.isdecimal() and int(suffix) >= 2:
            number = int(suffix)
            if suffix == f'{number:03d}':
                existing[number] = path
    if not existing:
        return None, base
    last = max(existing)
    return existing[last], base.with_name(f'{prefix}{last + 1:03d}')


def _matching_record(directory: Path, identity: dict) -> dict:
    if directory.is_symlink() or not directory.is_dir():
        return {}
    try:
        doc = result_io.read(directory)
        if not doc:
            return {}
        subject = doc['subject']
        computation = doc['computation']
        if Step.SAMPLE in doc['measurements'] and 'compare_sample' not in computation:
            computation = dict(computation, compare_sample=True)
        saved = dict(operation=doc['operation'], inputs=doc['inputs'], computation=computation,
                     subject={key: subject.get(key) for key in identity['subject']})
        if provenance.canonical_hash(saved) != provenance.canonical_hash(identity):
            return {}
        if doc['execution']['status'] == 'complete':
            Decision(doc['evaluation']['decision'])
        if finished(doc):
            result_io.validate_evidence(directory, doc, complete=True)
            if provenance.canonical_hash(doc['evaluation']['configuration']) != provenance.canonical_hash(
                    identity['computation']['configuration']):
                return {}
            decision = Decision(doc['evaluation']['decision'])
            if not (directory / 'report.txt').is_file():
                return {}
            if subject['lens'] == 'rear' and decision.installable and not (directory / 'rear_calibration.json').is_file():
                return {}
            artifact_sets = [step['artifacts'] for step in doc['execution']['steps'].values()]
            if Step.FACTORY in doc['measurements']:
                artifact_sets.append(result_io.measurement(doc, Step.FACTORY).get('artifacts', {}))
            for artifacts in artifact_sets:
                for relative, digest in artifacts.items():
                    if provenance.file_hash(result_io.local_path(directory, relative)) != digest:
                        return {}
        return doc
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def output_plan(base: Path, identity: dict, *, recompute: bool) -> dict:
    latest, new = _directories(base)
    if latest is not None and not recompute:
        previous = _matching_record(latest, identity)
        if previous:
            return dict(output=str(latest), action=_action(previous))
    return dict(output=str(new), action='new')


def _action(previous: dict) -> str:
    return 'reuse' if finished(previous) else 'resume'


def finished(doc: dict) -> bool:
    evaluation = doc.get('evaluation')
    return (doc.get('execution', {}).get('status') == 'complete' and isinstance(evaluation, dict)
            and evaluation.get('decision') in (Decision.ACCEPTED, Decision.WARNINGS, Decision.REJECTED))


@contextmanager
def open_run(base: Path, identity: dict, *, recompute: bool):
    # Serialize allocation across a series and protect each result against recheck/install.
    with locked(base.parent, name=f'.{base.name}.calibration-series.lock'):
        latest, new = _directories(base)
        if latest is not None and not recompute:
            with locked(latest.parent, name=f'.{latest.name}.calibration.lock'):
                previous = _matching_record(latest, identity)
                if previous:
                    runtime.console.write(f'结果目录：{latest}', state='复用' if _action(previous) == 'reuse' else '继续')
                    paths = RunPaths(latest)
                    if not finished(previous):
                        paths.logs.mkdir(parents=True, exist_ok=True)
                    yield paths, previous
                    return
        paths = RunPaths(new)
        with locked(new.parent, name=f'.{new.name}.calibration.lock'):
            new.mkdir()
            paths.logs.mkdir(parents=True)
            runtime.console.write(f'结果目录：{new}', state='新版本')
            yield paths, {}
