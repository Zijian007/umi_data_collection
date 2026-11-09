"""Explicit checkpoint steps. Only JSON measurements are persisted; no runtime class decoding."""
from __future__ import annotations
import time
from software import config
from software.calibration import result_io
from software.calibration.models import Step
from software.calibration.pipeline import runtime

DEPENDENTS = {
    Step.PREPARE_INTRINSICS: (Step.INTRINSICS,), Step.PREPARE_CAMERA_IMU: (Step.CAMERA_IMU,),
    Step.PREPARE_VALIDATION: (Step.TRAJECTORY, Step.SAMPLE), Step.PREPARE_GRAVITY: (Step.GRAVITY,),
    Step.INTRINSICS: (Step.CAMERA_IMU,),
    Step.CAMERA_IMU: (Step.GRAVITY, Step.TRAJECTORY),
}


class StageRunner:
    def __init__(self, output, document):
        self.root = output.resolve()
        self.document = document
        self.timings = {}
        self.document['execution']['timings'] = self.timings

    def invalidate(self, step):
        self.document['measurements'].pop(str(step), None)
        self.document['execution']['steps'].pop(str(step), None)
        for child in DEPENDENTS.get(step, ()):
            self.invalidate(child)

    def run(self, step: Step, identity: dict, operation, *, artifacts=(), reusable=lambda value: True):
        key = config.canonical_hash(identity)
        doc = self.document
        start = time.monotonic()
        entry = doc['execution']['steps'].get(str(step))
        if entry and entry['input_hash'] == key and str(step) in doc['measurements']:
            try:
                for relative, digest in entry['artifacts'].items():
                    if config.file_hash(result_io.local_path(self.root, relative)) != digest:
                        raise ValueError('步骤产物已变化')
                value = result_io.measurement(doc, step)
                if reusable(value):
                    self.timings[str(step)] = dict(status='reused', elapsed_s=round(time.monotonic()-start, 3),
                                                  original_elapsed_s=entry['elapsed_s'])
                    runtime.console.write(f'{step.label}；使用已有结果', state='复用')
                    return value
            except (OSError, ValueError):
                pass
        self.invalidate(step)
        if doc.get('evaluation'):
            doc.setdefault('evaluation_history', []).append(doc['evaluation'])
        doc['evaluation'] = None
        doc['execution']['status'] = 'running'
        result_io.write(self.root, doc)
        runtime.console.write(step.label, state='开始')
        try:
            value = operation()
            elapsed = round(time.monotonic()-start, 3)
            # Validate serializability before committing a successful step.
            digest = config.canonical_hash(value)
            doc['measurements'][str(step)] = dict(data=value, sha256=digest, input_hash=key)
            if reusable(value):
                doc['execution']['steps'][str(step)] = dict(input_hash=key, elapsed_s=elapsed,
                    artifacts={str(path.resolve().relative_to(self.root)): config.file_hash(path) for path in artifacts})
            self.timings[str(step)] = dict(status='computed', elapsed_s=elapsed)
        except BaseException as error:
            doc['execution'].update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                    error=dict(step=str(step), reason=str(error) or type(error).__name__))
            self.timings[str(step)] = dict(status='failed', elapsed_s=round(time.monotonic()-start, 3))
            runtime.console.write(f"{step.label}；耗时 {self.timings[str(step)]['elapsed_s']:.1f} 秒；"
                f"{str(error) or type(error).__name__}", state='中断' if isinstance(error, KeyboardInterrupt) else '失败')
            raise
        finally:
            doc['execution']['timings'] = self.timings
            result_io.write(self.root, doc)
        runtime.console.write(f'{step.label}；耗时 {elapsed:.1f} 秒', state='结束')
        return value
