"""Task output coordinates and the optional map used to estimate them."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from umi.infra import provenance

if TYPE_CHECKING:
    from umi.store.models import Episode


@dataclass(frozen=True)
class MapReference:
    version: str | None
    frame_id: str | None


@dataclass(frozen=True)
class OutputCoordinates:
    frame_id: str | None
    scope: Literal['shared_scene', 'task_poster', 'local']
    backend: str | None
    source_map: MapReference | None
    T_output_map: list[list[float]] | None

    @classmethod
    def from_record(cls, record: dict, *, source_map: dict | None = None) -> OutputCoordinates:
        scope = record.get('scope')
        if scope not in ('local', 'task_poster'):
            scope = 'shared_scene'
        reference = None
        if scope == 'shared_scene' or (scope == 'task_poster' and 'T_board_scene' in record):
            source = source_map if source_map is not None else record
            reference = MapReference(source.get('version'), source.get('frame_id'))
        return cls(record.get('frame_id'), scope, record.get('reconstruction_backend'),
                   reference, record.get('T_board_scene'))

    @property
    def uses_map(self) -> bool:
        return self.source_map is not None


def is_local_scene(record: dict) -> bool:
    return OutputCoordinates.from_record(record).scope == 'local'


def is_poster_scene(record: dict) -> bool:
    return OutputCoordinates.from_record(record).scope == 'task_poster'


def uses_scene_map(record: dict) -> bool:
    return OutputCoordinates.from_record(record).uses_map


def validate_poster_placement(matrix) -> None:
    if (not isinstance(matrix, list) or len(matrix) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
            or any(type(x) not in (float, int) or not math.isfinite(x) for row in matrix for x in row)
            or matrix[3] != [0, 0, 0, 1]):
        raise ValueError('海报坐标变换无效')
    x, y, z = [row[:3] for row in matrix[:3]]
    cross = [x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0]]
    if (any(abs(math.hypot(*axis)-1) > 1e-6 for axis in (x, y, z))
            or abs(sum(a*b for a, b in zip(x, y))) > 1e-6
            or math.dist(cross, z) > 1e-6):
        raise ValueError('海报坐标旋转必须为右手正交矩阵')


def task_poster_frame(episode: Episode, layout_hash: str) -> dict:
    recordings = {arm: ref['sha256'] for arm, ref in episode.recording_refs.items()}
    identity = provenance.canonical_hash(dict(recordings=recordings, layout=layout_hash))
    return dict(frame_id=f'poster:{identity}', scope='task_poster', axes='board_layout', units='metres')
