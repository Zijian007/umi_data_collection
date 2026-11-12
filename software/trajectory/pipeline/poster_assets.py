"""冻结已选海报资产、缓存原始角点并读取融合输入；不选择镜头版本。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable
import json
import shutil

import numpy as np

from software import provenance
from software.parameters import EnvironmentTagParameters
from software.storage import write_json
from software.trajectory.calibration.assets import read_basalt
from software.vision import tag_board, tags


if TYPE_CHECKING:
    from software.trajectory.algorithms.fusion_poster import PosterInputs


@dataclass(frozen=True)
class PosterAssetHashes:
    layout_sha256: str
    rear_sha256: str


def prepare(video: Path, directory: Path, layout: Path, rear: Path, parameters: EnvironmentTagParameters,
            *, expected: PosterAssetHashes | None = None, progress: Callable[[str], None] | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    frozen_layout, frozen_rear = directory / 'layout.json', directory / 'rear.json'
    shutil.copyfile(layout, frozen_layout)
    shutil.copyfile(rear, frozen_rear)
    board = tag_board.load(frozen_layout)
    if expected is not None and (board.asset_hash != expected.layout_sha256
                                 or provenance.file_hash(frozen_rear) != expected.rear_sha256):
        raise ValueError('海报布局或 rear 标定在预检后发生变化')
    detection_settings = tags.DetectionSettings(
        border_bits=board.border_bits, ids=board.ids, dictionary=board.dictionary,
        minimum_side_px=parameters.tag_minimum_side_px)
    detection_report = tags.cached(video, directory / 'detection', detection_settings, layout=board, progress=progress)
    manifest_path = directory / 'input.json'
    write_json(manifest_path, dict(
        schema_version=1, layout='layout.json', rear='rear.json', observations='detection/observations.npz',
        assets={name: provenance.file_hash(directory / name)
                for name in ('layout.json', 'rear.json', 'detection/observations.npz')},
        parameters=asdict(parameters), detection_identity=detection_report['identity_hash']))
    return manifest_path


def load(manifest_path: Path) -> PosterInputs:
    from software.trajectory.algorithms.fusion_poster import PosterInputs
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema_version'] != 1:
        raise ValueError('未知海报输入版本')
    snapshot_directory = manifest_path.parent.resolve()
    for asset_name, expected_hash in manifest['assets'].items():
        asset_path = (snapshot_directory / asset_name).resolve()
        if not asset_path.is_relative_to(snapshot_directory) or provenance.file_hash(asset_path) != expected_hash:
            raise ValueError('冻结海报输入哈希不匹配')
    for asset_kind in ('layout', 'rear', 'observations'):
        if manifest[asset_kind] not in manifest['assets']:
            raise ValueError('海报输入未被哈希覆盖')
    with np.load(snapshot_directory / manifest['observations'], allow_pickle=False) as observation_archive:
        observations = tags.TagObservations(**{
            field: observation_archive[field] for field in tags.TagObservations.__dataclass_fields__})
    observations.validate()
    return PosterInputs(tag_board.load(snapshot_directory / manifest['layout']), observations,
                        read_basalt(snapshot_directory / manifest['rear']))
