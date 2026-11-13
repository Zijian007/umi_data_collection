"""本次场景算法覆盖：一次读取、显式传递，不修改场景/批次/套件声明。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
from typing import TYPE_CHECKING

from software import config, provenance

if TYPE_CHECKING:
    from software.scene.models import SceneParameters


@dataclass(frozen=True)
class SceneOverrides:
    path: Path
    sha256: str
    values_json: str

    @classmethod
    def load(cls, path: Path) -> SceneOverrides:
        from software.scene.management.inputs import read_yaml
        path = path.expanduser().resolve()
        declaration = read_yaml(path)
        return cls.from_snapshot(dict(path=str(path), sha256=provenance.file_hash(path), values=declaration))

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> SceneOverrides:
        """验收/导出使用结果冻结的覆盖内容，不依赖临时实验文件仍然存在。"""
        from software.scene.management.inputs import require_keys, load_scene_parameters
        require_keys(snapshot, {'path', 'sha256', 'values'}, {'path', 'sha256', 'values'}, '场景覆盖记录')
        declaration = snapshot['values']
        if not isinstance(declaration, dict) or declaration.keys() - {'trajectory', 'parameters'}:
            raise ValueError('场景 --config 只允许 trajectory 和 parameters；设备/布局/地图请写入 kit、scene 或 batch')
        if any(not isinstance(section, dict) for section in declaration.values()):
            raise ValueError('场景 --config 的 trajectory / parameters 必须为参数映射')
        # Check names/structure now; validate numerical relationships after merging
        # with the selected scene (a partial overlay can depend on its values).
        config.merge_overrides(asdict(config.load('trajectory').values), declaration.get('trajectory', {}))
        config.merge_overrides(asdict(load_scene_parameters()), declaration.get('parameters', {}), 'parameters.')
        return cls(Path(snapshot['path']), snapshot['sha256'], json.dumps(declaration, sort_keys=True))

    @property
    def values(self) -> dict:
        return json.loads(self.values_json)

    def snapshot(self) -> dict:
        return dict(path=str(self.path), sha256=self.sha256, values=self.values)


def apply_overrides(trajectory: config.Configuration, parameters: SceneParameters,
                    overrides: SceneOverrides | None) -> tuple[config.Configuration, SceneParameters]:
    if overrides is None:
        return trajectory, parameters
    from software.scene.management.inputs import load_scene_parameters
    declared = overrides.values
    trajectory_values = config.merge_overrides(asdict(trajectory.values), declared.get('trajectory', {}))
    effective_trajectory = config.load('trajectory', inline=trajectory_values)
    effective_trajectory = replace(effective_trajectory, sources=(*trajectory.sources, str(overrides.path)))
    parameter_values = config.merge_overrides(asdict(parameters), declared.get('parameters', {}), 'parameters.')
    return effective_trajectory, load_scene_parameters(overrides=parameter_values)


def effective_report(paths: list[Path], *, overrides: SceneOverrides | None = None,
                     intrinsics_source: str | None = None) -> list[dict]:
    """读取声明与算法配置；不扫描原片、不生成批次快照或地图。"""
    from software.scene import layout
    from software.scene.management import batches, episodes, inputs, processing

    targets = set()
    for requested_path in paths:
        requested_path = requested_path.expanduser().resolve()
        if not requested_path.exists():
            raise ValueError(f'配置展示路径不存在：{requested_path}')
        if requested_path.is_file() and requested_path.name not in {'scene.yaml', 'batch.yaml', 'manifest.yaml', 'episode.yaml', 'task.yaml'}:
            raise ValueError('请指定场景、批次、任务目录或对应 YAML 清单')
        directory = layout.owner(requested_path) if requested_path.is_file() else requested_path
        if directory.name == layout.INTERNAL:
            directory = directory.parent
        if ((directory.parent/'batch.yaml').is_file()
                or layout.manifest(directory, 'episode.yaml').is_file()):
            targets.add(directory.parent)
        else:
            discovered = processing.configured_batches([requested_path])
            discovered.extend(layout.owner(path).parent for path in episodes.episode_paths([requested_path], exclude=discovered))
            targets.update(discovered or [directory])
    reports = []
    for directory in sorted(targets):
        batch_path = directory/'batch.yaml'
        selection = None
        declarations = []
        frozen_mapping = {}
        map_version = None
        if batch_path.is_file():
            declaration = inputs.read_yaml(batch_path)
            selection = batches.select_kit(declaration, intrinsics_source=intrinsics_source)
            map_version = declaration.get('map_version')
            scene_path = directory.parent/'scene.yaml'
            declarations = [str(batch_path), str(inputs.KITS/declaration['kit']/'kit.yaml')]
        elif layout.manifest(directory, 'manifest.yaml').is_file():
            manifest = inputs.read_yaml(layout.manifest(directory, 'manifest.yaml'))
            frozen_mapping = inputs.mapping_records(manifest.get('mapping', {}))
            map_version = (manifest.get('map') or {}).get('version')
            # A frozen batch does not switch calibration versions. The requested
            # source is verified against installed assets during input checks.
            kit = inputs.validate_kit(manifest['kit'])
            selection = batches.KitSelection(kit, inputs.read_arms(manifest.get('expected_arms', list(kit['arms'])), kit['arms']), manifest.get('intrinsics_source'))
            scene_path = directory.parent/'scene.yaml'
            declarations = [str(layout.manifest(directory, 'manifest.yaml'))]
        else:
            scene_path = directory/'scene.yaml'
        scene = inputs.read_scene(scene_path, overrides=overrides)
        if frozen_mapping:
            scene = replace(scene, mode='shared_front' if len(frozen_mapping) == 1 else 'dual',
                            reference_arm=manifest.get('mapping_reference_arm', next(iter(frozen_mapping))))
        reports.append(dict(
            target=str(directory), sources=dict(
                defaults=[str(inputs.ROOT/'config/trajectory.yaml'), str(inputs.ROOT/'config/scene_parameters.yaml')],
                scene=str(scene_path) if scene_path.is_file() else None, declarations=declarations,
                override=overrides.snapshot() if overrides else None),
            effective=dict(mode=scene.mode, reference_arm=scene.reference_arm,
                tag_layout=str(scene.tag_layout) if scene.tag_layout else None,
                trajectory=scene.trajectory.snapshot(), parameters=asdict(scene.parameters)),
            selection=(dict(kit=selection.kit, arms=list(selection.expected_arms),
                            intrinsics_source=selection.intrinsics_source,
                            map_version=map_version,
                            requested_intrinsics_source=intrinsics_source) if selection else None)))
    return reports
