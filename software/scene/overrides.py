"""本次场景算法覆盖：一次读取、显式传递，不修改场景/批次/套件声明。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
from typing import TYPE_CHECKING

from software.configuration import loader as config
from software.infrastructure import provenance

if TYPE_CHECKING:
    from software.scene.models import SceneParameters


@dataclass(frozen=True)
class SceneOverrides:
    path: Path
    sha256: str
    values_json: str

    @classmethod
    def load(cls, path: Path) -> SceneOverrides:
        from software.infrastructure.file_store import read_yaml
        path = path.expanduser().resolve()
        declaration = read_yaml(path)
        return cls.from_snapshot(dict(path=str(path), sha256=provenance.file_hash(path), values=declaration))

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> SceneOverrides:
        """验收/导出使用结果冻结的覆盖内容，不依赖临时实验文件仍然存在。"""
        from software.scene.management.inputs.scene import require_keys, load_scene_parameters
        from software.scene.parameter_schema import flatten as flatten_scene_parameters
        require_keys(snapshot, {'path', 'sha256', 'values'}, {'path', 'sha256', 'values'}, '场景覆盖记录')
        declaration = snapshot['values']
        if not isinstance(declaration, dict) or declaration.keys() - {'trajectory', 'parameters'}:
            raise ValueError('场景 --expert-config 只允许 trajectory 和 parameters；设备/布局/地图请写入 kit 或 collection')
        if any(not isinstance(section, dict) for section in declaration.values()):
            raise ValueError('场景 --expert-config 的 trajectory / parameters 必须为参数映射')
        # Check names/structure now; validate numerical relationships after merging
        # with the selected scene (a partial overlay can depend on its values).
        config.merge_overrides(asdict(config.load('trajectory').values), declaration.get('trajectory', {}))
        parameter_overlay = flatten_scene_parameters(declaration.get('parameters', {}), label='parameters')
        config.merge_overrides(asdict(load_scene_parameters()), parameter_overlay, 'parameters.')
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
    from software.scene.management.inputs.scene import load_scene_parameters
    from software.scene.parameter_schema import flatten as flatten_scene_parameters
    declared = overrides.values
    trajectory_values = config.merge_overrides(asdict(trajectory.values), declared.get('trajectory', {}))
    effective_trajectory = config.load('trajectory', inline=trajectory_values)
    effective_trajectory = replace(effective_trajectory, sources=(*trajectory.sources, str(overrides.path)))
    parameter_overlay = flatten_scene_parameters(declared.get('parameters', {}), label='parameters')
    parameter_values = config.merge_overrides(asdict(parameters), parameter_overlay, 'parameters.')
    return effective_trajectory, load_scene_parameters(overrides=parameter_values)

