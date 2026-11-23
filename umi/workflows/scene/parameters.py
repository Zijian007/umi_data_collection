"""本次场景算法覆盖：一次读取、显式传递，不修改场景/批次/套件声明。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
from typing import TYPE_CHECKING
from umi.config import loader as config
from umi.infra import provenance
from umi.config.scene_parameters import flatten


if TYPE_CHECKING:
    from umi.store.models import SceneParameters


@dataclass(frozen=True)
class SceneOverrides:
    path: Path
    sha256: str
    values_json: str

    @classmethod
    def load(cls, path: Path) -> SceneOverrides:
        from umi.infra.file_store import read_yaml
        path = path.expanduser().resolve()
        declaration = read_yaml(path)
        return cls.from_snapshot(dict(path=str(path), sha256=provenance.file_hash(path), values=declaration),
                                 historical=False)

    @classmethod
    def from_snapshot(cls, snapshot: dict, *, historical: bool = True) -> SceneOverrides:
        """验收/导出使用结果冻结的覆盖内容，不依赖临时实验文件仍然存在。"""
        from umi.workflows.scene.inputs.scene import require_keys, load_scene_parameters
        require_keys(snapshot, {'path', 'sha256', 'values'}, {'path', 'sha256', 'values'}, '场景覆盖记录')
        declaration = snapshot['values']
        if not isinstance(declaration, dict) or declaration.keys() - {'trajectory', 'parameters'}:
            raise ValueError('场景 --expert-config 只允许 trajectory 和 parameters；设备/布局/地图请写入 kit 或 collection')
        if any(not isinstance(section, dict) for section in declaration.values()):
            raise ValueError('场景 --expert-config 的 trajectory / parameters 必须为参数映射')
        # Check names/structure now; validate numerical relationships after merging
        # with the selected scene (a partial overlay can depend on its values).
        trajectory_overlay = config.normalize_parameter_names(
            'trajectory', declaration.get('trajectory', {}), public=not historical)
        config.merge_overrides(asdict(config.load('trajectory').values), trajectory_overlay)
        if not historical and 'tag_fusion_enabled' in trajectory_overlay.get('environment_tags', {}):
            raise ValueError('海报用途开关请在 collection.yaml 的 backend.scene_map 中配置')
        parameter_overlay = flatten(declaration.get('parameters', {}), label='parameters',
                                    allow_poster_policy=historical, historical=historical)
        config.merge_overrides(asdict(load_scene_parameters()), parameter_overlay, 'parameters.')
        return cls(Path(snapshot['path']), snapshot['sha256'], json.dumps(declaration, sort_keys=True))

    @property
    def values(self) -> dict:
        return json.loads(self.values_json)

    def snapshot(self) -> dict:
        return dict(path=str(self.path), sha256=self.sha256, values=self.values)


def apply_overrides(trajectory: config.Configuration, parameters: SceneParameters,
                    overrides: SceneOverrides | None, *,
                    gap_sources: dict[str, str] | None = None) -> tuple[config.Configuration, SceneParameters]:
    if overrides is None:
        return trajectory, parameters
    from umi.workflows.scene.inputs.scene import load_scene_parameters
    declared = overrides.values
    trajectory_overlay = config.normalize_parameter_names('trajectory', declared.get('trajectory', {}))
    trajectory_values = config.merge_overrides(asdict(trajectory.values), trajectory_overlay)
    effective_trajectory = config.load('trajectory', inline=trajectory_values)
    effective_trajectory = replace(effective_trajectory, sources=(*trajectory.sources, str(overrides.path)))
    parameter_overlay = flatten(declared.get('parameters', {}),
                                label=f'{overrides.path}:parameters', gap_sources=gap_sources, historical=True)
    parameter_values = config.merge_overrides(asdict(parameters), parameter_overlay, 'parameters.')
    return effective_trajectory, load_scene_parameters(overrides=parameter_values)
