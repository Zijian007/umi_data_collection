"""单批次准备与建图入口。"""

from __future__ import annotations

from pathlib import Path

from . import batches, camera_assets, cleanup, ingest
from software.scene import layout
from software.scene.configuration import SceneOverrides
from software.scene.pipeline import map_lifecycle
from software.infrastructure.file_store import read_yaml
from software.scene.management.batch_inputs import batch_from_record, read_batch
from .batch_discovery import batch_root, require_collection


def prepare_batch(batch: Path, *, dry_run: bool = False, intrinsics_source: str | None = None, overrides: SceneOverrides | None = None) -> dict:
    """预览与实际运行共用原片规划；预览只构造内存中的批次快照。"""
    if not dry_run:
        from .import_manifest import ensure_epoch_plan
        ensure_epoch_plan(batch)
    epoch_mode = layout.metadata(batch, 'map_epochs.yaml').is_file()
    # 普通批次建图后，mapping/ 可能已经整理或清理；内部清单中的冻结地图
    # 指针足以支持后续 take。此时再次强制要求建图原片，会让有效地图的
    # 批量 CLI 运行无故失败。
    record_path = layout.metadata(batch, 'manifest.yaml')
    previous = read_yaml(record_path) if record_path.is_file() else {}
    collection = read_yaml(batch / 'collection.yaml') if (batch / 'collection.yaml').is_file() else {}
    has_bound_map = bool(previous.get('map') or collection.get('map_version'))
    batch_inputs = batches.read_inputs(
        batch,
        require_mapping=not epoch_mode and not has_bound_map,
        include_mapping=not epoch_mode,
        intrinsics_source=intrinsics_source,
        overrides=overrides,
    )
    if dry_run:
        snapshot = batch_from_record(layout.metadata(batch, 'manifest.yaml'), batches.make_record(batch_inputs), overrides=overrides)
    else:
        snapshot = read_batch(batches.save_record(batch_inputs), overrides=overrides)
    return ingest.register_recordings(snapshot, dry_run=dry_run)


def build(batch: Path, *, force: bool = False, keep_work: bool = False,
          cores: str = '', dry_run: bool = False, intrinsics_source: str | None = None,
          overrides: SceneOverrides | None = None, epochs: bool = False) -> dict:
    """Build only from a batch; run and build share configuration resolution."""
    batch = batch_root(batch)
    if (batch / 'batch.yaml').is_file():
        layout.reject_legacy_user_file(batch / 'batch.yaml')
    if epochs:
        from .epoch_processing import build_epochs
        require_collection(batch, 'build --epochs')
        batch_inputs = batches.read_inputs(batch, require_mapping=False, include_mapping=False,
                                           intrinsics_source=intrinsics_source, overrides=overrides)
        if dry_run:
            session = batch_from_record(layout.metadata(batch, 'manifest.yaml'),
                                        batches.make_record(batch_inputs), overrides=overrides)
        else:
            batches.save_record(batch_inputs)
            session = None
        return build_epochs(batch, force=force, keep_work=keep_work, cores=cores,
                            dry_run=dry_run, intrinsics_source=intrinsics_source,
                            overrides=overrides, session=session)
    if (batch / 'collection.yaml').is_file():
        batch_inputs = batches.read_inputs(batch, intrinsics_source=intrinsics_source, overrides=overrides)
        if dry_run:
            snapshot = batch_from_record(layout.metadata(batch, 'manifest.yaml'),
                                         batches.make_record(batch_inputs), verify_mapping=True,
                                         overrides=overrides)
            return map_lifecycle._build_scene(snapshot.scene, force=force, keep_work=keep_work, cores=cores, dry_run=True)
        path = batches.save_record(batch_inputs)
    else:
        path = layout.manifest(batch, 'manifest.yaml')
        if not path.is_file():
            raise ValueError('build 需要采集批次目录及 collection.yaml；将建图原片放入该批次的 mapping/')
        if intrinsics_source is not None:
            camera_assets.require_calibration_source(read_batch(path, overrides=overrides).kit, intrinsics_source)
    result = map_lifecycle.build_scene(path, force=force, keep_work=keep_work, cores=cores, dry_run=dry_run,
                                       overrides=overrides)
    if not dry_run:
        cleanup.after_batch(layout.owner(path))
    return result
