"""Publish one completed scene episode after final input verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from software.infrastructure import provenance as artifact_identity
from software.infrastructure.file_store import publish_link
from software.scene import layout
from software.scene.management import cleanup, episodes
from software.scene.asset_identity import recording_identity
from software.scene.pipeline import reporting, episode_result_writer
from software.scene.result_io import is_local_scene


@dataclass(frozen=True)
class PublicationResult:
    output: Path
    is_local: bool


def _verify_frozen_inputs(episode, input_identity: dict, *, batch_map) -> None:
    """Reject publication if source inputs changed after reconstruction started."""
    current_recordings = {
        arm: recording_identity(video) for arm, video in episode.recordings.items()
    }
    if input_identity['recordings'] != current_recordings:
        raise ValueError('任务期间录像发生变化，拒绝发布')

    epoch_plan = layout.metadata(episode.session.root, 'map_epochs.yaml')
    current_epoch_plan = artifact_identity.file_hash(epoch_plan) if epoch_plan.is_file() else None
    if input_identity.get('map_epoch_plan_sha256') != current_epoch_plan:
        raise ValueError('地图 epoch 清单在任务期间发生变化，拒绝发布')

    if batch_map is not None:
        batch_map.validate(check_configuration=False)


def publish_episode(*, episode, input_identity: dict, map_manifest: dict, tracks: dict,
                    video_offsets_s: dict[str, float], scene_record: dict,
                    run_dir: Path, map_path: Path | None, widths: dict,
                    distances: dict, cache_events, common_interval_s: list[float],
                    elapsed_s: float, started_at: float, batch_map) -> PublicationResult:
    """Write reports, atomically publish the result, and record task state."""
    _verify_frozen_inputs(
        episode,
        input_identity,
        batch_map=batch_map,
    )
    reporting.write_run_report(
        run_dir,
        episode=episode,
        map_manifest=map_manifest,
        input_identity=input_identity,
        tracks=tracks,
        video_offsets_s=video_offsets_s,
        scene_record=scene_record,
        gripper_width=widths,
        tag_distance=distances,
        common_interval_s=common_interval_s,
        cache_audit=cache_events,
    )
    episode_result_writer.write_episode_result(
        episode,
        tracks,
        run_dir,
        common_interval_s=common_interval_s,
        scene_record=scene_record,
        input_identity=input_identity,
        video_offsets_s=video_offsets_s,
        elapsed_s=elapsed_s,
        map_path=map_path,
        gripper_width=widths,
        tag_distance=distances,
    )

    is_local = is_local_scene(scene_record)
    output = episode.root / 'local_results' if is_local else episode.output
    publish_link(run_dir, output)
    episodes.record_attempt(episode, 'local_ok' if is_local else 'ok', run_dir)
    cleanup.after_task(episode.root, used_since=started_at)
    return PublicationResult(output=output, is_local=is_local)
