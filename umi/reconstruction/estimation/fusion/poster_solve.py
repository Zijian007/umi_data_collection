"""Prepare poster factors and apply the existing within-attempt culling policy."""

from dataclasses import dataclass, replace

from umi.config.parameter_models import TrajectoryParameters
from umi.perception.poster.quality import cumulative_culling_record
from umi.reconstruction.estimation.fusion import poster as fusion_poster
from umi.reconstruction.estimation.fusion.alignment import AlignmentResult
from umi.reconstruction.estimation.fusion.data import FusionInputs, MapPlacements, TrajectorySeed
from umi.reconstruction.estimation.fusion.graph import TrajectoryOptimizationResult, optimize_fused_trajectory


@dataclass(frozen=True)
class PosterSolveResult:
    optimized: TrajectoryOptimizationResult | None
    evidence: dict
    sampling: dict


def solve_poster_graph(inputs: FusionInputs, aligned: AlignmentResult, placements: MapPlacements,
                      seed: TrajectorySeed, parameters: TrajectoryParameters, *, log,
                      poster_keyframe_maximum_dt_s: float, confirm_zupt_visually: bool) -> PosterSolveResult:
    """Select exposures once and preserve the original budget across hot-start solves."""
    exposure_times = None
    sampling = {}
    if poster_keyframe_maximum_dt_s and parameters.environment_tags.tag_fusion_enabled:
        exposure_times, eligible_frames = fusion_poster.sample_exposures(
            inputs, seed, parameters.environment_tags, poster_keyframe_maximum_dt_s)
        sampling['exposure_sampling'] = dict(maximum_dt_s=poster_keyframe_maximum_dt_s,
                                             eligible_frames=eligible_frames, selected_frames=len(exposure_times))
        log('poster_exposure_sampling', sampling['exposure_sampling'], flush=True)
    poster, evidence = fusion_poster.prepare(inputs, seed, parameters, exposure_times=exposure_times)
    if poster is None:
        return PosterSolveResult(None, evidence, sampling)
    stages = [evidence] if evidence.get('culled_observations') else []
    observations = inputs.poster.observations
    # One initial graph and at most two post-fit culls. Reuse the same alignment,
    # placements and trajectory seed. A second cull must be smaller than the first.
    warm_start = None
    for solve_index in range(3):
        try:
            optimized = optimize_fused_trajectory(aligned, placements, parameters, log=log, poster=poster,
                                                  warm_start=warm_start, confirm_zupt_visually=confirm_zupt_visually)
            break
        except fusion_poster.PosterCullRequest as request:
            warm_start = request.warm_start
            step = request.audit
            if solve_index == 2 or (solve_index and step['culled_observations'] >= stages[-1]['culled_observations']):
                raise fusion_poster.PosterGeometryError('独立海报优化后异常观测未在有限轮次内消除',
                                                       metrics=step) from request
            stages.append(step)
            selected, times = fusion_poster.eligible_sightings(
                inputs, seed.video_times_s, parameters.environment_tags, exposure_times=exposure_times)
            total = cumulative_culling_record(stages, selected, observations.sample_indices[selected],
                                              observations.tag_ids[selected], times, parameters.environment_tags)
            removed = [entry['observation'] for entry in total['culling']['removed']]
            usable = observations.usable.copy()
            usable[removed] = False
            filtered = replace(inputs, poster=replace(inputs.poster, observations=replace(observations, usable=usable)))
            poster, _ = fusion_poster.prepare(filtered, seed, parameters, exposure_times=exposure_times, _allow_cull=False)
            if poster is None:
                raise fusion_poster.PosterGeometryError('独立海报剔除后观测不足')
            poster.evidence.update(total, pre_cull_reprojection_rms_px=stages[0]['pre_cull_reprojection_rms_px'],
                                   pre_cull_reprojection_metrics=stages[0]['pre_cull_reprojection_metrics'],
                                   pre_cull_invalid_depth_sightings=step['pre_cull_invalid_depth_sightings'])
    return PosterSolveResult(optimized, optimized.evidence['environment_poster'], sampling)
