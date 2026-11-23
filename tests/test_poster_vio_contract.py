from __future__ import annotations

import unittest

import numpy as np

from umi.workflows.scene.tasks.poster_vio import PosterVioRequest
from umi.estimation.poster_vio.models import PosterVioParameters
from umi.estimation.poster_vio import PosterVioObservation, PosterVioQuality, PosterVioProblem, PosterVioResult, validate_problem, validate_result, PosterVioPolicy, assess_observations, ArmSolveInput, solve_joint_poster_imu


def observation(arm: str) -> PosterVioObservation:
    return PosterVioObservation(
        arm,
        np.array([0., .1]),
        np.array([12, 12]),
        np.zeros((2, 4, 2)),
    )


class PosterVioContractTests(unittest.TestCase):
    def request(self) -> PosterVioRequest:
        return PosterVioRequest(
            ('left', 'right'), 'board-sha', (observation('left'), observation('right')),
            {'left': {}, 'right': {}}, {'left': {}, 'right': {}},
            {'scope': 'task_poster', 'frame_id': 'poster:run-1'},
        )

    def test_request_requires_task_poster_and_all_assets(self):
        self.request().validate_assets()
        for change in (
            {'coordinate_record': {'scope': 'shared_scene', 'frame_id': 'scene/map'}},
            {'imu_assets': {'left': {}}},
            {'board_asset_hash': ''},
        ):
            with self.subTest(change=change):
                values = self.request().__dict__ | change
                with self.assertRaises(ValueError):
                    PosterVioRequest(**values).validate_assets()

        with self.assertRaisesRegex(ValueError, '观测对象'):
            validate_problem(PosterVioProblem(('left', 'right'), (observation('left'),)))

    def test_result_requires_all_numerical_trajectories(self):
        trajectory = np.tile(np.eye(4), (2, 1, 1))
        result = PosterVioResult(
            {'left': trajectory, 'right': trajectory.copy()},
            PosterVioQuality('solved', {'left': 2, 'right': 2}, 1.2, {'left': 1., 'right': 1.}),
        )
        validate_result(result, ('left', 'right'))
        with self.assertRaises(ValueError):
            validate_result(result, ('left',))

    def test_quality_gate_rejects_sparse_or_discontinuous_observations(self):
        request = self.request().problem()
        quality = assess_observations(request, PosterVioPolicy(minimum_tag_frames=2,
                                                               maximum_observation_gap_s=.2,
                                                               minimum_time_span_s=.1))
        self.assertEqual(quality.status, 'ready')
        sparse = PosterVioProblem(request.arms,
            tuple(PosterVioObservation(item.arm, np.array([0., .1]), item.tag_ids[:2], item.tag_corners_px[:2])
                  for item in request.observations))
        quality = assess_observations(sparse, PosterVioPolicy(minimum_tag_frames=3))
        self.assertEqual(quality.status, 'insufficient_observations')

    def test_joint_poster_imu_solver_optimizes_shared_board_on_synthetic_observations(self):
        from test_poster_fusion import fixture
        from umi.core.geometry import resample_pose_series

        data = fixture()
        baseline, times, imu, observations, extrinsics, trajectory, parameters, detections, board, measured, truth, board_pose = data
        rear_times = detections.imu_times_s(measured['cam_time_offset_ns'])
        node_times, tag_indices = np.unique(rear_times, return_inverse=True)
        seed = resample_pose_series(times, truth, node_times)
        points = np.array([board.corners_m[int(tag_id)] for tag_id in detections.tag_ids])
        model = __import__('umi.perception.poster.geometry', fromlist=['camera_model']).camera_model(measured)
        world_point = board_pose[:3, :3] @ np.array([.2, .2, 0.]) + board_pose[:3, 3]
        natural_nodes = np.array([0, len(node_times) // 2, len(node_times) - 1])
        natural_pixels = []
        for node in natural_nodes:
            in_body = seed[node, :3, :3].T @ (world_point - seed[node, :3, 3])
            in_camera = extrinsics['rear'][:3, :3] @ in_body + extrinsics['rear'][:3, 3]
            natural_pixels.append(model.project(in_camera[None, ...])[0])
        natural_tracks = (dict(track_id=7, node_indices=natural_nodes,
                               pixels=np.asarray(natural_pixels), initial_point_m=world_point),)
        problem = PosterVioProblem(('left',), (PosterVioObservation('left', rear_times,
            detections.tag_ids, detections.corners_px, natural_tracks),))
        validate_problem(problem)
        result = solve_joint_poster_imu(problem, (ArmSolveInput(
            'left', node_times, imu, seed, extrinsics['rear'],
            model,
            points, detections.corners_px, tag_indices,
            trajectory.fusion, parameters.for_algorithm(PosterVioParameters)),), board_pose)
        self.assertEqual(result.quality.status, 'solved')
        self.assertLess(result.quality.reprojection_rms_px, 1e-3)


if __name__ == '__main__':
    unittest.main()
