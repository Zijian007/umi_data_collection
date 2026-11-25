"""双手共享海报：各手只交出 PosterInputs，最终轨迹来自一次联合求解。"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from umi.estimation.models import FusedTrajectory
from umi.workflows.scene.tasks import scene_map as episode_execution, scene_map as scene_reconstruction
from umi.workflows.scene.tasks import measurements
from umi.workflows.scene.tasks import trajectory_writers
from umi.estimation.fixed_map import poster
from umi.store.models import TaskInputs


class SharedPosterExecutionTests(unittest.TestCase):
    def run_arms(self, *, tag_fusion_enabled: bool, observed=('left', 'right')):
        scene = SimpleNamespace(name='scene', environment_tag_detection=True,
                                parameters=SimpleNamespace(tag_fusion_enabled=tag_fusion_enabled))
        episode = SimpleNamespace(expected_arms=('left', 'right'), session=SimpleNamespace(name='batch'),
                                  scene=scene, root=Path('take'),
                                  recording_refs={'left': {'sha256': 'a'}, 'right': {'sha256': 'b'}})
        baselines = {arm: SimpleNamespace(name=f'{arm}-baseline') for arm in ('left', 'right')}
        inputs = {arm: SimpleNamespace(arm=arm) for arm in ('left', 'right')}
        board = np.eye(4)
        board[:3, 3] = [1., 2., 3.]
        refined = {arm: FusedTrajectory(np.eye(4)[None], np.ones(1, bool), np.ones(1, bool),
                       dict(environment_poster=dict(status='fused', T_scene_board=board.tolist(),
                                                    layout=dict(asset_hash='fixture'))))
                   for arm in ('left', 'right')}
        for arm in refined.keys() - set(observed):
            refined[arm].evidence['environment_poster'] = dict(status='not_observed_shared_board')
        calls = []

        def reconstruct(episode, arm, *args, defer_poster_fusion, **kwargs):
            calls.append((arm, defer_poster_fusion))
            return scene_reconstruction.ArmReconstruction(baselines[arm], inputs[arm] if defer_poster_fusion else None)

        written = {}

        def write(task_inputs, arm, fusion_result, *args):
            written[arm] = fusion_result
            return f'{arm}-track', None, None

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(measurements, 'prepare_detections'), \
                patch.object(scene_reconstruction, 'reconstruct_arm_in_scene', side_effect=reconstruct), \
                patch.object(trajectory_writers, 'write_arm_result', side_effect=write), \
                patch.object(poster, 'refine_shared_poster_pose', return_value=refined) as shared:
            task_inputs = TaskInputs(episode, {'left': object(), 'right': object()},
                {'left': 0., 'right': 0.}, Path(directory), None, None,
                SimpleNamespace(write=lambda *args, **kwargs: None))
            result = episode_execution.execute_task(task_inputs, map_dir=Path(directory),
                map_manifest=dict(frame_id='scene/v1'), coordinate_record=dict(frame_id='scene/v1'))
        return result, calls, written, shared, inputs, refined, baselines

    def test_shared_poster_solves_once_and_writes_refined_tracks(self):
        result, calls, written, shared, inputs, refined, _ = self.run_arms(tag_fusion_enabled=True)
        self.assertEqual(calls, [('left', True), ('right', True)])
        shared.assert_called_once_with([inputs['left'], inputs['right']])
        for arm in refined:
            np.testing.assert_allclose(written[arm].T_reference_imu[0, :3, 3], [-1., -2., -3.])
        self.assertEqual(result.tracks, {'left': 'left-track', 'right': 'right-track'})
        self.assertEqual(result.coordinate_record['scope'], 'task_poster')

    def test_one_observing_hand_places_both_trajectories(self):
        result, _, written, *_ = self.run_arms(tag_fusion_enabled=True, observed=('left',))
        self.assertEqual(result.coordinate_record['scope'], 'task_poster')
        np.testing.assert_allclose(written['right'].T_reference_imu[0, :3, 3], [-1., -2., -3.])
        self.assertEqual(written['right'].evidence['environment_poster']['status'], 'not_observed_shared_board')

    def test_missing_alignment_rejects_map_candidate(self):
        with self.assertRaises(episode_execution.SceneFusionQualityError):
            self.run_arms(tag_fusion_enabled=True, observed=())

    def test_without_poster_fusion_each_arm_writes_its_own_result(self):
        _, calls, written, shared, _, _, baselines = self.run_arms(tag_fusion_enabled=False)
        self.assertEqual(calls, [('left', False), ('right', False)])
        shared.assert_not_called()
        self.assertEqual(written, baselines)


if __name__ == '__main__':
    unittest.main()
