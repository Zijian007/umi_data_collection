"""双手共享海报：各手只交出 PosterInputs，最终轨迹来自一次联合求解。"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from software.scene.pipeline import episode_execution, measurements, reconstruction
from software.scene.spatial import poster


class SharedPosterExecutionTests(unittest.TestCase):
    def run_arms(self, *, tag_fusion_enabled: bool):
        scene = SimpleNamespace(name='scene', environment_tag_detection=True,
                                parameters=SimpleNamespace(tag_fusion_enabled=tag_fusion_enabled))
        episode = SimpleNamespace(expected_arms=('left', 'right'), session=SimpleNamespace(name='batch'))
        baselines = {arm: SimpleNamespace(name=f'{arm}-baseline') for arm in ('left', 'right')}
        inputs = {arm: SimpleNamespace(arm=arm) for arm in ('left', 'right')}
        refined = {arm: SimpleNamespace(name=f'{arm}-refined') for arm in ('left', 'right')}
        calls = []

        def reconstruct(episode, arm, *args, share_poster, **kwargs):
            calls.append((arm, share_poster))
            return reconstruction.ArmReconstruction(baselines[arm], inputs[arm] if share_poster else None)

        written = {}

        def write(episode, arm, recording, fusion_result, *args):
            written[arm] = fusion_result
            return f'{arm}-track', None, None

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(measurements, 'prepare_detections'), \
                patch.object(reconstruction, 'reconstruct_arm_in_scene', side_effect=reconstruct), \
                patch.object(episode_execution, 'write_arm_result', side_effect=write), \
                patch.object(poster, 'refine_shared_poster_pose', return_value=refined) as shared:
            result = episode_execution.execute_arm_tracks(
                episode=episode, prepared_recordings={'left': object(), 'right': object()},
                video_offsets_s={'left': 0., 'right': 0.}, map_dir=Path(directory),
                map_manifest=dict(frame_id='scene/v1'), run_dir=Path(directory), scene=scene, name='take',
                allocation=False, cache_events=None, scene_metadata=dict(frame_id='scene/v1'),
                console=SimpleNamespace(write=lambda *args, **kwargs: None))
        return result, calls, written, shared, inputs, refined, baselines

    def test_shared_poster_solves_once_and_writes_refined_tracks(self):
        result, calls, written, shared, inputs, refined, _ = self.run_arms(tag_fusion_enabled=True)
        self.assertEqual(calls, [('left', True), ('right', True)])
        shared.assert_called_once_with([inputs['left'], inputs['right']])
        self.assertEqual(written, refined)
        self.assertEqual(result.tracks, {'left': 'left-track', 'right': 'right-track'})
        self.assertEqual(result.scene_metadata, dict(frame_id='scene/v1'))

    def test_without_poster_fusion_each_arm_writes_its_own_result(self):
        _, calls, written, shared, _, _, baselines = self.run_arms(tag_fusion_enabled=False)
        self.assertEqual(calls, [('left', False), ('right', False)])
        shared.assert_not_called()
        self.assertEqual(written, baselines)


if __name__ == '__main__':
    unittest.main()
