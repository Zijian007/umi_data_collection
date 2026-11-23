from umi.dataset import reader
"""Poster independent backend: board frame, whole-task routing and publication."""

from dataclasses import replace
import json
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from umi.infra.file_store import read_yaml, write_yaml
from umi.workflows.scene import workflow
from umi.workflows.scene.inputs import batch as episode_inputs
from umi.workflows.scene import episodes
from umi.workflows.scene import batch_snapshot as batches
from umi.estimation.models import FusedTrajectory
from umi.workflows.scene.tasks import poster_independent as independent_reconstruction, poster_independent, poster_vio, scene_map as scene_reconstruction
from umi.workflows.scene.maps import repository as map_lifecycle
from umi.workflows.scene.maps import repository as map_repository
from umi.workflows.scene.tasks import measurements
from umi.workflows.scene.tasks.backend_routing import MapSelection
from umi.store.result_io import read_arm_summary, scene_reference
from umi.estimation.fixed_map.fusion import SceneFusionQualityError
from umi.workflows.scene.errors import MapUnavailableError, MapCompatibilityError, PosterVioQualityError
from umi.perception.poster import layout as tag_board


class MapSelectionTests(unittest.TestCase):
    def test_map_directory_and_manifest_form_one_state(self):
        directory, manifest = Path('/unread/map'), {'version': 'one'}
        selection = MapSelection.available(directory, manifest)
        self.assertEqual((selection.directory, selection.manifest), (directory, manifest))
        reason = {'category': 'map_unavailable'}
        self.assertEqual(MapSelection.without_map(reason).failure, reason)
        self.assertIsNone(MapSelection.without_map().manifest)
        with self.assertRaises(ValueError):
            MapSelection.available(None, None)
        for values in (dict(directory=directory), dict(manifest=manifest),
                       dict(directory=directory, manifest=manifest, failure=reason)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                MapSelection(**values)


class AutomaticBackendChainTests(unittest.TestCase):
    def test_auto_generated_poster_vio_preparation_quality_can_fall_through(self):
        episode = SimpleNamespace(
            scene=SimpleNamespace(environment_tag_detection=True,
                                  parameters=SimpleNamespace(tag_fusion_enabled=True),
                                  reconstruction_mode='auto'))
        inputs = SimpleNamespace(episode=episode)
        with patch.object(poster_vio, 'prepare_task_inputs', side_effect=ValueError('too few tags')):
            with self.assertRaises(PosterVioQualityError):
                poster_vio.execute_task(inputs)

    def test_explicit_poster_vio_preparation_error_remains_input_error(self):
        episode = SimpleNamespace(
            scene=SimpleNamespace(environment_tag_detection=True,
                                  parameters=SimpleNamespace(tag_fusion_enabled=True),
                                  reconstruction_mode='poster_vio'))
        inputs = SimpleNamespace(episode=episode)
        with patch.object(poster_vio, 'prepare_task_inputs', side_effect=ValueError('bad frozen input')):
            with self.assertRaisesRegex(ValueError, 'bad frozen input'):
                poster_vio.execute_task(inputs)

    def test_map_quality_then_poster_vio_quality_then_independent(self):
        from umi.workflows.scene.tasks import backend_routing as estimation
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            calls = []
            @dataclass(frozen=True)
            class Inputs:
                directory: Path
                episode: object
                console: object
            console = SimpleNamespace(messages=[], write=lambda message, **_: console.messages.append(message))
            inputs = Inputs(root, SimpleNamespace(scene=SimpleNamespace(reconstruction_mode='auto')), console)
            expected = SimpleNamespace(coordinate_record={})

            def scene(inputs, selection):
                calls.append('scene_map')
                raise SceneFusionQualityError('map coverage too low')

            def vio(inputs, selection):
                calls.append('poster_vio')
                raise PosterVioQualityError('poster support too low')

            def independent(inputs, selection):
                calls.append(('poster_independent', selection.failure))
                return expected

            with patch.dict(estimation.ESTIMATORS, dict(scene_map=scene, poster_vio=vio,
                                                        poster_independent=independent)):
                result = estimation.execute(inputs, MapSelection.available(root / 'map', {'version': 'v'}))
            self.assertIs(result, expected)
            self.assertEqual([item if isinstance(item, str) else item[0] for item in calls],
                             ['scene_map', 'poster_vio', 'poster_independent'])
            failure = calls[-1][1]
            self.assertEqual(failure['category'], 'scene_quality')
            self.assertEqual([item['backend'] for item in failure['attempts']],
                             ['scene_map', 'poster_vio'])
            attempts = json.loads((root / 'reconstruction_attempts/attempts.json').read_text())
            self.assertEqual(attempts['selected_backend'], 'poster_independent')
            self.assertEqual([item['status'] for item in attempts['attempts']],
                             ['failed', 'failed', 'succeeded'])
            self.assertIn('自动后端路由', console.messages[0])
            self.assertIn('后端切换：scene_map → poster_vio', console.messages[2])
            self.assertIn('后端选定：poster_independent', console.messages[-1])


class PosterIndependentWorkflowTests(unittest.TestCase):
    def setUp(self):
        import test_scene
        self.helper = test_scene.SceneWorkflowTests('runTest')
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        # Workflow fixtures replace the solver and use placeholder INSV files.
        detector = patch.object(measurements, 'prepare_detections', return_value=None)
        detector.start()
        self.addCleanup(detector.stop)

    def configure(self, mode='poster_independent', *, confirmed=False):
        h = self.helper
        collection = h.batch_root / 'collection.yaml'
        doc = read_yaml(collection)
        doc['backend'] = dict(mode=mode, scene_map=dict(poster_constraint=dict(enabled=True, method='joint')))
        doc['environment_poster'] = dict(
            layout_path=str(tag_board.DEFAULT_LAYOUT),
            detection=dict(enabled=True))
        doc['tasks'] = dict(take_001=dict(instruction='move both grippers', arms=['left', 'right'], synchronization=dict(
            mode='manual', offsets_s=dict(left=0., right=0.), confirmed=confirmed,
            note='fixture common clock' if confirmed else '')))
        write_yaml(collection, doc)
        for video, camera in list(h.cameras.items()):
            h.cameras[video] = replace(camera, files=dict(camera.files, rear=camera.files['front']),
                                      measurements=dict(camera.measurements, rear=camera.measurements['front']))

    def solved(self, episode, arm, recording, directory, *, require_poster=False):
        self.assertTrue(require_poster)
        raw = self.helper.raw[arm].T_reference_camera.copy()
        raw[:, :3, 3] *= 2.3
        return FusedTrajectory(raw, np.ones(len(raw), bool), np.ones(len(raw), bool), dict(
            mode='poster_independent', scene_map_priors_used=False,
            localization_fraction=1., maximum_visual_gap_s=0.,
            independent=dict(gravity_output_m_s2=[0., 0., -9.81]),
            environment_poster=dict(status='fused', layout=tag_board.load().record(),
                observation_video_times_s=recording.video_times_s.tolist(),
                position_support_video_times_s=recording.video_times_s.tolist())))

    def run_poster(self):
        with patch.object(independent_reconstruction, 'reconstruct', side_effect=self.solved):
            return workflow.run_episode(self.helper.episode_path)

    def test_no_map_bimanual_publication_reuse_review_and_viewer_contract(self):
        self.configure(confirmed=True)
        h = self.helper
        with patch.object(map_lifecycle, 'bind_map', side_effect=AssertionError('must not bind map')):
            result = self.run_poster()
        self.assertEqual(result['status'], 'ok')
        manifest = json.loads((h.task_root / 'results/data/episode.json').read_text())
        self.assertEqual(manifest['kind'], 'umi_poster_episode')
        self.assertIsNone(manifest['map'])
        self.assertIsNone(manifest['inputs']['scene_map_sha256'])
        self.assertEqual(scene_reference(manifest)['scope'], 'task_poster')
        self.assertTrue((h.task_root / 'results/data/bimanual.csv').is_file())
        summary = read_arm_summary(h.task_root / 'results/data/left')
        self.assertEqual(summary['mode'], 'poster')
        from umi.apps.viewer.resources import maps
        self.assertEqual(maps.load(h.task_root / 'results/data/left', summary)['maps'], [])
        from umi.apps.viewer.results.catalog import RecordingCatalog
        catalog = RecordingCatalog(h.root)
        entries = catalog.scan()['recordings']
        entry = next(item for item in entries if item.get('available') and item['serial'] == '任务海报轨迹')
        loaded = catalog.load(entry['id'])
        self.assertEqual(loaded['summary']['mode'], 'poster')
        self.assertIsNotNone(loaded['companion'])
        self.assertIn('imu', loaded['tracks'])
        self.assertEqual(workflow.run_episode(h.episode_path)['status'], 'skipped')
        episodes.review(h.episode_path, accept=True, reviewer='tester')
        from umi.dataset.export import read_accepted_episode
        self.assertEqual(read_accepted_episode(h.episode_path).manifest['kind'], 'umi_poster_episode')
        exported = episodes.export([h.episode_path], h.root / 'trajectory-export')
        self.assertEqual(exported['episodes'], 1)
        from umi.dataset import export, sampling
        with patch.object(sampling, 'video_timestamps',
                          return_value=h.prepared_recordings[h.recording_paths['left']].video_times_s):
            exported = export.build([h.episode_path], h.root / 'training-export')
        self.assertEqual(exported['episodes'], 1)
        dataset_manifest = json.loads((h.root / 'training-export/manifest.json').read_text())
        training, arrays = reader.load_episode(h.root / 'training-export', dataset_manifest['episodes'][0]['path'])
        self.assertEqual(training['pose_representation'], 'T_board_tcp')
        self.assertEqual(training['coordinate_scope'], 'task_poster')
        self.assertTrue(arrays['left_position_supported'].all())
        self.assertTrue(arrays['right_poster_position_supported'].all())
        self.assertTrue(training['quality']['arms']['left']['position_support_available'])
        np.testing.assert_allclose(arrays['T_left_tcp_right_tcp'],
            np.linalg.inv(arrays['left_T_scene_tcp']) @ arrays['right_T_scene_tcp'])

    def test_auto_missing_map_falls_back_without_building(self):
        self.configure('auto')
        with patch.object(map_lifecycle, 'build_scene', side_effect=AssertionError('must not build')):
            result = self.run_poster()
        self.assertEqual(result['status'], 'ok')
        scene = json.loads((self.helper.task_root / 'results/data/episode.json').read_text())['scene']
        self.assertEqual(scene['reconstruction_backend'], 'poster_independent')
        self.assertIn('地图不可用', scene['fallback']['reason'])
        self.assertEqual(scene['fallback']['category'], 'map_unavailable')

    def test_unexpected_map_reader_errors_do_not_trigger_fallback(self):
        self.configure('auto')
        episode = episode_inputs.read_episode(self.helper.episode_path)
        from umi.workflows.scene.tasks import backend_routing as estimation
        for error in (ValueError('program error'), KeyError('program key'), RuntimeError('engine crash')):
            with self.subTest(error=error), patch.object(map_repository, 'load_published', side_effect=error):
                with self.assertRaises(type(error)):
                    estimation.select_map(episode, None)

    def test_map_compatibility_failure_preserves_category_and_error_type(self):
        self.configure('auto')
        episode = episode_inputs.read_episode(self.helper.episode_path)
        from umi.workflows.scene.tasks import backend_routing as estimation
        with patch.object(map_repository, 'load_published', side_effect=MapCompatibilityError('engine changed')):
            _, selection = estimation.select_map(episode, None)
        self.assertEqual(selection.failure['category'], 'map_incompatible')
        self.assertEqual(selection.failure['error_type'], 'MapCompatibilityError')

    def test_auto_corrupt_map_falls_back_without_using_any_scene_pose(self):
        h = self.helper
        h.build_scene()
        directory, _ = map_repository.load_published(h.scene)
        (directory / 'maps/shared_front/map.osa').write_bytes(b'corrupt map')
        self.configure('auto')
        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene', side_effect=AssertionError('no scene poses')):
            result = self.run_poster()
        self.assertEqual(result['status'], 'ok')
        manifest = json.loads((h.task_root / 'results/data/episode.json').read_text())
        self.assertIsNone(manifest['inputs']['scene_map_sha256'])
        self.assertIsNone(manifest['map'])

    def test_auto_missing_declared_map_does_not_block_layout_inference(self):
        self.configure('auto')
        h = self.helper
        path = h.batch_root / 'collection.yaml'
        doc = read_yaml(path)
        doc['map']['layout'] = 'auto'
        doc['map']['selection']['version'] = '20261117_000000-deadbeef'
        write_yaml(path, doc)
        batch = batches.read_inputs(h.batch_root, include_mapping=False)
        self.assertEqual(batch.scene.mode, 'shared_front')
        batches.save_record(batch)
        episode = episode_inputs.read_episode(h.episode_path)
        self.assertEqual(episode.map_version, doc['map']['selection']['version'])
        from umi.workflows.scene.tasks import backend_routing as estimation
        with patch.object(map_repository, 'load_published', side_effect=MapUnavailableError('missing')) as load:
            _, selected = estimation.select_map(episode, None)
        self.assertEqual(load.call_args.kwargs['version'],
                         doc['map']['selection']['version'])
        self.assertIsNone(selected.manifest)

    def test_missing_poster_configuration_fails_before_any_engine_call(self):
        self.configure()
        path = self.helper.batch_root / 'collection.yaml'
        doc = read_yaml(path)
        doc['environment_poster']['detection']['enabled'] = False
        write_yaml(path, doc)
        with patch.object(independent_reconstruction, 'reconstruct') as solve:
            with self.assertRaisesRegex(ValueError, 'detection.enabled'):
                workflow.run_episode(self.helper.episode_path)
        solve.assert_not_called()

    def test_auto_scene_quality_failure_recomputes_both_hands(self):
        h = self.helper
        h.build_scene()
        self.configure('auto')
        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene',
                          side_effect=SceneFusionQualityError('scene coverage too low')), \
                patch.object(independent_reconstruction, 'reconstruct', side_effect=self.solved) as solve:
            result = workflow.run_episode(h.episode_path)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual({call.args[1] for call in solve.call_args_list}, {'left', 'right'})
        manifest = json.loads((h.task_root / 'results/data/episode.json').read_text())
        self.assertIsNone(manifest['map'])
        self.assertIsNotNone(manifest['inputs']['scene_map_sha256'])
        self.assertEqual(manifest['scene']['fallback']['category'], 'scene_quality')
        episodes.review(h.episode_path, accept=True, reviewer='tester')

    def test_auto_input_error_does_not_trigger_poster_fallback(self):
        h = self.helper
        h.build_scene()
        self.configure('auto')
        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene', side_effect=ValueError('bad IMU')), \
                patch.object(independent_reconstruction, 'reconstruct') as solve:
            with self.assertRaisesRegex(ValueError, 'bad IMU'):
                workflow.run_episode(h.episode_path)
        solve.assert_not_called()

    def test_auto_success_publishes_scene_candidate_without_local_fallback(self):
        h = self.helper
        h.build_scene()
        self.configure('auto')

        def scene_solve(episode, arm, recording, *args, **kwargs):
            trajectory = self.solved(episode, arm, recording, h.root, require_poster=True)
            trajectory = replace(trajectory, evidence=dict(trajectory.evidence, visual_consistency_rms_m=0.))
            return scene_reconstruction.ArmReconstruction(trajectory)

        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene', side_effect=scene_solve), \
                patch.object(independent_reconstruction, 'reconstruct') as solve:
            result = workflow.run_episode(h.episode_path)
        solve.assert_not_called()
        self.assertEqual(result['status'], 'ok')
        manifest = json.loads((h.task_root / 'results/data/episode.json').read_text())
        self.assertEqual(manifest['kind'], 'umi_scene_episode')
        self.assertIsNotNone(manifest['map'])

    def test_missing_poster_arm_never_replaces_previous_success(self):
        self.configure()
        h = self.helper
        self.run_poster()
        previous = (h.task_root / 'results').resolve()

        def broken(*args, **kwargs):
            result = self.solved(*args, **kwargs)
            if args[1] == 'right':
                return replace(result, evidence=dict(result.evidence, environment_poster=dict(status='insufficient_observations')))
            return result

        with patch.object(independent_reconstruction, 'reconstruct', side_effect=broken):
            with self.assertRaisesRegex(ValueError, '未完成海报融合'):
                workflow.run_episode(h.episode_path, force=True)
        self.assertEqual((h.task_root / 'results').resolve(), previous)

    def test_disagreeing_gravity_rejects_bimanual_anchor(self):
        self.configure()

        def broken(*args, **kwargs):
            result = self.solved(*args, **kwargs)
            if args[1] == 'right':
                return replace(result, evidence=dict(result.evidence,
                    independent=dict(gravity_output_m_s2=[9.81, 0., 0.])))
            return result

        with patch.object(independent_reconstruction, 'reconstruct', side_effect=broken):
            with self.assertRaisesRegex(ValueError, '重力方向不一致'):
                workflow.run_episode(self.helper.episode_path)

    def test_backend_gravity_gate_uses_its_own_parameter(self):
        self.configure()
        h = self.helper
        episode = episode_inputs.read_episode(h.episode_path)
        parameters = replace(episode.scene.parameters, maximum_placement_rotation_deg=.1,
                             poster_maximum_gravity_difference_deg=3.)
        episode = replace(episode, session=replace(episode.session,
            scene=replace(episode.scene, parameters=parameters)))

        def tilted(*args, **kwargs):
            result = self.solved(*args, **kwargs)
            if args[1] == 'right':
                gravity = Rotation.from_euler('x', 2., degrees=True).apply([0., 0., -9.81])
                result = replace(result, evidence=dict(result.evidence,
                    independent=dict(gravity_output_m_s2=gravity.tolist())))
            return result

        prepared = {arm: h.prepared_recordings[path] for arm, path in h.recording_paths.items()}
        with patch.object(independent_reconstruction, 'reconstruct', side_effect=tilted):
            poster_independent.solve_task(episode, prepared, h.root / 'solve')
            episode = replace(episode, session=replace(episode.session, scene=replace(episode.scene,
                parameters=replace(parameters, poster_maximum_gravity_difference_deg=1.))))
            with self.assertRaisesRegex(ValueError, '重力方向不一致'):
                poster_independent.solve_task(episode, prepared, h.root / 'solve')

    def test_unconfirmed_sync_is_not_training_accepted(self):
        self.configure()
        self.run_poster()
        episodes.review(self.helper.episode_path, accept=True, reviewer='tester')
        from umi.dataset.export import read_accepted_episode
        with self.assertRaisesRegex(ValueError, 'synchronization'):
            read_accepted_episode(self.helper.episode_path)

    def test_task_override_and_mapless_batch_input(self):
        self.configure()
        h = self.helper
        collection = h.batch_root / 'collection.yaml'
        doc = read_yaml(collection)
        doc['map']['layout'] = 'auto'
        write_yaml(collection, doc)
        inputs = batches.read_inputs(h.batch_root)
        self.assertEqual(inputs.mapping_records, {})
        self.assertEqual(inputs.scene.mode, 'shared_front')
        doc['backend'] = dict(mode='scene_map')
        doc['tasks']['take_001']['backend'] = dict(mode='poster_independent')
        write_yaml(collection, doc)
        episode = episode_inputs.read_episode(h.episode_path)
        self.assertEqual(episode.scene.reconstruction_mode, 'poster_independent')


class PosterIndependentAdapterTests(unittest.TestCase):
    def setUp(self):
        import test_scene_fallback
        self.helper = test_scene_fallback.SceneFallbackAdapterTests('runTest')
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        from test_independent_poster import fixture, write_camera
        inputs, _ = fixture()
        h = self.helper
        h.episode.root = h.root / 'take_001'
        h.episode.scene.tag_layout = tag_board.DEFAULT_LAYOUT
        h.episode.scene.environment_tag_detection = True
        h.episode.scene.parameters = replace(h.episode.scene.parameters, tag_fusion_enabled=True,
                                              tag_fusion_method='joint')
        h.episode.recordings['right'].write_bytes(b'fixture')
        rear = h.root / 'rear.json'
        write_camera(rear, inputs.poster.measured, rear=True)
        from types import SimpleNamespace
        h.recording = replace(h.recording, camera=SimpleNamespace(
            files=dict(h.recording.camera.files, rear=rear), measurements=dict(rear=inputs.poster.measured)))
        self.observations = inputs.poster.observations
        self.board_pose = np.eye(4)
        self.board_pose[:3, :3] = Rotation.from_euler('x', 30, degrees=True).as_matrix()
        self.board_pose[:3, 3] = [1., 2., 3.]

    def execute(self, *, status='fused', times=True, sparse_visual=False, sparse_tags=False, weak_tags=False):
        from umi.perception.poster import detection as environment_tags
        h = self.helper

        from umi.workflows.segment import fusion_runner

        def fuse(request, **options):
            h.engines.fuse(request, **options)
            if True:
                path = request.output / 'fuse_info.json'
                evidence = json.loads(path.read_text())
                if sparse_visual:
                    evidence['chains']['front'] = [([0, 1], 'full')]
                evidence['environment_poster'] = dict(status=status,
                    T_world_board=self.board_pose.tolist(),
                    observation_video_times_s=(h.recording.video_times_s[:3].tolist() if sparse_tags else
                                              h.recording.video_times_s.tolist()) if times else [],
                    position_support_video_times_s=[] if weak_tags or not times else (
                        h.recording.video_times_s[:3].tolist() if sparse_tags else h.recording.video_times_s.tolist()))
                path.write_text(json.dumps(evidence))

        with patch.object(environment_tags, 'detect', return_value=self.observations), \
                patch.object(fusion_runner, 'execute', side_effect=fuse):
            return independent_reconstruction.reconstruct(h.episode, 'right', h.recording,
                                        h.root / 'poster-independent', require_poster=True)

    def test_local_world_to_board_rotation_translation_and_gravity(self):
        result = self.execute()
        local = np.tile(np.eye(4), (len(result.tracked), 1, 1))
        local[:, 0, 3] = np.arange(len(local)) * .01
        np.testing.assert_allclose(result.T_reference_imu, np.linalg.inv(self.board_pose) @ local, atol=1e-6)
        np.testing.assert_allclose(result.evidence['independent']['gravity_output_m_s2'],
            self.board_pose[:3, :3].T @ [0., 0., -9.81], atol=1e-6)
        self.assertFalse(result.evidence['scene_map_priors_used'])

    def test_diagnostics_only_does_not_count_as_poster_fusion(self):
        with self.assertRaisesRegex(ValueError, '未实际融合'):
            self.execute(status='diagnostics_only')

    def test_missing_observation_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '时间证据不足'):
            self.execute(times=False)

    def test_continuous_poster_constraints_cover_a_natural_visual_gap(self):
        result = self.execute(sparse_visual=True)
        self.assertGreater(result.evidence['maximum_visual_gap_s'], 2.)
        self.assertLess(result.evidence['independent']['maximum_position_support_gap_s'], .04)

    def test_many_weak_poster_sightings_are_not_position_support(self):
        with self.assertRaisesRegex(ValueError, '几何有效的位置支持不足'):
            self.execute(sparse_visual=True, weak_tags=True)

    def test_missing_natural_and_poster_support_rejects_long_gap(self):
        with self.assertRaisesRegex(ValueError, '共同约束缺口过长'):
            self.execute(sparse_visual=True, sparse_tags=True)
