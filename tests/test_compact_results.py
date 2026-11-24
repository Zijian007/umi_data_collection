"""精简结果的坐标、可选诊断、输入身份和旧格式读取。"""

import csv
import json
import shutil
import subprocess
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from umi.workflows.scene import episodes, workflow
from umi.workflows.scene.parameters import SceneOverrides
from umi.store.result_io import arm_evidence, result_paths, scene_reference
from umi.infra.file_store import publish_link, write_json, write_yaml
from umi.apps.viewer.results.catalog import RecordingCatalog
from umi.apps.viewer.results.trajectory_reader import derive_body_tracks

import test_scene as fixtures
from umi.store import result_io as asset_identity


class BodyTrackTests(unittest.TestCase):
    def test_derived_tracks_match_rigid_transforms_with_rotating_levers(self):
        rotations = Rotation.from_euler('xyz', [[.1, -.2, .3], [2., -.7, 1.], [-1., .4, -.6]])
        positions = np.array([[1., 2., 3.], [-2., .5, 4.], [0., -3., 2.]])
        camera_rotation = Rotation.from_euler('xyz', [np.pi, .2, -.3])
        camera_translation = np.array([.02, -.04, .1])
        imu_rotation = Rotation.from_euler('xyz', [-.6, .8, 2.])
        imu_translation = np.array([-.01, .03, .02])
        settings = dict(T_camera_tcp=dict(R_rows=camera_rotation.as_matrix().tolist(), t_m=camera_translation.tolist()),
            measurements=dict(front=dict(q_xyzw=imu_rotation.as_quat().tolist(), p=imu_translation.tolist())))
        derived = derive_body_tracks([0., .1, .2],
            dict(position=positions.tolist(), quaternion=rotations.as_quat().tolist()), settings)
        camera_expected_rotation = rotations * camera_rotation.inv()
        camera_expected_position = positions - camera_expected_rotation.apply(camera_translation)
        imu_expected_rotation = camera_expected_rotation * imu_rotation.inv()
        imu_expected_position = camera_expected_position - imu_expected_rotation.apply(imu_translation)
        for name, expected_rotation, expected_position in (
                ('camera', camera_expected_rotation, camera_expected_position),
                ('imu', imu_expected_rotation, imu_expected_position)):
            np.testing.assert_allclose(derived[name]['position'], expected_position, atol=1e-12)
            np.testing.assert_allclose(Rotation.from_quat(derived[name]['quaternion']).as_matrix(),
                                       expected_rotation.as_matrix(), atol=1e-12)

    def test_viewer_pose_reader_import_does_not_load_numerical_packages(self):
        command = ('import sys; from umi.apps.viewer.results.trajectory_reader import derive_body_tracks; assert not ({"numpy", "scipy", "gtsam"} & sys.modules.keys())')
        subprocess.run([sys.executable, '-c', command], check=True)

    def test_invalid_external_rotations_are_rejected(self):
        track = dict(position=[[0., 0., 0.]] * 2, quaternion=[[0., 0., 0., 1.]] * 2)
        for rotation in (np.diag([-1., 1., 1.]), np.diag([1., 1., 1.01]), np.full((3, 3), np.nan)):
            with self.subTest(rotation=rotation):
                settings = dict(T_camera_tcp=dict(R_rows=rotation.tolist(), t_m=[0., 0., 0.]))
                with self.assertRaises(ValueError):
                    derive_body_tracks([0., .1], track, settings)


class CompactResultTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SceneWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_output_policy_preserves_poses_and_map_and_replaces_previous_run(self):
        fixture = self.fixture
        expert = fixture.root / 'compact-expert.yaml'
        write_yaml(expert, dict(parameters=dict(diagnostics=dict(export_auxiliary_trajectories=True))))
        fixture.scene_overrides = SceneOverrides.load(expert)
        built = fixture.build_scene()
        workflow.run_episode(fixture.episode_path, overrides=fixture.scene_overrides)
        output = fixture.task_root / 'results'
        original = output.resolve()
        trajectory = (output / 'data/left/trajectory.csv').read_bytes()
        expected_camera = np.loadtxt(arm_evidence(output / 'data/left', 'traj_camera.txt'))
        expected_imu = np.loadtxt(arm_evidence(output / 'data/left', 'traj_imu.txt'))

        write_yaml(expert, dict(parameters=dict(diagnostics=dict(export_auxiliary_trajectories=False))))
        fixture.scene_overrides = SceneOverrides.load(expert)
        self.assertEqual(fixture.build_scene()['status'], 'skipped')
        workflow.run_episode(fixture.episode_path, overrides=fixture.scene_overrides)
        metadata = json.loads((output / 'data/episode.json').read_text())
        self.assertEqual(metadata['schema_version'], 5)
        self.assertNotEqual(output.resolve(), original)
        self.assertEqual(scene_reference(metadata)['frame_id'], built['frame_id'])
        self.assertEqual((output / 'data/left/trajectory.csv').read_bytes(), trajectory)
        for filename in ('traj_camera.txt', 'traj_imu.txt'):
            self.assertFalse(arm_evidence(output / 'data/left', filename).exists())
        self.assertFalse(original.exists())  # 被取代的运行由自动清理删除
        asset_identity.validate_assets(output, metadata['assets'])
        catalog = RecordingCatalog(output)
        loaded = catalog.load(catalog.scan()['recordings'][0]['id'])
        for name, expected in (('camera', expected_camera), ('imu', expected_imu)):
            np.testing.assert_allclose(loaded['tracks'][name]['position'], expected[:, 1:4], atol=2e-6)
            np.testing.assert_allclose(Rotation.from_quat(loaded['tracks'][name]['quaternion']).as_matrix(),
                                       Rotation.from_quat(expected[:, 4:]).as_matrix(), atol=1e-8)

    def test_metadata_has_one_identity_and_records_code_revision(self):
        fixture = self.fixture
        fixture.build_scene()
        workflow.run_episode(fixture.episode_path)
        output = fixture.task_root / 'results'
        metadata = json.loads((output / 'data/episode.json').read_text())
        self.assertNotIn('frame_id', metadata)
        self.assertNotIn('session', metadata['scene'])
        self.assertNotIn('gripper_width', metadata)
        self.assertNotIn('tag_distance', metadata)
        for duplicate in ('frame_id', 'scene_version', 'session', 'synchronization', 'trajectory_code'):
            self.assertNotIn(duplicate, metadata['inputs'])
        self.assertFalse({'code', 'engines'} & metadata['inputs'].keys())
        self.assertEqual(set(metadata['code_revision']), {'commit', 'dirty'})
        with (output / 'data/left/trajectory.csv').open() as stream:
            self.assertEqual(csv.DictReader(stream).fieldnames,
                             ['row', 't_video_s', 't_episode_s', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

    def test_retired_schema_three_results_are_rejected(self):
        legacy = dict(schema_version=3, frame_id='scene/old', scene=dict(frame_id='scene/old', version='old'))
        with self.assertRaisesRegex(ValueError, '受支持的结果版本'):
            scene_reference(legacy)

    def test_v4_result_can_still_be_reused_viewed_and_exported(self):
        fixture = self.fixture
        fixture.build_scene()
        workflow.run_episode(fixture.episode_path)
        output = fixture.task_root / 'results'
        paths = result_paths(output)
        manifest = json.loads(paths.manifest.read_text())
        for arm in manifest['arms']:
            shutil.copytree(paths.data / arm, paths.run / arm, dirs_exist_ok=True)
        shutil.copy2(paths.data / 'bimanual.csv', paths.run)
        shutil.copy2(paths.published / 'report.txt', paths.run)
        manifest['schema_version'] = 4
        manifest.pop('evidence_assets')
        manifest.pop('diagnostics')
        manifest['assets'] = asset_identity.artifact_hashes(
            paths.run, ('report.txt', 'bimanual.csv', 'devices', *manifest['arms']))
        legacy = paths.run / 'episode.json'
        write_json(legacy, manifest)
        shutil.rmtree(paths.published)
        publish_link(paths.run, output)

        self.assertEqual(workflow.run_episode(fixture.episode_path)['status'], 'skipped')
        catalog = RecordingCatalog(output)
        entry, = catalog.scan()['recordings']
        self.assertIn('camera', catalog.load(entry['id'])['tracks'])
        self.assertEqual(catalog.map_data(entry['id'])['kind'], 'umi_map_view')
        episodes.review(fixture.episode_path, accept=True, reviewer='test')
        export = fixture.root / 'legacy_export'
        episodes.export([fixture.episode_path], export)
        copied, = export.rglob('episode.json')
        self.assertEqual(copied.read_bytes(), legacy.read_bytes())
