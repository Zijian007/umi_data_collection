"""首尾规则、同步交集、原片帧映射和真实视频裁剪回归。"""
from dataclasses import replace
import csv
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np

from umi.core.frame import output_frame_rate
from umi.config.loader import load
from umi.store.cache import StageCache
from umi.config.collection_schema import normalize_collection
from umi.store.models import TaskInputs, ArmExecutionResult
from umi.workflows.scene import recordings
from umi.workflows.scene.tasks import recording_intervals as trim, backend_routing as estimation
from umi.workflows.scene.errors import PosterVioQualityError
from umi.estimation.fixed_map.fusion import SceneFusionQualityError
from umi.perception.models import RecordingData, RecordingPaths
from umi.perception.tools import make_lens_video


class TagTrimmingTests(unittest.TestCase):
    def test_configuration_defaults_and_invalid_values(self):
        doc = dict(kit='rig', scene={}, recording=dict(arms=['right']), backend=dict(mode='poster_vio'))
        self.assertNotIn('poster_edge_trim', normalize_collection(doc)['reconstruction'])
        doc['recording']['trim'] = dict(enabled=True)
        with self.assertRaisesRegex(ValueError, 'recording 含未知字段'):
            normalize_collection(doc)
        doc['recording'].pop('trim')
        doc['backend']['poster_edge_trim'] = dict(enabled=True)
        self.assertEqual(normalize_collection(doc)['reconstruction']['poster_edge_trim'], dict(enabled=True))
        for value in (False, 0, -1, float('nan'), float('inf'), '1'):
            doc['backend']['poster_edge_trim']['min_missing_duration_s'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_collection(doc)
        doc['backend']['poster_edge_trim'] = dict(enabled='true')
        with self.assertRaises(ValueError):
            normalize_collection(doc)

    def test_stable_boundaries_ignore_isolated_detections_and_keep_internal_gap(self):
        self.assertEqual(trim.stable_interval([.1, 2., 2.1, 2.2, 7., 7.1, 7.2, 9.]), (2., 7.2))
        for times in ([], [1, 1.1], [1, 2, 3], [1, float('nan')]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                trim.stable_interval(times)
        self.assertEqual(trim.retained_interval((0., 10.), (.5, 9.5), 1.), (0., 10.))
        self.assertEqual(trim.retained_interval((0., 10.), (2., 8.), 1.), (2., 8.))

    def test_geometry_boundary_uses_two_tags_and_lens_clock_offsets(self):
        from test_poster_fusion import fixture
        from scipy.spatial.transform import Rotation

        _, times, imu, _, extrinsics, trajectory, parameters, observations, board, measured, _, _ = fixture()
        imu_rear = np.linalg.inv(extrinsics['rear'])
        rear = dict(measured, q_xyzw=Rotation.from_matrix(imu_rear[:3, :3]).as_quat().tolist(),
                    p=imu_rear[:3, 3].tolist())
        front = dict(rear, cam_time_offset_ns=23000000)
        recording = SimpleNamespace(camera=SimpleNamespace(measurements=dict(front=front, rear=rear)),
                                    video_times_s=times, imu=imu,
                                    context=SimpleNamespace(parameters=trajectory))
        accepted, rejected = trim.reliable_times(observations, board, recording, parameters)
        self.assertGreater(len(accepted), 3)
        np.testing.assert_allclose(accepted, observations.video_times_s - .01, atol=1e-9)
        # One tag cannot start/end the work interval, regardless of its pixel quality.
        single = replace(observations, usable=observations.usable & (np.arange(len(observations.tag_ids)) % 3 == 0))
        accepted, rejected = trim.reliable_times(single, board, recording, parameters)
        self.assertEqual(len(accepted), 0)
        self.assertEqual(rejected['insufficient_tags'], len(observations.video_times_s))

    def test_trim_and_vio_share_exact_pnp_hypotheses(self):
        from test_poster_fusion import fixture
        from umi.perception.poster import pnp
        from umi.perception.poster.geometry import camera_model

        _, _, _, _, _, _, parameters, observations, board, measured, _, _ = fixture()
        rows = observations.sample_indices == 0
        points = np.array([board.corners_m[tag] for tag in observations.tag_ids[rows]]).reshape(-1, 3)
        pixels = observations.corners_px[rows].reshape(-1, 2)
        solve = pnp.FrameCandidateCache()
        with patch.object(pnp, 'solve_frame_candidates', wraps=pnp.solve_frame_candidates) as solver:
            pnp.solve_frame(points, pixels, camera_model(measured),
                            parameters.tag_maximum_reprojection_rms_px, solve_candidates=solve)
            pnp.solve_frame(points.copy(), pixels.copy(), camera_model(measured),
                            2 * parameters.tag_maximum_reprojection_rms_px, solve_candidates=solve)
            solver.assert_called_once()

    def test_pnp_branch_jump_does_not_form_stable_boundary(self):
        from test_poster_fusion import fixture
        _, times, imu, _, _, trajectory, parameters, observations, board, measured, _, _ = fixture()
        measured = dict(measured, q_xyzw=[0., 0., 0., 1.], p=[0., 0., 0.])
        recording = SimpleNamespace(camera=SimpleNamespace(measurements=dict(front=measured, rear=measured)),
                                    video_times_s=times, imu=imu,
                                    context=SimpleNamespace(parameters=trajectory))
        bad = np.eye(4)
        bad[:3, :3] = np.diag([-1., -1., 1.])
        candidates = [(np.eye(4) if i % 2 == 0 else bad, 0.) for i in range(len(observations.video_times_s))]
        with patch.object(trim, 'solve_frame', side_effect=candidates):
            accepted, rejected = trim.reliable_times(observations, board, recording, parameters)
        self.assertGreater(rejected['pnp_branch_jump'], 0)
        with self.assertRaises(ValueError):
            trim.stable_interval(accepted)

    def inputs(self, root, mode='poster_vio', arms=('left', 'right')):
        scene = SimpleNamespace(recording_trim=dict(enabled=True, minimum_missing_duration_s=1.),
                                environment_tag_detection=True,
                                reconstruction_mode=mode, tag_layout='board',
                                parameters=SimpleNamespace(tag_minimum_side_px=6.))
        episode = SimpleNamespace(root=root / 'take_001', scene=scene,
                                  recordings={arm: root / f'{arm}.insv' for arm in arms})
        recordings = {arm: SimpleNamespace(video_times_s=np.arange(101) / 10., camera=SimpleNamespace(files={}))
                      for arm in arms}
        return TaskInputs(episode, recordings, {arm: (0. if arm == 'left' else .5) for arm in arms},
                          root, None, None, SimpleNamespace(write=lambda *a, **kw: None))

    def test_two_hands_use_sync_intersection_and_preserve_internal_missing_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self.inputs(Path(temporary))
            board = SimpleNamespace(ids=(12, 13), border_bits=1, dictionary='DICT_4X4_100')
            # Left 2..8, right 1..9 in native front clocks => common 2..8.
            times = [np.array([2., 2.1, 2.2, 7.8, 7.9, 8.]),
                     np.array([1., 1.1, 1.2, 8.8, 8.9, 9.])]
            snapshots = {arm: SimpleNamespace(layout=board, observations=None) for arm in inputs.recordings}
            with patch.object(trim.measurements, 'prepare_task_detections',
                              side_effect=lambda current, arm, **kwargs: snapshots[arm]), \
                 patch.object(trim, 'reliable_times', side_effect=[(t, {}) for t in times]), \
                 patch.object(trim, 'crop_recording', side_effect=lambda r, a, b: (a, b)):
                result, report = trim.prepare(inputs)
            self.assertEqual(result.recordings, dict(left=(20, 81), right=(15, 76)))
            self.assertEqual(report['common_episode_interval_s'], [2., 8.])
            self.assertEqual(inputs.recordings['left'].video_times_s.size, 101)
            self.assertEqual(report['status'], 'applied')
            self.assertEqual(json.loads((Path(temporary) / 'tag_trim.json').read_text()), report)

    def test_disabled_requires_no_detector_and_scene_map_bypasses_trim(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self.inputs(Path(temporary), mode='scene_map')
            inputs.episode.scene.recording_trim = {}
            self.assertEqual(trim.prepare(inputs), (inputs, None))
            inputs.episode.scene.recording_trim = dict(enabled=True)
            result = ArmExecutionResult({}, {}, {}, {})
            with patch.object(trim, 'prepare', side_effect=AssertionError('must not trim')), \
                 patch.dict(estimation.ESTIMATORS, scene_map=lambda *args: result):
                self.assertIs(estimation.execute(inputs, estimation.MapSelection.without_map()), result)

    def test_auto_reuses_identical_trimmed_inputs_for_both_poster_backends(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self.inputs(Path(temporary), mode='auto')
            recordings = {'right': object()}
            trimmed = replace(inputs, recordings=recordings)
            result = ArmExecutionResult({}, {}, {}, {})
            seen = []
            def vio(current, selection):
                seen.append(current.recordings)
                raise PosterVioQualityError('quality')
            def independent(current, selection):
                seen.append(current.recordings)
                return result
            with patch.object(trim, 'prepare', return_value=(trimmed, dict(status='applied'))) as prepare, \
                 patch.dict(estimation.ESTIMATORS, poster_vio=vio, poster_independent=independent):
                estimation.execute(inputs, estimation.MapSelection.without_map())
            prepare.assert_called_once_with(inputs)
            self.assertTrue(all(record is recordings for record in seen))
            self.assertEqual(result.coordinate_record['recording_trim']['status'], 'applied')

    def test_auto_reuses_scene_map_edge_trim_when_poster_constraint_is_enabled(self):
        for constraint in (True, False):
            with self.subTest(constraint=constraint), tempfile.TemporaryDirectory() as temporary:
                inputs = self.inputs(Path(temporary), mode='auto')
                scene = inputs.episode.scene
                scene.scene_map_edge_trim = True
                scene.scene_map_poster_constraint_enabled = constraint
                cropped = replace(inputs, recordings={'left': object(), 'right': object()})
                result = ArmExecutionResult({}, {}, {}, {})
                seen = {}

                def scene_map(current, selection):
                    seen['map'] = current.recordings
                    raise SceneFusionQualityError('map coverage')

                def vio(current, selection):
                    seen['vio'] = current.recordings
                    return result

                with patch.object(trim, 'prepare', return_value=(cropped, dict(status='applied'))) as prepare, \
                     patch.dict(estimation.ESTIMATORS, scene_map=scene_map, poster_vio=vio):
                    estimation.execute(inputs, estimation.MapSelection.available(Path(temporary), {}))
                prepare.assert_called_once_with(inputs)
                self.assertIs(seen['map'], cropped.recordings if constraint else inputs.recordings)
                self.assertIs(seen['vio'], cropped.recordings)
                self.assertEqual(result.coordinate_record['recording_trim']['status'], 'applied')

    def test_output_frame_provenance_and_viewer_clock_after_head_trim(self):
        from test_scene import SceneWorkflowTests
        from umi.estimation.models import FusedTrajectory
        from umi.workflows.scene.tasks.trajectory_writers import write_arm_trajectory
        from umi.store.result_io import read_arm_summary, read_arm_settings

        fixture = SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        original = fixture.prepared_recordings[fixture.recording_paths['right']]
        step = original.context.parameters.video.frame_step
        recording = replace(original, video_times_s=original.video_times_s[3:9], source_frame_start=3 * step)
        coordinate = dict(scope='local', frame_id='local:test', version=None)
        result = FusedTrajectory(np.tile(np.eye(4), (6, 1, 1)), np.ones(6, bool), np.ones(6, bool),
                                 dict(maximum_visual_gap_s=0.))
        directory = fixture.root / 'output/right'
        track = write_arm_trajectory(recording, result, .5, coordinate, directory)
        with (directory / 'details/frames.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row['video_frame']) for row in rows], list(range(6)))
        self.assertEqual([int(row['source_video_frame']) for row in rows], list(range(3 * step, 9 * step, step)))
        np.testing.assert_allclose([float(row['t_video_s']) for row in rows], recording.video_times_s)
        manifest = dict(kind='umi_local_episode', schema_version=4, status='local_ok', scene=coordinate,
            map=None, arms=dict(right='right'), arm_results=dict(right=track.metadata),
            tcp_frames=dict(right=track.tcp_frame), synchronization=dict(offsets_s=dict(right=.5)),
            inputs=dict(cameras=dict(right=dict(serial='SERIAL_right')), trajectory_parameters=dict(video={})))
        expected = float(recording.video_times_s[0] + .5)
        self.assertEqual(read_arm_summary(directory, manifest)['video_time_offset_s'], expected)
        self.assertEqual(read_arm_settings(directory, manifest)['video_time_offset_s'], expected)
        self.assertEqual(manifest['synchronization']['offsets_s']['right'], .5)

    def test_cropped_vio_features_keep_original_insv_indices(self):
        from test_poster_vio_workflow import PosterVioWorkflowTests
        from umi.workflows.scene.tasks import poster_vio_inputs
        fixture = PosterVioWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for recording in fixture.inputs.recordings.values():
            recording.source_frame_start = 60
        captured = []
        def capture(video, streams, *args, **kwargs):
            captured.extend(streams)
            return dict(front=(), rear=())
        with patch.object(poster_vio_inputs, 'track_cameras', side_effect=capture):
            poster_vio_inputs.prepare_task_inputs(fixture.inputs)
        fronts = [item for item in captured if item.camera == 'front']
        self.assertEqual(len(fronts), 2)
        for item in fronts:
            self.assertGreaterEqual(int(item.source_frames[0]), 60)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires ffmpeg')
    def test_real_video_crop_preserves_original_clock_imu_and_source_frame_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = RecordingPaths(root / 'prepared')
            paths.lens_videos.mkdir(parents=True)
            cam = paths.dataset / 'mav0/cam0'
            cam.mkdir(parents=True)
            parameters = load('trajectory').values
            parameters = replace(parameters, video=replace(parameters.video, frame_step=2,
                codec='libx264', crf=12, preset='ultrafast', keep_audio=False))
            rate = output_frame_rate(parameters.video.frame_step)
            times = 5. + np.arange(12) / float(rate)
            rows = [f'{round((t + 1) * 1e9)},unused.png' for t in times]
            (cam / 'data.csv').write_text('# timestamp,filename\n' + '\n'.join(rows) + '\n')
            imu_file = paths.dataset / 'mav0/imu0/data.csv'
            imu_file.parent.mkdir()
            imu_file.write_text('1000000000,0,0,0,0,0,9.81\n11000000000,0,0,0,0,0,9.81\n')
            manifest = dict(source=dict(serial='fixture'), frames=dict(n_frames_used=12, duration_s=1.))
            (paths.dataset / 'extract_manifest.json').write_text(json.dumps(manifest))
            video = root / 'source.insv'
            with av.open(str(video), 'w', format='mp4') as container:
                stream = container.add_stream('libx264', rate=rate * 2)
                stream.width = stream.height = 32
                stream.pix_fmt = 'yuv420p'
                for index in range(24):
                    frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 9, np.uint8), format='rgb24')
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            original_bytes = video.read_bytes()
            def render(source, context, *, lenses, source_frame_start=0):
                self.assertEqual(lenses, ('front',))
                target = context.paths.lens_videos / 'front.mp4'
                target.parent.mkdir(parents=True)
                with patch.object(make_lens_video, 'crop_xywh', return_value=(0, 0, 32, 32)), \
                     patch.object(make_lens_video, 'validate_video_parameters'):
                    make_lens_video.render(str(source), str(context.paths.dataset / 'mav0'), str(target), 'right',
                                           video=replace(parameters.video, size_px=32),
                                           source_frame_start=source_frame_start)

            camera = SimpleNamespace(serial='fixture', measurements={'front': {}, 'rear': {}},
                                     selected=SimpleNamespace(assets=SimpleNamespace(front_mask=None)))
            scene = SimpleNamespace(trajectory=SimpleNamespace(values=parameters), reconstruction_mode='poster_vio')
            settings = SimpleNamespace(front_settings=None, rear_settings=None, front_mask=None)
            cache = StageCache(root / 'cache')
            with patch.object(recordings.recording_data, 'prepare', return_value=RecordingData(manifest, imu_file)), \
                 patch.object(recordings.runtime, 'visual_settings', return_value=settings), \
                 patch.object(recordings.calibration_assets, 'slam_settings_text', return_value=''), \
                 patch.object(recordings.recording_data, 'prepare_lens_videos', side_effect=render) as encode:
                original = recordings.prepare_recording(video, camera, scene, paths.output,
                                                         cache=cache, defer_lens_videos=True)
                cropped = trim.crop_recording(original, 3, 9)
                self.assertFalse(cropped.context.paths.lens_videos.exists())
                self.assertIs(cropped.full_video_times_s, original.video_times_s)
                encode.assert_not_called()
                output_video = cropped.lens_video('front').path
                self.assertEqual(cropped.lens_video('front').path, output_video)
                rerun = recordings.prepare_recording(video, camera, scene, root / 'rerun',
                                                      cache=cache, defer_lens_videos=True)
                cached_video = trim.crop_recording(rerun, 3, 9).lens_video('front').path
                self.assertEqual(cached_video.read_bytes(), output_video.read_bytes())
                encode.assert_called_once()
            self.assertFalse((cropped.context.paths.lens_videos / 'rear.mp4').exists())
            np.testing.assert_allclose(cropped.video_times_s, times[3:9], atol=1e-9, rtol=0)
            self.assertIs(cropped.imu, original.imu)
            self.assertEqual(cropped.source_frame_start, 6)
            self.assertEqual(cropped.context.frame_count, 6)
            self.assertEqual(video.read_bytes(), original_bytes)
            self.assertEqual(json.loads((paths.dataset / 'extract_manifest.json').read_text()), manifest)
            with av.open(str(output_video)) as container:
                frames = list(container.decode(video=0))
                self.assertEqual(len(frames), 6)
                self.assertAlmostEqual(float(frames[0].pts * frames[0].time_base), 0.)
                for index, frame in enumerate(frames):
                    self.assertAlmostEqual(frame.to_ndarray(format='rgb24').mean(), (index + 3) * 18, delta=3)
            selected = (cropped.context.paths.dataset / 'mav0/cam0/data.csv').read_text().splitlines()[1:]
            self.assertEqual(selected, rows[3:9])
