"""场景结果、视频与轨迹查看、缓存和相对位姿评估。"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_scene as scene_tests
from registry_fakes import write_batch, write_kit, write_task
from scipy.spatial.transform import Rotation

from umi.apps.viewer import ViewerDataError, cache, maps
from umi.apps.viewer import video as media
from umi.apps.viewer.catalog import RecordingCatalog, capture_status, read_manifest
from umi.apps.viewer.server import byte_range, make_handler
from umi.apps.viewer.trajectory_reader import derive_body_tracks, read_trajectory
from umi.apps.viewer.video import decode_frame
from umi.config.collection_schema import normalize_collection
from umi.core.geometry import relative_pose_series
from umi.devices import registry
from umi.infra import provenance
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import publish_link, write_json, write_yaml
from umi.scene import run as workflow
from umi.scene.collection import batch as batch_inputs
from umi.scene.collection import batch as batch_inputs_task_manifest
from umi.scene.collection import batch as batches
from umi.scene.collection import batch as episode_inputs
from umi.scene.collection import batch as processing_batch
from umi.scene.collection import ingest
from umi.scene.collection import kit as kit_inputs
from umi.scene.inputs import recordings
from umi.scene.relative_pose_evaluation import evaluate
from umi.store import layout
from umi.store.result_io import arm_evidence, result_paths, scene_reference

# UMI reader and streaming contracts without requiring a browser or listening socket.


TUM = '# t x y z qx qy qz qw\n0 0 0 0 0 0 0 1\n0.5 0.3 0.4 0 0 0 0 1\n1 0.6 0.8 0 0 0 0 1\n'


class ViewerDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, layout="details", chosen=1):
        output = self.root / 'M4CP/tasks/umi_traj_out/VID_001'
        output.mkdir(parents=True)
        details = output / 'details'
        details.mkdir()
        (output / 'traj_tcp.txt').write_text(TUM)
        for name in ('camera', 'imu'):
            (details / f'traj_{name}.txt').write_text(TUM)
        (details / 'frames.csv').write_text('row,video_frame,t_video_s,constraint,front_lens_tracked\n0,0,0,full,1\n1,2,0.5,rot,1\n2,4,1,imu,0\n')
        (details / 'summary.json').write_text(json.dumps(dict(status='ok', duration_s=1, rows=3, chosen_try=chosen, calibration_used=dict(serial='camera', id='test'))))
        video = output / 'debug/reconstruction' / f'try_{chosen}' / 'front/front.mp4'
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b'0123456789')
        return output, details, video

    def test_all_layouts_and_selected_attempt_video(self):
        output, details, video = self.fixture(chosen=2)
        catalog = RecordingCatalog(self.root)
        listing = catalog.scan()
        self.assertEqual(len(listing['recordings']), 1)
        key = listing['recordings'][0]['id']
        data = catalog.load(key)
        self.assertEqual(set(data['tracks']), {'tcp', 'camera', 'imu'})
        self.assertEqual(data['times'], [0, .5, 1])
        self.assertEqual(data['tracks']['tcp']['path_length'], 1)
        self.assertEqual(data['tracks']['tcp']['speed'], [0, 1, 1])
        self.assertEqual([f['constraint'] for f in data['frames']], ['full', 'rot', 'imu'])
        self.assertEqual(catalog.video_path(key), video)

    def test_nonmonotonic_and_nonfinite_trajectories_are_rejected(self):
        path = self.root / 'traj_tcp.txt'
        for content in ('0 0 0 0 0 0 0 1\n0 1 1 1 0 0 0 1',
                        '0 nan 0 0 0 0 0 1\n1 0 0 0 0 0 0 1',
                        '0 0 0 0 0 0 0 0\n1 0 0 0 0 0 0 1'):
            path.write_text(content)
            with self.assertRaises(ViewerDataError):
                read_trajectory(path)

    def test_shared_manifest_cache_is_invalidated_and_returns_independent_data(self):
        path = self.root / 'episode.json'
        path.write_text('{"arms": {"left": "left"}}')
        first = read_manifest(path)
        first['arms']['left'] = 'changed'
        with patch.object(Path, 'read_text', side_effect=AssertionError('manifest was read again')):
            self.assertEqual(read_manifest(path)['arms'], {'left': 'left'})
        replacement = self.root / 'replacement.json'
        replacement.write_text('{"arms": {"right": "right"}}')
        replacement.replace(path)
        self.assertEqual(read_manifest(path)['arms'], {'right': 'right'})

    def test_capture_manifests_use_the_shared_yaml_version_check(self):
        task = self.root / 'scene/batch/take_001'
        task.mkdir(parents=True)
        write_yaml(task.parent / 'collection.yaml', dict(kit='test', arms=['left']))
        write_yaml(layout.metadata(task.parent, 'manifest.yaml'), dict(kit=dict(id='test', arms={'left': {}})))
        manifest = layout.metadata(task, 'episode.yaml')
        write_yaml(manifest, {'recordings': {'left': {}}})
        self.assertEqual(capture_status(task)['status'], 'pending')

        manifest.write_text('%YAML 1.1\n---\nrecordings: {left: {}}\n')
        with self.assertRaisesRegex(ViewerDataError, 'YAML 1.2'):
            capture_status(task)

    def test_missing_video_and_unaligned_optional_track(self):
        output, details, video = self.fixture()
        video.unlink()
        (details / 'traj_camera.txt').write_text(TUM.replace('0.5 ', '0.6 '))
        catalog = RecordingCatalog(self.root)
        key = catalog.scan()['recordings'][0]['id']
        data = catalog.load(key)
        self.assertIsNone(data['video_url'])
        self.assertNotIn('camera', data['tracks'])
        self.assertTrue(data['notes'])

    def test_outside_summary_symlink_is_not_exposed(self):
        output, details, video = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / 'private.json'
            outside.write_text('{"secret":true}')
            (details / 'summary.json').unlink()
            (details / 'summary.json').symlink_to(outside)
            self.assertEqual(RecordingCatalog(self.root).scan()['recordings'], [])

    def test_frame_reader_retains_poster_support_and_native_source_frame(self):
        output, details, _ = self.fixture()
        (details / 'frames.csv').write_text(
            'row,video_frame,source_video_frame,t_video_s,constraint,front_lens_tracked,position_source\n'
            '0,0,120,0,imu,0,poster\n1,1,122,0.5,full,1,visual_and_poster\n2,2,124,1,imu,0,imu\n')
        catalog = RecordingCatalog(self.root)
        data = catalog.playback(catalog.scan()['recordings'][0]['id'])
        self.assertEqual([frame['source_video_frame'] for frame in data['frames']], [120, 122, 124])
        self.assertEqual([frame['position_source'] for frame in data['frames']], ['poster', 'visual_and_poster', 'imu'])

    def test_rear_source_requires_frozen_raw_identity_and_original_frame_grid(self):
        from umi.apps.viewer.video import rear_source
        from umi.store.result_io import recording_identity

        take = self.root / 'take_001'
        run = self.root / '.pipeline/tasks' / take.name / 'runs/completed'
        arm = run / 'published/data/left'
        arm.mkdir(parents=True)
        frames = run / 'left/details/frames.csv'
        frames.parent.mkdir(parents=True)
        frames.write_text('video_frame,source_video_frame\n0,12\n1,14\n2,16\n')
        raw = take / 'left/camera.insv'
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b'fixture')
        metadata = dict(schema_version=5, inputs=dict(recordings=dict(left=recording_identity(raw))),
            arm_results=dict(left=dict(rows=3, frame_geometry=dict(
                profile='x3_360_4k_60', source_px=[3840, 1920], frame_step=2,
                rear_lens=dict(crop_xywh=[240, 240, 1440, 1440])))))
        manifest = arm.parent / 'episode.json'
        manifest.write_text(json.dumps(metadata))
        contains = RecordingCatalog(self.root).contains
        source, clip = rear_source(arm, contains)
        self.assertEqual(source, raw)
        self.assertEqual((clip.first_frame, clip.frame_step, clip.frames), (12, 2, 3))
        self.assertIn('trim=start_frame=12', clip.filter())
        self.assertIn('crop=1440:1440:240:240', clip.filter())
        frames.write_text('video_frame,source_video_frame\n0,12\n1,160\n2,180\n')
        self.assertIsNone(rear_source(arm, contains))
        frames.write_text('video_frame,source_video_frame\n0,12\n1,14\n2,16\n')
        raw.write_bytes(b'changed source')
        self.assertIsNone(rear_source(arm, contains))
        (arm / 'rear.mp4').write_bytes(b'exported')
        self.assertEqual(rear_source(arm, contains), (arm / 'rear.mp4', None))

    def test_opening_published_results_can_read_only_matching_task_arm_raw_video(self):
        from umi.store.result_io import recording_identity

        take = self.root / 'take_001'
        run = self.root / '.pipeline' / 'tasks' / take.name / 'runs' / 'completed'
        arm = run / 'published/data/left'
        arm.mkdir(parents=True)
        frames = run / 'left/details/frames.csv'
        frames.parent.mkdir(parents=True)
        frames.write_text('video_frame,source_video_frame\n0,0\n1,2\n')
        raw = take / 'left' / 'camera.insv'
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b'fixture')
        (take / 'results').symlink_to(run / 'published', target_is_directory=True)
        catalog = RecordingCatalog(take / 'results')
        catalog.directories['left'] = arm
        manifest = arm.parent / 'episode.json'
        geometry = dict(profile='x3_360_4k_60', source_px=[3840, 1920], frame_step=2,
                        rear_lens=dict(crop_xywh=[240, 240, 1440, 1440]))

        def record(path):
            manifest.write_text(json.dumps(dict(schema_version=5,
                inputs=dict(recordings=dict(left=recording_identity(path))),
                arm_results=dict(left=dict(rows=2, frame_geometry=geometry)))))

        record(raw)
        self.assertFalse(catalog.contains(raw))
        self.assertEqual(catalog.rear_source('left')[0], raw)

        for outside in (take / 'right' / 'other_arm.insv', self.root / 'take_002' / 'left' / 'different_task.insv'):
            outside.parent.mkdir(parents=True)
            outside.write_bytes(b'same file size')
            record(outside)
            self.assertIsNone(catalog.rear_source('left'))

    def test_bad_frame_times_are_unknown_instead_of_misleading(self):
        output, details, video = self.fixture()
        path = details / 'frames.csv'
        path.write_text(path.read_text().replace('1,2,0.5', '1,2,0.9'))
        catalog = RecordingCatalog(self.root)
        data = catalog.load(catalog.scan()['recordings'][0]['id'])
        self.assertEqual(data['frames'][1]['constraint'], 'unknown')
        self.assertTrue(data['notes'])


class HTTPContractTests(unittest.TestCase):
    def test_video_ranges(self):
        for header, expected in [('bytes=0-3', (0, 3)), ('bytes=8-', (8, 9)), ('bytes=-3', (7, 9)), ('bytes=0-99', (0, 9))]:
            self.assertEqual(byte_range(header, 10), expected)
        for header in ('bytes=10-', 'bytes=3-1', 'bytes=-0', 'bytes=0-1,4-5', 'invalid'):
            with self.assertRaises(ValueError):
                byte_range(header, 10)

    def test_static_listing_and_range_routes_without_socket(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'traj_tcp.txt').write_text(TUM)
            (root / 'details').mkdir()
            (root / 'details/summary.json').write_text(json.dumps(dict(status='ok')))
            path = root / 'debug/reconstruction/try_1/front/front.mp4'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'0123456789')
            catalog = RecordingCatalog(root)
            key = catalog.scan()['recordings'][0]['id']
            handler_class = make_handler(catalog)

            def request(route, headers=None, head=False):
                handler = object.__new__(handler_class)
                handler.path = route
                handler.command = 'HEAD' if head else 'GET'
                handler.headers = headers or {}
                handler.wfile = io.BytesIO()
                captured = {}
                handler.send_content_headers = lambda status, content_type, length, extra=None: captured.update(status=status, content_type=content_type, length=length, extra=extra)
                handler.do_GET()
                return captured, handler.wfile.getvalue()

            status, body = request('/api/recordings')
            self.assertEqual(status['status'], 200)
            self.assertEqual(len(json.loads(body)['recordings']), 1)
            with patch.object(catalog, 'map_data', return_value={'maps': []}) as read_map:
                status, body = request(f'/api/recordings/{key}/map')
                self.assertEqual((status['status'], json.loads(body)), (200, {'maps': []}))
                read_map.assert_called_once_with(key)
            status, body = request(f'/api/recordings/{key}/playback')
            self.assertEqual(status['status'], 200)
            self.assertEqual(json.loads(body)['times'], [0, .5, 1])
            status, body = request(f'/api/recordings/{key}/details')
            self.assertEqual(status['status'], 200)
            self.assertEqual(json.loads(body)['summary'], {'status': 'ok'})
            status, _ = request(f'/api/recordings/{key}/rear/preview')
            self.assertEqual(status['status'], 404)
            rear = root / 'rear.mp4'
            rear.write_bytes(b'rear')
            status, body = request(f'/api/recordings/{key}/playback')
            self.assertEqual(json.loads(body)['rear']['preview_url'], f'/api/recordings/{key}/rear/preview')
            status, body = request(f'/api/recordings/{key}/rear/video', {'Range': 'bytes=1-2'})
            self.assertEqual((status['status'], body), (206, b'ea'))
            with patch('umi.apps.viewer.server.preview_status', return_value={'status': 'ready', 'version': 'rear'}) as prepare:
                status, body = request(f'/api/recordings/{key}/rear/preview')
                self.assertEqual(json.loads(body)['url'], f'/api/recordings/{key}/rear/preview/video?v=rear')
                prepare.assert_called_once_with(rear, None)
            status, body = request(f'/api/recordings/{key}/video', {'Range': 'bytes=2-5'})
            self.assertEqual((status['status'], body), (206, b'2345'))
            self.assertEqual(status['extra']['Content-Range'], 'bytes 2-5/10')
            status, body = request(f'/api/recordings/{key}/video', head=True)
            self.assertEqual((status['length'], body), (10, b''))
            etag = status['extra']['ETag']
            status, body = request(f'/api/recordings/{key}/video', {'If-None-Match': etag})
            self.assertEqual((status['status'], body), (304, b''))
            with patch('umi.apps.viewer.server.preview_status', return_value={'status': 'ready', 'version': 'test'}):
                status, body = request(f'/api/recordings/{key}/preview')
                self.assertEqual(json.loads(body)['url'], f'/api/recordings/{key}/preview/video?v=test')
            cached = root/'preview.webm'
            cached.write_bytes(b'\x1aE\xdf\xa3preview')
            with patch('umi.apps.viewer.server.preview_file', return_value=cached):
                status, body = request(f'/api/recordings/{key}/preview/video?v=test', {'Range': 'bytes=0-3'})
                self.assertEqual((status['status'], status['content_type'], body), (206, 'video/webm', b'\x1aE\xdf\xa3'))
                self.assertIn('immutable', status['extra']['Cache-Control'])
            with patch('umi.apps.viewer.server.decode_frame', return_value=b'\xff\xd8jpeg') as decode:
                status, body = request(f'/api/recordings/{key}/frame?t=0.5')
                self.assertEqual((status['status'], status['content_type'], body), (200, 'image/jpeg', b'\xff\xd8jpeg'))
                decode.assert_called_once_with(path, .5)
            status, body = request(f'/api/recordings/{key}/frame?t=invalid')
            self.assertEqual(status['status'], 400)
            status, body = request('/../data.py')
            self.assertEqual(status['status'], 404)
            status, body = request('/app.mjs')
            self.assertEqual(status['content_type'], 'text/javascript')


class FrameDecodeTests(unittest.TestCase):
    def setUp(self):
        from umi.apps.viewer import video as media
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        environment = patch.dict('os.environ', {'UMI_VIEW_CACHE': self.temp.name})
        environment.start(); self.addCleanup(environment.stop)
        media._decode_cached.cache_clear()
        self.addCleanup(media._decode_cached.cache_clear)

    def test_invalid_times_are_rejected_before_starting_decoder(self):
        for seconds in (-1, float('nan'), float('inf'), 86401):
            with patch('umi.apps.viewer.video.subprocess.run') as execute:
                with self.assertRaises(ViewerDataError):
                    decode_frame(Path('input.mp4'), seconds)
                execute.assert_not_called()

    def test_decoder_outputs_jpeg_without_writing_video(self):
        with patch('umi.apps.viewer.video.find_ffmpeg', return_value='/ffmpeg'), patch('umi.apps.viewer.video.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=b'\xff\xd8jpeg', stderr=b'')) as execute:
            self.assertEqual(decode_frame(Path('input.mp4'), 17.35), b'\xff\xd8jpeg')
            command = execute.call_args.args[0]
            self.assertEqual(command[-1], 'pipe:1')
            self.assertEqual(command[command.index('-ss') + 1], '17.350000')

    def test_empty_decoded_frame_reports_error(self):
        with patch('umi.apps.viewer.video.find_ffmpeg', return_value='/ffmpeg'), patch('umi.apps.viewer.video.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=b'', stderr=b'')):
            with self.assertRaisesRegex(ViewerDataError, '没有可用画面'):
                decode_frame(Path('input.mp4'), 0)


# 查看器缓存跨重启复用、失效、原子发布和后台任务去重。


class ViewerCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        environment = patch.dict('os.environ', {'UMI_VIEW_CACHE': str(self.root/'cache')})
        environment.start(); self.addCleanup(environment.stop)
        self.video = self.root/'source.mp4'
        self.video.write_bytes(b'original video')
        media._decode_cached.cache_clear()
        maps._convert.cache_clear()
        self.addCleanup(media._decode_cached.cache_clear)
        self.addCleanup(maps._convert.cache_clear)

    def test_map_survives_memory_cache_reset_and_invalidates_on_input_change(self):
        first = dict(kind='umi_map_view', schema_version=1, maps=[], note='first')
        second = dict(first, note='second')
        identity = (cache.file_identity(self.video),)
        with patch.object(maps, '_export', return_value=first) as export:
            self.assertEqual(maps._convert(self.root, identity), first)
        export.assert_called_once()
        maps._convert.cache_clear()
        with patch.object(maps, '_export', side_effect=AssertionError('must reuse disk')):
            self.assertEqual(maps._convert(self.root, identity), first)
        self.video.write_bytes(b'updated video')
        with patch.object(maps, '_export', return_value=second) as export:
            self.assertEqual(maps._convert(self.root, (cache.file_identity(self.video),)), second)
        export.assert_called_once()

    def test_corrupt_map_cache_is_regenerated(self):
        data = dict(kind='umi_map_view', schema_version=1, maps=[])
        identity = (cache.file_identity(self.video),)
        target = cache.path('maps', identity, '.json')
        target.parent.mkdir(parents=True)
        target.write_text('{broken')
        with patch.object(maps, '_export', return_value=data):
            self.assertEqual(maps._convert(self.root, identity), data)
        self.assertEqual(json.loads(target.read_text()), data)

    def test_failed_publish_keeps_old_cache_and_removes_partial_file(self):
        target = self.root/'saved.json'
        target.write_bytes(b'complete')
        with self.assertRaises(RuntimeError):
            with cache.atomic_output(target) as temporary:
                temporary.write_bytes(b'partial')
                raise RuntimeError('interrupted')
        self.assertEqual(target.read_bytes(), b'complete')
        self.assertEqual(list(self.root.glob('.building-*')), [])

    def test_first_frame_survives_memory_reset_and_repeated_frames_are_cached(self):
        response = SimpleNamespace(returncode=0, stdout=b'\xff\xd8jpeg', stderr=b'')
        with patch.object(media, 'find_ffmpeg', return_value='/ffmpeg'), patch.object(media.subprocess, 'run', return_value=response) as run:
            media.decode_frame(self.video, 0)
            media._decode_cached.cache_clear()
            media.decode_frame(self.video, 0)
            self.assertEqual(run.call_count, 1)
            media.decode_frame(self.video, 1.5)
            media.decode_frame(self.video, 1.5)
            self.assertEqual(run.call_count, 2)
            self.video.write_bytes(b'changed input')
            media.decode_frame(self.video, 0)
            self.assertEqual(run.call_count, 3)

    def test_preview_work_is_deduplicated_and_ready_file_reused(self):
        entered, release = threading.Event(), threading.Event()
        def encode(video, target):
            entered.set(); release.wait(5)
            with cache.atomic_output(target) as temporary:
                temporary.write_bytes(b'\x1aE\xdf\xa3preview')
        with patch.object(media, 'find_ffmpeg', return_value='/ffmpeg'), patch.object(media, '_encode_preview', side_effect=encode) as encoder:
            try:
                self.assertEqual(media.preview_status(self.video)['status'], 'preparing')
                self.assertTrue(entered.wait(3))
                self.assertEqual(media.preview_status(self.video)['status'], 'preparing')
                self.assertEqual(encoder.call_count, 1)
            finally:
                release.set()
                media._jobs[media._preview_target(self.video)].result(timeout=5)
            state = media.preview_status(self.video)
            self.assertEqual(state['status'], 'ready')
            self.assertEqual(media.preview_file(self.video, state['version']).read_bytes(), b'\x1aE\xdf\xa3preview')
            with self.assertRaises(ValueError):
                media.preview_file(self.video, 'stale-version')
            self.assertEqual(encoder.call_count, 1)

    def test_rear_preview_retains_crop_and_frame_grid_in_background_job(self):
        clip = media.RearClip((240, 240, 1440, 1440), 12, 2, 3)

        def run(command, **kwargs):
            video_filter = command[command.index('-vf') + 1]
            self.assertIn('trim=start_frame=12', video_filter)
            self.assertIn('crop=1440:1440:240:240', video_filter)
            self.assertEqual(command[command.index('-frames:v') + 1], '3')
            Path(command[-1]).write_bytes(b'\x1aE\xdf\xa3preview')
            return SimpleNamespace(returncode=0, stderr=b'')

        with patch.object(media, 'find_ffmpeg', return_value='/ffmpeg'), patch.object(media.subprocess, 'run', side_effect=run):
            target = media._preview_target(self.video, clip)
            self.assertNotEqual(target, media._preview_target(self.video))
            media.preview_status(self.video, clip)
            media._jobs[target].result(timeout=5)
            self.assertEqual(media.preview_status(self.video, clip)['status'], 'ready')
            self.assertEqual(media.preview_file(self.video, target.stem, clip), target)

    def test_failed_encoder_never_publishes_partial_preview(self):
        def run(command, **kwargs):
            Path(command[-1]).write_bytes(b'partial')
            return SimpleNamespace(returncode=1, stderr=b'encoder failed')
        with patch.object(media, 'find_ffmpeg', return_value='/ffmpeg'), patch.object(media.subprocess, 'run', side_effect=run):
            target = media._preview_target(self.video)
            with self.assertRaisesRegex(ValueError, 'encoder failed'):
                media._encode_preview(self.video, target)
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.glob('.building-*')), [])
        self.assertEqual(self.video.read_bytes(), b'original video')


# 当前场景结果的坐标、输入身份和读取边界。


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
        command = ('import sys; from umi.apps.viewer.trajectory_reader import derive_body_tracks; assert not ({"numpy", "scipy", "gtsam"} & sys.modules.keys())')
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
        self.fixture = scene_tests.SceneWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

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
        for filename in ('traj_camera.txt', 'traj_imu.txt'):
            self.assertFalse(arm_evidence(output / 'data/left', filename).exists())
        catalog = RecordingCatalog(output)
        loaded = catalog.load(catalog.scan()['recordings'][0]['id'])
        self.assertEqual(set(loaded['tracks']), {'tcp', 'camera', 'imu'})
        with (output / 'data/left/trajectory.csv').open() as stream:
            self.assertEqual(csv.DictReader(stream).fieldnames,
                             ['row', 't_video_s', 't_episode_s', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        (output / 'data/left/trajectory.csv').unlink()
        (output / 'data/left/traj_tcp.txt').write_text('0 0 0 0 0 0 0 1\n')
        self.assertEqual(catalog.scan()['recordings'], [])

    def test_retired_results_are_hidden_and_recomputed(self):
        fixture = self.fixture
        fixture.build_scene()
        workflow.run_episode(fixture.episode_path)
        output = fixture.task_root / 'results'
        paths = result_paths(output)
        manifest = json.loads(paths.manifest.read_text())
        for version in (3, 4):
            manifest['schema_version'] = version
            with self.assertRaisesRegex(ValueError, '受支持的结果版本'):
                scene_reference(manifest)
        write_json(paths.manifest, manifest)
        self.assertEqual(RecordingCatalog(output).scan()['recordings'], [])

        write_json(paths.run / 'episode.json', manifest)
        (paths.run / 'left/traj_tcp.txt').write_text('0 0 0 0 0 0 0 1\n')
        shutil.rmtree(paths.published)
        publish_link(paths.run, output)

        self.assertEqual(RecordingCatalog(output).scan()['recordings'], [])
        pending, = RecordingCatalog(fixture.batch_root).scan()['recordings']
        self.assertEqual(pending['status'], 'pending')
        self.assertFalse(pending['available'])
        self.assertEqual(workflow.run_episode(fixture.episode_path)['status'], 'ok')
        catalog = RecordingCatalog(output)
        entry, = catalog.scan()['recordings']
        self.assertIn('camera', catalog.load(entry['id'])['tracks'])
        self.assertEqual(catalog.map_data(entry['id'])['kind'], 'umi_map_view')
        self.assertNotEqual(result_paths(output).run, paths.run)
        self.assertEqual(json.loads(result_paths(output).manifest.read_text())['schema_version'], 5)


"""场景内原片：先放视频、后登记，保留原片身份并避免重复复制。"""


class SceneVideoLayoutTests(unittest.TestCase):
    def setUp(self):
        self.f = scene_tests.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        for override in (
            patch.object(kit_inputs, 'KITS', self.f.root / 'kits'),
            patch.object(registry, 'get', return_value={}),
            patch.object(registry, 'pin', side_effect=lambda serial: self.f.device_pins[serial]),
            patch.object(registry, 'read_insv_serial', side_effect=lambda path: 'SERIAL_' + Path(path).read_text().split('_')[0]),
        ):
            override.start()
            self.addCleanup(override.stop)
        write_kit('rig', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(left='SERIAL_left', right='SERIAL_right')).items()})

    def test_auto_uses_mapping_sources_when_the_batch_has_them(self):
        batch = self.f.scene.root / 'auto_mapping'
        mapping = batch / 'mapping/map.insv'
        mapping.parent.mkdir(parents=True)
        mapping.write_text('mapping')
        collection = normalize_collection(dict(kit='rig', arms=['left'], backend=dict(mode='auto')))
        for directory, expected in ((batch, True), (self.f.scene.root / 'auto_without_mapping', False)):
            with self.subTest(directory=directory):
                has_mapping = bool(recordings.insv_files(directory / 'mapping'))
                self.assertEqual(batches.mapping_requirements(collection, has_mapping=has_mapping),
                                 (expected, False))

    def test_prepared_scene_and_batch_are_discovered_without_copying_local_videos(self):
        scene = self.f.root / 'scenes/shelf'
        batch = scene / '20261104_001'
        mapping = batch / 'mapping/VID_20261104_090000_00_001.INSV'
        mapping.parent.mkdir(parents=True)
        mapping.write_text('left_mapping')
        infos = []
        for arm in ('left', 'right'):
            path = batch / 'take_001' / arm / 'VID_20261104_100000_00_001.insv'
            path.parent.mkdir(parents=True)
            path.write_text(arm)
            infos.append(ingest.RecordingMetadata(path, 'SERIAL_' + arm, artifact_identity.file_hash(path), 1000., 8.))
        before = {path: (path.read_bytes(), path.stat().st_ino) for path in scene.rglob('*.insv')}
        write_yaml(batch / 'collection.yaml', dict(description='', kit='rig',
                   arms=['left', 'right'],
                   map=dict(layout='auto', reference_arm='left'),
                   backend=dict(mode='scene_map')))
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
            preview = processing_batch.prepare_batch(batch, dry_run=True)
        self.assertEqual(len(preview['episodes']), 1)
        self.assertEqual(preview['episodes'][0]['name'], 'take_001')
        self.assertFalse((scene / 'collection.yaml').exists())
        self.assertFalse((batch / '.pipeline/manifest.yaml').exists())
        self.assertFalse((scene / '.scene.lock').exists())
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
            processing_batch.prepare_batch(batch)
        manifest = batch / '.pipeline/manifest.yaml'
        episode = episode_inputs.read_episode(batch / 'take_001')
        self.assertEqual(episode.recordings, {arm: infos[index].path for index, arm in enumerate(('left', 'right'))})
        self.assertEqual(batch_inputs.read_batch(manifest).scene.recordings['left'], mapping)
        self.assertEqual({path: (path.read_bytes(), path.stat().st_ino) for path in scene.rglob('*.insv')}, before)
        self.assertEqual(episode.recording_refs['left']['path'], 'left/' + infos[0].path.name)

    def test_registering_prepared_batch_preserves_videos_and_rejects_other_content(self):
        batch = self.f.scene.root / 'prepared'
        video = batch / 'take_001/left/VID.insv'
        video.parent.mkdir(parents=True)
        video.write_text('left')
        manifest = write_batch(self.f.scene.root, batch.name, kit='rig')
        self.assertEqual(video.read_text(), 'left')
        self.assertEqual(layout.owner(manifest), batch)
        before = {path: (path.read_bytes(), path.stat().st_ino) for path in batch.rglob('*') if path.is_file()}
        self.assertEqual(batches.save_record(batches.read_inputs(batch, require_mapping=False)), manifest)
        self.assertEqual({path: (path.read_bytes(), path.stat().st_ino) for path in batch.rglob('*') if path.is_file()}, before)
        other = self.f.scene.root / 'other'
        write_yaml(other / 'take_001/.pipeline/episode.yaml', dict(schema_version=5))
        with self.assertRaisesRegex(ValueError, '只允许'):
            write_batch(self.f.scene.root, other.name, kit='rig')

    def test_external_same_named_videos_are_separated_by_actual_arm(self):
        manifest = write_batch(self.f.scene.root, 'external', kit='rig')
        session = batch_inputs.read_batch(manifest)
        paths = {}
        for arm in ('left', 'right'):
            path = self.f.root / 'sd' / arm / 'VID.insv'
            path.parent.mkdir(parents=True)
            path.write_text(arm)
            paths[arm] = path
        episode = episode_inputs.read_episode(write_task(session, 'take_001', paths))
        for arm in ('left', 'right'):
            self.assertEqual(episode.recordings[arm], layout.owner(manifest) / 'take_001' / arm / 'VID.insv')
            self.assertEqual(paths[arm].read_bytes(), episode.recordings[arm].read_bytes())

    def test_pending_pair_is_relocated_after_success_and_dry_run_keeps_sources(self):
        manifest = write_batch(self.f.scene.root, 'pair_later', kit='rig')
        infos = [ingest.RecordingMetadata(path, 'SERIAL_' + arm, artifact_identity.file_hash(path), 1000., 8.)
                 for arm, path in self.f.recording_paths.items()]
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos[:1], [])):
            first = ingest.register_recordings(batch_inputs.read_batch(manifest))
        pending = Path(first['pending'][0]['path'])
        self.assertEqual(pending.parent, layout.owner(manifest) / 'pending/left')
        self.assertTrue(pending.is_file())
        self.assertFalse((layout.owner(manifest) / 'take_001').exists())
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
            ingest.register_recordings(batch_inputs.read_batch(manifest), dry_run=True)
            self.assertTrue(pending.is_file())
            self.assertFalse((layout.owner(manifest) / 'take_001').exists())
            result = ingest.register_recordings(batch_inputs.read_batch(manifest))
        episode = episode_inputs.read_episode(Path(result['episodes'][0]['path']))
        self.assertFalse((layout.owner(manifest) / 'pending').exists())
        for arm in ('left', 'right'):
            self.assertEqual(episode.recordings[arm].parent, episode.root / arm)
            self.assertTrue(self.f.recording_paths[arm].is_file())

    def test_prearranged_tasks_do_not_mix_even_when_file_times_are_unknown(self):
        batch = self.f.scene.root / 'prearranged'
        infos = []
        for name in ('take_010', 'take_011'):
            for arm in ('left', 'right'):
                path = batch / name / arm / 'VID.insv'
                path.parent.mkdir(parents=True)
                path.write_text(arm + '_' + name)
                infos.append(ingest.RecordingMetadata(path, 'SERIAL_' + arm, artifact_identity.file_hash(path), None, 8.))
        manifest = write_batch(self.f.scene.root, batch.name, kit='rig')
        before = {info.path: info.path.stat().st_ino for info in infos}
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
            result = ingest.register_recordings(batch_inputs.read_batch(manifest))
        self.assertEqual(result['status'], 'ok')
        self.assertEqual([item['name'] for item in result['episodes']], ['take_010', 'take_011'])
        for item in result['episodes']:
            episode = episode_inputs.read_episode(Path(item['path']))
            for arm, path in episode.recordings.items():
                self.assertEqual(path.read_text(), arm + '_' + item['name'])
        self.assertEqual({info.path: info.path.stat().st_ino for info in infos}, before)

    def test_multiple_candidates_in_one_task_are_retained_for_confirmation(self):
        for mode in ('flat', 'nested'):
            with self.subTest(mode=mode):
                batch = self.f.scene.root / ('ambiguous_' + mode)
                infos = []
                for arm, name in (('left', 'one'), ('left', 'two'), ('right', 'one')):
                    root = batch / 'take_001'
                    root /= arm
                    if mode == 'nested' and name == 'two':
                        root /= 'extra'
                    path = root / (name + '.insv')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(arm + '_' + name)
                    infos.append(ingest.RecordingMetadata(path, 'SERIAL_' + arm, artifact_identity.file_hash(path), None, 8.))
                manifest = write_batch(self.f.scene.root, batch.name, kit='rig')
                with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
                    result = ingest.register_recordings(batch_inputs.read_batch(manifest))
                self.assertEqual(result['episodes'], [])
                self.assertEqual({item['status'] for item in result['pending']}, {'ambiguous_task'})
                self.assertEqual(len(result['pending']), 3)
                self.assertFalse((batch / 'pending').exists())
                self.assertTrue(all(info.path.exists() for info in infos))

    def test_import_does_not_reference_another_batchs_mutable_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / 'first/left/VID.insv'
            original.parent.mkdir(parents=True)
            original.write_text('left')
            target = root / 'second/left'
            imported = recordings.import_video(original, target, metadata=recordings.inspect(original))
            self.assertEqual(imported, target / original.name)
            original.write_text('changed')
            self.assertEqual(imported.read_text(), 'left')

    def test_batch_discovery_scans_only_raw_arm_directories_and_pending(self):
        batch = self.f.root / 'discovery'
        wanted = ['take_001/left/a.INSV', 'take_001/right/b.insv',
                  'take_002/left/extra/c.insv', 'pending/right/d.insv']
        ignored = ['take_001/results/left/output.insv', 'take_001/.pipeline/cache/raw.insv',
                   'take_002/videos/left/old.insv',
                   'mapping/left/map.insv', '.hidden/left/hidden.insv',
                   'results/left/result.insv', 'cache/left/cache.insv']
        for name in wanted + ignored:
            path = batch / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        def inspect(path):
            return ingest.RecordingMetadata(path, 'SERIAL_left', artifact_identity.file_hash(path), None, 8.)
        with patch.object(ingest, 'inspect_recording', side_effect=inspect):
            found, errors = ingest.scan_raw_videos(batch_inputs_task_manifest.batch_video_sources(batch))
        self.assertEqual(errors, [])
        self.assertEqual({item.path for item in found}, {batch / name for name in wanted})

    def test_legacy_and_mixed_tasks_are_rejected_without_changing_videos(self):
        for mode in ('legacy', 'mixed'):
            with self.subTest(mode=mode):
                batch = self.f.scene.root / mode
                infos = []
                for arm in ('left', 'right'):
                    root = batch / 'take_001'
                    if mode == 'legacy' or arm == 'left':
                        root /= 'videos'
                    path = root / arm / 'VID.insv'
                    path.parent.mkdir(parents=True)
                    path.write_text(arm)
                    infos.append(ingest.RecordingMetadata(path, 'SERIAL_' + arm, artifact_identity.file_hash(path), None, 8.))
                before = {item.path: (item.path.read_bytes(), item.path.stat().st_ino) for item in infos}
                with self.assertRaisesRegex(ValueError, '未登记任务只允许'):
                    write_batch(self.f.scene.root, batch.name, kit='rig')
                self.assertEqual({item.path: (item.path.read_bytes(), item.path.stat().st_ino) for item in infos}, before)
                self.assertFalse(layout.metadata(batch, 'manifest.yaml').exists())

    def test_incomplete_task_accepts_peer_without_copying_original(self):
        batch = self.f.scene.root / 'supplement'
        left = batch / 'take_001/left/VID.insv'
        left.parent.mkdir(parents=True)
        left.write_text('left')
        manifest = write_batch(self.f.scene.root, batch.name, kit='rig')
        inode = left.stat().st_ino
        session = batch_inputs.read_batch(manifest)
        task = write_task(session, 'take_001', dict(left=left))
        current = episode_inputs.read_episode(task)
        clips = dict(current.recordings, right=self.f.recording_paths['right'])
        metadata = [ingest.RecordingMetadata(clip, 'SERIAL_' + arm, artifact_identity.file_hash(clip), 1000., 8.)
                    for arm, clip in clips.items()]
        with patch.object(ingest, 'scan_raw_videos', return_value=(metadata, [])):
            ingest.register_recordings(session)
        episode = episode_inputs.read_episode(task)
        self.assertEqual(episode.recordings['left'], left)
        self.assertEqual(episode.recordings['right'], batch / 'take_001/right/right.insv')
        self.assertEqual(left.stat().st_ino, inode)
        self.assertFalse((batch / 'take_001/videos').exists())

    def test_flat_right_only_task_preserves_hand_identity_and_has_no_videos_layer(self):
        batch = self.f.scene.root / 'right_only'
        path = batch / 'take_001/right/VID.insv'
        path.parent.mkdir(parents=True)
        path.write_text('right')
        manifest = write_batch(self.f.scene.root, batch.name, kit='rig', arms=('right',))
        info = ingest.RecordingMetadata(path, 'SERIAL_right', artifact_identity.file_hash(path), None, 8.)
        with patch.object(ingest, 'scan_raw_videos', return_value=([info], [])):
            result = ingest.register_recordings(batch_inputs.read_batch(manifest))
        self.assertEqual(result['status'], 'ok')
        task = episode_inputs.read_episode(Path(result['episodes'][0]['path']))
        self.assertEqual(task.expected_arms, ('right',))
        self.assertEqual(task.recording_refs['right']['path'], 'right/VID.insv')
        self.assertFalse((task.root / 'videos').exists())
        self.assertFalse((task.root / 'left').exists())


# 固定夹具实测参考：不使用拟合对齐隐藏相对平移、尺度和 yaw 误差。


class RelativeEvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run = self.root/'run'
        self.run.mkdir()
        self.data = self.run/'published/data'
        self.times = np.arange(31)/20.
        self.left = np.tile(np.eye(4), (len(self.times), 1, 1))
        self.left[:, :3, :3] = Rotation.from_euler('z', self.times[:, None]).as_matrix()
        self.left[:, :3, 3] = np.column_stack([np.sin(self.times), self.times*.2, self.times*0])
        self.reference_pose = np.eye(4)
        self.reference_pose[:3, 3] = [.3, .1, .05]
        self.reference_pose[:3, :3] = Rotation.from_euler('z', 25., degrees=True).as_matrix()
        self.right = self.left @ self.reference_pose
        self.reference = dict(kind='umi_rigid_pair_reference', schema_version=1,
            measurement_note='synthetic independently specified fixture',
            T_left_tcp_right_tcp=self.reference_pose.tolist(), interval_s=[.1, 1.4],
            maximum_position_error_m=.01, maximum_rotation_error_deg=1., maximum_interpolation_gap_s=.25)
        self.reference_path = self.root/'reference.json'
        self.reference_path.write_text(json.dumps(self.reference))

    def write(self, *, confirmed=True, keep=None):
        assets = {}
        for arm, matrices in [('left', self.left), ('right', self.right)]:
            path = self.data/arm/'trajectory.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            indices = np.arange(len(self.times)) if arm == 'left' or keep is None else keep
            with path.open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['t_episode_s', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
                for i in indices:
                    writer.writerow([self.times[i], *matrices[i, :3, 3], *Rotation.from_matrix(matrices[i, :3, :3]).as_quat()])
            assets[f'data/{arm}/trajectory.csv'] = provenance.file_hash(path)
        diagnostics = self.run/'diagnostics.json'
        diagnostics.write_text('{}')
        manifest = dict(kind='umi_poster_episode', schema_version=6, status='ok',
            scene=dict(scope='task_poster', frame_id='poster:fixture', version=None), map=None,
            arms=dict(left='left', right='right'), assets=assets,
            evidence_assets={'diagnostics.json': provenance.file_hash(diagnostics)},
            synchronization=dict(confirmed=confirmed, offsets_s=dict(left=0., right=0.)),
            arm_results={arm: dict(fusion=dict(scene_map_priors_used=False,
                environment_poster=dict(status='fused'))) for arm in ('left', 'right')})
        (self.data/'episode.json').write_text(json.dumps(manifest))

    def test_common_motion_and_coordinate_changes_preserve_relative_accuracy(self):
        common = np.eye(4)
        common[:3, :3] = Rotation.from_euler('xyz', [.4, -.2, 1.]).as_matrix()
        common[:3, 3] = [15., -3., 8.]
        self.left = common @ self.left
        self.right = common @ self.right
        self.write()
        result = evaluate(self.run, self.reference_path)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['alignment'], 'none')
        self.assertLess(result['position_error_m']['maximum'], 1e-12)
        self.assertLess(result['rotation_error_deg']['maximum'], 1e-10)
        np.testing.assert_allclose(relative_pose_series(self.left, self.right),
                                   np.tile(self.reference_pose, (len(self.times), 1, 1)), atol=1e-12)

    def test_translation_scale_and_yaw_biases_fail_without_fitting(self):
        for kind in ('position', 'scale', 'yaw'):
            with self.subTest(kind=kind):
                wrong = self.reference_pose.copy()
                if kind == 'position':
                    wrong[0, 3] += .03
                elif kind == 'scale':
                    wrong[:3, 3] *= 1.2
                else:
                    wrong[:3, :3] = wrong[:3, :3] @ Rotation.from_euler('z', 3., degrees=True).as_matrix()
                self.right = self.left @ wrong
                self.write()
                self.assertEqual(evaluate(self.run, self.reference_path)['status'], 'failed')

    def test_unconfirmed_sync_and_sparse_data_cannot_pass(self):
        self.write(confirmed=False)
        result = evaluate(self.run, self.reference_path)
        self.assertEqual(result['status'], 'not_ready')
        self.assertIn('synchronization_unconfirmed', result['not_ready_reasons'])
        self.write(keep=np.r_[0:5, 14:31])
        self.assertIn('interpolation_gap_exceeded', evaluate(self.run, self.reference_path)['not_ready_reasons'])

    def test_corrupt_assets_and_out_of_range_interval_are_rejected(self):
        self.write()
        (self.data/'right/trajectory.csv').write_text('corrupt')
        with self.assertRaisesRegex(ValueError, '内容已变'):
            evaluate(self.run, self.reference_path)
        self.write()
        self.reference['interval_s'] = [0., 2.]
        self.reference_path.write_text(json.dumps(self.reference))
        with self.assertRaisesRegex(ValueError, '完整落在'):
            evaluate(self.run, self.reference_path)


if __name__ == '__main__':
    unittest.main()
