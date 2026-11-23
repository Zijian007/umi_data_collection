"""UMI reader and streaming contracts without requiring a browser or listening socket."""
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

from umi.apps.viewer.results.catalog import RecordingCatalog
from umi.apps.viewer.results.catalog import read_manifest
from umi.apps.viewer import ViewerDataError
from umi.apps.viewer.results.trajectory_reader import read_trajectory
from umi.apps.viewer.server import byte_range, make_handler
from umi.apps.viewer.resources.video import decode_frame

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
        from umi.store.result_io import recording_identity
        from umi.apps.viewer.resources.video import rear_source

        arm = self.root / 'take_001' / 'results' / 'left'
        (arm / 'details').mkdir(parents=True)
        frames = arm / 'details' / 'frames.csv'
        frames.write_text('video_frame,source_video_frame\n0,12\n1,14\n2,16\n')
        raw = self.root / 'take_001' / 'camera.insv'
        raw.write_bytes(b'fixture')
        metadata = dict(schema_version=4, inputs=dict(recordings=dict(left=recording_identity(raw))),
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
        arm = run / 'left'
        (arm / 'details').mkdir(parents=True)
        (arm / 'details' / 'frames.csv').write_text('video_frame,source_video_frame\n0,0\n1,2\n')
        raw = take / 'left' / 'camera.insv'
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b'fixture')
        (take / 'results').symlink_to(run, target_is_directory=True)
        catalog = RecordingCatalog(take / 'results')
        catalog.directories['left'] = arm
        manifest = run / 'episode.json'
        geometry = dict(profile='x3_360_4k_60', source_px=[3840, 1920], frame_step=2,
                        rear_lens=dict(crop_xywh=[240, 240, 1440, 1440]))

        def record(path):
            manifest.write_text(json.dumps(dict(schema_version=4,
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
        from umi.apps.viewer.resources import video as media
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        environment = patch.dict('os.environ', {'UMI_VIEW_CACHE': self.temp.name})
        environment.start(); self.addCleanup(environment.stop)
        media._decode_cached.cache_clear()
        self.addCleanup(media._decode_cached.cache_clear)

    def test_invalid_times_are_rejected_before_starting_decoder(self):
        for seconds in (-1, float('nan'), float('inf'), 86401):
            with patch('umi.apps.viewer.resources.video.subprocess.run') as execute:
                with self.assertRaises(ViewerDataError):
                    decode_frame(Path('input.mp4'), seconds)
                execute.assert_not_called()

    def test_decoder_outputs_jpeg_without_writing_video(self):
        with patch('umi.apps.viewer.resources.video.find_ffmpeg', return_value='/ffmpeg'), patch('umi.apps.viewer.resources.video.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=b'\xff\xd8jpeg', stderr=b'')) as execute:
            self.assertEqual(decode_frame(Path('input.mp4'), 17.35), b'\xff\xd8jpeg')
            command = execute.call_args.args[0]
            self.assertEqual(command[-1], 'pipe:1')
            self.assertEqual(command[command.index('-ss') + 1], '17.350000')

    def test_empty_decoded_frame_reports_error(self):
        with patch('umi.apps.viewer.resources.video.find_ffmpeg', return_value='/ffmpeg'), patch('umi.apps.viewer.resources.video.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=b'', stderr=b'')):
            with self.assertRaisesRegex(ViewerDataError, '没有可用画面'):
                decode_frame(Path('input.mp4'), 0)


if __name__ == '__main__':
    unittest.main()
