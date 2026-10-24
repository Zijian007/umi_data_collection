"""UMI reader and streaming contracts without requiring a browser or listening socket."""
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

from software.viewer.data import RecordingCatalog, ViewerDataError, read_trajectory
from software.viewer.server import byte_range, make_handler
from software.viewer.media import decode_frame

TUM = '# t x y z qx qy qz qw\n0 0 0 0 0 0 0 1\n0.5 0.3 0.4 0 0 0 0 1\n1 0.6 0.8 0 0 0 0 1\n'


class ViewerDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, layout="details", chosen=1):
        output = self.root / 'M4CP/tasks/umi_traj_out/VID_001'
        output.mkdir(parents=True)
        details = output if layout == 'flat' else output / layout
        details.mkdir(exist_ok=True)
        (output / 'traj_tcp.txt').write_text(TUM)
        for name in ('camera', 'imu'):
            (details / f'traj_{name}.txt').write_text(TUM)
        (details / 'frames.csv').write_text('row,video_frame,t_video_s,constraint,front_lens_tracked\n0,0,0,full,1\n1,2,0.5,rot,1\n2,4,1,imu,0\n')
        (details / 'summary.json').write_text(json.dumps(dict(status='ok', duration_s=1, rows=3, chosen_try=chosen, calibration_used=dict(serial='camera', id='test'))))
        attempt = 'kiwi' if chosen == 1 else f'kiwi_try{chosen}'
        video = output / 'front.mp4' if layout == 'details' else details / 'work' / attempt / 'front/front.mp4'
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b'0123456789')
        return output, details, video

    def test_all_layouts_and_selected_attempt_video(self):
        for layout in ('flat', 'diagnostics', 'details'):
            with self.subTest(layout=layout):
                if self.root.exists():
                    import shutil
                    shutil.rmtree(self.root)
                output, details, video = self.fixture(layout=layout, chosen=2)
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
            path = root / 'work/kiwi/front/front.mp4'
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
            status, body = request(f'/api/recordings/{key}/video', {'Range': 'bytes=2-5'})
            self.assertEqual((status['status'], body), (206, b'2345'))
            self.assertEqual(status['extra']['Content-Range'], 'bytes 2-5/10')
            status, body = request(f'/api/recordings/{key}/video', head=True)
            self.assertEqual((status['length'], body), (10, b''))
            with patch('software.viewer.server.decode_frame', return_value=b'\xff\xd8jpeg') as decode:
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
    def test_invalid_times_are_rejected_before_starting_decoder(self):
        for seconds in (-1, float('nan'), float('inf'), 86401):
            with patch('software.viewer.media.subprocess.run') as execute:
                with self.assertRaises(ViewerDataError):
                    decode_frame(Path('input.mp4'), seconds)
                execute.assert_not_called()

    def test_decoder_outputs_jpeg_without_writing_video(self):
        with patch('software.viewer.media.find_ffmpeg', return_value='/ffmpeg'), patch('software.viewer.media.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=b'\xff\xd8jpeg', stderr=b'')) as execute:
            self.assertEqual(decode_frame(Path('input.mp4'), 17.35), b'\xff\xd8jpeg')
            command = execute.call_args.args[0]
            self.assertEqual(command[-1], 'pipe:1')
            self.assertEqual(command[command.index('-ss') + 1], '17.350000')

    def test_empty_decoded_frame_reports_error(self):
        with patch('software.viewer.media.find_ffmpeg', return_value='/ffmpeg'), patch('software.viewer.media.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=b'', stderr=b'')):
            with self.assertRaisesRegex(ViewerDataError, '没有可用画面'):
                decode_frame(Path('input.mp4'), 0)


if __name__ == '__main__':
    unittest.main()
