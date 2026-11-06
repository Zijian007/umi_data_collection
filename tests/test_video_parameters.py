"""可配置裁切、取帧与编码：验证真实画面、相机几何、时钟和资产不变性。"""
from contextlib import redirect_stdout
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from software import config
from software.camera.images import prepare_mask
from software.camera.x3 import extraction, frame
from software.trajectory.calibration import assets
from software.vision.tools import make_lens_video

PROJECT = Path(__file__).resolve().parents[1]
TOOLS_PATH = str(Path(os.environ.get('UMI_DEPS', PROJECT / '.deps')) / 'env/bin') + os.pathsep + os.environ.get('PATH', '')
FFMPEG = shutil.which('ffmpeg', path=TOOLS_PATH)
FFPROBE = shutil.which('ffprobe', path=TOOLS_PATH)

class VideoParametersTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def video(self, **values):
        return config.load('trajectory', overrides={'video': values}).values.video

    def test_crop_resize_preserves_camera_rays_for_both_templates(self):
        import cv2
        video = self.video(crop=dict(size_px=1200, offset_x_px=-37, offset_y_px=51), size_px=600, frame_step=4)
        camera = assets.read_basalt(assets.SAMPLE_CAMERA_JSON)
        parameters = config.load('trajectory', overrides={'video': {'crop': dict(size_px=1200, offset_x_px=-37, offset_y_px=51), 'size_px': 600, 'frame_step': 4}}).values
        native_pixel = [875., 413.]
        x, y, size, _ = frame.calibrated_crop_xywh(video.crop)
        for rear in (False, True):
            base = cv2.FileStorage(assets.slam_settings_text(camera, config.load('trajectory').values, rear=rear),
                                  cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
            generated = cv2.FileStorage(assets.slam_settings_text(camera, parameters, rear=rear),
                                       cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
            for pixel, offset, focal, centre in zip(native_pixel, (x, y), ('fx', 'fy'), ('cx', 'cy')):
                output_pixel = (pixel-offset+.5)*video.size_px/size-.5
                old_pixel = (pixel+.5)*720/1440-.5
                expected_ray = (old_pixel-base.getNode('Camera1.'+centre).real())/base.getNode('Camera1.'+focal).real()
                actual_ray = (output_pixel-generated.getNode('Camera1.'+centre).real())/generated.getNode('Camera1.'+focal).real()
                self.assertAlmostEqual(actual_ray, expected_ray, places=10)
            self.assertEqual(generated.getNode('Camera.fps').real(), 15)
            self.assertEqual([generated.getNode(f'Camera1.k{k}').real() for k in range(1, 5)],
                             [base.getNode(f'Camera1.k{k}').real() for k in range(1, 5)])

    def test_orb_requires_integer_nominal_fps_while_frame_clock_stays_exact(self):
        import cv2
        camera = assets.read_basalt(assets.SAMPLE_CAMERA_JSON)
        for step, expected in ((1, 60), (2, 30), (4, 15), (120, 1)):
            parameters = config.load('trajectory', overrides={'video': {'frame_step': step}}).values
            for rear in (False, True):
                with self.subTest(step=step, rear=rear):
                    settings = cv2.FileStorage(assets.slam_settings_text(camera, parameters, rear=rear),
                                              cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
                    node = settings.getNode('Camera.fps')
                    self.assertTrue(node.isInt())
                    self.assertEqual(node.real(), expected)
            self.assertEqual(frame.output_frame_rate(step), Fraction(60000, 1001 * step))

    def test_crop_mask_uses_same_region_and_preserves_source(self):
        import cv2
        source = self.root / 'mask.png'
        mask = np.zeros((1440, 1440), dtype=np.uint8)
        mask[600:, 700:] = 255
        cv2.imwrite(str(source), mask)
        before = source.read_bytes()
        video = self.video(crop=dict(size_px=1000, offset_x_px=20, offset_y_px=-30), size_px=500)
        x, y, size, _ = frame.calibrated_crop_xywh(video.crop)
        target = prepare_mask(source, self.root / 'processed.png', video.size_px, crop=video.crop)
        expected = cv2.resize(mask[y:y+size, x:x+size], (500, 500), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(cv2.imread(str(target), cv2.IMREAD_GRAYSCALE), expected)
        self.assertEqual(source.read_bytes(), before)

    def test_invalid_combinations_are_rejected_before_processing(self):
        cases = [dict(frame_step=0), dict(frame_step=True), dict(crop=dict(size_px=1500)),
                 dict(crop=dict(offset_x_px=1)), dict(crop=dict(size_px=1200, offset_y_px=-121)),
                 dict(crop=dict(size_px=600), size_px=720), dict(codec='shell-command'),
                 dict(pixel_format='rgb'), dict(resize_filter='unknown'), dict(keep_audio='yes'),
                 dict(audio_bitrate_kbps=0)]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.video(**values)

    def test_extracted_timestamps_use_configured_step(self):
        raw = self.root / 'source.insv'
        raw.write_bytes(b'raw')
        times = np.arange(-.05, 1.1, .005)
        imu = dict(t_s=times, gyr=np.zeros((len(times), 3)), acc=np.zeros((len(times), 3)),
                   acc_range_g=16, gyro_range_dps=2000)
        info = dict(video_streams=[dict(w=3840, h=1920)], average_rate=60000/1001)
        for step in (1, 3, 4):
            with self.subTest(step=step), patch.object(extraction, 'video_stream_info', return_value=info), \
                    patch.object(extraction, 'read_imu_and_meta', return_value=imu), \
                    patch.object(extraction, 'presentation_pts', return_value=(list(range(12)), 1001/60000)):
                output = self.root / f'frames-{step}'
                manifest = extraction.extract(raw, output, frame_step=step)
                stored = np.loadtxt(output / 'mav0/cam0/data.csv', delimiter=',', comments='#', usecols=0)/1e9-1
                np.testing.assert_allclose(stored, np.arange(0, 12, step)*1001/60000, atol=1e-9)
                self.assertEqual(manifest['frames']['step'], step)
                self.assertTrue(manifest['frames']['regular'])
                self.assertAlmostEqual(manifest['frames']['fps_used'], 60000/1001/step)

    def test_video_changes_invalidate_computation_identity(self):
        original = config.load('trajectory')
        for values in (dict(frame_step=4), dict(crop=dict(size_px=1200)), dict(codec='libx265'),
                       dict(pixel_format='yuv444p'), dict(resize_filter='lanczos'), dict(keep_audio=True)):
            with self.subTest(values=values):
                changed = config.load('trajectory', overrides={'video': values})
                self.assertNotEqual(changed.computation_hash, original.computation_hash)

    @unittest.skipUnless(FFMPEG and FFPROBE, 'FFmpeg required')
    def test_real_mp4_frame_rate_dimensions_and_audio_follow_yaml(self):
        import av
        environment = patch.dict(os.environ, PATH=TOOLS_PATH)
        environment.start()
        self.addCleanup(environment.stop)
        raw = self.root / 'source.insv'
        generated = subprocess.run([FFMPEG, '-v', 'error', '-f', 'lavfi', '-i',
            'color=gray:size=3840x1920:rate=60000/1001', '-f', 'lavfi', '-i',
            'sine=frequency=440:sample_rate=16000', '-frames:v', '12', '-t', '0.2002',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', '-f', 'mp4', str(raw)],
            capture_output=True, text=True, timeout=40)
        self.assertEqual(generated.returncode, 0, generated.stderr)
        for step, audio in ((1, True), (4, False)):
            video = self.video(frame_step=step, crop=dict(size_px=1200, offset_x_px=11, offset_y_px=-9),
                               size_px=64, resize_filter='lanczos', keep_audio=audio)
            mav = self.root / f'mav-{step}/cam0'
            mav.mkdir(parents=True)
            count = len(range(0, 12, step))
            (mav / 'data.csv').write_text('#timestamp,filename\n'+'1000000000,frame.png\n'*count)
            target = self.root / f'front-{step}.mp4'
            with redirect_stdout(io.StringIO()):
                make_lens_video.render(str(raw), str(mav.parent), str(target), 'right', video=video)
            with av.open(str(target)) as container:
                stream = container.streams.video[0]
                self.assertEqual((stream.width, stream.height), (64, 64))
                self.assertEqual(stream.average_rate, Fraction(60000, 1001*step))
                self.assertEqual(bool(container.streams.audio), audio)
                decoded = list(container.decode(video=0))
                self.assertEqual(len(decoded), count)
            manifest = json.loads(target.with_name(target.stem+'_manifest.json').read_text())
            self.assertEqual(manifest['crop_xywh'], list(frame.crop_xywh('right', video.crop)))
            self.assertEqual(manifest['parameters']['keep_audio'], audio)
