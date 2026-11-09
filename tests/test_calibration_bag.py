"""Parallel PNG decoding must preserve ROS messages and close failed outputs."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from software.calibration.tools.calib_folder_to_bag import write_mono_bag


class BagConversionTests(unittest.TestCase):
    def make_dataset(self, root):
        cam, imu = root/'mav0/cam0', root/'mav0/imu0'
        (cam/'data').mkdir(parents=True)
        imu.mkdir(parents=True)
        images = {}
        for k in range(12):
            timestamp = 1_000_000_000 + k * 10_000_000
            pixels = np.arange(48, dtype=np.uint8).reshape(6, 8) + k
            if k % 3 == 1:
                pixels = pixels.astype(np.uint16) * 1000
            elif k % 3 == 2:
                pixels = np.stack((pixels, pixels + 2, pixels + 4), axis=-1)
            cv2.imwrite(str(cam/f'data/{k}.png'), pixels)
            images[timestamp] = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY) if pixels.ndim == 3 else pixels
        # Deliberately unsorted CSV: prefetch order must follow sorted events.
        (cam/'data.csv').write_text(''.join(f'{1_000_000_000 + k * 10_000_000},{k}.png\n' for k in reversed(range(12))))
        (imu/'data.csv').write_text(''.join(f'{t+5_000_000},1,2,3,4,5,6\n' for t in images))
        return images

    def test_parallel_conversion_preserves_bag_bytes_pixels_and_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.make_dataset(root)
            serial = write_mono_bag(root, root/'one.bag', cam_time_offset_ns=5_000_000, workers=1)
            parallel = write_mono_bag(root, root/'four.bag', cam_time_offset_ns=5_000_000, workers=4)
            self.assertEqual(serial['bag_sha256'], parallel['bag_sha256'])
            self.assertEqual((parallel['n_images'], parallel['n_imu']), (12, 12))
            types = get_typestore(Stores.ROS1_NOETIC)
            times, image_count, imu_count = [], 0, 0
            with Reader(root/'four.bag') as reader:
                for connection, timestamp, raw in reader.messages():
                    msg = types.deserialize_ros1(raw, connection.msgtype)
                    times.append(timestamp)
                    self.assertEqual(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec, timestamp)
                    if connection.topic == '/cam0/image_raw':
                        pixels = expected[timestamp-5_000_000]
                        dtype = np.uint16 if msg.encoding == 'mono16' else np.uint8
                        np.testing.assert_array_equal(msg.data.view(dtype).reshape(msg.height, msg.width), pixels)
                        self.assertEqual(msg.header.seq, image_count)
                        self.assertEqual(msg.step, pixels.shape[1] * pixels.dtype.itemsize)
                        image_count += 1
                    else:
                        self.assertEqual((msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z), (1, 2, 3))
                        self.assertEqual((msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z), (4, 5, 6))
                        self.assertEqual(msg.header.seq, imu_count)
                        imu_count += 1
            self.assertEqual(times, sorted(times))
            self.assertEqual((image_count, imu_count), (12, 12))

    def test_missing_image_propagates_and_closes_partial_bag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            (root/'mav0/cam0/data/6.png').unlink()
            with self.assertRaisesRegex(FileNotFoundError, '6.png'):
                write_mono_bag(root, root/'partial.bag', workers=4)
            self.assertFalse((root/'partial.bag.manifest.json').exists())
            with Reader(root/'partial.bag') as reader:
                self.assertTrue(all(t < 1_060_000_000 for _, t, _ in reader.messages()))
