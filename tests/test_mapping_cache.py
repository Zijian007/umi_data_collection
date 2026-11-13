"""建图内容复用、阶段缓存失效、失败保护和最终地图定位回归。"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from software import config, provenance
from software.scene.map_cache import MapBuildCache, source_identity
from software.scene.pipeline import map_lifecycle
from software.scene.spatial import slam
from software.vision.models import LensVideo, MapAtlas
import test_scene as fixtures


class CacheStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = MapBuildCache(self.root / 'cache', {})

    def run_stage(self, name, *, identity=None, fail=False, cache=None, refresh=False):
        output = self.root / name
        selected = cache or self.cache
        with selected.stage('build', '视觉建图', identity or {'video': 'one'}, output,
                            roots=('result',), refresh=refresh) as reused:
            if not reused:
                (output / 'result').mkdir(parents=True)
                (output / 'result/map.osa').write_bytes(b'complete atlas')
                if fail:
                    raise RuntimeError('interrupted')
        return output, reused

    def test_reuse_validates_and_copies_outputs_without_sharing_writable_files(self):
        original, reused = self.run_stage('first')
        self.assertFalse(reused)
        shutil.rmtree(original)
        restored, reused = self.run_stage('second')
        self.assertTrue(reused)
        (restored / 'result/map.osa').write_bytes(b'work changed')
        final, reused = self.run_stage('third')
        self.assertTrue(reused)
        self.assertEqual((final / 'result/map.osa').read_bytes(), b'complete atlas')
        self.assertEqual([entry['status'] for entry in self.cache.timings], ['computed', 'cached', 'cached'])

    def test_linked_stage_shares_read_only_files_and_rejects_writes(self):
        def run_linked(name):
            output = self.root / name
            with self.cache.stage('prepare', '录像准备', {'video': 'one'}, output,
                                  roots=('result',), link=True) as reused:
                if not reused:
                    (output / 'result').mkdir(parents=True)
                    (output / 'result/front.mp4').write_bytes(b'decoded')
            return output / 'result/front.mp4', reused
        first, reused = run_linked('first')
        self.assertFalse(reused)
        second, reused = run_linked('second')
        self.assertTrue(reused)
        cached = next((self.root / 'cache/prepare').rglob('front.mp4'))
        self.assertEqual({first.stat().st_ino, second.stat().st_ino}, {cached.stat().st_ino})
        with self.assertRaises(PermissionError):
            second.write_bytes(b'changed')
        self.assertEqual(cached.read_bytes(), b'decoded')

    def test_corrupted_cache_recomputes_instead_of_returning_damaged_atlas(self):
        self.run_stage('first')
        next((self.root / 'cache').rglob('map.osa')).write_bytes(b'broken')
        _, reused = self.run_stage('second')
        self.assertFalse(reused)
        self.assertTrue(self.run_stage('third')[1])

    def test_interrupted_stage_never_becomes_ready_and_can_be_retried(self):
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.run_stage('failed', fail=True)
        self.assertEqual(list((self.root / 'cache').rglob('ready.json')), [])
        self.assertEqual(self.cache.timings[-1]['status'], 'failed')
        self.assertFalse(self.run_stage('retry')[1])

    def test_force_and_retry_refresh_recompute_but_failed_refresh_preserves_cache(self):
        self.run_stage('first')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.run_stage('failed', fail=True, refresh=True)
        self.assertTrue(self.run_stage('after_failure')[1])
        forced = MapBuildCache(self.root / 'cache', {}, force=True)
        self.assertFalse(self.run_stage('forced', cache=forced)[1])
        self.assertFalse(self.run_stage('retry', refresh=True)[1])

    def test_input_change_and_incomplete_manifest_cannot_hit_cache(self):
        self.run_stage('first')
        self.assertFalse(self.run_stage('changed', identity={'video': 'two'})[1])
        for ready in (self.root / 'cache').rglob('ready.json'):
            ready.unlink()
        self.assertFalse(self.run_stage('partial')[1])


class MapReuseTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SceneWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_same_recording_in_another_batch_reuses_map_and_keeps_original_provenance(self):
        first = self.fixture.build_scene()
        source = self.fixture.recording_paths['left']
        copied = self.fixture.scene.root / 'another_batch/mapping/copy.insv'
        copied.parent.mkdir(parents=True)
        shutil.copyfile(source, copied)
        self.fixture.cameras[copied] = self.fixture.cameras[source]
        moved_scene = replace(self.fixture.scene, recordings={'left': copied})
        with patch.object(slam, 'build', side_effect=AssertionError('same content rebuilt')):
            second = map_lifecycle._build_scene(moved_scene)
        self.assertEqual(second['status'], 'skipped')
        self.assertEqual(second['version'], first['version'])
        _, manifest = map_lifecycle.load_published(moved_scene)
        self.assertEqual(manifest['inputs']['recordings']['left']['path'], str(source))
        self.assertEqual(manifest['inputs']['recordings']['left']['sha256'], provenance.file_hash(copied))

    def test_changed_content_with_same_size_and_mtime_requires_new_map(self):
        import os
        first = self.fixture.build_scene()
        source = self.fixture.recording_paths['left']
        before = source.stat()
        source.write_bytes(b'LEFT')
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        second = map_lifecycle._build_scene(self.fixture.scene)
        self.assertNotEqual(first['version'], second['version'])

    def test_map_identity_excludes_cli_and_task_only_changes_but_tracks_algorithm_sources(self):
        sources = source_identity('map')
        self.assertNotIn('software/scene/cli.py', sources)
        self.assertNotIn('software/scene/pipeline/reconstruction.py', sources)
        self.assertNotIn('software/viewer/server.py', sources)
        for dependency in ('software/vision/slam.py', 'software/camera/x3/extraction.py',
                           'software/trajectory/calibration/assets.py',
                           'software/trajectory/algorithms/fusion_alignment.py',
                           'software/trajectory/algorithms/fusion_alignment_solver.py',
                           'software/trajectory/algorithms/inertial.py', 'software/vision/decoding.py'):
            self.assertIn(dependency, sources)
        original_hash = provenance.source_hash
        for module in ('fusion_alignment.py', 'fusion_alignment_solver.py', 'inertial.py', 'decoding.py'):
            with self.subTest(module=module):
                def changed_hash(path):
                    return 'changed' if path.name == module else original_hash(path)
                with patch.object(provenance, 'source_hash', side_effect=changed_hash):
                    self.assertNotEqual(source_identity('map'), sources)

    def test_tcp_change_keeps_map_but_calibration_and_threshold_changes_invalidate(self):
        first = self.fixture.build_scene()
        tcp = self.fixture.cameras[self.fixture.recording_paths['left']].files['tcp']
        measurement = json.loads(tcp.read_text())
        measurement['T_camera_tcp']['t_m'] = [0.01, 0, 0]
        tcp.write_text(json.dumps(measurement))
        self.assertEqual(self.fixture.build_scene()['version'], first['version'])
        tuned = replace(self.fixture.scene, parameters=replace(self.fixture.scene.parameters,
            maximum_alignment_condition=self.fixture.scene.parameters.maximum_alignment_condition * 2))
        self.assertNotEqual(map_lifecycle._build_scene(tuned)['version'], first['version'])
        camera = self.fixture.cameras[self.fixture.recording_paths['left']]
        self.fixture.cameras[self.fixture.recording_paths['left']] = replace(camera,
            selected=replace(camera.selected, calibration_id='new-calibration'))
        self.assertNotEqual(self.fixture.build_scene()['version'], first['version'])

    def test_preparation_ignores_alignment_and_orb_parameters_but_not_video_changes(self):
        scene = self.fixture.scene
        source = self.fixture.recording_paths['left']
        camera = self.fixture.cameras[source]
        cache = MapBuildCache(self.fixture.root / 'cache', {})
        original = cache.preparation_inputs(source, camera, scene)
        tuned = replace(scene, parameters=replace(scene.parameters, maximum_scale_relative_std=.4),
                        trajectory=replace(scene.trajectory, values=replace(scene.trajectory.values,
                            orb=replace(scene.trajectory.values.orb, features=999))))
        self.assertEqual(cache.preparation_inputs(source, camera, tuned), original)
        resized = replace(scene, trajectory=replace(scene.trajectory, values=replace(scene.trajectory.values,
                          video=replace(scene.trajectory.values.video, size_px=640))))
        self.assertNotEqual(cache.preparation_inputs(source, camera, resized), original)

    def test_preparation_ignores_calibration_but_tracks_decoding(self):
        source = self.fixture.recording_paths['left']
        camera = self.fixture.cameras[source]
        cache = MapBuildCache(self.fixture.root / 'cache', {'ffmpeg': 'one', 'orb_library': 'x'})
        original = cache.preparation_inputs(source, camera, self.fixture.scene)
        recalibrated = replace(camera, selected=replace(camera.selected, calibration_id='new-calibration'))
        self.assertEqual(cache.preparation_inputs(source, recalibrated, self.fixture.scene), original)
        self.assertEqual(original['engines'], {'ffmpeg': 'one'})
        sources = source_identity('prepare')
        for unrelated in ('software/registry.py', 'software/camera/calibration_source.py',
                          'software/trajectory/calibration/assets.py', 'software/scene/map_cache.py'):
            self.assertNotIn(unrelated, sources)
        for decoder in ('software/camera/x3/extraction.py', 'software/vision/tools/make_lens_video.py'):
            self.assertIn(decoder, sources)
        newer_ffmpeg = MapBuildCache(self.fixture.root / 'cache', {'ffmpeg': 'two'})
        self.assertNotEqual(newer_ffmpeg.preparation_inputs(source, camera, self.fixture.scene), original)


class VisualStageCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, content in [('video.mp4', 'pixels'), ('front.yaml', 'intrinsics'), ('mask.png', 'mask')]:
            (self.root / name).write_text(content)
        self.video = LensVideo('front', self.root / 'video.mp4', self.root / 'front.yaml', self.root / 'mask.png')
        self.recording = SimpleNamespace(
            lens_video=lambda lens: self.video, video_times_s=np.array([0., 1.]),
            context=SimpleNamespace(frame_count=2, parameters=config.load('trajectory').values),
            camera=SimpleNamespace(measurements={'front': {'cam_time_offset_ns': 0}}))
        self.cache = MapBuildCache(self.root / 'cache', {'orb_library': 'engine-one'})
        self.build_patch = patch.object(slam.visual_slam, 'build_maps', side_effect=self.build_engine)
        self.localize_patch = patch.object(slam.visual_slam, 'localize_map', side_effect=self.localize_engine)
        self.build = self.build_patch.start()
        self.localize = self.localize_patch.start()
        self.addCleanup(self.build_patch.stop)
        self.addCleanup(self.localize_patch.stop)

    def build_engine(self, videos, context, directory):
        archive = directory / 'front/map/map.osa'
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b'atlas')
        return (MapAtlas(videos[0], archive),)

    def localize_engine(self, atlas, index, output, *args, **kwargs):
        output.mkdir(parents=True)
        (output / 'frames.csv').write_text('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw\n'
                                           '0,0,2,1,0,0,0,0,0,0,1\n1,1,2,1,1,0,0,0,0,0,1\n')
        (output / 'log.txt').write_text('atlas_map index 0 id 0 keyframes 100\n'
                                       'atlas_map index 1 id 1 keyframes 20\n')

    def run_visual(self, name, *, cache=None):
        cache = cache or self.cache
        directory = self.root / name
        atlas = slam.build(self.recording, 'front', directory / 'build', cache=cache)
        return list(slam.atlas_candidates(self.recording, 'front', atlas.archive,
                                         directory / 'localize', cache=cache))

    def test_reuses_atlas_and_all_submap_localizations_with_identical_observations(self):
        original = self.run_visual('first')
        restored = self.run_visual('second')
        self.assertEqual(self.build.call_count, 1)
        self.assertEqual(self.localize.call_count, 2)
        self.assertEqual([index for index, _ in restored], [0, 1])
        for (_, before), (_, after) in zip(original, restored):
            np.testing.assert_array_equal(before.indices, after.indices)
            np.testing.assert_array_equal(before.T_reference_camera, after.T_reference_camera)
        # 同一固定地图观测按本次标定时间偏移读取，不缓存已偏移的 IMU 时间。
        self.recording.camera.measurements['front']['cam_time_offset_ns'] = 1000000
        shifted = self.run_visual('shifted')
        np.testing.assert_allclose(shifted[0][1].imu_times_s, original[0][1].imu_times_s + .001)
        self.assertEqual(self.localize.call_count, 2)

    def test_camera_settings_engine_and_pixels_invalidate_visual_stages(self):
        self.run_visual('first')
        self.video.settings.write_text('new intrinsics')
        self.run_visual('new_intrinsics')
        self.video.path.write_text('new pixels')
        self.run_visual('new_pixels')
        changed_engine = MapBuildCache(self.root / 'cache', {'orb_library': 'engine-two'})
        self.run_visual('new_engine', cache=changed_engine)
        self.assertEqual(self.build.call_count, 4)
        self.assertEqual(self.localize.call_count, 8)

    def test_mask_affects_build_and_force_recomputes_all_visual_stages(self):
        self.run_visual('first')
        self.video.mask.write_text('new mask')
        self.run_visual('new_mask')
        self.assertEqual(self.build.call_count, 2)
        # 测试引擎产生完全相同的地图字节，因此定位可以复用；真实地图改变则哈希改变。
        self.assertEqual(self.localize.call_count, 2)
        forced = MapBuildCache(self.root / 'cache', self.cache.engines, force=True)
        self.run_visual('forced', cache=forced)
        self.assertEqual(self.build.call_count, 3)
        self.assertEqual(self.localize.call_count, 4)
