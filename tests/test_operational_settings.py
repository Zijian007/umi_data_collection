"""操作配置的冻结、覆盖、资源预算与缓存边界。"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from software import settings
from software.gripper_width import detection, pipeline, tag_distance
from software.scene import scheduling
from software.storage import write_yaml
from software.viewer import cache, maps, media, server


class OperationalSettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for kind in settings.SCHEMA:
            (self.root / f'{kind}.yaml').write_bytes((settings.DIRECTORY / f'{kind}.yaml').read_bytes())
        self.defaults = {kind: dict(settings.load(kind, directory=self.root, environment={}).values)
                         for kind in settings.SCHEMA}
        directory = patch.object(settings, 'DIRECTORY', self.root)
        directory.start()
        self.addCleanup(directory.stop)
        environment = patch.dict(os.environ, dict(os.environ))
        environment.start()
        self.addCleanup(environment.stop)
        for name in (settings.SNAPSHOT_ENV, 'UMI_OPENCV_THREADS'):
            os.environ.pop(name, None)
        settings.current.cache_clear()
        self.addCleanup(settings.current.cache_clear)
        media.shutdown()
        self.addCleanup(media.shutdown)

    def overlay(self, kind, **values):
        write_yaml(self.root / f'{kind}.local.yaml', values)
        settings.current.cache_clear()

    def test_defaults_local_environment_cli_and_origins(self):
        self.overlay('runtime', opencv_threads=3, decode_threads=2)
        configured = settings.load('runtime', environment={'UMI_OPENCV_THREADS': '6'})
        self.assertEqual(configured.values['opencv_threads'], 6)
        self.assertEqual(configured.values['decode_threads'], 2)
        self.assertEqual(configured.origins['opencv_threads'], 'environment:UMI_OPENCV_THREADS')
        self.assertTrue(configured.origins['decode_threads'].endswith('runtime.local.yaml'))
        with self.assertRaises(TypeError):
            configured.values['decode_threads'] = 7
        self.overlay('viewer', port=8000)
        configured = settings.load('viewer', overrides={'port': 0})
        self.assertEqual(configured.values['port'], 0)
        self.assertEqual(configured.origins['port'], 'command_line:--port')
        self.assertEqual(self.defaults['viewer']['port'], 9092)

    def test_invalid_fields_types_ranges_duplicates_and_relations(self):
        cases = [('runtime', {'decode_threads': True}), ('runtime', {'decode_threads': 0}),
                 ('runtime', {'decode_threads': 1.5}), ('runtime', {'decode_threads': None}),
                 ('runtime', {'unknown': 2}), ('viewer', {'port': 65536}),
                 ('viewer', {'preview_workers': 3, 'preview_pending': 2}),
                 ('viewer', {'preview_width': 721}), ('viewer', {'map_max_points': 50001}),
                 ('viewer', {'host': ''}), ('viewer', {'preview_timeout_s': float('inf')}),
                 ('tag_detection', {'recovery_batch_size': -1})]
        for kind, overlay in cases:
            with self.subTest(kind=kind, overlay=overlay):
                self.overlay(kind, **overlay)
                with self.assertRaises(ValueError):
                    settings.load(kind, environment={})
                (self.root / f'{kind}.local.yaml').unlink()
        (self.root / 'viewer.local.yaml').write_text('port: 9092\nport: 9093\n')
        with self.assertRaisesRegex(ValueError, 'viewer.local.yaml'):
            settings.load('viewer')

    def test_snapshot_is_frozen_in_process_and_in_child(self):
        first = settings.current('runtime')
        write_yaml(self.root / 'runtime.local.yaml', {'decode_threads': 2})
        self.assertIs(settings.current('runtime'), first)
        env = settings.child_environment()
        # The child uses the snapshot even though the YAML now differs.
        code = "from software import settings; print(settings.values('runtime')['decode_threads'])"
        child = subprocess.run([sys.executable, '-c', code], env=env, text=True,
                               capture_output=True, check=True)
        self.assertEqual(int(child.stdout), first.values['decode_threads'])
        self.assertEqual(settings.load('runtime').values['decode_threads'], 2)

    def test_budget_respects_cpu_affinity_cli_and_custom_threshold(self):
        self.overlay('runtime', calibration_max_workers=6, calibration_reserved_cpus=2, arm_min_cpus=3)
        with patch.object(settings.os, 'sched_getaffinity', return_value=set(range(16))):
            report = settings.execution_report(cores='2-7', workers=20, calibration=True)
            self.assertEqual(report['resources']['available_cpus'], list(range(2, 8)))
            self.assertEqual(report['resources']['calibration_workers'], 6)
            self.assertEqual(report['resources']['opencv_threads'], 1)
            self.assertEqual(settings.calibration_workers(cores='2-7'), 4)
            self.assertEqual(scheduling.arm_cores(('left', 'right'), '2-7', parallel=True),
                             {'left': '2,4,6', 'right': '3,5,7'})
            self.assertFalse(settings.execution_report(jobs=2)['resources']['dual_arm_eligible'])
            with self.assertRaisesRegex(ValueError, '--cores'):
                settings.execution_report(cores='40-42')

    def test_runtime_changes_reuse_detection_but_detection_changes_invalidate_final_inputs(self):
        original = [function() for function in (detection.identity, pipeline.identity, tag_distance.identity)]
        self.overlay('runtime', decode_threads=2, slam_timeout_base_s=900)
        self.assertEqual(original, [function() for function in (detection.identity, pipeline.identity, tag_distance.identity)])
        self.overlay('tag_detection', candidate_limit=48, full_search_interval=10)
        for before, function in zip(original, (detection.identity, pipeline.identity, tag_distance.identity)):
            self.assertNotEqual(before, function())
        detector = detection.MarkerDetector()
        self.assertEqual(detector._candidate_limit, 48)
        self.assertEqual(detector._full_search_interval, 10)

    def test_viewer_content_changes_invalidate_only_related_artifacts(self):
        video = self.root / 'video.mp4'
        video.write_bytes(b'video')
        with patch.object(media, 'find_ffmpeg', return_value='/ffmpeg'):
            preview, frame = media._preview_target(video), media.video_version(video)
            self.overlay('viewer', preview_workers=2, preview_pending=3, preview_timeout_s=900,
                         preview_encode_threads=2, frame_cache_entries=1)
            self.assertEqual(media._preview_target(video), preview)
            self.assertEqual(media.video_version(video), frame)
            self.overlay('viewer', preview_crf=20)
            self.assertNotEqual(media._preview_target(video), preview)
            self.assertEqual(media.video_version(video), frame)
            self.overlay('viewer', frame_quality=8)
            self.assertEqual(media._preview_target(video), preview)
            self.assertNotEqual(media.video_version(video), frame)

    def test_viewer_configured_cache_and_single_resource_bundle(self):
        self.overlay('viewer', frame_cache_entries=1, preview_workers=2, preview_pending=3,
                     decode_concurrency=3)
        with ThreadPoolExecutor(max_workers=8) as threads:
            bundles = list(threads.map(lambda _: media._resources(), range(20)))
        self.assertTrue(all(bundle is bundles[0] for bundle in bundles))
        semaphore, executor = bundles[0]
        self.assertEqual(executor._max_workers, 2)
        for _ in range(3):
            self.assertTrue(semaphore.acquire(blocking=False))
        self.assertFalse(semaphore.acquire(blocking=False))
        for _ in range(3):
            semaphore.release()
        calls = []
        @cache.bounded('frame_cache_entries')
        def invoke(key):
            calls.append(key)
            return key
        for key in (1, 1, 2, 1):
            invoke(key)
        self.assertEqual(calls, [1, 2, 1])

    def test_display_budget_does_not_reject_or_mutate_published_map(self):
        self.overlay('viewer', map_max_points=2)
        document = dict(kind='umi_map_view', schema_version=1, maps=[dict(index=0,
            coordinates='map', alignment=dict(status='unaligned'), points=[[i, 0, 0] for i in range(5)],
            keyframes=[], bounds=None)])
        self.assertEqual(len(maps.validate(document)['maps'][0]['points']), 5)
        with patch.object(maps, '_load', return_value=document):
            self.assertEqual(len(maps.load(self.root, {})['maps'][0]['points']), 2)
        self.assertEqual(len(document['maps'][0]['points']), 5)

    def test_viewer_show_config_uses_local_and_cli_without_resources(self):
        self.overlay('viewer', port=8000, preview_workers=2, preview_pending=3)
        with patch.object(server, 'RecordingCatalog') as catalog, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(server.main(['--show-config', '--port', '0']), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report['effective']['port'], 0)
        self.assertEqual(report['effective']['preview_workers'], 2)
        self.assertEqual(report['map_export']['timeout_s'], 180)
        self.assertEqual(report['configuration']['origins']['port'], 'command_line:--port')
        self.assertIsNone(media._resource_pair)
        catalog.assert_not_called()

    def test_invalid_config_is_reported_before_listen_and_help_still_works(self):
        self.overlay('viewer', port='oops')
        with patch.object(server, 'ThreadingHTTPServer') as listen, redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as error:
                server.main([])
        self.assertEqual(error.exception.code, 2)
        self.assertIn('viewer.local.yaml', output.getvalue())
        listen.assert_not_called()
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as help_exit:
            server.main(['--help'])
        self.assertEqual(help_exit.exception.code, 0)
