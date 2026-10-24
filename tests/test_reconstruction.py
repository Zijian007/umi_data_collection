"""重建阶段协议与执行顺序；替代 FFmpeg/SLAM/GTSAM，实际运行定位合并。"""
import os
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from software import config
from software.trajectory import calibration_assets, recording_data, reconstruction, runtime
from software.trajectory.models import (
    FusionResult, RecordingData, RecordingPaths, ReconstructionContext, ReconstructionError, TrajectoryRequest,
)
from trajectory_fakes import FakeEngines


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engines = FakeEngines(self.root)
        self.paths = RecordingPaths(self.root / 'output')
        self.paths.logs.mkdir(parents=True)
        environment = dict(os.environ, **self.engines.environment,
                           KIWI_FRONT_SETTING=str(calibration_assets.FACTORY_YAML),
                           KIWI_FUSER=str(calibration_assets.FACTORY_FUSER))
        self.settings = runtime.reconstruction_settings(environment, camera_json=calibration_assets.FACTORY_JSON,
                                                        front_mask=runtime.HERE / "assets/config/front_hand_mask.png",
                                                        output=self.paths.debug / "settings", parameters=config.load("trajectory").values)
        self.context = ReconstructionContext(self.paths, config.resolve_trajectory_request(TrajectoryRequest()), self.settings, 100)
        self.engines.write_dataset(self.paths.dataset)
        self.data = RecordingData(dict(source={'name': 'VID.insv'}), self.paths.dataset / 'mav0/imu0/data.csv')
        self.video = self.root / 'VID.insv'
        self.video.write_bytes(b'raw recording')
        execute = patch.object(runtime.subprocess, 'run', side_effect=self.engines)
        execute.start()
        self.addCleanup(execute.stop)
        quiet = patch.object(runtime.console, 'write')
        quiet.start()
        self.addCleanup(quiet.stop)

    def run_stages(self, context=None, data=None):
        context = context or self.context
        videos = recording_data.prepare_lens_videos(self.video, context)
        return reconstruction.run_attempts(videos, data or self.data, context, recording_name=self.video.name)

    def test_explicit_stage_order_and_forwarded_data(self):
        result = self.run_stages()
        self.assertEqual([kind for kind, *_ in self.engines.calls], ['video', 'map', 'localize', 'merge', 'fuse'])
        self.assertEqual(result.chosen, 1)
        self.assertEqual(result.attempts[0]['full_fraction'], 1.0)
        calls = {kind: (command, options) for kind, command, options in self.engines.calls}
        mapping, _ = calls['map']
        locating, _ = calls['localize']
        fusion, _ = calls['fuse']
        self.assertEqual(mapping[mapping.index('--input_video')+1], locating[locating.index('--input_video')+1])
        self.assertEqual(mapping[mapping.index('--save_map')+1], locating[locating.index('--load_map')+1])
        self.assertIn(str(self.paths.attempt(1) / 'front/loc/frames.csv'), fusion)
        self.assertIn(str(self.data.imu_csv), fusion)
        self.assertEqual(json.loads(self.settings.parameters_file.read_text())['input']['mode'], 'handheld')
        self.assertIn('kiwi_fuse3s.py ', (result.directory / 'logs/run.txt').read_text())

    def test_output_size_reaches_video_slam_mask_and_run_record(self):
        import cv2
        from software.trajectory.algorithms import postprocess
        from software.trajectory import reports

        configuration = config.load('trajectory', overrides={'video': {'size_px': 960}})
        request = config.resolve_trajectory_request(TrajectoryRequest(configuration=configuration, mode='handheld'))
        settings = runtime.reconstruction_settings(dict(os.environ, **self.engines.environment),
            camera_json=calibration_assets.FACTORY_JSON,
            front_mask=runtime.HERE / 'assets/config/front_hand_mask.png',
            output=self.paths.debug / 'settings-960', parameters=configuration.values)
        context = replace(self.context, request=request, settings=settings)
        videos = recording_data.prepare_lens_videos(self.video, context)
        video_command = self.engines.calls[0][1]
        self.assertEqual(video_command[video_command.index('--size-px')+1], '960')
        self.assertEqual(cv2.imread(str(videos[0].mask), 0).shape, (960, 960))
        self.assertEqual(videos[0].settings, settings.front_settings)

        record = postprocess.collect_run_settings(self.paths.output, {}, 'handheld', 'VID.insv',
            calib=dict(yaml=str(settings.front_settings), mask=str(settings.front_mask),
                       source_mask=str(runtime.HERE / 'assets/config/front_hand_mask.png')),
            command=dict(parameters={'configuration': configuration.snapshot()}))
        self.assertEqual(record['frame_geometry']['slam_video_px'], [960, 960])
        self.assertEqual(record['assets']['orbslam3_front_settings']['scalars']['Camera.width'], '960')
        mask_asset = record['assets']['front_hand_mask']
        self.assertEqual(mask_asset['path'], str(settings.front_mask))
        self.assertNotEqual(mask_asset['sha256_12'], mask_asset['source']['sha256_12'])
        (self.paths.output / 'front.mp4').write_bytes(b'video')
        reports.write_result_report(self.paths.output, dict(status='ok', parameters=record['parameters']))
        self.assertIn('960×960', (self.paths.output / 'report.txt').read_text())

    def test_dual_lenses_visit_every_map_and_use_mapping_mask_only(self):
        self.engines.map_count = 2
        request = TrajectoryRequest(mode='dual')
        context = replace(self.context, request=config.resolve_trajectory_request(request))
        result = self.run_stages(context)
        kinds = [kind for kind, *_ in self.engines.calls]
        self.assertEqual(kinds, ['video', 'video', 'map', 'map', 'localize', 'localize', 'merge', 'localize', 'localize', 'merge', 'fuse'])
        for kind, command, options in self.engines.calls:
            if kind == 'map':
                front = '/front/' in command[command.index('--input_video')+1]
                self.assertEqual('--mask_img' in command, front)
            elif kind == 'localize':
                self.assertNotIn('--mask_img', command)
                self.assertIn(options['env']['ORB_MAP_INDEX'], ('0', '1'))
            elif kind == 'fuse':
                self.assertEqual(context.request.mode, 'dual')
                self.assertIn(str(result.directory / 'rear/loc/frames.csv'), command)
        # 第 1 张图关键帧更多，原合并算法应优先使用 m1。
        self.assertTrue((result.directory / 'front/loc/frames.csv').read_text().splitlines()[1].endswith(',m1'))

    def test_retry_reuses_video_and_reruns_all_three_core_stages(self):
        self.engines.bad_tries = {1}
        result = self.run_stages()
        self.assertEqual(result.chosen, 2)
        kinds = [kind for kind, *_ in self.engines.calls]
        self.assertEqual(kinds.count('video'), 1)
        for kind in ('map', 'localize', 'fuse'):
            self.assertEqual(kinds.count(kind), 2)
        for number in (1, 2):
            self.assertEqual((self.paths.attempt(number) / 'front/front.mp4').resolve(), self.paths.lens_videos / 'front.mp4')

    def test_engine_returncode_with_valid_artifact_matches_legacy_policy(self):
        self.engines.map_returncode = 139
        result = self.run_stages()
        self.assertEqual(result.chosen, 1)
        self.assertIn('map: rc=139 TRACKED', (result.directory / 'logs/run.txt').read_text())

    def test_video_failure_stops_before_any_mapping_attempt(self):
        self.engines.failures['video'] = 9
        with self.assertRaises(ReconstructionError) as caught:
            self.run_stages()
        self.assertEqual(caught.exception.stage, '准备镜头视频（front）')
        self.assertEqual([kind for kind, *_ in self.engines.calls], ['video'])
        self.assertFalse(self.paths.reconstruction.exists())

    def test_each_core_failure_is_retried_and_attributes_its_stage(self):
        for kind, stage in (('map', '建图'), ('localize', '定位'), ('merge', '合并定位'), ('fuse', 'IMU 融合')):
            with self.subTest(kind=kind):
                paths = RecordingPaths(self.root / f'failed-{kind}')
                paths.logs.mkdir(parents=True)
                self.engines.write_dataset(paths.dataset)
                self.engines.failures = {kind: 9}
                self.engines.calls.clear()
                context = replace(self.context, paths=paths, request=config.resolve_trajectory_request(TrajectoryRequest(max_tries=2)))
                data = replace(self.data, imu_csv=paths.dataset / 'mav0/imu0/data.csv')
                result = self.run_stages(context, data)
                self.assertIsNone(result.chosen)
                self.assertEqual(len(result.attempts), 2)
                self.assertIn(stage, result.failed_stage)
                self.assertIn('2 次重建均失败', result.reason)
                self.assertIn(stage, (paths.logs / 'runner_try2.log').read_text())
                self.assertEqual([name for name, *_ in self.engines.calls].count('video'), 1)
                if kind != 'fuse':
                    self.assertNotIn('fuse', [name for name, *_ in self.engines.calls])

    def test_missing_atlas_metadata_is_localization_failure(self):
        self.engines.map_count = 0
        result = self.run_stages(replace(self.context, request=config.resolve_trajectory_request(TrajectoryRequest(max_tries=1))))
        self.assertIsNone(result.chosen)
        self.assertIn('定位', result.failed_stage)
        self.assertIn('无法读取图集中的地图列表', result.reason)
        self.assertNotIn('fuse', [kind for kind, *_ in self.engines.calls])

    def test_unreadable_quality_preserves_cause_and_stage(self):
        info = self.root / 'fuse_info.json'
        for content in (None, '{invalid', json.dumps([]), json.dumps({'frames': 'invalid'})):
            with self.subTest(content=content):
                if content is not None:
                    info.write_text(content)
                with self.assertRaises(ReconstructionError) as caught:
                    reconstruction.assess_try(FusionResult(self.root / 'traj_tum.txt', info))
                self.assertEqual(caught.exception.stage, '重建质量检查')
                self.assertEqual(caught.exception.return_code, 1)
                self.assertIn(str(info), str(caught.exception))
                self.assertIsNotNone(caught.exception.__cause__)

    def test_real_atlas_log_protocol_is_parsed_into_map_records(self):
        log = self.root / 'map.log'
        log.write_text('noise\natlas_map index 0 id 23 keyframes 81 mappoints 2492\natlas_map index 1 id 42 keyframes 16 mappoints 520\natlas_map selected index 0\n')
        maps = reconstruction.read_atlas_maps(log, lens='front')
        self.assertEqual([(item.index, item.keyframes) for item in maps], [(0, 81), (1, 16)])

    def test_irregular_times_cpu_pinning_and_fusion_parameters_are_preserved(self):
        settings = replace(self.settings, environment=dict(self.settings.environment, ORB3_EXTRA_LD='/engine/extra'))
        context = replace(self.context, request=config.resolve_trajectory_request(TrajectoryRequest(cores='0-3', allow_irregular=True)), settings=settings)
        times = self.paths.debug / 'times.csv'
        times.write_text('t_canonical_s\n1\n1.1\n')
        self.run_stages(context, replace(self.data, times_csv=times))
        for kind, command, options in self.engines.calls:
            if kind in ('map', 'localize', 'fuse'):
                self.assertEqual(command[:5], ['nice', '-n', '10', 'taskset', '-c'])
                self.assertEqual(command[5], '0-3')
            if kind in ('map', 'localize'):
                self.assertEqual(command[6:8], ['timeout', '348'])
                self.assertTrue(options['env']['LD_LIBRARY_PATH'].endswith(':/engine/extra'))
            if kind == 'fuse':
                self.assertEqual(command[command.index('--times')+1], str(times))
                actual = json.loads(Path(command[command.index('--parameters')+1]).read_text())
                expected = context.parameters
                self.assertEqual(actual['fusion']['step'], expected.fusion.step)
                self.assertEqual(actual['fusion']['node_stride'], expected.fusion.node_stride)
                self.assertEqual(actual['fusion']['split_rotation_deg'], expected.fusion.split_rotation_deg)
                self.assertTrue(actual['fusion']['map_residual'])
                self.assertTrue(actual['fusion']['adaptive_sigma'])
                self.assertTrue(actual['zupt']['enabled'])
                self.assertEqual(Path(command[command.index('--calibration')+1]), self.settings.camera_json)


if __name__ == '__main__':
    unittest.main()
