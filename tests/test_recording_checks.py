"""原片检查、静止任务、曝光未知与成功发布保护。"""
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from software.scene import inputs, recording_checks, workflow
import test_scene as fixtures


class RecordingCheckTests(unittest.TestCase):
    def setUp(self):
        self.cfg=inputs.load_scene_parameters(None)

    def test_static_hand_in_fixed_metric_map_is_allowed(self):
        t=np.arange(0,5,.001)
        imu=dict(t_s=t,acc=np.tile([0.,0.,9.81],(len(t),1)),gyr=np.zeros((len(t),3)))
        self.assertEqual(recording_checks.check_initial_static(imu,self.cfg).status,'PASS')
        imu['gyr'][:]=2.
        self.assertEqual(recording_checks.check_initial_static(imu,self.cfg).status,'WARN')

    def test_exposure_missing_or_invalid_is_unknown_and_slow_is_fail(self):
        path=Path('/dummy.insv')
        with patch.object(recording_checks.insv_reader,'parse_trailer',return_value=dict(records=[])):
            self.assertEqual(recording_checks.check_exposure(path,self.cfg).status,'UNKNOWN')
        records=dict(records=[dict(id=4,offset=100,size=1600)])
        data=b''.join(struct.pack('<Qd',i,0.05) for i in range(100))
        with patch.object(recording_checks.insv_reader,'parse_trailer',return_value=records),patch.object(recording_checks.insv_reader,'read_record_bytes',return_value=data):
            self.assertEqual(recording_checks.check_exposure(path,self.cfg).status,'FAIL')

    def test_stream_tolerance_is_absolute_and_matches_extraction_contract(self):
        container=MagicMock();container.__enter__.return_value=container
        stream=MagicMock();stream.codec_context.width=3840;stream.codec_context.height=1920;stream.average_rate=57.
        container.streams.video=[stream]
        with patch('av.open',return_value=container):
            self.assertFalse(recording_checks.check_stream(Path('/dummy.insv')).passed)

    def test_invalid_imu_and_short_recording_are_rejected(self):
        t=np.arange(0,5,.001)
        imu=dict(t_s=t,acc=np.tile([0.,0.,9.81],(len(t),1)),gyr=np.zeros((len(t),3)),integrity=dict(status='OK'))
        with patch.object(recording_checks.imu_reader,'read_imu_and_meta',return_value=imu):
            self.assertTrue(recording_checks.check_imu(Path('/dummy'),self.cfg)[0].passed)
            imu['acc'][5,0]=np.nan
            self.assertFalse(recording_checks.check_imu(Path('/dummy'),self.cfg)[0].passed)

    def test_failed_qa_preserves_success_and_dry_run_is_read_only(self):
        f=fixtures.SceneWorkflowTests();f.setUp()
        try:
            f.build_scene();workflow.run_episode(f.episode_path)
            output=f.episode_path.parent/'results';previous=output.resolve()
            before={str(path):path.read_bytes() for path in f.root.rglob('*') if path.is_file() and not path.is_symlink()}
            failure=recording_checks.EpisodeCheckReport(False,['review rejection'])
            with patch.object(recording_checks,'check_episode',return_value=failure):
                with self.assertRaises(recording_checks.RecordingCheckError):workflow.run_episode(f.episode_path,dry_run=True)
                after={str(path):path.read_bytes() for path in f.root.rglob('*') if path.is_file() and not path.is_symlink()}
                self.assertEqual(before,after)
                with self.assertRaises(recording_checks.RecordingCheckError):workflow.run_episode(f.episode_path)
            self.assertEqual(output.resolve(),previous)
            state=json.loads((f.episode_path.parent/'state.json').read_text())
            self.assertEqual(state['attempt']['status'],'rejected_preflight')
            self.assertEqual(json.loads((output/'episode.json').read_text())['status'],'ok')
        finally:f.doCleanups()
