"""独立图海报：实际 GTSAM 图、偏移时间、分段尺度与空观测回归。"""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from software.configuration import loader as config
from software.camera import registry
from software.camera.geometry import resample_pose_series
from software.trajectory.algorithms.fusion import solver as fusion, poster as fusion_poster
from software.trajectory.algorithms.fusion.data import FusionInputs
from software.vision.poster import layout as tag_board, detection as environment_tags, geometry as poster_pose
tags = environment_tags


def fixture():
    from test_scene import motion
    from test_poster_fusion import pose
    times,truth,imu = motion()
    parameters = config.load('trajectory').values
    parameters = replace(parameters,fusion=replace(parameters.fusion,node_stride=3),
                         environment_tags=replace(parameters.environment_tags,tag_fusion_enabled=True))
    camera = dict(camera_type='kb4',resolution=[1440,1440],fx=520.,fy=525.,cx=720.,cy=720.,k=[.01,-.001,.0001,-.00001],
                  q_xyzw=[0.,0.,0.,1.],p=[0.,0.,0.],cam_time_offset_ns=17000000)
    T_imu_rear = pose((.12,-.08,.05),(.04,-.03,.02))
    rear = dict(camera,q_xyzw=Rotation.from_matrix(T_imu_rear[:3,:3]).as_quat().tolist(),p=T_imu_rear[:3,3].tolist(),cam_time_offset_ns=30000000)
    noisy = truth.copy(); noisy[:,1,3] += .008*np.sin(times*3)
    observations = {k:(p[:3,3]/2.3,p[:3,:3]) for k,p in enumerate(noisy)}
    board = tag_board.load()
    board_pose = pose((.1,.08,0.),(2.5,-2.7,3.5))
    rear_times = times[1:-1:3]+.013
    sample = np.repeat(np.arange(len(rear_times)),3)
    ids = np.tile([12,48,88],len(rear_times))
    points = np.array([board.corners_m[i] for i in ids])
    cameras = resample_pose_series(times,truth,rear_times[sample]) @ T_imu_rear
    pixels,_ = poster_pose.project_observations(board_pose,cameras,points,poster_pose.camera_model(rear))
    side = np.linalg.norm(pixels-np.roll(pixels,-1,axis=1),axis=2).min(axis=1)
    detection = environment_tags.TagObservations(np.arange(len(rear_times))*6,rear_times-.013,sample,ids,pixels+240.,side,
        np.ones(len(ids),bool),np.ones(len(ids),bool),np.zeros(len(rear_times),int),np.zeros(len(rear_times),int))
    inputs = FusionInputs(dict(front=observations,rear={}),dict(front={},rear={}),imu,times,camera,
        T_imu_rear[:3,:3],np.linalg.inv(T_imu_rear)[:3,3],fusion_poster.PosterInputs(board,detection,rear))
    return inputs,parameters


class IndependentPosterTests(unittest.TestCase):
    def solve(self, inputs, parameters):
        return fusion.solve(inputs,parameters,log=lambda *a,**k:None)

    def test_joint_native_graph_preserves_timeline_and_optimizes_scale(self):
        inputs,parameters = fixture()
        result = self.solve(inputs,parameters)
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['status'],'fused')
        self.assertGreater(evidence['added_nodes'],0)
        self.assertEqual(evidence['frames'],len(inputs.poster.observations.video_times_s))
        np.testing.assert_array_equal(result.video_times_s,inputs.video_times_s)
        self.assertTrue(np.isfinite(result.T_world_imu).all())
        self.assertLess(evidence['final_reprojection_rms_px'],evidence['initial_reprojection_rms_px'])
        self.assertGreater(len(result.segments),0)
        self.assertEqual(evidence['map_scale'],'joint_segment_variables')
        self.assertAlmostEqual(evidence['first_video_clock_time_s'],inputs.poster.observations.video_times_s[0]+.013)

    def test_empty_and_diagnostics_keep_baseline_poses_exact(self):
        inputs,parameters = fixture()
        disabled = replace(parameters,environment_tags=replace(parameters.environment_tags,tag_fusion_enabled=False))
        baseline = self.solve(replace(inputs,poster=None),disabled)
        diagnosed = self.solve(inputs,disabled)
        self.assertEqual(diagnosed.evidence['environment_poster']['status'],'diagnostics_only')
        np.testing.assert_array_equal(diagnosed.T_world_imu,baseline.T_world_imu)
        detections = replace(inputs.poster.observations,usable=np.zeros(len(inputs.poster.observations.usable),bool))
        empty = self.solve(replace(inputs,poster=replace(inputs.poster,observations=detections)),parameters)
        np.testing.assert_array_equal(empty.T_world_imu,baseline.T_world_imu)
        self.assertEqual(empty.evidence['environment_poster']['status'],'insufficient_observations')

    def test_fixed_rear_rotation_and_correction_limit_use_the_shared_factors(self):
        from software.camera.geometry import imu_from_camera
        inputs, parameters = fixture()
        fixed = replace(parameters, environment_tags=replace(parameters.environment_tags,
            tag_rear_rotation_refinement_enabled=False))
        evidence = self.solve(inputs, fixed).evidence['environment_poster']
        np.testing.assert_array_equal(evidence['T_rear_camera_imu_optimized'], np.linalg.inv(imu_from_camera(inputs.poster.measured)))
        self.assertEqual(evidence['rear_rotation_correction_deg'], 0.)
        limited = replace(parameters, environment_tags=replace(parameters.environment_tags,
            tag_rear_rotation_maximum_correction_deg=1e-8))
        with self.assertRaisesRegex(poster_pose.PosterGeometryError, '旋转修正超限'):
            self.solve(inputs, limited)

    def test_post_solve_cull_reuses_the_initialization_and_keeps_one_budget(self):
        inputs, parameters = fixture()
        corners = inputs.poster.observations.corners_half_px.copy()
        corners[0] += [25., 0.]
        inputs = replace(inputs, poster=replace(inputs.poster,
            observations=replace(inputs.poster.observations, corners_half_px=corners)))
        original = fusion_poster.isolated_cull
        calls = []

        def skip_seed_once(*args, **kwargs):
            calls.append(None)
            return False if len(calls) == 1 else original(*args, **kwargs)

        with patch.object(fusion_poster, 'isolated_cull', side_effect=skip_seed_once), \
                patch.object(fusion, 'align_recording', wraps=fusion.align_recording) as alignment:
            result = self.solve(inputs, parameters)
        self.assertEqual(alignment.call_count, 1)
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['culled_observations'], 1)
        self.assertEqual(evidence['culling']['stage'], 'optimization')
        self.assertEqual(evidence['culling']['passes'], 1)

    def test_conflicting_board_rejects_and_missing_assets_cannot_silently_disable(self):
        inputs,parameters = fixture()
        with self.assertRaisesRegex(ValueError,'布局'):
            self.solve(replace(inputs,poster=None),parameters)
        pixels = inputs.poster.observations.corners_half_px.copy(); pixels[len(pixels)//2:] += [80.,0.]
        detections = replace(inputs.poster.observations,corners_half_px=pixels)
        with self.assertRaises(poster_pose.PosterGeometryError):
            self.solve(replace(inputs,poster=replace(inputs.poster,observations=detections)),parameters)

    def test_dual_observations_retimed_without_crossing_map_switch_or_gaps(self):
        inputs,_ = fixture()
        times = np.array([0.,.1,.2,.3,.4])
        rear = {k:(np.array([t+.013,0.,0.]),np.eye(3)) for k,t in enumerate(times)}
        inputs = replace(inputs,video_times_s=times,observations=dict(front={},rear=rear),map_ids=dict(front={},rear={0:'a',1:'a',2:'b',3:'b',4:'b'}))
        from software.trajectory.algorithms.fusion.data import retime_rear
        result = retime_rear(inputs, inputs.poster.measured)
        self.assertNotIn(0,result.observations['rear'])
        self.assertNotIn(2,result.observations['rear'])
        self.assertAlmostEqual(result.observations['rear'][1][0][0],.1)
        self.assertAlmostEqual(result.observations['rear'][3][0][0],.3)


def write_camera(path, measured, *, serial='test-camera', rear=False):
    import json
    document = dict(value0=dict(T_imu_cam=[dict(zip(('px','py','pz','qx','qy','qz','qw'),[*measured['p'],*measured['q_xyzw']]))],
        intrinsics=[dict(camera_type='kb4',intrinsics={**{k:measured[k] for k in ('fx','fy','cx','cy')},**dict(zip(('k1','k2','k3','k4'),measured['k']))})],
        resolution=[[1440,1440]],cam_time_offset_ns=measured['cam_time_offset_ns']))
    if rear: document.update(kind='umi_rear_calibration',camera_serial=serial,lens='rear',validation=dict(verdict='PASS'))
    path.write_text(json.dumps(document))


class IndependentPosterPipelineTests(unittest.TestCase):
    def test_frozen_input_reaches_real_fusion_cli_and_detects_tampering(self):
        import csv
        import io
        import json
        from contextlib import redirect_stdout
        from dataclasses import asdict
        from pathlib import Path
        import tempfile
        from software.trajectory import environment_assets as calibration_assets
        from software.trajectory.algorithms.fusion import cli as fusion_cli
        inputs,parameters = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video,front,rear = root/'raw.insv',root/'front.json',root/'rear.json'
            video.write_bytes(b'synthetic container; detector mocked')
            write_camera(front,inputs.camera)
            write_camera(rear,inputs.poster.measured,rear=True)
            with patch.object(tags,'detect',return_value=inputs.poster.observations):
                manifest = calibration_assets.prepare_poster(video,root/'poster',tag_board.DEFAULT_LAYOUT,rear,parameters.environment_tags)
            loaded = calibration_assets.load_poster(manifest)
            self.assertEqual(loaded.measured['cam_time_offset_ns'],30000000)
            frames,imu,times = root/'frames.csv',root/'imu.csv',root/'times.csv'
            with frames.open('w') as stream:
                writer = csv.writer(stream); writer.writerow('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw,map'.split(','))
                for k,(p,r) in inputs.observations['front'].items():
                    writer.writerow([k,inputs.video_times_s[k],2,1,*p,*Rotation.from_matrix(r).as_quat(),'map0'])
            with imu.open('w') as stream:
                writer=csv.writer(stream);writer.writerow(['#t','wx','wy','wz','ax','ay','az'])
                for row in inputs.imu_video_clock:
                    writer.writerow([round((row[0]+1)*1e9)+17000000,*row[1:]])
            times.write_text('t_canonical_s\n'+''.join(f'{t+1:.12f}\n' for t in inputs.video_times_s))
            parameter_file=root/'parameters.json';parameter_file.write_text(json.dumps(asdict(parameters)))
            with redirect_stdout(io.StringIO()):
                result = fusion_cli.main([str(frames),str(frames),str(imu),'17000000','0,0,0,1',str(root/'out'),
                    '--lever_prior','0,0,0','--calibration',str(front),'--parameters',str(parameter_file),'--times',str(times),'--poster',str(manifest)])
            self.assertEqual(result,0)
            evidence=json.loads((root/'out/fuse_info.json').read_text())
            self.assertEqual(evidence['environment_poster']['status'],'fused')
            self.assertEqual(np.loadtxt(root/'out/traj_tum.txt').shape,(len(inputs.video_times_s),8))
            (manifest.parent/'rear.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'哈希'):
                calibration_assets.load_poster(manifest)

    def test_preflight_requires_registered_matching_rear(self):
        from pathlib import Path
        from types import SimpleNamespace
        import tempfile
        from software.trajectory.models import TrajectoryRequest
        from software.trajectory import environment_assets as calibration_assets
        inputs,_ = fixture()
        request=TrajectoryRequest(configuration=config.load('trajectory'),tag_layout=tag_board.DEFAULT_LAYOUT)
        selected=SimpleNamespace(camera_serial='test-camera',intrinsics_source='measured',
                                 assets=SimpleNamespace(camera_json=Path('front.json')))
        with patch('software.camera.rear_fallback.derive', side_effect=ValueError('无法读取 INSV 工厂参数')):
            with self.assertRaisesRegex(ValueError,'工厂参数'):
                calibration_assets.select_environment_assets(Path('raw.insv'),selected,request)
        with tempfile.TemporaryDirectory() as temporary:
            rear=Path(temporary)/'rear.json';write_camera(rear,inputs.poster.measured,rear=True)
            request=replace(request,rear_calibration='002')
            with patch.object(registry,'rear_path',return_value=rear):
                valid=calibration_assets.select_environment_assets(Path('raw.insv'),selected,request)
                self.assertEqual(valid.rear.identity.version,'002')
                write_camera(rear,inputs.poster.measured,rear=True,serial='another-camera')
                with self.assertRaisesRegex(ValueError,'camera_serial'):
                    calibration_assets.select_environment_assets(Path('raw.insv'),selected,request)

    def test_single_arm_fallback_carries_scene_tags_to_native_engine(self):
        import json
        from pathlib import Path
        from types import SimpleNamespace
        import test_scene_fallback
        from software.trajectory import environment_assets as calibration_assets, reconstruction as runtime
        helper=test_scene_fallback.SceneFallbackAdapterTests('runTest')
        helper.setUp();self.addCleanup(helper.doCleanups)
        inputs,_=fixture()
        helper.episode.recordings['right'].write_bytes(b'synthetic container')
        rear=helper.root/'rear.json';write_camera(rear,inputs.poster.measured,rear=True)
        helper.episode.scene.tag_layout=tag_board.DEFAULT_LAYOUT
        helper.episode.scene.environment_tag_detection=True
        helper.episode.scene.parameters=replace(helper.episode.scene.parameters,tag_fusion_enabled=True)
        helper.recording=replace(helper.recording,camera=SimpleNamespace(
            files=dict(helper.recording.camera.files,rear=rear),measurements=dict(rear=inputs.poster.measured)))
        def engine(command,**kwargs):
            result=helper.engines(command,**kwargs)
            if 'software.trajectory.algorithms.fusion.cli' in command:
                manifest=Path(command[command.index('--poster')+1]);calibration_assets.load_poster(manifest)
                parameters=json.loads(Path(command[command.index('--parameters')+1]).read_text())
                self.assertTrue(parameters['environment_tags']['tag_fusion_enabled'])
                pos=command.index('software.trajectory.algorithms.fusion.cli')
                evidence=Path(command[pos+6])/'fuse_info.json'
                document=json.loads(evidence.read_text());document['environment_poster']=dict(status='fused',actual_error='not_measured')
                evidence.write_text(json.dumps(document))
            return result
        with patch.object(tags,'detect',return_value=inputs.poster.observations),patch.object(runtime.subprocess,'run',side_effect=engine):
            result=helper.run_fallback()
        self.assertEqual(result.evidence['environment_poster']['status'],'fused')
        self.assertEqual(result.evidence['mode'],'independent')

    def test_dual_native_graph_uses_measured_rear_and_distinct_board_key(self):
        from software.camera.geometry import imu_from_camera
        inputs,parameters=fixture()
        # Supply a second independent visual-map scale, in the rear camera's own clock.
        poses=np.tile(np.eye(4),(len(inputs.video_times_s),1,1))
        for k,(position,rotation) in inputs.observations['front'].items():
            poses[k,:3,:3]=rotation;poses[k,:3,3]=position*2.3
        shifted=inputs.video_times_s[:-1]+.013
        rear_poses=resample_pose_series(inputs.video_times_s,poses,shifted) @ imu_from_camera(inputs.poster.measured)
        rear_obs={k:(pose[:3,3]/1.7,pose[:3,:3]) for k,pose in enumerate(rear_poses)}
        inputs=replace(inputs,observations=dict(inputs.observations,rear=rear_obs))
        parameters=replace(parameters,input=replace(parameters.input,mode='dual'))
        result=fusion.solve(inputs,parameters,log=lambda *a,**k:None)
        self.assertEqual(result.evidence['environment_poster']['status'],'fused')
        self.assertGreater(result.evidence['obs']['rear'],0)
        self.assertIsNotNone(result.evidence['rear_lever_final'])
        self.assertTrue(np.isfinite(result.T_world_imu).all())
