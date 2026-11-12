"""Source selection must keep B, lenses, maps, and immutable registry versions paired."""
from __future__ import annotations
from software import provenance as artifact_identity
from software.scene.pipeline import map_lifecycle
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from software import config, registry
from software.camera import calibration_source as origin
from software.camera.x3 import factory as optical
from software.calibration import acceptance, cli, result_io, workflow
from software.calibration.models import CalibrationInputs, RunPaths, SegmentPaths, Step
from software.calibration.solve import factory as solver, basalt
from software.calibration.validation import factory as diagnostic
from software.scene import cli as scene_cli
from software.scene.management import inputs as scene_inputs, batches
from software.storage import write_json, write_yaml
from software.trajectory.calibration import assets
from test_factory_comparison import offset
import test_calibration_workflow as calibration_tests


def native():
    data = optical.parse_offset(offset())
    data.update(serial='camera', firmware='test', camera_model='Insta360 X3')
    return data


def factory_bundle(root, *, source='factory', lens='front'):
    raw = json.loads(assets.SAMPLE_CAMERA_JSON.read_text())
    path = root/'input.json';write_json(path,raw)
    camera = assets.read_basalt(path)
    digest = origin.optical_hash(camera)
    details = dict(camera_serial='camera', lens=lens, lens_index=1 if lens=='front' else 0,
        b_intrinsics_sha256=digest, geometry=dict(kind='kb4_forward', maximum_ray_angle_degrees=85.),
        native=dict(recordings=dict(B=native())), limits=dict(maximum_recording_ray_difference_deg=.1))
    proof = dict(camera_serial='camera', lens=lens, intrinsics_source=source,
        b_intrinsics_sha256=digest, extrinsics_source='measured_with_selected_intrinsics', factory_intrinsics=details)
    bundle = root/'bundle'
    assets.make_result(path,bundle,provenance=proof)
    return bundle


class SourceTests(unittest.TestCase):
    def test_both_calibration_commands_and_scene_commands_share_the_flag(self):
        parser=cli.build_parser()
        for lens in ('front','rear'):
            self.assertEqual(parser.parse_args(['run',lens,'--intrinsics-source','factory']).intrinsics_source,'factory')
        for command in ('run','build'):
            self.assertEqual(scene_cli.build_parser().parse_args([command,'batch','--intrinsics-source','factory']).intrinsics_source,'factory')

    def test_measured_and_factory_versions_coexist_and_wrong_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);idx=root/'registry/index.json';write_json(idx,dict(kind='umi_device_registry',version=2,devices={}))
            with patch.multiple(registry,ROOT=idx.parent,INDEX=idx,CAMERAS=idx.parent/'cameras'):
                measured=factory_bundle(root/'measured',source='measured')
                first=registry.register('camera',measured)
                factory=factory_bundle(root/'factory')
                second=registry.register('camera',factory)
                self.assertNotEqual(first['calibration_version'],second['calibration_version'])
                self.assertEqual(registry.get('camera',intrinsics_source='measured')['calibration_version'],first['calibration_version'])
                self.assertEqual(registry.get('camera',intrinsics_source='factory')['calibration_version'],second['calibration_version'])
                with self.assertRaisesRegex(ValueError,'来源'):
                    registry.get('camera',calibration=first['calibration_version'],intrinsics_source='factory')
                self.assertEqual(registry.get('camera')['calibration_version'],second['calibration_version'])

    def test_factory_bundle_refuses_intrinsics_replaced_without_matching_B(self):
        with tempfile.TemporaryDirectory() as d:
            bundle=factory_bundle(Path(d))
            self.assertEqual(assets.describe(bundle,'user')['intrinsics_source'],'factory')
            raw=json.loads((bundle/'calibration.json').read_text());raw['value0']['intrinsics'][0]['intrinsics']['fx']+=2
            write_json(bundle/'calibration.json',raw)
            manifest=json.loads((bundle/'manifest.json').read_text());manifest['files']['calibration.json']=artifact_identity.file_hash(bundle/'calibration.json');write_json(bundle/'manifest.json',manifest)
            with self.assertRaisesRegex(assets.CalibrationError,'匹配'):
                assets.describe(bundle,'user')

    def test_b_solver_cannot_publish_a_different_intrinsic_model(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=json.loads(assets.SAMPLE_CAMERA_JSON.read_text());path=root/'input.json';write_json(path,raw)
            camera=SimpleNamespace(calibration_json=path)
            changed=copy.deepcopy(raw);changed['value0']['intrinsics'][0]['intrinsics']['k1']+=.01;bad=root/'bad.json';write_json(bad,changed)
            execution=SimpleNamespace(paths=RunPaths(root/'result'))
            with patch.object(basalt,'_run_solver',return_value=(bad,{})),self.assertRaisesRegex(ValueError,'改动'):
                basalt.estimate_camera_imu(None,{},camera,config.load('calibration').values.imu,execution)
            self.assertFalse(execution.paths.calibration.exists())

    def test_kit_resolves_both_lenses_and_never_invents_a_missing_variant(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);tcp=p/'tcp.json';write_json(tcp,dict(units='metres',T_camera_tcp=dict(R_rows=np.eye(3).tolist(),t_m=[0.,0.,0.])))
            declaration=dict(id='kit',arms=dict(left=dict(serial='camera',gripper=dict(tcp=str(tcp)),
                calibrations=dict(measured=dict(front='001',rear='rear_001'),factory=dict(front='002',rear='rear_002')))))
            chosen=scene_inputs.validate_kit(declaration,intrinsics_source='factory')['arms']['left']
            self.assertEqual((chosen['calibration'],chosen['rear_calibration'],chosen['intrinsics_source']),('002','rear_002','factory'))
            declaration['arms']['left']['calibrations'].pop('factory')
            with self.assertRaisesRegex(ValueError,'缺少 factory'):
                scene_inputs.validate_kit(declaration,intrinsics_source='factory')

    def test_map_mismatch_is_rejected_even_with_identical_numeric_intrinsics(self):
        selected=SimpleNamespace(intrinsics_source='factory')
        camera=SimpleNamespace(serial='camera',selected=selected,asset_hashes=dict(front='same'))
        manifest=dict(inputs=dict(cameras=dict(left=dict(serial='camera',assets=dict(front='same'),intrinsics_source='measured'))))
        with self.assertRaisesRegex(ValueError,'来源不同'):
            map_lifecycle.validate_map_calibrations({'left':camera},manifest)
        manifest['inputs']['cameras']['left']['intrinsics_source']='factory'
        map_lifecycle.validate_map_calibrations({'left':camera},manifest)
        manifest['inputs']['cameras']['left']['assets']['front']='changed'
        with self.assertRaisesRegex(ValueError,'标定版本'):
            map_lifecycle.validate_map_calibrations({'left':camera},manifest)

    def test_factory_image_evidence_is_required_and_finite_large_errors_warn(self):
        p=config.load('calibration').values
        a=dict(factory=dict(conversion_forward_pixels=dict(rms=.5,p99=1.),
            recording_ray_difference_degrees=dict(B=dict(maximum=0.))),
            image_validation=dict(common_frames=0,heldout_pixels=dict(rms=None,count=0)),
            coverage=dict(reach_radius_percentile=.8,coverage_percentile=99,coverage_reach_threshold=.5,
                          frames_reaching_by_direction=dict(right=4,left=4,up=4,down=4)))
        self.assertEqual(acceptance.decision(acceptance.assess_factory_intrinsics(a,p.checks,p.factory)).value,'incomplete')
        a['image_validation']=dict(common_frames=12,heldout_pixels=dict(rms=8.,count=100))
        self.assertEqual(acceptance.decision(acceptance.assess_factory_intrinsics(a,p.checks,p.factory)).value,'accepted_with_warnings')
        a['image_validation']['heldout_pixels']['rms']=None
        self.assertEqual(acceptance.decision(acceptance.assess_factory_intrinsics(a,p.checks,p.factory)).value,'incomplete')
        a['image_validation']['heldout_pixels']['rms']=.7
        self.assertEqual(acceptance.decision(acceptance.assess_factory_intrinsics(a,p.checks,p.factory)).value,'accepted')

    def test_new_recording_cannot_silently_replace_the_frozen_factory_intrinsics(self):
        with tempfile.TemporaryDirectory() as d:
            bundle=factory_bundle(Path(d));proof=assets.describe(bundle,'user')['intrinsics_provenance']
            with patch.object(optical,'read_factory',return_value=native()):origin.check_recording(Path('video.insv'),proof)
            changed=native();changed['lenses'][1]['fx']*=1.02
            with patch.object(optical,'read_factory',return_value=changed),self.assertRaisesRegex(ValueError,'不一致'):
                origin.check_recording(Path('video.insv'),proof)


class FactoryWorkflowTests(unittest.TestCase):
    def test_both_lenses_skip_A_solver_validate_A_and_recompute_B(self):
        helper=calibration_tests.FullWorkflowTests()
        for lens in ('front','rear'):
            with self.subTest(lens=lens),tempfile.TemporaryDirectory() as d,helper.simulation(Path(d)) as (args,commands,_):
                from software.calibration.pipeline.inputs import parse_run_args
                request=parse_run_args(args)
                from dataclasses import replace
                request=replace(request,lens=lens,intrinsics_source='factory',keep_work=True)
                source=CalibrationInputs(SegmentPaths(*(Path(d)/f'{r}.insv' for r in 'ABCD')),'camera',
                    dict(tagCols=6,tagRows=6,tagSpacing=.3,tagSize=.024))
                original=solver.generate
                def generate(*a,**kw):
                    solved,evidence=original(*a,**kw)
                    return solved,evidence
                def solve_b(dataset,grid,initial,imu,execution,**kw):
                    raw=json.loads(initial.calibration_json.read_text());raw['value0']['cam_time_offset_ns']=1000000
                    write_json(execution.paths.calibration,raw)
                    from software.calibration.models import CameraImuResult
                    return CameraImuResult(execution.paths.calibration,assets.read_basalt(execution.paths.calibration),
                        dict(corners_per_frame=100,mean_reprojection_px=.3,rms_reprojection_px=.4))
                found=dict(n_frames=10,video_time_s=np.arange(10,dtype=float),dets={i:{} for i in range(10)})
                validation=dict(common_frames=10,heldout_pixels=dict(count=100,rms=.4),sufficient=True)
                with patch.object(diagnostic,'sources',return_value=dict(recordings={r:native() for r in 'ABCD'},consistent=True)), \
                     patch('software.vision.capabilities.require_forward_kb4',return_value={}), \
                     patch.object(workflow,'_detect_board',return_value=(found,dict(elapsed_s=0))), \
                     patch.object(solver,'validate_images',return_value=validation), \
                     patch.object(workflow,'_compare_factory',return_value={}), \
                     patch.object(basalt,'estimate_camera_imu',side_effect=solve_b) as b, \
                     patch.object(basalt,'estimate_intrinsics') as measured_a:
                    code=workflow.run(request,trajectory_generator=(lambda *a,**kw:None) if lens=='rear' else None)
                self.assertIn(code,(0,1))
                doc=result_io.load_result(Path(args.out),complete=True)
                self.assertEqual(doc['subject']['intrinsics_source'],'factory')
                self.assertEqual(doc['measurements'][Step.INTRINSICS]['data']['factory']['lens_index'],1 if lens=='front' else 0)
                self.assertIn(Step.GRAVITY,doc['measurements']);self.assertIn(Step.TRAJECTORY,doc['measurements'])
                measured_a.assert_not_called();b.assert_called_once()
                self.assertTrue((Path(args.out)/'factory/intrinsics.json').is_file())
                self.assertEqual(assets.describe(Path(args.out),'user')['intrinsics_source'],'factory')
                self.assertIn('A 独立验证',(Path(args.out)/'report.txt').read_text())

class BatchSourceTests(unittest.TestCase):
    def test_source_switch_archives_snapshot_and_clears_incompatible_map(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);batch=root/'shelf/batch_001';batch.mkdir(parents=True)
            kits=root/'kits';gripper=kits/'rig/tcp.json'
            write_json(gripper,dict(units='metres',T_camera_tcp=dict(R_rows=np.eye(3).tolist(),t_m=[0.,0.,0.])))
            write_yaml(kits/'rig/kit.yaml',dict(arms=dict(left=dict(serial='camera',gripper=dict(tcp='tcp.json'),
                calibrations=dict(measured=dict(front='001',rear='rear_001'),factory=dict(front='002',rear='rear_002'))))))
            write_yaml(batch/'batch.yaml',dict(kit='rig'))
            with patch.object(scene_inputs,'KITS',kits), patch.object(registry,'rear_path',side_effect=lambda serial,version:root/version):
                factory=scene_inputs.read_kit('rig',intrinsics_source='factory')
                video=batch/'mapping/video.insv';video.parent.mkdir();video.write_bytes(b'video')
                metadata=dict(serial='camera',sha256=artifact_identity.file_hash(video))
                with patch.object(batches.recordings,'inspect',return_value=metadata):
                    current=batches.read_inputs(batch,intrinsics_source='measured')
                    path=batches.save_record(current)
                    previous=scene_inputs.read_yaml(path);previous['map']=dict(version='deadbeef',manifest_sha256='0'*64);write_yaml(path,previous)
                    selected=replace(current,previous_record=previous,kit=factory,intrinsics_source='factory')
                    batches.save_record(selected)
                record=scene_inputs.read_yaml(path)
                self.assertNotIn('map',record)
                session=scene_inputs.read_batch(path)
                self.assertEqual(session.kit['arms']['left']['calibration'],'002')
                archives=path.parent/'snapshots'
                self.assertTrue((archives/(artifact_identity.canonical_hash(previous)+'.yaml')).is_file())
                self.assertTrue((archives/(artifact_identity.canonical_hash(record)+'.yaml')).is_file())
                self.assertEqual(scene_inputs.read_yaml(batch/'batch.yaml'),dict(kit='rig'))

    def test_legacy_sample_factory_override_cannot_select_insv_factory(self):
        import os
        with patch.dict(os.environ,{'UMI_TRAJ_CALIB':'factory'}),self.assertRaisesRegex(ValueError,'来源'):
            registry.resolve_insv(Path('video.insv'),intrinsics_source='factory')

    def test_fixed_map_source_is_checked_without_task_reconstruction(self):
        kit=dict(arms=dict(left=dict(serial='camera',calibration='002',intrinsics_source='factory')))
        manifest=dict(inputs=dict(cameras=dict(left=dict(serial='camera',intrinsics_source='measured',calibration_version='001'))))
        with self.assertRaisesRegex(ValueError,'来源不同'):
            map_lifecycle.validate_map_bindings(kit,manifest)

class SourceConfigurationTests(unittest.TestCase):
    def test_A_evidence_rules_can_be_rechecked_without_changing_computation(self):
        first=config.load('calibration')
        changed=config.load('calibration',overrides={'factory':{'holdout_warn_rms_px':3.,'minimum_holdout_frames':8}})
        self.assertEqual(first.computation_hash,changed.computation_hash)
        self.assertNotEqual(first.evaluation_hash,changed.evaluation_hash)
        geometry=config.load('calibration',overrides={'factory':{'maximum_recording_ray_difference_deg':.2}})
        self.assertNotEqual(first.computation_hash,geometry.computation_hash)
