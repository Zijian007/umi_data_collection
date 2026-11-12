from software import provenance as artifact_identity
"""Boundary tests for evidence freshness, installed identity and checkpoint dependencies."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from software import config, registry
from software.calibration import workflow, devices, result_io
from software.calibration.models import Step, RunPaths, ExecutionOptions
from software.calibration.pipeline import inputs, runtime
from software.calibration.validation import gravity
from software.trajectory.calibration import assets
from software.storage import locked
import test_calibration_workflow as workflow_fixtures


class CheckpointContractTests(unittest.TestCase):
    def test_gravity_parameters_only_recompute_gravity(self):
        with tempfile.TemporaryDirectory() as tmp, workflow_fixtures.FullWorkflowTests().simulation(Path(tmp)) as (args, commands, _):
            self.assertEqual(workflow.run(inputs.parse_run_args(args)),0)
            override=Path(tmp)/'d.yaml'
            override.write_text('gravity:\n  window_s: 0.7\n')
            args.resume,args.config=True,override
            commands.clear()
            self.assertEqual(workflow.run(inputs.parse_run_args(args)),0)
            self.assertEqual(len(commands),1)
            self.assertIn('imu_calib_physical_check.py',commands[0][1])
            timing=result_io.read(Path(args.out))['execution']['timings']
            for step in (Step.INTRINSICS,Step.CAMERA_IMU,Step.TRAJECTORY):
                self.assertEqual(timing[step]['status'],'reused')
            self.assertEqual(timing[Step.GRAVITY]['status'],'computed')

    def test_recheck_uses_result_lock_and_preserves_old_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp, workflow_fixtures.FullWorkflowTests().simulation(Path(tmp)) as (args, _, _):
            workflow.run(inputs.parse_run_args(args))
            out=Path(args.out)
            original=(out/'run/record.json').read_bytes()
            with locked(out.parent,name=f'.{out.name}.calibration.lock'):
                self.assertEqual(workflow.recheck(out,None),2)
            self.assertEqual((out/'run/record.json').read_bytes(),original)
            before=result_io.read(out)
            self.assertEqual(workflow.recheck(out,None),0)
            after=result_io.read(out)
            self.assertEqual(after['evaluation_history'][-1],before['evaluation'])
            self.assertEqual(after['measurements'],before['measurements'])

    def test_incompatible_method_version_refuses_read_resume_recheck_and_install(self):
        with tempfile.TemporaryDirectory() as tmp, workflow_fixtures.FullWorkflowTests().simulation(Path(tmp),mock_install=False) as (args, _, _):
            workflow.run(inputs.parse_run_args(args))
            out=Path(args.out)
            path=out/'run/record.json'
            doc=json.loads(path.read_text());doc['measurement_version']=999
            path.write_text(json.dumps(doc));original=path.read_bytes()
            with self.assertRaises(ValueError): result_io.load_result(out)
            args.resume=True
            self.assertEqual(workflow.run(inputs.parse_run_args(args)),2)
            self.assertEqual(workflow.recheck(out,None),2)
            self.assertEqual(devices.install_bundle(out,True),2)
            self.assertEqual(path.read_bytes(),original)

    def test_replacing_parameter_and_manifest_together_still_cannot_reuse_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp, workflow_fixtures.FullWorkflowTests().simulation(Path(tmp),mock_install=False) as (args, _, _):
            workflow.run(inputs.parse_run_args(args))
            out=Path(args.out)
            raw=json.loads((out/'calibration.json').read_text())
            (raw.get('value0') or raw)['intrinsics'][0]['intrinsics']['fx']+=1
            (out/'calibration.json').write_text(json.dumps(raw))
            manifest=json.loads((out/'manifest.json').read_text())
            manifest['files']['calibration.json']=artifact_identity.file_hash(out/'calibration.json')
            (out/'manifest.json').write_text(json.dumps(manifest))
            assets.describe(out,'registry') # standalone asset pair is consistent
            with patch.object(registry,'register') as register:
                self.assertEqual(devices.install_bundle(out,True),2)
                register.assert_not_called()

    def test_verify_launcher_never_installs_basalt(self):
        with tempfile.TemporaryDirectory() as tmp:
            deps=Path(tmp)
            fake=deps/'python'
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            fake.chmod(0o755)
            (deps/'env.sh').write_text(f'export UMI_PY="{fake}"\n')
            for command in ('verify', 'preview-width', 'detect-width'):
                with self.subTest(command=command):
                    result=subprocess.run([str(runtime.REPO/'umi-calib'),command,'missing.insv'],
                        env=dict(os.environ,UMI_DEPS=str(deps)),capture_output=True,text=True)
                    self.assertEqual(result.returncode,0)
                    self.assertIn('software.calibration',result.stdout)
                    self.assertIn(command,result.stdout)
                    self.assertNotIn('安装',result.stderr)
                    self.assertFalse((deps/'calib_env').exists())


class GravityFailureTests(unittest.TestCase):
    def test_missing_malformed_or_refused_output_does_not_reuse_an_old_success(self):
        cfg=config.load('calibration').values.gravity
        for mode in ('missing','malformed','refused'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                dataset=root/'ds';(dataset/'mav0/imu0').mkdir(parents=True)
                (dataset/'mav0/imu0/data.csv').write_text('0,0,0,0,0,0,9.81')
                execution=ExecutionOptions(RunPaths(root/'out'),None,1)
                old=execution.paths.work/'gravity_check_old'
                old.mkdir(parents=True)
                (old/'imu_calib_physical_check.json').write_text('{"status":"REPORT_ONLY"}')
                def command(argv,*args,**kwargs):
                    dest=Path(argv[argv.index('--out')+1])
                    self.assertNotEqual(dest,old)
                    if mode=='malformed': (dest/'imu_calib_physical_check.json').write_text('{')
                    return 5 if mode=='refused' else 0
                with patch.object(runtime,'run_logged',side_effect=command):
                    with self.assertRaises(json.JSONDecodeError if mode == 'malformed' else RuntimeError):
                        gravity.validate_gravity(dataset,assets.SAMPLE_CAMERA_JSON,cfg,execution)
                self.assertEqual((old/'imu_calib_physical_check.json').read_text(),'{"status":"REPORT_ONLY"}')

class SolverAdapterTests(unittest.TestCase):
    def test_b_receives_exact_a_snapshot_and_solver_dirs_do_not_share_caches(self):
        from software.calibration.solve import basalt
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            execution=ExecutionOptions(RunPaths(root/'out'),None,1)
            commands=[]
            initial=assets.SAMPLE_CAMERA_JSON.read_bytes()
            def command(argv,log,**kwargs):
                argv=list(map(str,argv));commands.append(argv)
                if '--result-path' in argv:
                    directory=Path(argv[argv.index('--result-path')+1])
                    if 'basalt_calibrate_imu' in ' '.join(argv):
                        self.assertEqual((directory/'calibration.json').read_bytes(),initial)
                        self.assertFalse((directory/'corners.cereal').exists())
                    (directory/'calibration.json').write_bytes(initial)
                    (directory/'corners.cereal').write_text('solver cache')
                return 0
            with patch.object(runtime,'run_logged',side_effect=command):
                a=basalt.estimate_intrinsics(root/'A',{},execution)
                self.assertFalse((a.calibration_json.parent/'corners.cereal').exists())
                b=basalt.estimate_camera_imu(root/'B',{},a,config.load('calibration').values.imu,execution)
            self.assertEqual(b.calibration_json.read_bytes(),initial)
            self.assertEqual(len(commands),4)
            self.assertIn('--fixed-scale-opt-offset',commands[-1])
            self.assertIn('--cam-types',commands[1])

    def test_rear_cli_uses_explicit_exit_code_and_preserves_options(self):
        from software.calibration import cli
        with patch('software.scene.spatial.rear_calibration.run',return_value=1) as run:
            self.assertEqual(cli.main(['run','/data/rear','--resume','--workers','2']),1)
            self.assertTrue(run.call_args.args[0].resume)
            self.assertEqual(run.call_args.args[0].workers,2)
