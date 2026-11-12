"""Report ordering, activation status, installation and refusal of old result layouts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from software import registry
from software import provenance as artifact_identity
from software.calibration import devices
from software.calibration import reports
from software.trajectory.calibration import assets as calibration


class ResultReportTests(unittest.TestCase):
    def test_modern_pair_is_portable_and_tampering_or_wrong_owner_is_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out, portable = root / 'result', root / 'portable'
            description = calibration.make_result(calibration.SAMPLE_CAMERA_JSON, out,
                provenance=dict(camera_serial='camera', lens='front', board=dict(tagSize=.024)))
            portable.mkdir()
            for name in ('calibration.json', 'manifest.json'):
                shutil.copy2(out / name, portable / name)
            self.assertEqual(calibration.describe(portable, 'registry')['id'], description['id'])
            (portable / 'calibration.json').write_text((portable / 'calibration.json').read_text() + ' ')
            with self.assertRaisesRegex(calibration.CalibrationError, '校验失败'):
                calibration.describe(portable, 'registry')
            with patch.object(registry, 'load_index') as load:
                with self.assertRaisesRegex(registry.RegistryError, '序列号'):
                    registry.register('other_camera', out)
            load.assert_not_called()
    def test_registry_always_checks_version_assets(self):
        from test_scene_management import RegistryVersionTests
        fixture = RegistryVersionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        sample = registry.get('CAMERA_A')
        for path in (Path(sample['json']),):
            original = artifact_identity.file_hash
            with self.subTest(path=path), patch.object(artifact_identity, 'file_hash', side_effect=lambda value: '0'*64 if Path(value) == path else original(value)):
                with self.assertRaisesRegex(registry.RegistryError, '哈希'):
                    registry.get('CAMERA_A')


    def test_rendering_and_reading_never_execute_or_query_devices(self):
        from software.calibration import result_io
        from software.calibration.pipeline.steps import StageRunner
        from test_calibration_workflow import FullWorkflowTests
        with tempfile.TemporaryDirectory() as tmp, FullWorkflowTests().simulation(Path(tmp)) as (args, _, _):
            from software.calibration import workflow
            from software.calibration.pipeline import inputs
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            out = Path(args.out)
            before = {str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()}
            with patch.object(registry, 'get', side_effect=AssertionError('reader must be pure')), \
                 patch.object(StageRunner, '__init__', side_effect=AssertionError('reader must not decode execution')):
                doc = result_io.load_result(out)
                text = reports.render(doc)
            self.assertIn('结论：PASS', text)
            self.assertIn('./umi-calib install', text)
            self.assertEqual(before, {str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()})
            self.assertFalse((out/'run/work').exists())

    @staticmethod
    def empty_registry(root: Path) -> Path:
        target = root/'registry'
        target.mkdir()
        (target/'index.json').write_text(json.dumps(dict(kind='umi_device_registry', version=2, devices={})))
        return target

    def test_manual_install_copies_result_and_updates_activation_snapshot(self):
        from test_calibration_workflow import FullWorkflowTests
        from software.calibration import workflow, result_io
        from software.calibration.pipeline import inputs
        with tempfile.TemporaryDirectory() as tmp, FullWorkflowTests().simulation(Path(tmp), mock_install=False) as (args, _, _):
            target = self.empty_registry(Path(tmp))
            with patch.multiple(registry, ROOT=target, INDEX=target/'index.json', CAMERAS=target/'cameras'):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
                out = Path(args.out)
                self.assertEqual(registry.load_index()['devices'], {})
                self.assertEqual(devices.install_bundle(out, force=False), 0)
                doc = result_io.load_result(out, complete=True)
                self.assertEqual(doc['activation']['state'], 'active')
                installed = Path(registry.get('camera')['dir'])
                self.assertEqual({p.name for p in installed.iterdir()}, {'manifest.json','calibration.json'})
                self.assertEqual((out/'report.txt').read_bytes(), (installed.parent/'report.txt').read_bytes())

    def test_force_cannot_bypass_incomplete_or_wrong_identity_and_no_registry_write(self):
        from test_calibration_workflow import FullWorkflowTests
        from software.calibration import workflow, result_io
        from software.calibration.pipeline import inputs
        for case in ('serial', 'incomplete', 'parameters', 'manifest', 'fingerprint', 'locked', 'verify'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp, FullWorkflowTests().simulation(Path(tmp), mock_install=False) as (args, _, _):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
                out = Path(args.out)
                doc = result_io.read(out)
                if case == 'incomplete':
                    doc['execution']['status'] = 'interrupted'
                    result_io.write(out, doc)
                elif case == 'parameters':
                    with (out/'calibration.json').open('a') as f: f.write(' ')
                elif case == 'manifest':
                    manifest = json.loads((out/'manifest.json').read_text())
                    manifest['lens'] = 'rear'
                    (out/'manifest.json').write_text(json.dumps(manifest))
                elif case == 'fingerprint':
                    doc['evaluation']['evidence_fingerprint'] = 'changed'
                    result_io.write(out, doc)
                elif case == 'verify':
                    doc['operation']='verify'
                    result_io.write(out, doc)
                from contextlib import nullcontext
                from software.storage import locked
                lock = locked(out.parent, name=f'.{out.name}.calibration.lock') if case == 'locked' else nullcontext()
                with lock, patch.object(registry, 'register') as register:
                    self.assertEqual(devices.install_bundle(out, True, camera_serial='other' if case == 'serial' else None), 2)
                    register.assert_not_called()

    def test_force_can_override_rejection_only_with_complete_evidence(self):
        from test_calibration_workflow import FullWorkflowTests, AcceptanceTests
        from software.calibration import workflow
        from software.calibration.pipeline import inputs
        with tempfile.TemporaryDirectory() as tmp, FullWorkflowTests().simulation(Path(tmp),
                trajectory=AcceptanceTests().trajectory(ate=40), mock_install=False) as (args, _, _):
            workflow.run(inputs.parse_run_args(args))
            doc = devices._installable(Path(args.out), 'front', force=True)
            self.assertEqual(doc['evaluation']['decision'], 'rejected')
            with self.assertRaises(ValueError): devices._installable(Path(args.out), 'front')
            self.assertEqual(devices.install_bundle(Path(args.out),False),1)

    def test_old_layouts_and_measurement_versions_are_refused_without_writing(self):
        from software.calibration import result_io
        for version in (1,2,4):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)
                path=out/'run/record.json'
                path.parent.mkdir()
                path.write_text(json.dumps(dict(kind='umi_calibration_run', schema_version=version)))
                original=path.read_bytes()
                with self.assertRaises(ValueError): result_io.load_result(out)
                self.assertEqual(devices.install_bundle(out, True), 2)
                self.assertEqual(path.read_bytes(), original)

    def test_errors_and_scale_are_presented_without_claiming_task_accuracy(self):
        data=dict(status='ok', ate_se3_rmse_mm=5.423, ate_se3_p95_mm=10.695,
                  ate_sim3_rmse_mm=1.827, scale_board_over_trajectory=1.076054,
                  board_frames=2076, n=2076, board_extent_m=.40477)
        text='\n'.join(reports.trajectory_error_lines(data))
        for expected in ('5.42', '10.70', '1.83', '1.0761', '-7.07%', '2076', '夹爪尖误差'):
            self.assertIn(expected,text)
        for data in (None,dict(status='trajectory_failed',reason='建图失败')):
            text='\n'.join(reports.trajectory_error_lines(data))
            self.assertIn('无法评估',text)
            self.assertNotIn('0.00',text)
