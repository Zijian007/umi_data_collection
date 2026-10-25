"""Report ordering, activation status and legacy installation compatibility."""
import json
import io
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

from software import registry
from software.calibration import devices, reports
from software.trajectory import calibration_assets as calibration


class ResultReportTests(unittest.TestCase):
    def test_registry_always_checks_version_assets(self):
        sample = registry.sample()
        for path in (Path(sample['json']), Path(sample['tcp_json']), Path(sample['mask'])):
            original = registry.sha256_file
            with self.subTest(path=path), patch.object(registry, 'sha256_file', side_effect=lambda value: '0'*64 if Path(value) == path else original(value)):
                with self.assertRaisesRegex(registry.RegistryError, '哈希'):
                    registry.sample()

    def test_terminal_distinguishes_acceptance_failure_from_execution_failure(self):
        output = Path('/results/calibration')
        cases = [
            (dict(verdict='FAIL', wall_s=12, why='尺度检查'), '[完成]', '验收结论：FAIL'),
            (dict(verdict='FAIL', wall_s=12, why='无法读取 IMU', failed_stage='读取 B 段'),
             '[失败]', '阶段：读取 B 段；原因：无法读取 IMU'),
            (dict(verdict='FAIL', wall_s=12, C=dict(reason='板检测失败'), failed_stage='C 标定板检测'),
             '[失败]', '阶段：C 标定板检测；原因：板检测失败'),
        ]
        for report, state, detail in cases:
            with self.subTest(report=report), redirect_stdout(io.StringIO()) as terminal:
                reports.show_conclusion(output, report)
                text = terminal.getvalue()
                self.assertIn(f'[标定]{state}', text)
                self.assertIn(detail, text)
                self.assertIn(str(output / 'report.txt'), text)
                self.assertIn(str(output / 'diagnostics/logs'), text)
                self.assertNotIn('手动启用', text)

    def test_actionable_warnings_precede_sample_differences(self):
        checks = [dict(check="内参：焦距与样例标定相差", result="WARN", value=0.04, limit="3%"),
                  dict(check="D 段：静止朝向数", result="WARN", value=4, limit="5"),
                  dict(check="C 段：尺度", result="FAIL", value=1.2, limit="10%")]
        ordered = sorted(checks, key=reports.check_priority)
        self.assertEqual([c['result'] for c in ordered], ['FAIL', 'WARN', 'WARN'])
        self.assertTrue(ordered[-1]['check'].startswith('内参'))
        self.assertIn('标签实测尺寸', '\n'.join(reports.attention_lines(checks)))

    def test_status_is_read_from_registry_and_ignores_environment_override(self):
        report = dict(serial="camera", bundle={"id": "new"})
        with patch.object(registry, "get", return_value={"id": "new"}):
            self.assertEqual(reports.activation_status(report)['state'], 'active')
        with patch.object(registry, "get", return_value={"id": "old"}):
            self.assertEqual(reports.activation_status(report)['state'], 'inactive')
        with patch.object(registry, "get", side_effect=registry.UnknownCameraError('missing')):
            self.assertEqual(reports.activation_status(report)['state'], 'inactive')
        with patch.object(registry, "get", side_effect=registry.RegistryError('retired')):
            self.assertEqual(reports.activation_status(report)['state'], 'unavailable')

    def test_manual_install_supports_old_and_new_layout(self):
        for legacy in (True, False):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                out = root / 'result'
                out.mkdir()
                target = root / 'registry'
                target.mkdir()
                (target / 'index.json').write_text(json.dumps(dict(kind='umi_device_registry', version=2, devices={}, default_gripper='handheld_v1')))
                with patch.multiple(registry, ROOT=target, INDEX=target / 'index.json', CAMERAS=target / 'cameras'):
                    bundle = calibration.make_bundle(calibration.FACTORY_JSON, out / 'bundle')
                    report = dict(kind='umi_calib_report', serial='camera', verdict='PASS', checks=[],
                                  bundle=dict(id=bundle['id']), segments=dict(A='A.insv'), tag_size_mm=24.0)
                    if legacy:
                        (out / 'report.json').write_text(json.dumps(report))
                        (out / 'report.txt').write_text('legacy report')
                    else:
                        reports.write_report(out, report)
                    self.assertEqual(registry.load_index()['devices'], {})  # Writing a report never activates.
                    self.assertEqual(devices.install_bundle(out, force=False), 0)
                    self.assertEqual(registry.get('camera')['id'], bundle['id'])
                    refreshed = reports.load_report(out)
                    self.assertEqual(refreshed['activation']['state'], 'active')
                    self.assertFalse((out / 'report.json').exists())
                    self.assertIn('已设为该设备默认标定', (out / 'report.txt').read_text())
                    self.assertEqual((out / 'report.txt').read_bytes(), (Path(registry.get('camera')['dir']).parent / 'report.txt').read_bytes())

    def test_text_report_includes_camera_errors_and_scale_interpretation(self):
        result = dict(status='ok', ate_se3_rmse_mm=5.423, ate_se3_p95_mm=10.695,
                      ate_sim3_rmse_mm=1.827, scale_board_over_trajectory=1.076054,
                      board_frames=2076, n=2076, board_extent_m=0.40477)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            report = dict(verdict='PASS', checks=[], segments={}, tag_size_mm=24, C_new=result)
            reports.write_report(out, report)
            text = (out / 'report.txt').read_text()
            for expected in ('C 段轨迹误差', '5.42 mm', '10.70 mm', '1.83 mm',
                             '1.0761', '-7.07%', '2076', '0.405 m', '不是实际输出精度', '不是夹爪尖误差'):
                self.assertIn(expected, text)

    def test_missing_or_failed_validation_is_not_reported_as_zero_error(self):
        for result in (None, dict(status='trajectory_failed', reason='建图失败'),
                       dict(status='ok', note='有效帧不足')):
            text = '\n'.join(reports.trajectory_error_lines(result))
            self.assertNotIn('0.00', text)
            self.assertTrue('尚未执行' in text or '无法评估' in text)

    def test_failure_report_does_not_offer_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            report = dict(verdict='FAIL', checks=[], segments={}, tag_size_mm=24)
            reports.write_report(out, report)
            self.assertNotIn('手动启用', (out / 'report.txt').read_text())
            with patch.object(registry, 'register') as register:
                calibration.make_bundle(calibration.FACTORY_JSON, out / 'bundle')
                self.assertEqual(devices.install_bundle(out, force=False), 1)
                register.assert_not_called()


if __name__ == '__main__':
    unittest.main()
