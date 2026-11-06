"""在临时目录生成测试标定包，不依赖工作区里的设备登记或演示数据。"""
import json
from pathlib import Path

from software import registry
from software.storage import write_json
from software.trajectory.calibration import assets as calibration_assets


def legacy_bundle(path: Path) -> Path:
    calibration_assets.make_bundle(calibration_assets.SAMPLE_CAMERA_JSON, path)
    measured = calibration_assets.read_basalt(path / 'calibration_r1440.json')
    metadata = json.loads((path / 'bundle.json').read_text())
    metadata['schema_version'] = 1
    write_json(path / 'bundle.json', metadata)
    (path / 'front_x3_r1440_720.yaml').write_text(calibration_assets.yaml_text(measured))
    (path / 'kiwi_fuse3s.py').write_text(
        f'P_CfinI = np.array({measured["p"]!r})\n'
        f'R_ICf = Rot.from_quat({measured["q_xyzw"]!r}).as_matrix()\n')
    return path


def device_record() -> dict:
    record = registry.attach_gripper(calibration_assets.describe(calibration_assets.HERE, 'factory'))
    return dict(record, serial='TEST_CAMERA', status='active')
