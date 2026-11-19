"""地图复用专用身份：镜头资产选择、内容复用键与身份比较。"""

from software.infrastructure import provenance as artifact_identity
from software.scene.models import CalibratedCamera


# 地图只依赖镜头标定和遮手图；TCP、夹爪宽度和标签测距不参与建图，也不冻结进地图版本。
MAP_CAMERA_ASSETS = ('front', 'front_manifest', 'rear', 'mask')


def camera_identity(camera: CalibratedCamera) -> dict:
    return dict(serial=camera.serial, intrinsics_source=camera.selected.intrinsics_source,
                calibration_version=camera.selected.calibration_id,
                assets={name: digest for name, digest in camera.asset_hashes.items()
                        if name in MAP_CAMERA_ASSETS})


def reuse_key(input_identity: dict) -> str:
    """路径留在输入记录中供追溯，复用比较使用内容和影响计算的配置。"""
    definition = {name: value for name, value in input_identity['definition'].items()
                  if name not in ('mapping', 'devices')}
    definition['devices'] = {
        serial: {name: value for name, value in binding.items() if name != 'gripper'}
        for serial, binding in input_identity['definition']['devices'].items()}
    return artifact_identity.canonical_hash(dict(
        schema_version=1, definition=definition,
        recordings={arm: recording['sha256'] for arm, recording in input_identity['recordings'].items()},
        cameras=input_identity['cameras'], code=input_identity['code'], engines=input_identity['engines']))


def matches(previous: dict, current: dict) -> bool:
    return previous.get('reuse_key') == current['reuse_key']
