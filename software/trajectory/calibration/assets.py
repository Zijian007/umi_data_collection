"""Camera measurements and read-only legacy bundle validation.

Bundles contain measured Basalt JSON and provenance. SLAM settings are generated
for each run; fusion always executes the shared algorithm, never a bundle script.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from software.camera.x3.frame import CROP, scale_intrinsics, validate_video_size

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
SAMPLE_CAMERA_JSON = HERE / "assets" / "calib" / "calibration_r1440.json"
FRONT_SETTINGS_TEMPLATE = HERE / "assets" / "config" / "front_x3_r1440_720.yaml"
FUSION_SCRIPT = HERE / "algorithms" / "fusion_cli.py"
USER_DIR = REPO / "calibration_user"


class CalibrationError(Exception):
    pass


def _sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def read_basalt(path: Path) -> dict:
    """The fields the pipeline needs from a Basalt calibration.json (single kb4 camera, 1440x1440 crop)."""
    raw_data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_data = raw_data.get("value0") or raw_data
    try:
        imu_cam_extrinsics = raw_data["T_imu_cam"][0]
        camera_intrinsics = raw_data["intrinsics"][0]
        resolution = raw_data["resolution"][0]
        cam_time_offset_ns = int(raw_data["cam_time_offset_ns"])
        intrinsics_params = camera_intrinsics["intrinsics"]
        parsed_calibration = {
            "q_xyzw": [imu_cam_extrinsics["qx"], imu_cam_extrinsics["qy"], imu_cam_extrinsics["qz"], imu_cam_extrinsics["qw"]],
            "p": [imu_cam_extrinsics["px"], imu_cam_extrinsics["py"], imu_cam_extrinsics["pz"]],
            "fx": intrinsics_params["fx"], "fy": intrinsics_params["fy"],
            "cx": intrinsics_params["cx"], "cy": intrinsics_params["cy"],
            "k": [intrinsics_params["k1"], intrinsics_params["k2"], intrinsics_params["k3"], intrinsics_params["k4"]],
            "cam_time_offset_ns": cam_time_offset_ns,
            "camera_type": camera_intrinsics["camera_type"],
            "resolution": list(resolution),
        }
    except (KeyError, IndexError, TypeError) as error:
        raise CalibrationError(f"{path} 不是有效的 Basalt 相机–IMU 标定：{error}") from error
    if parsed_calibration["camera_type"] != "kb4" or parsed_calibration["resolution"] != [CROP, CROP]:
        raise CalibrationError(f"{path} 需要单相机 kb4 标定，分辨率 {CROP}×{CROP}；实际为 {parsed_calibration['camera_type']} {parsed_calibration['resolution']}")
    numeric = [*parsed_calibration['q_xyzw'], *parsed_calibration['p'],
               parsed_calibration['fx'], parsed_calibration['fy'], parsed_calibration['cx'], parsed_calibration['cy'],
               *parsed_calibration['k']]
    if not np.all(np.isfinite(numeric)) or parsed_calibration['fx'] <= 0 or parsed_calibration['fy'] <= 0:
        raise CalibrationError(f'{path} 的标定数值必须有限，焦距必须为正')
    if len(raw_data["T_imu_cam"]) != 1 or abs(float(np.linalg.norm(parsed_calibration["q_xyzw"])) - 1.0) > 1e-6:
        raise CalibrationError(f"{path} 的 T_imu_cam 必须且只能包含一个单位四元数")
    return parsed_calibration


def _replace_scalars(text: str, values: dict) -> str:
    for key, value in values.items():
        text, count = re.subn(rf'(?m)^({re.escape(key)}: )\S+$', lambda m: m[1] + str(value), text)
        if count != 1:
            raise CalibrationError(f'SLAM 模板应且仅应包含一行 {key}')
    return text


def _image_size(text: str) -> int:
    width = re.findall(r'(?m)^Camera.width: (\d+)$', text)
    height = re.findall(r'(?m)^Camera.height: (\d+)$', text)
    if len(width) != 1 or width != height:
        raise CalibrationError('SLAM 模板必须包含唯一的正方形图像尺寸')
    size = int(width[0])
    validate_video_size(size)
    return size


def yaml_text(calib_info: dict, *, size_px: int | None = None) -> str:
    """实测标定 → 指定尺寸的 SLAM 内参；省略尺寸仅用于核对旧包的模板规格。"""
    template_text = FRONT_SETTINGS_TEMPLATE.read_text(encoding="utf-8")
    size_px = _image_size(template_text) if size_px is None else size_px
    validate_video_size(size_px)
    T_imu_camera = np.eye(4)
    T_imu_camera[:3, :3] = Rot.from_quat(calib_info["q_xyzw"]).as_matrix()
    T_imu_camera[:3, 3] = calib_info["p"]
    vals = {**scale_intrinsics({key: calib_info[key] for key in ('fx', 'fy', 'cx', 'cy')}, CROP, size_px),
            "k1": calib_info["k"][0], "k2": calib_info["k"][1], "k3": calib_info["k"][2], "k4": calib_info["k"][3]}
    template_text = _replace_scalars(template_text, {**{f'Camera1.{key}': f'{value:.12f}' for key, value in vals.items()},
                                                     'Camera.width': size_px, 'Camera.height': size_px})
    rows = [", ".join("%.12f" % x for x in T_imu_camera[i]) for i in range(4)]
    block = "data: [" + ",\n          ".join(rows) + "]"
    template_text, match_count = re.subn(r"(?s)(IMU\.T_b_c1:.*?)data: \[.*?\]", lambda m: m.group(1) + block, template_text, count=1)
    if match_count != 1:
        raise CalibrationError("样例 YAML 缺少 IMU.T_b_c1 配置块")
    return template_text


def make_bundle(calib_json: Path, out_dir: Path, note: dict | None = None) -> dict:
    """Create a v2 measurement bundle; never copy algorithm code into it."""
    calib_info = read_basalt(calib_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / 'calibration_r1440.json'
    if calib_json.resolve() != target.resolve():
        shutil.copyfile(calib_json, target)
    meta = {**(note or {}), 'kind': 'umi_traj_calibration_bundle', 'schema_version': 2,
            'calibration_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'T_imu_cam': {'p': calib_info['p'], 'q_xyzw': calib_info['q_xyzw']},
            'cam_time_offset_ns': calib_info['cam_time_offset_ns']}
    (out_dir / 'bundle.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding='utf-8')
    return describe(out_dir, 'user')


def _check_legacy(bundle_dir: Path, calib_info: dict) -> None:
    """Read literal extrinsics from v1 code without importing or executing it.

    Legacy algorithm copies are retired. Verify the measurement-bearing fields;
    differences in old algorithm implementations no longer prevent using a camera.
    """
    yaml_path = bundle_dir / 'front_x3_r1440_720.yaml'
    fuser_path = bundle_dir / 'kiwi_fuse3s.py'
    for path in (yaml_path, fuser_path):
        if not path.is_file():
            raise CalibrationError(f'旧标定包缺少 {path.name}：{bundle_dir}')
    expected = yaml_text(calib_info)
    for key in ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'):
        pattern = rf'(?m)^Camera1\.{key}: (\S+)$'
        actual, want = re.findall(pattern, yaml_path.read_text()), re.findall(pattern, expected)
        if len(actual) != 1 or not np.isclose(float(actual[0]), float(want[0]), rtol=0, atol=1e-9):
            raise CalibrationError(f'旧标定包 {bundle_dir} 的 YAML 内参与原始标定不一致')
    try:
        tree = ast.parse(fuser_path.read_text())
        literals = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name == 'P_CfinI':
                    literals[name] = ast.literal_eval(node.value.args[0])
                elif name == 'R_ICf':
                    literals[name] = ast.literal_eval(node.value.func.value.args[0])
        if not np.allclose(literals['P_CfinI'], calib_info['p'], rtol=0, atol=1e-12) or not np.allclose(literals['R_ICf'], calib_info['q_xyzw'], rtol=0, atol=1e-12):
            raise ValueError('相机–IMU 外参不一致')
    except (ValueError, KeyError, AttributeError, TypeError, SyntaxError) as error:
        raise CalibrationError(f'旧标定包 {bundle_dir} 的融合外参与原始标定不一致：{error}') from error


def describe(bundle_dir: Path, source: str) -> dict:
    json_path = SAMPLE_CAMERA_JSON if source == 'factory' else bundle_dir / 'calibration_r1440.json'
    calib_info = read_basalt(json_path)
    version = 2
    if source != 'factory':
        try:
            meta = json.loads((bundle_dir / 'bundle.json').read_text())
            if not isinstance(meta, dict):
                raise ValueError('标定包元数据必须是对象')
            version = meta.get('schema_version', 1)
            if type(version) is not int or version not in (1, 2):
                raise ValueError(f'不支持标定包版本 {version}')
            if meta['calibration_sha256'] != hashlib.sha256(json_path.read_bytes()).hexdigest():
                raise ValueError('原始标定 JSON 校验失败')
            if version == 1:
                _check_legacy(bundle_dir, calib_info)
        except (OSError, ValueError, KeyError) as error:
            raise CalibrationError(f'无法读取标定包 {bundle_dir}：{error}') from error
    return {'source': source, 'dir': str(bundle_dir), 'json': str(json_path),
            'yaml': str(FRONT_SETTINGS_TEMPLATE), 'fuser': str(FUSION_SCRIPT), 'id': _sha12(json_path),
            'bundle_schema_version': version,
            'cam_time_offset_ns': calib_info['cam_time_offset_ns'],
            'T_imu_cam': {'p': calib_info['p'], 'q_xyzw': calib_info['q_xyzw']}}


def slam_settings_text(calib_info: dict, parameters, *, rear: bool = False) -> str:
    """Measured camera geometry + this run's algorithm parameters -> ORB YAML."""
    size = parameters.video.size_px
    validate_video_size(size)
    if rear:
        # 后镜头只有样例模板；按其原始尺寸转换，不借用前镜头的实测内参。
        txt = (HERE / 'assets/config/rear_basalt065_720.yaml').read_text()
        intrinsics = {key: float(re.search(rf'(?m)^Camera1\.{key}: (\S+)$', txt)[1])
                      for key in ('fx', 'fy', 'cx', 'cy')}
        scaled = scale_intrinsics(intrinsics, _image_size(txt), size)
        txt = _replace_scalars(txt, {**{f'Camera1.{key}': f'{value:.12f}' for key, value in scaled.items()},
                                     'Camera.width': size, 'Camera.height': size})
    else:
        txt = yaml_text(calib_info, size_px=size)
    orb_params = parameters.orb
    knobs = {'ORBextractor.nFeatures': orb_params.features, 'ORBextractor.scaleFactor': orb_params.scale_factor,
             'ORBextractor.nLevels': orb_params.levels, 'ORBextractor.iniThFAST': orb_params.initial_fast,
             'ORBextractor.minThFAST': orb_params.minimum_fast, 'System.thFarPoints': orb_params.far_points_map_units}
    return txt + '\n' + '\n'.join(f'{key}: {value}' for key, value in knobs.items()) + '\n'




def active() -> dict:
    """Engineer override only. ``umi-traj`` normally calls ``software.registry.resolve_insv``."""
    env = os.environ.get("UMI_TRAJ_CALIB", "").strip()
    if env == "factory":
        return describe(HERE, "factory")
    if env:
        return describe(Path(env).expanduser().resolve(), "registry")
    raise CalibrationError(
        "未设置 $UMI_TRAJ_CALIB；umi-traj 默认按录像中的相机序列号选择标定；"
        "使用 ./umi-calib devices 查看设备档案"
    )
