"""场景清单、显式设备资产和参数；读取不创建目录，不启动算法。"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path

from ruamel.yaml import YAML

from software import config, registry
from software.trajectory import calibration_assets, selection

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('left', 'right')


def read_yaml(path: Path) -> dict:
    parser = YAML(typ='safe', pure=True)
    parser.version = (1, 2)
    parser.allow_duplicate_keys = False
    with path.open(encoding='utf-8') as stream:
        data = parser.load(stream)
    if parser.version != (1, 2) or not isinstance(data, dict):
        raise ValueError(f'{path} 需要 YAML 1.2 映射')
    return data


def require_keys(data: dict, allowed: set, required: set, label: str) -> None:
    if not isinstance(data, dict) or data.keys() - allowed or required - data.keys():
        raise ValueError(f'{label} 字段缺失或含未知项；允许 {sorted(allowed)}，必需 {sorted(required)}')


def resolve_path(base: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('文件路径必须为非空字符串')
    return (base / Path(value).expanduser()).resolve()


def validate_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[\w-]+', value, re.UNICODE):
        raise ValueError('场景和任务名称只允许字母、数字、中文、下划线和连字符')
    return value


def read_recordings(base: Path, data: dict) -> dict[str, Path]:
    require_keys(data, set(ARMS), set(), 'recordings')
    if not data:
        raise ValueError('至少需要一只手的录像')
    videos = {arm: resolve_path(base, value) for arm, value in data.items()}
    if len(set(videos.values())) != len(videos):
        raise ValueError('左右手不能引用同一个录像文件')
    for path in videos.values():
        if not path.is_file() or path.suffix.lower() != '.insv':
            raise ValueError(f'需要原始 .insv 文件：{path}')
    return videos


@dataclass(frozen=True)
class Parameters:
    schema_version: int
    alignment_node_dt_s: float
    alignment_max_gap_s: float
    minimum_alignment_nodes: int
    maximum_alignment_condition: float
    maximum_scale_relative_std: float
    gravity_tolerance_m_s2: float
    maximum_alignment_rms_m: float
    maximum_acc_bias_m_s2: float
    minimum_pose_pairs: int
    placement_position_sigma_m: float
    placement_rotation_sigma_deg: float
    maximum_placement_rms_m: float
    maximum_placement_rotation_deg: float
    minimum_localization_fraction: float
    maximum_visual_gap_s: float
    front_position_sigma_m: float
    rear_position_sigma_m: float
    visual_rotation_sigma_deg: float
    huber: float
    audio_sample_rate: int
    audio_duration_s: float
    audio_maximum_offset_s: float
    audio_minimum_overlap_s: float
    audio_minimum_correlation: float
    audio_minimum_peak_ratio: float
    audio_peak_exclusion_s: float


def load_parameters(path: Path | None) -> Parameters:
    data = read_yaml(ROOT / 'config/scene.yaml')
    if path is not None:
        overlay = read_yaml(path)
        if overlay.keys() - data.keys():
            raise ValueError(f'未知场景参数：{sorted(overlay.keys()-data.keys())}')
        data.update(overlay)
    if data.keys() != {field.name for field in fields(Parameters)}:
        raise ValueError('场景参数字段不完整')
    for field in fields(Parameters):
        value = data[field.name]
        typ = int if field.type == 'int' else float
        if type(value) not in ((int,) if typ is int else (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{field.name} 必须为正的 {typ.__name__}')
        data[field.name] = typ(value)
    p = Parameters(**data)
    if p.schema_version != 1 or p.minimum_alignment_nodes < 5 or p.minimum_pose_pairs < 3:
        raise ValueError('场景参数版本无效，或最少对齐节点/位姿对不足')
    if not all(0 < value <= 1 for value in (p.minimum_localization_fraction, p.audio_minimum_correlation)):
        raise ValueError('定位比例与音频相关性阈值必须在 (0, 1] 内')
    if p.audio_minimum_peak_ratio <= 1 or p.maximum_alignment_condition <= 1:
        raise ValueError('音频峰值比和最大对齐条件数必须大于 1')
    if p.audio_minimum_overlap_s >= p.audio_duration_s:
        raise ValueError('音频最短重叠时间必须小于解码时长')
    return p


@dataclass(frozen=True)
class Scene:
    path: Path
    name: str
    mode: str
    reference_arm: str
    recordings: dict[str, Path]
    rear_calibrations: dict[str, Path]
    trajectory: config.Configuration
    parameters: Parameters

    @property
    def root(self) -> Path:
        return self.path.parent

    def role(self, arm: str, lens: str) -> str:
        return 'shared_front' if self.mode == 'shared_front' else ('shared_rear' if lens == 'rear' else f'{arm}_front')


def read_scene(path: Path) -> Scene:
    path = path.expanduser().resolve()
    if path.is_dir():
        path /= 'scene.yaml'
    data = read_yaml(path)
    require_keys(data, {'schema_version', 'name', 'mode', 'reference_arm', 'mapping', 'rear_calibrations',
                        'trajectory_config', 'parameters'},
                 {'schema_version', 'name', 'mode', 'reference_arm', 'mapping'}, 'scene')
    if type(data['schema_version']) is not int or data['schema_version'] != 1:
        raise ValueError('不支持的场景清单版本')
    if data['mode'] not in ('shared_front', 'dual'):
        raise ValueError('场景 mode 必须为 shared_front 或 dual')
    videos = read_recordings(path.parent, data['mapping'])
    if data['mode'] == 'dual' and set(videos) != set(ARMS):
        raise ValueError('dual 场景需要左右手两段建图录像；单相机验证可使用 shared_front')
    if not isinstance(data['reference_arm'], str) or data['reference_arm'] not in videos:
        raise ValueError('reference_arm 必须指向一段建图录像')
    rear = data.get('rear_calibrations', {})
    if not isinstance(rear, dict) or any(not isinstance(serial, str) for serial in rear):
        raise ValueError('rear_calibrations 必须按完整相机序列号配置')
    optional = lambda name: resolve_path(path.parent, data[name]) if data.get(name) else None
    trajectory = config.load('trajectory', optional('trajectory_config'))
    return Scene(path, validate_name(data['name']), data['mode'], data['reference_arm'], videos,
                 {serial: resolve_path(path.parent, value) for serial, value in rear.items()},
                 trajectory, load_parameters(optional('parameters')))


@dataclass(frozen=True)
class Episode:
    path: Path
    scene: Scene
    recordings: dict[str, Path]
    synchronization: dict

    @property
    def output(self) -> Path:
        return self.path.parent / 'results'

    def offsets(self) -> dict[str, float]:
        sync = self.synchronization
        require_keys(sync, {'method', 'offsets_s', 'evidence'}, {'method', 'offsets_s'}, 'synchronization')
        if sync['method'] not in ('manual', 'audio') or not isinstance(sync['offsets_s'], dict):
            raise ValueError('时间同步需要 manual/audio 方法及 offsets_s 映射')
        if sync['offsets_s'].keys() != self.recordings.keys():
            raise ValueError('每只手都需要显式时间偏移；先运行 umi-scene sync')
        result = {}
        for arm, value in sync['offsets_s'].items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{arm} 时间偏移未确认；请运行 umi-scene sync 或显式填写 offsets_s')
            result[arm] = float(value)
        evidence = sync.get('evidence', {})
        if not isinstance(evidence, dict):
            raise ValueError('同步证据 evidence 必须为映射')
        if sync['method'] == 'audio' or 'recordings' in evidence:
            if not isinstance(evidence.get('recordings'), dict):
                raise ValueError('同步证据缺少对应的录像身份，请重新运行 umi-scene sync')
            for arm, video in self.recordings.items():
                stat = video.stat()
                if evidence.get('recordings', {}).get(arm) != dict(path=str(video), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns):
                    raise ValueError('同步对应的录像已变化，请重新运行 umi-scene sync')
        return result


def read_episode(path: Path) -> Episode:
    path = path.expanduser().resolve()
    if path.is_dir():
        path /= 'episode.yaml'
    data = read_yaml(path)
    require_keys(data, {'schema_version', 'scene', 'recordings', 'synchronization'},
                 {'schema_version', 'scene', 'recordings', 'synchronization'}, 'episode')
    if type(data['schema_version']) is not int or data['schema_version'] != 1:
        raise ValueError('不支持的任务清单版本')
    scene = read_scene(resolve_path(path.parent, data['scene']))
    return Episode(path, scene, read_recordings(path.parent, data['recordings']), data['synchronization'])


@dataclass(frozen=True)
class Camera:
    serial: str
    selected: selection.SelectedCalibration
    measurements: dict[str, dict]
    files: dict[str, Path]

    def identity(self) -> dict:
        return dict(serial=self.serial, gripper=self.selected.gripper_id,
                    assets={name: config.file_hash(path) for name, path in sorted(self.files.items())})


def select_camera(video: Path, scene: Scene) -> Camera:
    # 场景路径不接受工程样例/跨设备覆盖；身份始终由原始录像决定。
    record = registry.get(registry.read_insv_serial(video))
    if not re.fullmatch(r'[A-Za-z0-9_-]+', record['serial']):
        raise ValueError('相机序列号包含非法路径字符')
    if record['status'] != 'active':
        raise ValueError('共享场景需要已启用的实测相机标定，不能使用 sample')
    assets = selection.CalibrationAssets(Path(record['json']), Path(record['tcp_json']), Path(record['mask']))
    selected = selection.SelectedCalibration(assets, Path(record['dir']), record['id'], record['serial'],
        record['gripper'], record['source'], 'scene_registry', record['status'], int(record['cam_time_offset_ns']))
    files = dict(front=assets.camera_json, tcp=assets.tcp_geometry, mask=assets.front_mask)
    measured = dict(front=calibration_assets.read_basalt(assets.camera_json))
    if scene.mode == 'dual':
        rear = scene.rear_calibrations.get(record['serial'])
        if rear is None:
            raise ValueError(f"相机 {record['serial']} 缺少后镜头实测标定；在 rear_calibrations 中指定 rear_calibration.json，不能使用 rear_sample.json")
        metadata = json.loads(rear.read_text())
        if not isinstance(metadata, dict):
            raise ValueError(f'{rear} 的后镜头测量必须为 JSON 对象')
        if metadata.get('camera_serial') != record['serial'] or metadata.get('lens') != 'rear':
            raise ValueError(f'{rear} 必须明确包含匹配的 camera_serial 和 lens: rear')
        if metadata.get('kind') != 'umi_rear_calibration' or not str(metadata.get('validation', {}).get('verdict', '')).startswith('PASS'):
            raise ValueError(f'{rear} 缺少通过验收的后镜头测量；请运行 umi-scene calibrate-rear')
        measured['rear'] = calibration_assets.read_basalt(rear)
        files['rear'] = rear
    return Camera(record['serial'], selected, measured, files)
