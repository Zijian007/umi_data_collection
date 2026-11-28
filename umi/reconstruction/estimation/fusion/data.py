"""独立融合的显式观测、求解结果与文件边界。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from umi.core.geometry import imu_from_camera
from umi.core.timing import canonical_to_video_seconds, euroc_ns_to_seconds, video_to_canonical_seconds
from umi.devices.calibration_assets import read_basalt
from umi.reconstruction.estimation.map_information import camera_covariance


@dataclass(frozen=True)
class FusionInputs:
    observations: dict
    map_ids: dict
    imu_video_clock: np.ndarray  # t_video_s, gyro xyz, acc xyz
    video_times_s: np.ndarray | None
    camera: dict
    rear_rotation: np.ndarray
    rear_lever: np.ndarray
    poster: object | None = None
    rear_camera: dict | None = None
    front_covariances: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectorySeed:
    """Pose and velocity guesses on the video timeline for the poster graph."""

    video_times_s: np.ndarray
    T_world_imu: np.ndarray
    velocities: np.ndarray


@dataclass(frozen=True)
class FusionSolution:
    video_times_s: np.ndarray
    T_world_imu: np.ndarray
    velocities: np.ndarray
    gyro_biases: np.ndarray
    acc_biases: np.ndarray
    segments: tuple
    evidence: dict


@dataclass(frozen=True)
class FusionOutput:
    solution: FusionSolution
    tum: np.ndarray

    @property
    def video_times_s(self) -> np.ndarray:
        return canonical_to_video_seconds(self.tum[:, 0])

    @property
    def evidence(self) -> dict:
        return self.solution.evidence


def read_imu_video_clock(path: Path, offset_ns: int) -> np.ndarray:
    with path.open() as stream:
        imu = np.array([[float(x) for x in row] for row in csv.reader(stream) if row and not row[0].startswith('#')])
    if imu.ndim != 2 or imu.shape[1] != 7:
        raise ValueError('IMU 文件缺少时间或三轴角速度及加速度')
    imu[:, 0] = euroc_ns_to_seconds(imu[:, 0], camera_offset_ns=offset_ns)
    return imu


def read_video_times(path: Path) -> np.ndarray:
    with path.open() as stream:
        return canonical_to_video_seconds([float(row['t_canonical_s']) for row in csv.DictReader(stream)])


def load_inputs(*, front_frames: Path | None = None, rear_frames: Path | None = None, imu_csv: Path, camera_json: Path,
                offset_ns: int, rear_quaternion: list[float], rear_lever: list[float],
                use_rear: bool, times_csv: Path | None = None) -> FusionInputs:
    camera = read_basalt(camera_json)
    if camera['cam_time_offset_ns'] != offset_ns:
        raise ValueError('时间偏移与所选相机标定不一致')
    observations, map_ids, front_covariances = {}, {}, {}
    for lens, path in (('rear', rear_frames), ('front', front_frames)):
        obs, maps = {}, {}
        if path is not None and (lens == 'front' or use_rear):
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    if row['ok'] == '1' and row.get('state', '2') == '2':
                        k = int(row['frame_idx'])
                        obs[k] = (np.array([float(row[c]) for c in ('tx','ty','tz')]),
                                  Rot.from_quat([float(row[c]) for c in ('qx','qy','qz','qw')]).as_matrix())
                        if lens == 'front' and row.get('camera_information'):
                            covariance = camera_covariance(row['camera_information'])
                            if covariance is not None:
                                front_covariances[k] = covariance
                        if row.get('map'):
                            maps[k] = row['map']
        observations[lens], map_ids[lens] = obs, maps
    imu = read_imu_video_clock(imu_csv, offset_ns)
    times = read_video_times(times_csv) if times_csv is not None else None
    return FusionInputs(observations, map_ids, imu, times, camera,
                        Rot.from_quat(rear_quaternion).as_matrix(), np.array(rear_lever), front_covariances=front_covariances)


def write_solution(solution: FusionSolution, output: Path) -> FusionOutput:
    """保持原轨迹精度、辅助 CSV 字段与 JSON 结构；求解成功后才创建输出。"""
    output.mkdir(parents=True, exist_ok=True)
    tum = np.empty((len(solution.video_times_s), 8))
    with (output/'traj_tum.txt').open('w') as fo:
        fo.write('# t x y z qx qy qz qw | T_WI, W gravity-aligned metric, front+rear relocalization + IMU, piecewise map placement (KIWI-style, ours) | canonical s\n')
        for k, time in enumerate(video_to_canonical_seconds(solution.video_times_s)):
            q = Rot.from_matrix(solution.T_world_imu[k,:3,:3]).as_quat()
            p = solution.T_world_imu[k,:3,3]
            line = '%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n' % (time, *p, *q)
            fo.write(line)
            tum[k] = [float(value) for value in line.split()]
    cols = 't,px,py,pz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz,qx,qy,qz,qw,pstd_x,pstd_y,pstd_z,n_msckf,n_slam,dt_cam_imu,rItoC_x,rItoC_y,rItoC_z,pIinC_x,pIinC_y,pIinC_z,fx,fy,cx,cy,k1,k2,k3,k4,init_time,imu_t_last'.split(',')
    with (output/'state.csv').open('w') as fo:
        fo.write(','.join(cols)+'\n')
        for k, time in enumerate(video_to_canonical_seconds(solution.video_times_s)):
            p = solution.T_world_imu[k,:3,3]
            q = Rot.from_matrix(solution.T_world_imu[k,:3,:3]).as_quat()
            v = solution.velocities[k]
            bg = solution.gyro_biases[k]
            ba = solution.acc_biases[k]
            d = dict.fromkeys(cols, 0.0)
            d.update(t=time, px=p[0], py=p[1], pz=p[2], vx=v[0], vy=v[1], vz=v[2], qx=q[0], qy=q[1], qz=q[2], qw=q[3],
                     bgx=bg[0], bgy=bg[1], bgz=bg[2], bax=ba[0], bay=ba[1], baz=ba[2], init_time=float(video_to_canonical_seconds(0.)), imu_t_last=time)
            fo.write(','.join('%.9g' % d[c] for c in cols)+'\n')
    with (output/'seg_placement.csv').open('w') as fo:
        fo.write('lens,chain,sid,k0,k1,t0_s,t1_s,scale_init,scale,R_change_deg,tx,ty,tz\n')
        for row in solution.segments:
            fo.write('%s,%d,%d,%d,%d,%.3f,%.3f,%.6g,%.6g,%.3f,%.5f,%.5f,%.5f\n' % row)
    (output/'fuse_info.json').write_text(json.dumps(solution.evidence, indent=1))
    return FusionOutput(solution, tum)


def retime_rear(inputs, measured):
    """dual 地图定位也换到工作镜头视频时钟，不跨丢帧或地图切换插值。"""
    transform = imu_from_camera(measured)
    observations, maps = inputs.observations['rear'], inputs.map_ids['rear']
    offset = (measured['cam_time_offset_ns']-inputs.camera['cam_time_offset_ns'])/1e9
    if inputs.video_times_s is None:
        raise ValueError('正式 rear 时差修正必须提供真实视频时间轴')
    indices = sorted(observations)
    times = inputs.video_times_s[indices]+offset
    result, result_maps = {}, {}
    insertion_points = np.searchsorted(times, inputs.video_times_s) if indices else ()
    for k, right in enumerate(insertion_points):
        query = inputs.video_times_s[k]
        exact = next((i for i in (right,right-1) if 0 <= i < len(times) and abs(times[i]-query) <= 1e-9), None)
        if exact is not None:
            source = indices[exact]
            result[k] = observations[source]
            if source in maps: result_maps[k] = maps[source]
        elif 0 < right < len(times):
            a,b = indices[right-1:right+1]
            if b != a+1 or maps.get(a) != maps.get(b): continue
            fraction = (query-times[right-1])/(times[right]-times[right-1])
            pa,ra = observations[a]; pb,rb = observations[b]
            rotation = Slerp(times[right-1:right+1],Rot.from_matrix([ra,rb]))([query]).as_matrix()[0]
            result[k] = ((1-fraction)*pa+fraction*pb, rotation)
            if a in maps: result_maps[k] = maps[a]
    return replace(inputs, observations=dict(inputs.observations,rear=result), map_ids=dict(inputs.map_ids,rear=result_maps),
                   rear_rotation=transform[:3,:3], rear_lever=np.linalg.inv(transform)[:3,3])


@dataclass(frozen=True)
class MapAnchor:
    lens: str
    chain: int


@dataclass(frozen=True)
class PoseAnchor:
    T_world_imu_at_start: np.ndarray


@dataclass(frozen=True)
class MapPlacements:
    chains: dict
    groups: dict
    segments: dict
    initial_imu_poses: dict
    world_anchor: MapAnchor | PoseAnchor
    dropped: list
    gates: list
    initial_velocities: np.ndarray | None = None
