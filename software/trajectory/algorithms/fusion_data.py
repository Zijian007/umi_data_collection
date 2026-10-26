"""独立融合的显式观测、求解结果与文件边界。"""
from __future__ import annotations
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from software.camera.timing import canonical_to_video_seconds, euroc_ns_to_seconds, video_to_canonical_seconds
from software.trajectory.calibration_assets import read_basalt


@dataclass(frozen=True)
class FusionInputs:
    observations: dict
    map_ids: dict
    imu_video_clock: np.ndarray  # t_video_s, gyro xyz, acc xyz
    video_times_s: np.ndarray | None
    camera: dict
    rear_rotation: np.ndarray
    rear_lever: np.ndarray


@dataclass(frozen=True)
class FusionSolution:
    video_times_s: np.ndarray
    T_world_imu: np.ndarray
    velocities: np.ndarray
    gyro_biases: np.ndarray
    acc_biases: np.ndarray
    segments: tuple
    evidence: dict


def load_inputs(*, front_frames: Path, rear_frames: Path, imu_csv: Path, camera_json: Path,
                offset_ns: int, rear_quaternion: list[float], rear_lever: list[float],
                use_rear: bool, times_csv: Path | None = None) -> FusionInputs:
    camera = read_basalt(camera_json)
    if camera['cam_time_offset_ns'] != offset_ns:
        raise ValueError('时间偏移与所选相机标定不一致')
    observations, map_ids = {}, {}
    for lens, path in (('rear', rear_frames), ('front', front_frames)):
        obs, maps = {}, {}
        if lens == 'front' or use_rear:
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    if row['ok'] == '1' and row.get('state', '2') == '2':
                        k = int(row['frame_idx'])
                        obs[k] = (np.array([float(row[c]) for c in ('tx','ty','tz')]),
                                  Rot.from_quat([float(row[c]) for c in ('qx','qy','qz','qw')]).as_matrix())
                        if row.get('map'):
                            maps[k] = row['map']
        observations[lens], map_ids[lens] = obs, maps
    with imu_csv.open() as stream:
        imu = np.array([[float(x) for x in row] for row in csv.reader(stream) if row and not row[0].startswith('#')])
    imu[:, 0] = euroc_ns_to_seconds(imu[:, 0], camera_offset_ns=offset_ns)
    times = None
    if times_csv is not None:
        with times_csv.open() as stream:
            times = canonical_to_video_seconds([float(row['t_canonical_s']) for row in csv.DictReader(stream)])
    return FusionInputs(observations, map_ids, imu, times, camera,
                        Rot.from_quat(rear_quaternion).as_matrix(), np.array(rear_lever))


def write_solution(solution: FusionSolution, output: Path) -> None:
    """保持原轨迹精度、辅助 CSV 字段与 JSON 结构；求解成功后才创建输出。"""
    output.mkdir(parents=True, exist_ok=True)
    with (output/'traj_tum.txt').open('w') as fo:
        fo.write('# t x y z qx qy qz qw | T_WI, W gravity-aligned metric, front+rear relocalization + IMU, piecewise map placement (KIWI-style, ours) | canonical s\n')
        for k, time in enumerate(video_to_canonical_seconds(solution.video_times_s)):
            q = Rot.from_matrix(solution.T_world_imu[k,:3,:3]).as_quat()
            p = solution.T_world_imu[k,:3,3]
            fo.write('%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n' % (time, *p, *q))
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
