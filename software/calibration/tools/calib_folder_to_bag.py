"""calib_folder_to_bag — 把 recording_data.py 产出的 EuRoC 目录（mav0/cam0 + mav0/imu0，单目）转成 ROS1 bag，
供 Basalt 0.1.7 官方 `--dataset-type bag`（RosbagIO）与 Kalibr 使用。**不依赖 ROS**：用纯 Python 库 `rosbags`（Ternaris）
写 rosbag v2.0，连接头含正确 md5sum/msgdef（Basalt 内置 rosbag 读取器 instantiate<> 校验 md5）。

为什么必须走 bag（Basalt 0.1.7 源码事实）：
- `include/basalt/io/dataset_io_euroc.h:162` `EurocIO::read` 无条件 `data->num_cams = 2`；
- `include/basalt/optimization/poses_optimize.h:70` `resetCalib` 断言 `cam_types.size() == num_cams`；
  → 单目 EuRoC 目录 + `--cam-types kb4` 会断言失败；不得复制假 cam1。
- `include/basalt/io/dataset_io_rosbag.h`：`RosbagIO::read` 动态 `num_cams = cam_topics.size()`（所有 sensor_msgs/Image 话题），
  IMU = 首个非 /fcu 前缀的 sensor_msgs/Imu，图像编码仅 mono8/mono16，时间取 header.stamp，图像按相同 header 时间戳配对。

时间偏移视图（复测用）：`--cam-time-offset-ns N` 把图像 header.stamp（与 bag time）加 N（t_imu = t_cam + offset，
Basalt calibration.hpp 约定），IMU 不变——因为 basalt_vio 不应用 calibration.json 的 cam_time_offset_ns
（sqrt_keypoint_vio.cpp 中该行被注释）。生成 `<bag>.manifest.json` 记录源 sha256、偏移、计数、话题、编码。

若 `rosbags` 不可用：退出码 8 并打印 BLOCKED_FOR_MONO_BAG_CONVERSION（不得宣称一键可跑）。
用法：python -m software.calibration.tools.calib_folder_to_bag --euroc <root> --out <file.bag> [--cam-time-offset-ns 0] [--compression lz4|bz2|none] [--cam-topic /cam0/image_raw] [--imu-topic /imu0]
默认 LZ4 chunk 压缩（Basalt 内置 rosbag 读取器编译了 roslz4；未压缩 60–90 s×1440² mono 达数 GB）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

BLOCKED = "BLOCKED_FOR_MONO_BAG_CONVERSION"

try:  # 纯 Python ROS1 bag 写入
    from importlib.metadata import version as _pkg_version

    from rosbags.rosbag1 import Writer as _BagWriter
    from rosbags.typesys import Stores as _Stores
    from rosbags.typesys import get_typestore as _get_typestore

    BAG_BACKEND = f"rosbags {_pkg_version('rosbags')}"
except Exception as _e:  # noqa: BLE001
    _BagWriter = None
    BAG_BACKEND = None
    _BACKEND_ERR = repr(_e)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_euroc_folder(euroc: Path) -> tuple[list[tuple[int, str]], np.ndarray]:
    cam_csv = euroc / "mav0" / "cam0" / "data.csv"
    imu_csv = euroc / "mav0" / "imu0" / "data.csv"
    if not cam_csv.exists() or not imu_csv.exists():
        raise FileNotFoundError(f"need {cam_csv} and {imu_csv}")
    imgs = []
    for l in cam_csv.read_text(encoding="utf-8").splitlines():
        if not l.strip() or l.startswith("#"):
            continue
        t, name = l.split(",")[:2]
        imgs.append((int(t.strip()), name.strip()))
    imu = np.loadtxt(imu_csv, delimiter=",", comments="#", ndmin=2)
    if imu.shape[1] < 7:
        raise ValueError("imu0/data.csv needs t_ns,wx,wy,wz,ax,ay,az")
    return imgs, imu


COMPRESSIONS = ("lz4", "bz2", "none")
LZ4_FRAME_MAGIC = bytes.fromhex("04224d18")
ROS_LZ4_FLG = 0x64  # ROS roslz4 帧头 FLG：version 1 | block independence | content checksum（无 content-size、无 block checksum）


def ros_lz4_compress(x: bytes) -> bytes:
    """与 ROS roslz4 兼容的 LZ4 帧（Basalt 内置 rosbag 读取器可解）；写入前自检 FLG==0x64。"""
    import lz4.frame as lf

    out = lf.compress(x, block_size=lf.BLOCKSIZE_MAX4MB, block_linked=False, store_size=False, content_checksum=True)
    if out[:4] != LZ4_FRAME_MAGIC or out[4] != ROS_LZ4_FLG:
        raise RuntimeError(f"unexpected LZ4 frame header {out[:5].hex()} (need magic 04224d18 + FLG {ROS_LZ4_FLG:#x} for ROS/Basalt roslz4)")
    return out


def write_mono_bag(euroc: Path, out_bag: Path, cam_time_offset_ns: int = 0, cam_topic: str = "/cam0/image_raw", imu_topic: str = "/imu0",
                   frame_id_cam: str = "cam0", frame_id_imu: str = "imu0", compression: str = "lz4") -> dict:
    """compression: lz4（默认）| bz2 | none（仅测试）。60–90 s × 1440² mono8 未压缩约 4–5 GB。
    LZ4 帧标志必须与 ROS roslz4 写法一致（FLG=0x64：block independence=1、无 content-size 字段、content checksum=1、4 MB block）：
    Basalt 内置 rosbag 读取器（roslz4，libbasalt.so 含 rosbag::Bag::decompressLz4Chunk）在官方 0.1.7 二进制上实测——
    rosbags 默认 lz4.frame.compress(x)（linked blocks + content-size，FLG 0x68/0x60 变体）→ 'ROSLZ4_DATA_ERROR: malformed data'；
    FLG 0x64 → 读取正常且标定结果与未压缩 bag 逐位一致（4320 角点、0.349 px）。"""
    if _BagWriter is None:
        raise RuntimeError(f"{BLOCKED}: python package 'rosbags' unavailable ({_BACKEND_ERR})")
    if compression not in COMPRESSIONS:
        raise ValueError(f"compression must be one of {COMPRESSIONS}")
    euroc, out_bag = Path(euroc), Path(out_bag)
    if out_bag.exists():
        raise FileExistsError(f"{out_bag} exists — refuse to overwrite")
    imgs, imu = read_euroc_folder(euroc)
    ts = _get_typestore(_Stores.ROS1_NOETIC)
    Image = ts.types["sensor_msgs/msg/Image"]
    Imu = ts.types["sensor_msgs/msg/Imu"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    Vec3 = ts.types["geometry_msgs/msg/Vector3"]
    Quat = ts.types["geometry_msgs/msg/Quaternion"]

    def stamp(t_ns: int):
        return Time(sec=int(t_ns // 1_000_000_000), nanosec=int(t_ns % 1_000_000_000))

    n_img = 0
    shape = None
    out_bag.parent.mkdir(parents=True, exist_ok=True)
    w = _BagWriter(out_bag)
    if compression != "none":
        w.set_compression(getattr(_BagWriter.CompressionFormat, compression.upper()))  # 必须在 open 前
        if compression == "lz4":
            w.compressor = ros_lz4_compress  # 覆盖 rosbags 默认 lz4.frame.compress(x)（Basalt roslz4 无法解）
    w.open()
    try:
        c_img = w.add_connection(cam_topic, Image.__msgtype__, typestore=ts)
        c_imu = w.add_connection(imu_topic, Imu.__msgtype__, typestore=ts)
        # 按时间交错写入（bag time 单调有助于读取器）；图像时间加偏移
        events = [(t + cam_time_offset_ns, 0, i) for i, (t, _) in enumerate(imgs)] + [(int(imu[j, 0]), 1, j) for j in range(len(imu))]
        events.sort()
        seq_img = seq_imu = 0
        for t_ns, kind, idx in events:
            if kind == 0:
                name = imgs[idx][1]
                g = cv2.imread(str(euroc / "mav0" / "cam0" / "data" / name), cv2.IMREAD_UNCHANGED)
                if g is None:
                    raise FileNotFoundError(f"image {name} missing")
                if g.ndim == 3:
                    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
                if g.dtype == np.uint16:
                    enc, step, data = "mono16", g.shape[1] * 2, np.ascontiguousarray(g).view(np.uint8).reshape(-1)
                else:
                    g = g.astype(np.uint8)
                    enc, step, data = "mono8", g.shape[1], np.ascontiguousarray(g).reshape(-1)
                shape = [int(g.shape[1]), int(g.shape[0]), enc]
                msg = Image(header=Header(seq=seq_img, stamp=stamp(t_ns), frame_id=frame_id_cam), height=int(g.shape[0]), width=int(g.shape[1]),
                            encoding=enc, is_bigendian=0, step=int(step), data=data)
                w.write(c_img, int(t_ns), ts.serialize_ros1(msg, Image.__msgtype__))
                seq_img += 1
                n_img += 1
            else:
                r = imu[idx]
                msg = Imu(header=Header(seq=seq_imu, stamp=stamp(int(r[0])), frame_id=frame_id_imu),
                          orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0), orientation_covariance=np.full(9, -1.0),
                          angular_velocity=Vec3(x=float(r[1]), y=float(r[2]), z=float(r[3])), angular_velocity_covariance=np.zeros(9),
                          linear_acceleration=Vec3(x=float(r[4]), y=float(r[5]), z=float(r[6])), linear_acceleration_covariance=np.zeros(9))
                w.write(c_imu, int(r[0]), ts.serialize_ros1(msg, Imu.__msgtype__))
                seq_imu += 1
    finally:
        w.close()
    man = {"kind": "mono_rosbag_from_euroc_folder", "backend": BAG_BACKEND, "source_euroc": str(euroc.resolve()), "bag": str(out_bag.resolve()),
           "bag_sha256": _sha(out_bag), "cam_csv_sha256": _sha(euroc / "mav0" / "cam0" / "data.csv"), "imu_csv_sha256": _sha(euroc / "mav0" / "imu0" / "data.csv"),
           "topics": {"cam0": cam_topic, "imu0": imu_topic}, "n_images": n_img, "n_imu": int(len(imu)), "image_wh_encoding": shape,
           "cam_time_offset_ns_applied": int(cam_time_offset_ns), "compression": compression, "bag_bytes": int(out_bag.stat().st_size),
           "time_semantics": "image header.stamp = original t_ns + cam_time_offset_ns (t_imu = t_cam + offset, Basalt calibration.hpp); imu unchanged; "
                             "offset != 0 => shifted view for basalt_vio (which ignores calibration cam_time_offset_ns): output TUM t = IMU clock",
           "basalt_reader_facts": "RosbagIO: num_cams = #sensor_msgs/Image topics (mono OK); imu = first non-/fcu sensor_msgs/Imu; encodings mono8/mono16; header.stamp"}
    Path(str(out_bag) + ".manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--euroc", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="输出 .bag 路径（不存在）")
    ap.add_argument("--cam-time-offset-ns", type=int, default=0)
    ap.add_argument("--cam-topic", default="/cam0/image_raw")
    ap.add_argument("--imu-topic", default="/imu0")
    ap.add_argument("--compression", choices=COMPRESSIONS, default="lz4", help="rosbag chunk 压缩：lz4（默认，Basalt 可读）| bz2 | none（仅测试）")
    a = ap.parse_args()
    if _BagWriter is None:
        print(f"{BLOCKED}: python package 'rosbags' unavailable ({_BACKEND_ERR}); install with `pip install rosbags` (pure Python, ~150 KB)")
        return 8
    man = write_mono_bag(a.euroc, a.out, a.cam_time_offset_ns, a.cam_topic, a.imu_topic, compression=a.compression)
    print(json.dumps({k: man[k] for k in ("n_images", "n_imu", "image_wh_encoding", "cam_time_offset_ns_applied", "compression", "bag_bytes", "bag_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
