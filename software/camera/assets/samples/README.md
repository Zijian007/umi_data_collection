这里保留 `calibration_r1440.json`：某一台 Insta360 X3 的相机–IMU 样例标定（2026-09-04），用于算法比较。每台新 X3 自己完成标定，不会把样例当作未知相机的默认值。`UMI_TRAJ_CALIB=factory` 可显式选择它作对照。

夹爪几何保存在 `kits/<套件>/`，由 kit.yaml 的 `gripper.tcp` 选择。独立轨迹引擎默认输出相机轨迹；显式传入 `--tcp <几何 JSON>` 后才输出 TCP 轨迹。本目录不维护夹爪几何副本。坐标含义、CAD 依据和测量状态见 [TCP 坐标系](../../../../docs/设备与坐标.md)。
