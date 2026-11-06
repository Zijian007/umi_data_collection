这里保留 `calibration_r1440.json`：某一台 Insta360 X3 的相机–IMU 样例标定（2026-09-04），用于算法比较。每台新 X3 自己完成标定，不会把样例当作未知相机的默认值。`UMI_TRAJ_CALIB=factory` 可显式选择它作对照。

夹爪几何统一保存在 `registry/grippers/<夹爪ID>/<三位版本>/`。运行时必须传入 registry 或批次已选定的几何路径；本目录不再维护夹爪几何副本。坐标含义、CAD 依据和测量状态见 [TCP 坐标系](../../../../docs/设备与坐标.md)。
