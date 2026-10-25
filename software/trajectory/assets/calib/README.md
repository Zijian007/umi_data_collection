这个目录给 `umi-traj` 读，不是 `umi-calib` 的产出。

- `calibration_r1440.json`：某一台 Insta360 X3 的相机–IMU 标定（2026-09-04）。说明文档叫样例标定。设备档案里的 `registry/cameras/sample/` 从这份生成。每台新 X3 默认自己走全套标定，不会把这份当未知相机的默认值。`$UMI_TRAJ_CALIB=factory` 仍指向这里，给 `umi-calib run` 对照用。
- `t_tcp_camera_nominal_cad.json`：夹爪尖相对镜头的 CAD 名义值，还没有独立实测。`umi-calib` 不算它。采集用的副本在 `registry/grippers/handheld_v1/versions/<版本>/`。
