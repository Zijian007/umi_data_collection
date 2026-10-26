这个目录给 `umi-traj` 读，不是 `umi-calib` 的产出。

- `calibration_r1440.json`：某一台 Insta360 X3 的相机–IMU 标定（2026-09-04）。说明文档叫样例标定。设备档案里的 `registry/cameras/sample/` 从这份生成。每台新 X3 默认自己走全套标定，不会把这份当未知相机的默认值。`$UMI_TRAJ_CALIB=factory` 仍指向这里，给 `umi-calib run` 对照用。
- `t_tcp_camera_nominal_cad.json`：夹爪尖相对镜头的 CAD 名义值，还没有独立实测。`umi-calib` 不算它。采集用的副本在 `registry/grippers/handheld_v1/versions/<版本>/`。

当前 TCP 原点为闭合指尖接触中点，采用机器人机械轴向：+X 外侧指向内侧指（前镜头画面左侧）、+Z 本体指向指尖、+Y 按右手规则确定。平移和指尖倾角沿用原 CAD，旧注册版本保留原定义。详见 [TCP 坐标系](../../../../docs/TCP坐标系.md)。
