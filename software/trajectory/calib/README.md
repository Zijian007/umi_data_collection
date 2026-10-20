这个目录给 `umi-traj` 读，不是 `umi-calib` 的产出。

- `calibration_r1440.json`：某一台 Insta360 X3 的相机–IMU 标定（2026-09-04）。说明文档叫样例标定。代码和 `summary.json` 里的 `source` 仍写 `factory`，用来和装进仓库根目录 `calibration_user/` 的标定区分，这个字段不改名。换相机先 `./umi-calib verify`，见 `docs/标定/说明.md`。
- `t_tcp_camera_nominal_cad.json`：夹爪尖相对镜头的 CAD 名义值，还没有独立实测。`umi-calib` 不算它。
