# 固定地图融合整理前的数值基线

`scene_fusion_before_tags.npz` 在 2026-11-11 拆分场景 `fuse()` 前生成，使用本次修改前工作区中的实现（不是 Git HEAD）。输入来自 `tests/test_scene.py` 的 `motion()`：前三十五个视觉节点及覆盖它们的合成 IMU，单位为米、秒，单工作镜头且外参为单位矩阵。不是实际录像精度证据。

字段：`times` 为节点时刻，`poses` 为场景系相机观测，`imu` 为时间、角速度、加速度，`expected` 为原实现输出的 IMU 位姿。使用生成时的默认 trajectory 和 scene 参数。测试保留 1e-10 的跨环境浮点容差；本机完整 front、dual、短缺口三种输入的新旧实现比较均逐数值一致。
