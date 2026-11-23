"""共享场景：一次建图，多段任务，左右手在同一米制坐标和时间轴中处理。

架构分为三大模块：
- management: 采集任务生命周期与设备资产管理 (batches, ingest, episodes, inputs)
- spatial: 多相机空间几何、视觉定位与世界系 GTSAM 融合求解 (fusion, geometry, mapping, rear_calibration, slam)
- pipeline: 录像提取、质量预检与轨迹结果交付 (recordings, recording_checks, results)"""
