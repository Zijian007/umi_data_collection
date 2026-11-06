"""X3 录像转夹爪轨迹（独立单目/双目 + IMU 融合）。

阅读入口：cli.main() -> workflow.run_batch() -> workflow.run_recording()。
架构模块：
- calibration: 设备规格与实测标定资产管理
- pipeline:    重建多轮重试、输入校验、运行设置与复用策略
- reporting:   结果交付、展示地图、轨迹预览图与质量分析
- algorithms:  图优化多传感器融合求解器内核"""
