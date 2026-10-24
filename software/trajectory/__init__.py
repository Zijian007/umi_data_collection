"""X3 录像转夹爪轨迹。

阅读顺序：cli -> workflow.run_batch -> workflow.run_recording。
INSV 共享读取在 software.camera.x3；数值算法在 algorithms/，外部工具在 tools/。
calibration_assets 将已求出的标定封装为运行资产；标定求解在 software.calibration。
目录职责和流程说明见 README.md。
"""
