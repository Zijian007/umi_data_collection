"""双臂/单臂 UMI 移动操作数据采集与轨迹重建系统。

提供五个顶层命令行工具：
- ``umi-calib`` (``software.calibration``): 相机与机内 IMU 标定及设备档案登记。
- ``umi-traj`` (``software.trajectory``): 独立单录像视觉建图与 IMU 因子图轨迹融合。
- ``umi-scene`` (``software.scene``): 共享场景建图、多相机外参放置与双臂任务批处理。
- ``umi-dataset`` (``software.dataset``): 任务数据集整理、夹爪开合对齐与训练包导出。
- ``umi-view`` (``software.viewer``): 三维轨迹、建图点云与多路原片同步可视化查看器。
"""

