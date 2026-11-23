"""UMI 标定、场景轨迹处理与查看。

三个用户入口：umi-calib（umi.workflows.calibration）、umi-scene（umi.workflows.scene）、
umi-view（umi.apps.viewer）。umi.workflows.segment 为标定验证和共享算法提供内部引擎，
umi.dataset 为 umi-scene export 提供训练包导出。
配置实现位于 umi.config，跨领域基础设施位于 umi.infra，设备登记位于 umi.devices.registry。
"""
