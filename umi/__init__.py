"""UMI 标定、场景轨迹处理与查看。

三个用户入口：umi-calib（umi.calibration）、umi-scene（umi.scene）、
umi-view（umi.apps.viewer）。umi.reconstruction.segment 为标定验证和共享算法提供内部引擎，
配置实现位于 umi.config，跨领域基础设施位于 umi.infra，设备登记位于 umi.devices.registry。
"""
