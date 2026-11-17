"""UMI 标定、场景轨迹处理与查看。

三个用户入口：umi-calib（software.calibration）、umi-scene（software.scene）、
umi-view（software.viewer）。software.trajectory 为标定验证和共享算法提供内部引擎，
software.dataset 为 umi-scene export 提供训练包导出。
配置实现位于 software.configuration，跨领域基础设施位于 software.infrastructure，设备登记位于 software.camera.registry。
"""
