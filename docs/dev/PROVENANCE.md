# 第三方来源

[代码维护](README.md) · [当前文档](../README.md)

本页列出本项目使用或改编的外部代码与预编译程序，以及本地修改所在位置。改动历史见 git 提交记录。

| 来源 | 版本 | 用途 | 本地修改与说明 |
|---|---|---|---|
| 实验室 UMI 仓库 `github.com/Torbjorn-s-Apple-Americano/umi` | 分支 `dev_Zijian`，提交 `71a124d640b998dbbfcf35504dea830e179eddfc` | X3 读取、标定流程与逐段轨迹引擎的初始代码 | 导入后已按本仓库结构重写，不带原仓库提交历史 |
| ORB-SLAM3（`cheng-chi/ORB_SLAM3`，UMI 使用的 fork） | 提交 `b741dca39015330ef4bcc3a85f89493503ade04b` | 纯视觉单目建图与定位 | 补丁与驱动见 [setup/trajectory/orbslam3/](../../setup/trajectory/orbslam3)，下文列出 |
| Pangolin | 提交 `dd801d244db3a8e27b7fe8020cd751404aa818fd`（v0.6） | ORB-SLAM3 链接依赖，运行时不开窗口 | 无 |
| Basalt | 标签 `0.1.7`（提交 `6d8637b9d68ea18a1a63c1baa72818779e156932`），BSD 3-Clause | 相机内参 A 与相机–IMU B 标定 | 预编译包、补丁与逐文件哈希见 [setup/calibration/basalt/PROVENANCE.md](../../setup/calibration/basalt/PROVENANCE.md) |
| GTSAM | PyPI `gtsam==4.3.0` | 融合、固定地图与 poster_vio 的因子图优化 | 无；版本锁定在 [setup/trajectory/env/requirements.txt](../../setup/trajectory/env/requirements.txt) |
| KIWI 0.1.0（`TToTMooN/KIWI_software`） | 提交 `22860a30dc1c2e9e5f9a2e55c4955a4fba84646e`，Apache-2.0 | 共享地图米制对齐、镜头放置、固定场景融合与 KIWI 音频同步的参考实现 | 改编范围与许可见 [KIWI 改编说明](#kiwi)与[许可证](LICENSE.kiwi) |
| specialist（本地参考实现） | 见 NOTICE | poster_vio 初始化、特征几何与重关联、关键帧恢复的参考 | 改编范围与许可见 [umi/estimation/poster_vio/NOTICE.md](../../umi/estimation/poster_vio/NOTICE.md) |

ORB-SLAM3 与 Pangolin 先从组织下的镜像仓库拉取，失败时退回原作者仓库；两种来源都按上表提交号校验，由 [build_orbslam3.sh](../../setup/trajectory/orbslam3/build_orbslam3.sh) 构建。

## ORB-SLAM3 本地补丁

| 文件 | 作用 |
|---|---|
| `opencv47_aruco_api.patch` | 适配 OpenCV ≥ 4.7 的 ArUco 接口 |
| `orbslam3_map_select.patch` | 用环境变量 `ORB_MAP_INDEX` 选择定位使用的子地图 |
| `orbslam3_offline_sync.patch` | 离线单目逐帧完成局部建图与闭环，使结果不受线程调度影响 |
| `orbslam3_stable_order.patch` | 按稳定 ID 处理平分、观测与优化输入，保证同一输入重复运行结果一致 |
| `orbslam3_tag_init.patch` | 标签已校正坐标的畸变向量显式清零 |
| `orbslam3_kb4_forward.patch` | 修复前向 KB4 投影并限制射线 ≤ 85°（默认启用，`UMI_ORB_KB4_FORWARD`） |
| `orbslam3_feature_parallel.patch` | ORB 金字塔层并行提取，初始化后跳过无用途的标签检测 |
| `x3_mono_slam.cc`、`export_atlas.cc` | X3 纯视觉单目驱动（逐帧写位姿）与地图图集导出工具 |
| `*_check.cc` | 构建后自检：KB4 投影、离线同步、稳定顺序、特征提取 |

`orbslam3_offline_sync.patch` 与 `orbslam3_stable_order.patch` 是黄金回归能要求“逐位一致”的前提。

## KIWI

The scene metric alignment, front-to-rear placement, and per-arm fixed-frame
fusion in `umi/estimation/fixed_map/geometry.py`,
`umi/estimation/fixed_map/fusion.py`, and
`umi/estimation/inertial.py` adapt ideas and implementation structure
from **KIWI: Kinematic Interface for the Wild**, public reference release 0.1.0,
commit `22860a30dc1c2e9e5f9a2e55c4955a4fba84646e`:

- <https://github.com/TToTMooN/KIWI_software/blob/22860a30dc1c2e9e5f9a2e55c4955a4fba84646e/kiwi/track/metric_align.py>
- <https://github.com/TToTMooN/KIWI_software/blob/22860a30dc1c2e9e5f9a2e55c4955a4fba84646e/kiwi/track/scene_frame.py>
- <https://github.com/TToTMooN/KIWI_software/blob/22860a30dc1c2e9e5f9a2e55c4955a4fba84646e/kiwi/state/fusion.py>
- <https://github.com/TToTMooN/KIWI_software/blob/22860a30dc1c2e9e5f9a2e55c4955a4fba84646e/kiwi/sync/audio.py>

Authors: Benjamin C. Yang, Weiying Wang, Shenggao Li, Keming Yan,
Sasha Wilkinson, Zelin Wang, Yip Fun Yeung, and Lingfeng Sun.
Upstream is licensed under Apache-2.0; a copy is in [Apache-2.0 license](LICENSE.kiwi).

`umi/perception/audio/kiwi_audio.py` adapts KIWI's 48 kHz signed 16-bit decoder,
spectral landmarks, offset voting, GCC-PHAT refinement, and drift validation.
UMI checks matching camera models from INSV metadata and matching recording
modes before applying KIWI's zero A/V correction for a matched pair. It
converts the estimate to UMI's left-referenced task clock.

Local adaptations add exact IMU interval boundaries, observability and residual
gates, parallel/antiparallel gravity handling, explicit scene/calibration versions,
configuration-driven noise, and integration with the X3 timestamp convention.
These files are modified adaptations, not unmodified upstream KIWI code.
Shared-map scale estimation and motion-segment screening now call the common
solver in `umi/estimation/fusion/alignment.py`; the three-node system in
`geometry.py` is retained only as an observability check. Shared maps fit one
scale, gravity vector and accelerometer bias across accepted segments, rather
than adopting independent per-segment map placements.
The separate per-recording implementation is in `umi/estimation/fusion/solver.py`;
it runs in a worker process through `umi/workflows/segment/fusion_runner.py`.

The task-local poster-VIO implementation additionally references specialist
commit `da29719cdbaa366df0599984068ad4553d9cc86d` (MIT, Hybrid Robotics).
The reference paths, local adaptations and complete MIT notice are retained in
[poster_vio/NOTICE.md](../../umi/estimation/poster_vio/NOTICE.md).

## 样例资产

[umi/devices/assets/samples/calibration_r1440.json](../../umi/devices/assets/samples/calibration_r1440.json) 是实验室为某一台 X3 做的标定样例，由上表 Basalt 预编译程序算出，不是厂家出厂标定，也不是其他设备可用的默认标定。

## 维护

新增或升级第三方代码、预编译程序或改编实现时，在上表登记来源、版本与许可，并把本地修改放在 `setup/` 或对应模块旁的 NOTICE 中。
