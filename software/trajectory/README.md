# 独立轨迹代码指南

本模块服务 `umi-traj`：每段录像独立建图、定位并融合 IMU，不提供跨任务共享坐标。日常共享场景和双手采集由 [scene](../scene/README.md) 编排，共用的视觉能力位于 [vision](../vision/README.md)。

## 执行顺序

```text
cli.main() → workflow.run_batch() → workflow.run_recording()
    selection.select_calibration()       固定设备标定及夹爪几何
    reuse.run_fingerprint()              输入、资产、算法与引擎身份
    reuse.reuse_decision()               完整成功结果复用
    vision.recordings.prepare()          时间戳、IMU、丢帧策略
    runtime.reconstruction_settings()    视觉设置及独立融合配置
    vision.recordings.prepare_lens_videos()
    reconstruction.run_attempts()
        vision.slam.build_maps()         纯视觉建图
        vision.slam.relocalize_all_maps() 固定图集逐图定位并合并
        reconstruction.fuse_visual_imu() 调用独立融合 CLI
    algorithms.postprocess.export_trajectories()
    map_data.export_attempt()            仅用于展示的地图对齐
    results.finalize_result()            发布视频、摘要与报告
    results.cleanup_debug()              成功后按需清理中间产物
```

原片校验和设备选择发生在清理旧输出之前。每个镜头只裁切一次，所有重建尝试复用；可疑结果默认最多重建三轮，优先选择独立地图放置更少、位置约束覆盖更高的结果。参数位于 `config/trajectory.yaml`，指标不能代表实际误差。

## 文件职责

```text
software/trajectory/
├── cli.py                     参数解析与命令行入口
├── workflow.py                独立录像与批处理端到端编排
├── models.py                  强类型模型、请求与上下文契约
├── calibration/               设备规格与实测标定资产管理
│   ├── assets.py              标定包读取、测量到 SLAM 设置、旧包验证
│   └── selection.py           设备注册表与处理流程的边界
├── pipeline/                  重建调度与执行环境
│   ├── inputs.py              独立 CLI 输入发现和默认输出目录
│   ├── reconstruction.py      重建重试、融合调用与质量检查
│   ├── reuse.py               输入指纹、成功结果复用与提醒重评
│   └── runtime.py             独立融合配置和依赖诊断
├── reporting/                 结果交付、分析与报告
│   ├── map_data.py            图集展示导出、展示坐标对齐
│   ├── metadata.py            所用测量资产、外参和完整配置记录
│   ├── plots.py               轨迹预览图
│   ├── quality.py             跟踪统计、逐帧约束和可重评的质量提醒
│   ├── reports.py             用户报告和终端摘要
│   └── results.py             结果布局、旧结果迁移、发布与清理
└── algorithms/                图优化求解器内核
    ├── fusion.py              四阶段顺序入口 solve(...)
    ├── fusion_alignment.py    跟踪段细分、惯性尺度/重力对齐
    ├── fusion_placement.py    地图段放置和分段
    ├── fusion_graph.py        构建、优化因子图
    ├── fusion_recovery.py     恢复全帧率状态
    ├── fusion_factors.py      因子与解析 Jacobian
    ├── fusion_data.py         输入输出数据结构与文件边界
    ├── fusion_cli.py          独立求解 CLI
    └── postprocess.py         编排 IMU → 相机/TCP、指标与预览
```

`ReconstructionContext.visual` 将独立请求转换成不携带业务请求的 `VisualContext`。视觉程序、错误与路径由 `software/vision` 管理；外参和逐帧位姿右乘由 `software/camera/geometry.py` 管理；公共 TUM 读写位于 `software/pose_io.py`。

## 融合模块

```text
algorithms/
├── fusion_cli.py        参数解析 → 读取 → 求解 → 写结果
├── fusion_data.py       FusionInputs、FusionSolution、文件读写
├── fusion.py            四阶段顺序入口 solve(inputs, parameters, log=...)
├── fusion_alignment.py  跟踪段细分、惯性尺度/重力对齐 → AlignmentResult
├── fusion_placement.py  地图段放置和分段 → MapPlacements
├── fusion_graph.py      构建、优化因子图 → OptimizedGraph
├── fusion_recovery.py   恢复节点之间的全帧率状态 → FusionSolution
├── fusion_factors.py    因子与解析 Jacobian，自检 selftest()
├── postprocess.py       编排 IMU → 相机/TCP、指标、配置与预览输出
└── kiwi_fuse3s.py       旧脚本路径的薄兼容入口
```

求解器可直接导入，不解析命令行、不打开录像、不创建输出。输入和输出显式传递；尺度、跟踪段、偏置、GTSAM 图与优化结果都属于单次调用。文件边界独立，`write_solution()` 保留原轨迹、辅助 CSV 及 JSON 字段。`state.csv` 中历史兼容列仍保留，其零值不代表这些量已被估计。

算法内部使用视频秒。`FusionInputs.imu_video_clock` 已将 IMU 样本换到视频时钟，`FusionSolution.video_times_s` 与 `T_world_imu` 明确时间和位姿语义。`fusion_data` 在文件边界处理 EuRoC 人工 +1 秒，转换集中在 `software/camera/timing.py`；求解核心不处理文件时钟。

两套融合目标不同：独立模式估计跟踪段的尺度与放置，场景模式以已冻结的米制地图位姿为约束。两者分别维护，数值方法与历史验证见 [原理](../../docs/轨迹/原理.md)。

自检：

```bash
.deps/env/bin/python -m software.trajectory.algorithms.fusion_cli --selftest
```

## 运行记录与兼容

每轮 `logs/run.json` 记录运行身份、引擎哈希及完成状态；`logs/tracking.json` 记录各镜头各视觉阶段统计。结果生成优先读取结构化记录，历史结果仍可解析文本日志。原始文本日志继续保留。

`results.attempt_paths()` 统一解析选中轮次的当前与历史目录。旧平铺/diagnostics 结果迁移时保留独有文件并修复视频软链接；发布视频、摘要和报告成功后才能清理新运行的临时数据。

解释器使用 `UMI_PY`、`UMI_PY_TOOLS`，兼容旧 `KIWI_PY`、`KIWI_PY_TOOLS`；同用途的新变量优先。历史序列化字段 `source: factory` 表示样例测量，不是厂家标定。设备资产中的旧脚本仅用于 v1 外参校验，不执行。

## 输出

```text
umi_traj_out/<录像名>/
├── traj_tcp.txt
├── front.mp4
├── trajectory.png
├── report.txt
├── details/             相机/IMU 轨迹、逐帧信息、配置、摘要、展示地图
└── debug/
    ├── logs/            默认保留
    ├── dataset/         --keep-work 或失败时保留
    ├── lens_videos/
    └── reconstruction/try_N/
```

地图展示导出失败不会否定成功轨迹，错误单独记录。`--dry-run` 不创建输出，`--keep-work` 保留各轮计算现场。
