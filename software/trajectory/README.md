# 独立轨迹代码指南

本模块提供内部逐段轨迹引擎：每段录像独立建图、定位并融合 IMU，不提供跨任务共享坐标。日常共享场景和双手采集由 [scene](../scene/README.md) 编排，共用的视觉能力位于 [vision](../vision/README.md)。

各模块通过真实目录直接导入；包初始化不加载求解器、不注入旧模块别名。

## 执行顺序

```text
cli.main() → workflow.run_batch() → workflow.run_recording()
    selection.select_calibration()       选择相机标定及显式指定的 TCP 几何
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

原片校验和设备选择发生在清理旧输出之前。每个镜头只裁切一次，所有重建尝试复用；可疑结果默认最多重建三轮。先检查每个有位置约束的分段相对自身初始化的尺度变化，双向倍率超过 `retry.maximum_scale_change_ratio`（默认 2）或证据无效时，该轮不能选中；全部轮次因此被拒绝则执行失败并保留现场。其余轮次优先选择独立地图放置更少、位置约束覆盖更高的结果。各轮拒绝原因保存在 `tries`，融合证据包含 `scale_consistency`。该保护不估计真实误差，也不保证视觉地图正确；2 倍上限仍需多录像验证。参数位于 `config/trajectory.yaml`。

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
└── postprocess.py       编排 IMU → 相机/TCP、指标、配置与预览输出
```

求解器可直接导入，不解析命令行、不打开录像、不创建输出。输入和输出显式传递；尺度、跟踪段、偏置、GTSAM 图与优化结果都属于单次调用。文件边界独立，`write_solution()` 保留原轨迹、辅助 CSV 及 JSON 字段。`state.csv` 中历史兼容列仍保留，其零值不代表这些量已被估计。

算法内部使用视频秒。`FusionInputs.imu_video_clock` 已将 IMU 样本换到视频时钟，`FusionSolution.video_times_s` 与 `T_world_imu` 明确时间和位姿语义。`fusion_data` 在文件边界处理 EuRoC 人工 +1 秒，转换集中在 `software/camera/timing.py`；求解核心不处理文件时钟。

两套融合目标不同：独立模式估计跟踪段的尺度与放置，场景模式以已冻结的米制地图位姿为约束。两者分别维护，数值方法与历史验证见 [原理](../../docs/算法原理.md)。

自检：

```bash
.deps/env/bin/python -m software.trajectory.algorithms.fusion_cli --selftest
```

## 运行记录与兼容

每轮 `logs/run.json` 记录运行身份、引擎哈希及完成状态；`logs/tracking.json` 记录各镜头各视觉阶段统计。结果生成优先读取结构化记录，历史结果仍可解析文本日志。原始文本日志继续保留。

`results.attempt_paths()` 统一解析选中轮次的当前与历史目录。旧平铺/diagnostics 结果迁移时保留独有文件并修复视频软链接；发布视频、摘要和报告成功后才能清理新运行的临时数据。

解释器使用 `UMI_PY`、`UMI_PY_TOOLS`，兼容旧 `KIWI_PY`、`KIWI_PY_TOOLS`；同用途的新变量优先。历史序列化字段 `source: factory` 表示样例测量，不是厂家标定。设备资产中的旧脚本仅用于 v1 外参校验，不执行。

## 输出

标定 C 验证和开发测试使用内部逐段引擎。默认只输出相机/IMU 轨迹；需要 TCP 时显式提供 `--tcp kits/kit_001/tcp_camera.json`，可用 `--mask <PNG>` 覆盖通用遮手图。相机验收不依赖套件或夹爪。单段输出布局如下：

```text
umi_traj_out/VID_20261003_182246_00_067/
├── traj_tcp.txt           提供 --tcp 时生成的夹爪尖轨迹
├── front.mp4              前镜头视频，viewer 直接读取
├── trajectory.png         轨迹预览（绘图成功时生成）
├── report.txt             状态、设备、标定、运动统计和提醒
├── details/
│   ├── traj_camera.txt    相机轨迹，供标定验证和排查
│   ├── traj_imu.txt       IMU 轨迹
│   ├── frames.csv         逐帧对应关系和约束来源
│   ├── run_settings.json  完整配置及资产哈希
│   ├── summary.json       完整结果指标和复用判断信息
│   └── map.json           前镜头轻量地图展示数据（导出成功时）
└── debug/
    ├── logs/              建图、定位和融合日志，默认保留
    ├── dataset/           提取的时间戳与 IMU（按需保留）
    ├── lens_videos/       本段录像共用的镜头视频（按需保留）
    └── reconstruction/    各轮重建中间数据（按需保留）
```

内部逐段结果先看 `report.txt`、`trajectory.png`，相机轨迹位于 `details/traj_camera.txt`，提供 TCP 几何时还可使用 `traj_tcp.txt`，配合 `front.mp4` 回看画面。视频是前镜头裁切，默认 720×720、约 29.97 fps，保留鱼眼畸变；实际裁切、尺寸、取帧和编码参数见本次运行记录。`details/` 提供完整结果与配置，`debug/` 用于排查。

新运行成功后默认清理数据集与重建中间文件，根视频、轨迹、辅助结果和日志继续保留。`--keep-work` 保留完整中间数据；失败或中断时也保留。无独立真值的任务录像只报告跟踪和运动统计，不把地图约束比例当作真实误差。

`details/map.json` 保存选中轮次的前镜头稀疏点云、关键帧和展示对齐信息。它独立于被清理的 `.osa`，可直接由 `umi-view` 查看；每图最多展示 50000 点。对齐只用于展示，不改动轨迹或地图，拟合残差不代表真实精度。导出失败只增加提醒，不改变轨迹抽取的成功状态，详细原因见 `debug/logs/map_export_error.log`。

旧 `diagnostics/` 与平铺结果仍能读取；输入匹配而跳过计算时会自动整理为上述目录。整理旧结果只搬迁文件、发布根视频并修复引用，不重新建图、不改变轨迹数值，也不删除已有独有中间数据。

### details/summary.json

**总览**

| 字段 | 意思 |
|---|---|
| `status` | `ok` 或 `failed`；失败时 `reason` 写原因 |
| `pose_frame` | `camera` 或 `tcp`，由是否提供安装外参决定 |
| `camera_path_length_m` / `tcp_path_length_m`、相应的 `*_extent_m` | 所选坐标原点的路程、活动范围（米） |
| `notes` | 自动检查发现的问题（经验规则，不是精度测试） |
| `run_settings` | 相对结果根目录的 `details/run_settings.json` |

**建图**

| 字段 | 意思 |
|---|---|
| `tracking.front_map_build_tracked` | 建图那一趟跟上了多少帧 / 总帧数（`debug/logs/run.txt` 里的 `map: rc=… TRACKED a / b`）。偏低：开头没初始化、中途跟丢另起图 |
| `tries`、`chosen_try` | 重建失败或质量可疑时会重试完整的建图、定位和融合（默认最多 3 次）。`placements` 是融合里地图被摆了几块，大于 1 多半建图时跟丢、atlas 里不止一张图 |

**定位**

| 字段 | 意思 |
|---|---|
| `tracking.relocalization` | 每一张图单独查一遍。只有 `loc_m0` 表示全程一张图；出现 `loc_m1` 就是建图时另起过图，定位对每张图各播一遍 |
| `tracking.front_lens_tracked_fraction` | 定位成功的帧比例（和 `details/frames.csv` 的 `front_lens_tracked` 一致） |

**融合（最终轨迹）**

| 字段 | 意思 |
|---|---|
| `constraint_fraction.full` | 有多少比例的帧最终用地图位置（越高越好；验证过的录像在 0.85–1.0）。定位成功但和 IMU 对不上会变成 `rot`，不是建图数字 |
| `camera_map_chains` | 融合里连续「有地图位置」的段数。1 最好 |
| `longest_stretch_without_map_position_s` | 最长一段没有地图位置、靠 IMU 衔接的时长（秒） |

### debug/logs/：运行日志

默认保留镜头视频准备日志、选中轮次的运行/建图/定位/融合记录，以及各次重试摘要；整段失败时保留完整阶段数据。摘要的失败阶段会区分视频准备、建图、定位、合并定位和 IMU 融合。

| 文件 | 内容 |
|---|---|
| `run.txt` | 建图/定位摘要、工具哈希 |
| `front_map_log.txt` | 建图过程、初始化和新地图事件 |
| `front_loc_frames.csv` | 合并后的逐帧定位状态、位姿和地图来源 |
| `fuse_info.json` | 融合全文，包括 chains、尺度与参数 |
| `fuse.log` | 融合过程日志（源日志存在时保留） |
| `runner_tryN.log` | 各轮阶段摘要、失败阶段与原始日志位置 |

### debug/ 中间数据

新运行仅在 `--keep-work`、失败或中断时保留；迁移旧结果时保留原有独有数据。

| 路径 | 内容 |
|---|---|
| `debug/dataset/mav0/` | 每帧时间与 IMU |
| `debug/lens_videos/front.mp4` | 从原片生成一次的前镜头视频；dual 模式另有 rear.mp4 |
| `debug/reconstruction/try_N/front/front.mp4` | 该轮引用的共享前镜头视频，各轮不会重新裁切原片 |
| `debug/reconstruction/try_N/front/map/map.osa` | 保存的地图 |
| `debug/reconstruction/try_N/front/map/frames.csv` | 建图逐帧状态 |
| `debug/reconstruction/try_N/front/loc_m*/frames.csv` | 对各地图逐帧重定位的结果 |
| `debug/reconstruction/try_N/fuse/state.csv` | 融合内部状态，包括速度与 bias |

根目录 `front.mp4` 独立保留，不依赖这些中间目录。双镜头模式的后镜头视频属于中间数据，可用 `--keep-work` 保留。

### 参数与复用记录

`details/run_settings.json` 的 `parameters` 保存完整有效配置、覆盖来源、代码和引擎身份；TXT 显示配置 ID 和覆盖项。重建参数与提醒规则分别计算指纹，详见 [参数管理](../../config/README.md)。新运行的融合脚本来自项目，标定包仅提供测量；双镜头模式的后镜头仍为样例几何。

场景任务的结果布局与人工验收见[结果指南](../../docs/结果查看与验收.md)。
