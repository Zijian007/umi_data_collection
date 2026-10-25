# 轨迹代码阅读指南

先看 `workflow.py`。CLI 只解析参数，显式调用 `workflow.run_batch()`；批处理找到录像后调用 `run_recording()`，后者按执行顺序写出单段流程。

算法概念与 KIWI 官方实现的对照集中在 [原理与验证](../../docs/轨迹/原理.md)。本页对应本项目当前调用链；`algorithms/kiwi_fuse3s.py` 是本项目独立实现的融合程序。

共享场景的编排位于独立的 [scene 模块](../scene/README.md)，复用本模块的录像准备、`build_maps()`、`localize_map()` 与 `run_orb()`。单段 workflow 保持独立，不按场景标志分支。

```text
./umi-traj <录像或目录>
  cli.main()
    workflow.run_batch()
      recording_data.find_inputs()     找到录像（workflow 检查同名冲突）
      workflow.run_recording()         每段录像各自执行
        selection.select_calibration() 选择设备标定和夹爪几何
        reuse.run_fingerprint()        记录输入、资产和处理选项
        reuse.reuse_decision()         相同且完整的成功结果直接复用
        recording_data.prepare()       读帧时间与 IMU，检查丢帧策略
        recording_data.prepare_lens_videos()  原片裁切一次，准备共用视频
        reconstruction.run_attempts()  重建、重试，选择一轮结果
          reconstruction.run_attempt() 每轮按以下顺序执行
            build_maps()              第一轮视觉计算：建图
            relocalize_all_maps()     第二轮视觉计算：逐图定位并合并
            fuse_visual_imu()         视觉 + IMU -> IMU 世界位姿
        postprocess.export_trajectories()      IMU 位姿 -> 相机/TCP 轨迹
        map_data.export_attempt()      导出前镜头轻量地图，拟合展示变换
        results.finalize_result()      发布视频、辅助结果、摘要与 TXT 报告
        results.cleanup_debug()        成功后按配置清理临时数据
```

标定选择在处理、复用和清理旧目录之前完成。优先级仍为 `$UMI_TRAJ_CALIB` 工程覆盖、`--device` 显式设备、INSV 尾部设备序列号；夹爪来自 `--gripper` 或设备档案。`selection.py` 是设备注册表与轨迹处理的边界，`calibration_assets.py` 负责封装已求出的参数，求标定本身在 `software/calibration/`。

## 目录职责

```text
software/trajectory/
├── cli.py                 参数解析、命令分发
├── workflow.py            批处理与单段执行顺序、失败收尾
├── models.py              请求、路径、读取结果、重建结果
├── selection.py           选择相机标定与夹爪几何
├── calibration_assets.py  标定测量、旧包读取、生成 SLAM 输入
├── recording_data.py      查找录像、时间戳/IMU 提取与镜头视频准备
├── reconstruction.py      建图、逐图定位、IMU 融合、质量评估与重试
├── map_data.py            图集导出与地图展示对齐
├── reuse.py               输入指纹与成功结果复用规则
├── results.py             产物布局、日志归档、重复文件清理
├── reports.py             TXT 报告、终端摘要
├── runtime.py             依赖检查、运行路径、日志
├── algorithms/
│   ├── kiwi_fuse3s.py     本项目的离线 GTSAM 融合
│   └── postprocess.py     位姿转换、轨迹与指标、预览图
├── tools/
│   ├── make_lens_video.py   生成镜头视频
│   └── merge_frames.py      合并多张地图的逐帧定位
├── assets/
│   ├── calib/            样例相机标定与 TCP CAD 几何
│   └── config/           相机几何模板与遮手图
└── setup/
    ├── env/              依赖清单与锁定文件
    └── orbslam3/         构建脚本、补丁与 C++ 入口
```

相机共用的读取和画面约定位于 `software/camera/x3/`：`insv_reader.py` 解析容器尾部，`imu_reader.py` 解析 IMU，`extraction.py` 写时间戳/IMU 数据，`frame.py` 统一裁剪、帧率和时钟偏移。标定、轨迹都直接使用它们。

## 重建与重试

`workflow.run_recording()` 先提取时间戳和 IMU，再调用 `recording_data.prepare_lens_videos()` 从原片生成镜头视频，默认 720×720。`config/trajectory.yaml` 的 `video.size_px` 同时控制视频边长、SLAM 内参和遮手图尺寸；原片模式、标定裁切和取帧步长集中在 `software/camera/x3/frame.py`。视频放在 `debug/lens_videos/`，所有尝试只读共用这一份。

`reconstruction.run_attempts()` 管理重试；`run_attempt()` 是每轮重建的阅读入口，直接列出：

1. `build_maps()`：每个镜头使用 ORB-SLAM3 `MONOCULAR` 纯视觉建图，前镜头使用遮手图，返回 `MapAtlas`。
2. `relocalize_all_maps()`：加载完整图集，逐图重放同一视频并开启 `localization_only`，不新增关键帧、不遮手。当前来源仍有效时保持来源，失效后按关键帧数优先级切换，返回 `LensLocalization`。
3. `fuse_visual_imu()`：将逐帧视觉定位、原始 IMU、相机–IMU 外参与时差送入 `algorithms/kiwi_fuse3s.py`，返回 `FusionResult`，得到米制、重力对齐的 IMU 位姿。

前两步的 C++ 入口见 [x3_mono_slam.cc](setup/orbslam3/x3_mono_slam.cc)：只传图像和时间，不传 IMU。IMU 在第三步参与尺度与重力估计、运动约束和联合优化；融合不重新提取图像特征，也不修改 ORB 地图点。

双镜头模式先分别建好前后镜头的地图，再各自定位，最后共同融合；两个镜头的地图互相独立。地图列表由现有 ORB-SLAM3 日志接口提供，`read_atlas_maps()` 集中将其解析成带索引和关键帧数的 `AtlasMap`。

具体命令、日志重定向、nice/taskset/timeout 在 `runtime.py`；业务执行顺序由 Python 函数维护。旧的整段 Shell 编排已移除。

选中最佳融合结果后，`postprocess.export_trajectories()` 显式执行 IMU → 相机 → TCP 的位姿换算，写轨迹、统计和预览图。数值融合、裁剪、定位合并、位姿换算及算法参数保持原样。

`reconstruction.py` 内的 `assess_try()` 判定本轮是否可疑：独立地图摆放多于一次，或地图位置约束帧比例小于 95%。可疑时最多尝试 `--max-tries` 次，优先选摆放次数少的，再选地图位置约束比例高的；指标相同保留较早的一轮。达到非可疑结果即可提前结束。后续尝试共用第一次的镜头视频。

质量指标读取或计算失败会抛出带有原始原因的阶段异常，统一进入重试处理。复用判断返回已验证的摘要，workflow 直接使用；`--force` 跳过复用判断，仍先验证输入和标定。

这些指标用于重试与诊断，不能代表本次轨迹的实测误差。没有独立真值对照时，TXT 报告仍写“实际误差未测量”。

## 输出与失败

结果分为根目录交付文件、`details/` 辅助结果和 `debug/` 排查数据：

```text
umi_traj_out/<录像名>/
├── traj_tcp.txt
├── front.mp4
├── trajectory.png
├── report.txt
├── details/
│   ├── traj_camera.txt
│   ├── traj_imu.txt
│   ├── frames.csv
│   ├── run_settings.json
│   ├── summary.json
│   └── map.json                前镜头地图展示数据（导出成功时）
└── debug/
    ├── logs/                  默认保留运行日志
    ├── dataset/               --keep-work、失败或迁移旧结果时保留
    ├── lens_videos/           本段录像共用的前/后镜头视频
    └── reconstruction/
        └── try_N/             各次建图、重定位与融合的中间数据
```

默认新运行成功后保留根目录结果、`details/` 和 `debug/logs/`，清理数据集、共享镜头视频和重建中间文件。`front.mp4` 发布为独立文件，清理后 viewer 仍能播放；文件系统支持时使用硬链接，不额外占用一份视频空间。

`--keep-work` 保留所有轮次的数据集、地图、镜头视频、逐帧定位和融合内部状态；失败或中断也保留。轨迹数值和重试策略不变。

`map_data.export_attempt()` 在轨迹生成后导出选中轮次的前镜头图集，并以逐帧定位和最终相机轨迹拟合展示变换。`details/map.json` 在默认清理后保留，查看地图无需 `--keep-work`；只有保留完整原始图集才需要该选项。地图导出失败写提醒与独立日志，成功轨迹仍正常发布。原始坐标的子地图不会强行与米制轨迹叠加。

`results.result_path()` 同时支持 `details/`、旧 `diagnostics/` 与平铺结果，标定 C 验证和 viewer 使用同一读取入口。复用旧结果时自动迁移布局，迁移只搬文件、发布选中轮次的视频并修复引用，不删除已有独有中间数据。

阶段错误区分镜头视频准备、建图、逐图定位、合并定位和 IMU 融合；重试摘要也保留该归因。`debug/logs/runner_tryN.log` 记录每轮阶段摘要与失败原因，命令原始日志留在对应阶段目录。

进入处理后，失败阶段写入 `details/summary.json` 和 `report.txt`，堆栈位于 `debug/logs/stage_error.log`。标定或输入预检查失败时只在终端报告，保留旧输出。视频与报告发布成功后才清理临时数据，清理失败记为提醒并保留剩余数据。

## 修改位置

跨模块命名、终端输出与验证约定见 [代码维护](../../docs/dev/README.md)。

- 改执行顺序或失败策略：`workflow.py`。
- 改选择哪份标定：`selection.py` 和 `software/registry.py`。
- 改录像发现、默认输出目录或丢帧处理：`recording_data.py`；原始解析在 `camera/x3/`。
- 改算法数值：`config/trajectory.yaml`；改建图、定位、融合顺序：`reconstruction.py`；改进程执行：`runtime.py`；改裁切与定位合并算法：`tools/`。
- 改位姿数学：`algorithms/postprocess.py`；改融合：`algorithms/kiwi_fuse3s.py`。
- 改落盘目录或清理：`results.py`；改文字展示：`reports.py`。
- 改安装：根目录 `install.sh` 和 `setup/`。

## 配置入口

`workflow` 先解析 [统一参数配置](../../config/README.md)，再选择设备、检查复用并准备输出。`runtime.reconstruction_settings()` 组合本次测量与算法配置，生成 ORB YAML 和融合 JSON。融合显式读取相机 JSON，始终使用共享脚本。`--dry-run` 只读预检；`notes` 变化可以只更新解释，重试条件变化会重算。
