# 独立轨迹代码指南

每段录像由内部逐段轨迹引擎独立建图、定位并融合 IMU，轨迹使用本段局部坐标。场景和双手任务由 [scene](../../scene/README.md) 编排，共用视觉能力位于 [perception](../../perception/README.md)；融合数值实现位于 [umi/reconstruction/estimation/fusion](../estimation/fusion)。[算法实现约定](../../../docs/dev/算法实现约定.md)

各模块通过真实目录直接导入；包初始化只说明职责。

## 执行顺序

```text
umi/apps/segment.py main() → workflow.run_batch() → workflow.run_recording()
    selection.select_calibration()       选择相机标定及显式指定的 TCP 几何（umi/devices/selection.py）
    workflow.run_fingerprint()           输入、资产与算法参数（不含代码版本）
    workflow.reuse_decision()            完整成功结果复用；--force 时整段重算
    perception.recordings.prepare()      时间戳、IMU、丢帧策略
    workflow._reconstruction_settings()  冻结环境镜头标定与海报观测，生成视觉与融合设置
    perception.recordings.prepare_lens_videos()
    attempts.run_attempts()
        perception.slam.build_maps()     单目建图，可用陀螺仪旋转辅助
        perception.slam.relocalize_all_maps() 固定图集逐图定位并合并
        attempts.fuse_visual_imu() → fusion_runner.execute()  在 spawn 工作进程中调用 fuse()
    workflow._write_results()          trajectory_results.write_trajectories() 与展示地图
    reporting.artifacts.finalize_result() 发布视频、摘要与报告
    reporting.artifacts.cleanup_debug()  成功后按需清理中间产物
```

| 文件 | 职责 |
|---|---|
| `workflow.py` | 单段录像的阶段顺序、失败归因与复用判断 |
| `attempts.py` | 多轮“建图 → 定位 → 融合”、每轮质量评估与选择 |
| `fusion_runner.py` | 在独立工作进程执行 `FusionRequest`，经单向管道回传进度，并将求解器日志写入 `fuse.log` |
| [`estimation/fusion/solver.py`](../estimation/fusion/solver.py) | 尺度与重力对齐、选择初值、恢复全帧状态并检查位置支持 |
| [`estimation/fusion/poster_solve.py`](../estimation/fusion/poster_solve.py) | 海报曝光采样、联合图求解、按累计预算剔除观测并热启动重求解 |
| `environment_assets.py` | 环境镜头标定与海报输入的选择、冻结与加载（场景回退共用） |
| `trajectory_results.py` | 融合结果 → 相机/TCP 轨迹、逐帧证据与运行设置记录 |
| `results.py`、`map_data.py` | 结果读取、发布、报告与展示地图 |
| `tools/initialization_review.py` | 用保留的视觉/IMU 数据对照初始化的开发诊断 |

原片校验和设备选择发生在清理旧输出之前。每个镜头只裁切一次，所有重建尝试复用；可疑结果默认最多重建三轮。先检查每个有位置约束的分段相对自身初始化的尺度变化，双向倍率超过 `segment_engine.retry.maximum_scale_change_ratio`（默认 2）或证据无效时，该轮不能选中；全部轮次因此被拒绝则执行失败并保留现场。其余轮次优先选择独立地图放置更少、位置约束覆盖更高的结果。各轮拒绝原因保存在 `tries`，融合证据包含 `scale_consistency`。该保护不估计真实误差，也不保证视觉地图正确；2 倍上限仍需多录像验证。参数位于 `config/algorithms/reconstruction/common.yaml`。

独立初始化保留原尺度求解，并以起点偏移 `0`、`step//3`、`2*step//3` 的独立窗口复核每个足够长的连续段。尺度相对跨度、自由重力向量差沿用 `maximum_relative_scale_std` 和 `gravity_tolerance_m_s2`；原先合格的段若在复核窗口失效，也会被拒绝。不平均重叠窗口来降低不确定度，不选择恰好通过的窗口替代主解。短段无法提供至少两个窗口时标为 `insufficient_evidence`，仍执行原来的尺度/重力门限；全部窗口齐全且一致才标为 `stable`。检查只用于独立轨迹初始化，共享场景地图保留其原有验收流程。

初始化失败时，`debug/reconstruction/try_N/fuse/alignment_failure.json` 保存各段拒绝原因、门限、窗口估计和固定尺度下的速度衔接残差（`rms_m` 是位置等效残差，不是轨迹真值误差）。正常结果在 `fuse_info.json` 中保存 `sampling_stability`。结构化失败码仍为 `NO_USABLE_LENS`，不生成伪成功轨迹。若同一次任务的后续建图产生完全相同的融合输入，跳过重复融合并停止余下重试；外部程序崩溃、超时或输入改变仍按原策略重试。

实验性的重力方向联合细化由 `FusionRequest(refine_initialization=True)` 或库函数 `solve(..., refine_initialization=True)` 显式启用。普通独立轨迹入口和共享场景默认不启用。它从线性初值出发，在重力球面上优化两个方向自由度，同时优化正尺度、加速度计零偏和节点速度；视觉位姿、陀螺零偏、外参和时间偏移固定。位置残差使用米，速度残差乘以中位区间时长后使用米，Huber 转折点沿用 `huber * position_sigma_m`，属于实验权重而不是完整协方差白化。

候选必须收敛、不触及尺度/零偏边界、原始链验收与采样稳定性通过，而且两个未参与拟合的错位视觉端点窗口均有更低的速度衔接残差，才允许替换初值。多条有效链共用零偏时暂不替换；短链、不可观测链及超过 128 个训练节点的链也不细化。保留原始自由重力值和尺度绝对不确定度，不把强制 9.81 的结果当成通过重力检查的证据。原始初始化失败时仅记录候选，仍拒绝输出轨迹。`linear_stage1` 和 `initialization_refinement` 保存前后对照。复核窗口与拟合共用 IMU，不是独立精度真值。

已有 `--keep-work` 输出可直接生成初始化对照报告，无需重新建图或生成轨迹：

```bash
.deps/env/bin/python -m umi.reconstruction.segment.tools.initialization_review \
  --run-dir /path/to/trajectory_result \
  --calibration /path/to/calibration.json \
  --output refinement_review.json
```

此入口仅支持 handheld，默认读取第 1 轮，`--attempt N` 选择其他轮次；须使用本轮实际相机标定。报告同时保存视觉旋转与 IMU 的局部差异、两种初始化结论、输入哈希与 git 提交号。返回码 0 表示报告成功生成，报告内 `rejected` 仍表示初始化未通过。

## 依赖

`ReconstructionContext` 直接持有视觉阶段共用的路径、参数和 CPU 设置。独立命令和场景回退都将已解析的输入交给重建阶段。视觉程序、错误与路径由 `umi/perception` 管理。外参和逐帧位姿右乘由 `umi/core/geometry.py` 提供。TUM 轨迹读写位于 `umi/infra/file_store.py`。

## 融合模块

库入口为 `from umi.reconstruction.estimation.fusion.solver import solve`。重建通过 `umi/reconstruction/segment/fusion_runner.py` 的 `execute()` 调用：每次融合在一个 spawn 工作进程里执行 `fuse()`，保留双手并行、`--cores`、nice 和崩溃隔离（GTSAM 优化期间持有 GIL，线程内执行会让双手串行）。包初始化不转发 `solve`，也不导入数值求解器。

```text
umi/reconstruction/estimation/fusion/
├── data.py              FusionInputs、MapPlacements、FusionSolution 与文件读写
├── solver.py            顺序入口、地图段放置与全帧率恢复
├── alignment.py         运动段筛选、尺度与重力求解及初值细化
├── graph.py             视觉因子、图优化与 Jacobian 自检
└── poster.py            共享海报角点因子的独立图适配
```

求解器可直接导入，不解析命令行、不打开录像、不创建输出。输入和输出显式传递；尺度、跟踪段、偏置、GTSAM 图与优化结果都属于单次调用。文件边界独立，`write_solution()` 保留原轨迹、辅助 CSV 及 JSON 字段。`state.csv` 中历史兼容列仍保留，其零值不代表这些量已被估计。

算法内部使用视频秒。`FusionInputs.imu_video_clock` 已将 IMU 样本换到视频时钟，`FusionSolution.video_times_s` 与 `T_world_imu` 明确时间和位姿语义。`fusion.data` 在文件边界处理 EuRoC 人工 +1 秒，转换集中在 `umi/core/timing.py`；求解核心不处理文件时钟。

`RefinedChains` 汇集细分后通过独立复核的区间、估计与诊断。`RecoveredState` 明确区分每帧位姿、速度和 IMU 零偏；全帧恢复与段尺度统计分别实现。融合工作进程写出的文件路径使用 `umi.reconstruction.estimation.models.FusionArtifacts`，与算法内存中的 `FusionSolution` 分开。

两套融合目标不同：独立模式估计跟踪段的尺度与放置，场景模式以已冻结的米制地图位姿为约束。两者分别维护，数值方法见 [原理](../../../docs/principles/算法原理.md)。

自检：

```bash
.deps/env/bin/python -c 'from umi.reconstruction.estimation.fusion.graph import selftest; raise SystemExit(0 if selftest() else 1)'
```

## 运行记录与结果读取

每轮 `logs/run.json` 记录运行身份、引擎哈希及完成状态；`logs/tracking.json` 记录各镜头各视觉阶段统计。结果读取只接受当前结构化布局；原始文本日志继续保留用于审计。

项目构建的 `x3_mono_slam` 默认逐帧同步完成局部建图和闭环，并按稳定 ID 处理观测及共视权重平分，减少同一输入重复建图的差异。离线最小关键帧间隔默认 0.2 秒，按配置帧率向上取整为帧数，避免局部建图始终空闲时过密插帧。各阶段 `execution.json` 记录 `offline_sync`、`offline_keyframe_interval_s` 和实际最小帧数。底层 `--async_mapping`、`--offline_keyframe_interval` 用于调度对照。同步处理可能增加耗时；同一录像的可复现性不等于轨迹精度。源码升级后需要通过 `setup/trajectory/orbslam3/build_orbslam3.sh` 重编译，已安装的旧二进制不会仅因 Python 代码更新而改变。

ORB 金字塔各层的特征、方向和描述子并行计算，随后按原层级顺序合并；最多使用 4 个工作线程，`OMP_THREAD_LIMIT=1` 可用于串行对照。离线驱动关闭 OpenMP 动态缩减，避免运行库把请求的 4 个线程缩成 1 个。OpenMP 只编译到特征提取文件，BA 求解和关键帧策略保持原值。视觉启动器默认设置 `OMP_WAIT_POLICY=PASSIVE`，让特征线程在串行建图期间休眠。无原生窗口的纯单目模式在初始化完成后不再检测仅供初始化使用的标签，地图重置后恢复；原生窗口显示、IMU 单目路径和独立夹爪标签检测保持原行为。构建时自动检查串行/并行角点及描述子逐项相同，以及标签检测关闭与恢复。

`results.attempt_paths()` 只解析当前轮次目录。发布视频、摘要和报告成功后才能清理新运行的临时数据。

解释器使用 `UMI_PY`、`UMI_PY_TOOLS`；未显式设置基础解释器时，启动脚本使用当前依赖目录的 `env/bin/python`。历史序列化字段 `source: factory` 表示样例测量，不是厂家标定。设备资产只接受当前校验格式。

## 输出

标定 C 验证和开发测试使用内部逐段引擎。默认只输出相机/IMU 轨迹；需要 TCP 时显式提供 `--tcp hardware/kits/shared/tcp_camera.json`，可用 `--mask <PNG>` 覆盖硬件通用遮手图 `hardware/grippers/masks/front_hand_mask.png`。相机验收不依赖套件或夹爪。单段输出布局如下：

```text
umi_traj_out/VID_20261003_182246_00_067/
├── traj_tcp.txt           提供 --tcp 时生成的夹爪尖轨迹
├── front.mp4              工作镜头视频，viewer 直接读取
├── trajectory.png         轨迹预览（绘图成功时生成）
├── report.txt             状态、设备、标定、运动统计和提醒
├── details/
│   ├── traj_camera.txt    相机轨迹，供标定验证和排查
│   ├── traj_imu.txt       IMU 轨迹
│   ├── frames.csv         逐帧对应关系和约束来源
│   ├── run_settings.json  完整配置及资产哈希
│   ├── summary.json       完整结果指标和复用判断信息
│   └── map.json           工作镜头轻量地图展示数据（导出成功时）
└── debug/
    ├── logs/              建图、定位和融合日志，默认保留
    ├── dataset/           提取的时间戳与 IMU（按需保留）
    ├── lens_videos/       本段录像共用的镜头视频（按需保留）
    └── reconstruction/    各轮重建中间数据（按需保留）
```

内部逐段结果先看 `report.txt`、`trajectory.png`，相机轨迹位于 `details/traj_camera.txt`，提供 TCP 几何时还可使用 `traj_tcp.txt`，配合 `front.mp4` 回看画面。视频是工作镜头裁切，默认 720×720、约 29.97 fps，保留鱼眼畸变；实际裁切、尺寸、取帧和编码参数见本次运行记录。`details/` 提供完整结果与配置，`debug/` 用于排查。

新运行成功后默认清理数据集与重建中间文件，根视频、轨迹、辅助结果和日志继续保留。`--keep-work` 保留完整中间数据；失败或中断时也保留。无独立真值的任务录像只报告跟踪和运动统计，不把地图约束比例当作真实误差。

`details/map.json` 保存选中轮次的工作镜头稀疏点云、关键帧和展示对齐信息。它独立于被清理的 `.osa`，可直接由 `umi-view` 查看；每图最多展示 50000 点。对齐只用于展示，不改动轨迹或地图，拟合残差不代表真实精度。导出失败只增加提醒，不改变轨迹抽取的成功状态，详细原因见 `debug/logs/map_export_error.log`。

`details/` 保存发布结果，`debug/` 保存运行日志与中间数据。

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
| `fuse/poster_failure.json` | 独立海报优化失败条件、深度统计、像素 RMS、初末目标函数 |
| `runner_tryN.log` | 各轮阶段摘要、失败阶段与原始日志位置 |

### debug/ 中间数据

新运行仅在 `--keep-work`、失败或中断时保留。

| 路径 | 内容 |
|---|---|
| `debug/dataset/mav0/` | 每帧时间与 IMU |
| `debug/lens_videos/front.mp4` | 从原片生成一次的工作镜头视频；dual 模式另有 rear.mp4 |
| `debug/reconstruction/try_N/front/front.mp4` | 该轮引用的共享工作镜头视频，各轮不会重新裁切原片 |
| `debug/reconstruction/try_N/front/map/map.osa` | 保存的地图 |
| `debug/reconstruction/try_N/front/map/frames.csv` | 建图逐帧状态 |
| `debug/reconstruction/try_N/front/loc_m*/frames.csv` | 对各地图逐帧重定位的结果 |
| `debug/reconstruction/try_N/fuse/state.csv` | 融合内部状态，包括速度与 bias |

根目录 `front.mp4` 独立保留，不依赖这些中间目录。双镜头模式的环境镜头视频属于中间数据，可用 `--keep-work` 保留。

### 参数与复用记录

`details/run_settings.json` 的 `parameters` 保存完整有效配置、覆盖来源与 git 提交号；TXT 显示配置 ID 和覆盖项。重建参数与提醒规则分别计算指纹，详见 [参数管理](../../../config/README.md)。融合使用项目实现，标定包提供测量。双镜头模式的环境镜头使用本设备的登记参数或从 INSV 工厂参数推导。

场景任务的结果布局见[结果指南](../../../docs/04-结果查看.md)。

## 内参来源与引擎

独立工作镜头引擎可用 `--intrinsics-source measured|factory` 选择设备中对应的完整标定版本，已选目录（`UMI_TRAJ_CALIB`）也必须匹配来源。未指定时继续使用已启用版本。factory 是工厂 MEI 拟合 KB4 加重新实测的 B；历史样例选择值 `UMI_TRAJ_CALIB=factory` 含义保持独立。

双镜头参数来源选择可用 `umi-scene` 套件，或独立入口的 `--mode dual --rear-calibration measured`；不要求提供海报。未绑定正式 rear 时，尝试当前 X3 INSV 工厂内参/双镜头外参 + 本设备 front B 兜底；沿用 front 时差，标记 `NOT_EVALUATED`。`--intrinsics-source` 选择 front 参数来源并核对显式正式 rear 的来源一致；兜底的 rear 来源独立记录为 factory。项目构建默认启用已验证的 ≤85° 前向 KB4，两种来源使用同一几何与 0.2 秒选帧策略；1920 全镜头与 >180° 仍不属于正式范围。

```bash
python -m umi.apps.segment /path/to/task.insv --mode dual --rear-calibration rear_ACTUAL_HASH
```

`environment_assets.py` 集中选择 rear 与可选海报布局，用不可变 `EnvironmentAssets` 传递本次选择；指纹计算只消费已选资产，不再次查询登记表。rear 身份与文件路径分别建模，正式文件冻结在结果 `devices/rear/`，成功清理 debug 后仍保留。光学参数供 rear SLAM 使用；外参和时差传入原生融合，不论是否启用海报都换算到 front 视频时钟，禁止跨丢帧或地图切换插值。版本、来源和内容进入计算身份；复用结果时再次校验冻结资产。`handheld` 下仅在指定海报时接受 rear，避免选择了标定却没有使用。


## 陀螺仪辅助建图

独立重建默认使用 `segment_engine.orb.gyro_aided_mapping`。首遍建图在真实前镜头曝光时间之间积分陀螺仪，依据标定外参得到相机坐标系下的旋转增量。ORB 从最近可信位姿预测旋转，用视觉匹配求位姿，并检查约 0.3 秒内的相对姿态。验收失败时先尝试投影恢复，再使用现有重定位。恢复期间先确认连续帧，再插入关键帧。缺少兼容驱动或 IMU 连续覆盖时，相应录像或帧段走视觉跟踪。准备状态见 `debug/logs/mapping_gyro.json`，预测、拒绝、恢复及延后插入关键帧的次数见首遍的 `map/execution.json`。

## 环境海报

独立 `handheld` / `dual` 支持 `--tag-layout FILE`，可加 `--rear-calibration measured|factory` 选择本设备已登记的完整 rear 标定，要求序列号、参数来源和验收匹配。省略时按上述规则兜底，显式绑定损坏则拒绝。两种路径均冻结 rear 参数和来源到 `devices/rear/`；工厂转换不合格或元数据缺失时在预检失败，保留旧结果。指定资产默认只诊断；通过 YAML `standalone.poster_fusion: true` 显式启用。共享默认阈值在 `reconstruction/common.yaml` 的 `shared.poster` 下，单次覆盖通过 `--config` 传入相同结构。

`environment_assets.py` 选择环境镜头标定，并冻结标定、海报布局和原始角点；`umi/reconstruction/estimation/fusion/poster.py` 在原生图中添加单板角点因子。原有地图分段尺度仍是变量，不把上一轮融合轨迹作为固定地图先验再平滑一次。真实 rear 观测时刻插入 IMU 节点；恢复全帧轨迹时使用这些实际时间节点。dual 地图定位也按镜头时差重采样，不跨丢帧/地图切换。没有海报观测时保持原图输出；仍需通过惯性初始化和原有质量验收。

启用海报融合的 `handheld` 模式在 Stage 1 无合格 ORB 初始化段时，可用已知尺寸的多标签海报和 IMU 初始化米制位姿、重力及零偏。海报种子需通过实际时刻覆盖、重投影和几何可观测性检查。各前镜头子图使用重叠时刻拟合尺度和放置，符合门限的跟踪段进入联合图。后续用原始角点、IMU 和合格地图约束求解；最终还需通过几何有效的位置支持缺口与地图尺度保护。若没有任何地图尺度变量，质量判断使用合格的海报位置支持。成功时 `fuse_info.json` 的 `init_lens` 为 `poster`，`poster_initialization` 保存种子证据；种子被拒绝时在 `alignment_failure.json` 中记录相同字段和拒绝原因。

海报数量/时间剔除规则位于 `umi/perception/poster/quality.py`；`umi/reconstruction/estimation/poster_factors.py` 从原场景实现提取旋转 seat、鲁棒角点因子及解析 Jacobian，场景与独立图共同使用。rear 修正可严格关闭，开启时检查修正角上限；独立海报图在初始化后可剔除一次，优化后最多两次，第二次的异常数须少于第一次。各轮共用原始观测的数量与时间覆盖预算，优化后重求解复用已有对齐、地图放置和轨迹初值。正常路径的初值来自已求解基线，海报兜底路径的初值来自 PnP 与 IMU。两条路径显式选用参考地图段或首帧位姿作为世界系锚定。

`poster_independent` 用真实观测时刻检查海报位置支持，将发布门槛应用于每轮，再对合格候选排序。普通独立引擎和单手局部回退沿用各自的重试覆盖规则。尺度硬保护仍检查每个参与优化的地图尺度。

专家配置 `backends.poster_independent.poster_guidance: true` 将海报与 IMU 初始化提前到单目建图前。建图完成后，用最终图集中各子图的关键帧单独拟合米制尺度和坐标变换，并在留出关键帧上验证。第二遍定位仅在海报位置观测覆盖且该子图对齐通过时尝试位姿引导。ORB 用当前前镜头图像投影匹配并优化，投影验证失败时走原重定位。全段海报覆盖通过后，联合图可用该轨迹作为初值，海报因子继续使用原始角点。初始化证据保存在 `debug/logs/poster_initialization.json`，子图留出验证和引导次数见对应尝试的 `logs/poster_localization.json` 和 `logs/tracking.json`。

在项目锁定的 Linux x86_64 GTSAM 环境，`install.sh` 会编译独立融合使用的 KB4 角点和深度因子。角点按同一时刻每组最多八个标签共用 Huber 权重，深度因子按该时刻全部训练角点计算。没有原生扩展时使用 Python 实现。

输出 `environment_poster/` 保存布局、rear 标定、角点和哈希清单，默认清理 debug 后仍保留。结果的 `fusion.environment_poster` 记录初始/最终像素残差、板位姿和状态，`actual_error` 仍为 `not_measured`。单手共享失败回退自动继承场景中的海报参数及冻结资产，结果仍使用独立局部坐标。

无预建地图的任务海报联合图通过场景任务的 `poster_vio` 后端接入，见 [poster_vio 模块指南](../backends/poster_vio/README.md)。

固定地图的尺度对齐、定位合并与视觉惯性融合位于 `umi/reconstruction/estimation/fixed_map`。单独调用时显式传入观测、标定和数值参数。地图文件的读写与业务状态由场景流水线处理。
