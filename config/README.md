# 算法参数

标定和单段轨迹抽取共用 [software/config.py](../software/config.py) 加载配置。场景流程另由 `software/scene/management/inputs.py` 加载场景参数，并复用轨迹配置。数值默认值集中在下面的 YAML 中。

| 内容 | 管理位置 | 生效范围 |
|---|---|---|
| [calibration.yaml](calibration.yaml) | 标定算法、板检测、D 静止窗口、验收规则 | `umi-calib run / verify / recheck` |
| [trajectory.yaml](trajectory.yaml) | 视频尺寸与编码、ORB、重建重试、IMU 融合、ZUPT、质量提醒 | `umi-scene`、内部逐段轨迹引擎及标定 C 段验证 |
| [scene_parameters.yaml](scene_parameters.yaml) | 场景尺度/放置验收、固定坐标融合、原片配对、宽度处理与录制检查规则 | `umi-scene`；操作和覆盖方式见 [场景指南](../docs/场景处理.md) |
| 相机实测参数、时间差、TCP、遮手图 | `registry/` 中的相机与夹爪资产 | 按录像内部序列号选择，继续手动 `install` |
| 输入/输出路径、CPU、并行数、保留中间文件 | 命令行请求 | 只影响执行方式，不写入算法配置 |
| 原片规格、标定区域、像素变换与派生帧率规则 | `software/camera/x3/frame.py` | 相机图像规格；在结果中记录 |

`config/scene_parameters.yaml` 保存算法默认参数；`scenes/<场景>/scene.yaml` 保存具体场景的地图模式和算法覆盖；建图原片与设备快照保存在各批次 manifest.yaml。场景定义中的 `trajectory` 和 `parameters` 直接保存该场景的参数改动。

## 怎么使用

先只读检查，不创建输出或运行求解器：

```bash
./umi-calib run recordings/calibration/IAQEB26097M4CP/20261022/front --dry-run
./umi-scene run scenes/shelf/20261105_001 --dry-run
```

调整默认配置会影响以后使用默认配置的运行。针对某个场景做实验时，直接在它的 `scene.yaml` 中写改动项：

```yaml
trajectory:
  video:
    size_px: 960
  fusion:
    node_stride: 3
parameters:
  minimum_localization_fraction: 0.6
```

`trajectory` 覆盖本页的 `trajectory.yaml`，`parameters` 覆盖 `scene_parameters.yaml`；不需要额外的场景参数文件。场景配置版本默认 3，可省略 schema_version；历史 schema 2 的文件引用仍可读取，迁移时将原覆盖文件内容放入这两个映射。场景配置不再保存设备和建图录像，分别从 kit.yaml 和批次 mapping/ 读取。

优先级为项目默认 YAML → 场景内联覆盖 → 显式 CLI 覆盖。标定的实验覆盖仍使用 `umi-calib run --config <文件>`，内部独立轨迹引擎仍支持自己的 `--config`；这些不是新的场景配置来源。`--jobs`、`--cores`、`--force` 只影响执行方式。

## 参数按流程组织

| 阶段 | 配置表 | 主要内容 |
|---|---|---|
| A/C：覆盖评估与板参考 | `calibration.board` | 检测采样步长、放大倍率、PnP 标签数、归一化重投影阈值、角点视线限制、精化次数、比较帧数、覆盖指标 |
| B：相机–IMU 求解 | `calibration.imu` | 加速度/角速度噪声及 bias 随机游走先验 |
| D：多朝向静止检查 | `calibration.gravity` | 窗口时长、样本数、时间跨度、静止阈值、重力值、朝向聚类 |
| C：轨迹验证 | `calibration.trajectory_validation` | 轨迹覆盖配置文件；留空使用默认 `trajectory.yaml` |
| 标定验收 | `calibration.checks` | A/B/C/D 的质量门限和 C 证据完整性门槛，含和样例相机比较的提醒 |
| 镜头视频 | `trajectory.video` | 裁切范围、取帧步长、输出尺寸、缩放、编码器/像素格式、CRF/preset 和音轨 |
| 建图、逐图定位 | `trajectory.orb` | ORB 特征数量、金字塔、FAST、远点阈值、初始化标签参数 |
| 重建与选轮 | `trajectory.retry` | 尝试总次数、地图位置覆盖要求、独立放置数上限、尺度变化倍率上限 |
| IMU 融合 | `trajectory.fusion` | 对齐步长、图节点间隔、链拆分/分组/细分、噪声、连续性、图先验、Huber、求解迭代、静态初始 bias |
| 零速度约束 | `trajectory.zupt` | 开关、窗口、静止阈值与速度先验 |
| 轨迹质量解释 | `trajectory.notes` | 覆盖率、起始 IMU 桥接时长和建图跟踪率提醒 |

标定 A/B 的图像准备固定每 2 个原始帧取 1 帧；`calibration.board` 的步长在准备后的图像序列上采样。`trajectory.video` 控制轨迹引擎的视频准备，不改变 A/B 的标定图像。

`calibration.board` 只配置板检测和评估算法。板布局由 [`hardware/calibration_targets/aprilgrid.json`](../hardware/calibration_targets/aprilgrid.json) 定义，打印文件见 [标定板说明](../hardware/calibration_targets/README.md)；本次实测标签边长通过 `--tag-size-mm` 输入，无需修改板配置。

表中的 `calibration.*`、`trajectory.*` 表示所属文件；文件内直接写 `board:`、`fusion:` 等映射，参数使用两个空格缩进。字段名尽量带单位：`_m`、`_mm`、`_px`、`_s`、`_deg`、`_rad_s`。IMU 噪声使用现有 Basalt/GTSAM 连续时间模型的单位，四个 noise/RW 值不能为零。

C 段可以独立指定轨迹实验配置。在标定覆盖文件中写：

```yaml
trajectory_validation:
  config: "trajectory-tuning.yaml"
```

相对路径以这份标定覆盖文件所在目录为基准。标定开始前会同时校验两份配置；C 子进程读取完整解析结果，最终标定报告记录两套配置。

标定的 `run`、`verify` 和 `calibrate-rear` 会将本次实测板边长 `--tag-size-mm / 1000` 传给 C 段的 `orb.initial_tag_size_m`，优先于轨迹默认值和覆盖文件；候选与样例比较使用同一个尺寸。该派生覆盖会写入轨迹配置快照、内容指纹和子进程配置，`recheck` 从原运行的板尺寸恢复同一配置。切换 A4/A3 不用再手动修改 C 的初始化标签尺寸；日常场景建图和独立轨迹仍使用 `trajectory.yaml` 的 `orb.initial_tag_size_m`。

## 场景和单段轨迹的参数边界

场景复用轨迹算法中的部分计算步骤，但场景地图的验收门限和固定坐标融合模型由 `scene_parameters.yaml` 管理。下面按实际生效位置列出，不能只根据相似字段名选择要修改的文件。

| 计算位置 | 生效配置 |
|---|---|
| 场景建图：尺度相对标准差、重力一致性、加速度零偏门限 | `scene.maximum_scale_relative_std`、`scene.gravity_tolerance_m_s2`、`scene.maximum_acc_bias_m_s2`；通过统一的场景对齐参数转换覆盖轨迹中的对应字段 |
| 场景建图：对齐步长、转动断链、坏段细分、最少区间对、线性求解 | 复用 `trajectory.fusion.step / split_rotation_deg / refine_chain_pairs / minimum_alignment_pairs / minimum_chain_pairs / linear_iterations / linear_tolerance` |
| 场景任务：前后镜头视觉位置/旋转噪声、Huber | `scene.front_position_sigma_m / rear_position_sigma_m / visual_rotation_sigma_deg / huber`；不使用轨迹融合中的对应视觉参数 |
| 场景任务：IMU 噪声、bias 随机游走、初始 bias、速度/bias 先验、LM 迭代和 ZUPT | 复用 `trajectory.fusion` 中对应字段及 `trajectory.zupt` |
| 内部单段轨迹及前镜头标定 C | 使用 `trajectory.fusion` 自己的链/分段、尺度/放置和视觉噪声参数；不读取场景门限 |

这里的 `scene.*` 表示 `scene_parameters.yaml`，不是文件中的额外嵌套层。场景建图的内容指纹记录场景门限及实际复用的轨迹参数，单独修改被场景覆盖的轨迹门限不要求重新建图。

标定 B 的 `calibration.imu` 与轨迹/场景融合的 `trajectory.fusion` 噪声先验分别传给 Basalt 和 GTSAM，允许独立调节；当前相同的默认值不表示二者会自动同步。D 静止诊断与融合的重力参考也分别生效。`retry.minimum_full_fraction` 控制重试，`notes.minimum_full_fraction` 只控制提醒；D 静止筛选与 ZUPT 同样各有用途。

## 固定精度和图像规格

以下三项从标定配置中移除，保留原有计算数值：

- `board.time_match_tolerance_ns`：同一视频时间的匹配容差固定为 1000 ns，仅吸收文本精度差，不插值，定义于 `algorithms/trajectory_metrics.py`。
- `board.minimum_trajectory_spread_m`：轨迹去中心 RMS 的数值退化保护固定为 `1e-6 m`，同样定义于 `trajectory_metrics.py`。实际移动范围的验收仍通过 `checks.minimum_board_extent_m` 配置。
- `board.coverage_radius_px`：覆盖归一化半径从相机原始镜头半幅尺寸推导，当前为 `1920 / 2 = 960 px`，不随 1440 裁切或轨迹输出尺寸变化；实际分母写入覆盖指标。

旧覆盖文件若包含上述字段，需删除后再运行，否则按未知字段报错。已有结果不会自动改写；旧配置快照与当前配置不同时，`recheck` 会拒绝，可用 `run --resume` 或 `verify --resume` 更新受影响的计算后再验收。固定值和推导规则由代码内容指纹追踪。

## 夹爪宽度处理参数

config/scene_parameters.yaml 管理宽度的处理策略，可在 scene.yaml 的 parameters 中直接覆盖。测量 JSON 只保存 anchors、verified 等实测信息。

| 参数 | 默认值 | 作用 |
|---|---|---|
| gripper_width_max_gap_s | 0.2 | 宽度缺口两端最大间隔，秒；0 表示不填补 |
| gripper_width_closed_threshold_mm | 3.0 | 闭合判定阈值，毫米；非负且小于该手标定最大开口 |
| gripper_width_diagnostics | false | 输出原片逐帧诊断 CSV |

这些参数参与任务结果身份，不参与建图身份。原片检测缓存仅由原片内容和检测器决定；测量、阈值和输出时间轴变化会复用检测。宽度报告统一保存在 episode.json，格式见[训练数据](../docs/训练数据.md#夹爪宽度标定与接入)。

## 重试与配置校验

`trajectory.retry.maximum_scale_change_ratio` 控制重建的尺度发散保护：每个有位置约束的分段比较融合后尺度与自身初始化，使用 `max(后/前, 前/后)`，不同地图之间不直接比较。默认上限为 2 倍，超限轮次重试且不得选中，全部轮次不合格则失败。它不修改融合解，也不放宽 `calibration.checks`；当前上限是粗粒度保护，尚需多录像检验。C 的运行记录保存每轮视觉、对齐和尺度诊断，验收前保留子进程工作数据。

YAML 使用 1.2 规则，开关写 `true` / `false`，空覆盖写 `{}`；重复字段会报错。未知字段、类型错误、NaN/Inf、无效枚举、非正噪声/计数、越界比例、倒置的 WARN/FAIL 阈值都会在清理旧输出和执行算法之前报错。配置结构版本为 `schema_version: 1`。

## 视频处理参数

`config/trajectory.yaml` 的 `video:` 统一配置 INSV → 镜头视频的处理。默认仍为：**3840×1920 原片 → 前/后镜头的已标定 1440×1440 区域 → 720×720 H.264 视频，每两帧取一帧**。在 scene.yaml 的 trajectory.video 中覆盖时也使用相同字段。

```yaml
video:
  crop:
    size_px: 1440
    offset_x_px: 0
    offset_y_px: 0
  frame_step: 2
  size_px: 720
  resize_filter: area
  codec: libx264
  pixel_format: yuv420p
  crf: 10
  preset: medium
  keep_audio: false
  audio_bitrate_kbps: 128
```

| 参数 | 作用与约束 |
|---|---|
| `crop.size_px` | 已标定区域内的方形裁切边长，2–1440 的偶数 |
| `crop.offset_x_px` / `offset_y_px` | 相对居中裁切的偏移，可为负；正数向右/下，不能越出标定区域 |
| `frame_step` | 正整数，每 N 个原片帧取一个；视频帧率为 `(60000/1001)/N` |
| `size_px` | 输出正方形边长，偶数，不能大于裁切边长 |
| `resize_filter` | `area`、`bilinear`、`bicubic`、`lanczos` 或 `neighbor` |
| `codec` | `libx264`（H.264）或 `libx265`（HEVC），统一写 MP4 |
| `pixel_format` | `yuv420p` 或 `yuv444p`；默认 `yuv420p` 便于常规播放 |
| `crf` | 压缩画质，0–51，越小压缩损失越少 |
| `preset` | `ultrafast`、`superfast`、`veryfast`、`faster`、`fast`、`medium`、`slow`、`slower` 或 `veryslow` |
| `keep_audio` | 是否保留原片已有音轨；没有音轨时不会伪造 |
| `audio_bitrate_kbps` | 保留音轨时的 AAC 码率，正整数，单位 kbps |

裁切和缩放会同步变换 SLAM 内参与遮手图：`f'=s·f`，`c'=s·(c−裁切起点+0.5)−0.5`。KB4 畸变系数、相机–IMU 外参、相机–TCP 几何保持原值。遮手图只生成本次运行的处理副本，注册表资产不改写。场景双路模式使用各设备的实测后镜头标定；内部独立 dual 验证仍明确使用样例后镜头。

取帧步长同步用于时间戳提取、MP4 编码、SLAM 帧率、独立融合时间网格、结果中的原片帧号及训练对齐。`fusion.step`、`node_stride` 等仍按处理后的帧数计数，改变取帧密度后需结合录像长度核验求解质量。训练导出的 `--fps` 只设置训练样本时间轴，视频维持这里生成的帧率。

ORB 引擎要求 `Camera.fps` 为整数，因此自动使用四舍五入且至少为 1 的标称帧率（默认 30）；MP4 编码、时间戳和融合仍使用精确帧率（默认 `30000/1001`），无需额外配置。

原片尺寸、前后镜头的位置和鱼眼模型属于已登记标定的物理约定，不能靠视频配置改成另一种相机或录制模式。这里保留鱼眼画面，不提供未经标定验证的任意去畸变切换。MP4 是当前结果和训练包的文件协议。HEVC 或 yuv444p 需要对应播放器和训练解码器支持。

完整视频参数会写入镜头视频清单及处理配置快照；`frame_geometry` 记录实际裁切和帧率。参数变化使旧计算缓存失效，重新运行后仍需人工检查与验收。默认值之外的画面尺寸、视野或帧密度需用实际录像验证轨迹质量。

## 结果里记录什么

- 标定：`run/record.json` 的 `computation` 保存计算配置，`evaluation` 保存验收配置。
- 场景任务：统一保存在任务 `episode.json`；`inputs.trajectory_parameters.video` 保存视频参数，`arm_results.<手>.frame_geometry` 保存实际画面几何，`inputs` 保存处理身份与参数快照。旧结果继续支持各手 `details/run_settings.json`。
- 内部逐段轨迹：`details/run_settings.json` 的 `parameters`、`frame_geometry`，以及摘要中的运行指纹。
- TXT 报告：算法配置 ID、验收/提醒 ID、相对默认值的覆盖项和完整记录路径。

记录包含完整有效值、默认/覆盖文件来源、显式 CLI 覆盖、配置版本、实际相机和夹爪资产、算法源文件内容哈希、引擎/依赖身份。未提交的代码修改也会进入代码哈希。轨迹使用的 ORB YAML 和融合 JSON 在运行时生成；完整中间输入用 `--keep-work` 保留。

`configuration.computation_hash` 仅表示算法配置；外层 `computation_hash` 还包括输入、测量资产、代码及引擎。`evaluation_hash` 仅表示验收或提醒规则。配置注释、键顺序、覆盖文件的位置不改变配置数值指纹。

## 改参数后如何处理旧结果

**轨迹**：改变重建数值、重试条件、输入、设备资产、算法代码或引擎会重新计算；改变 `notes` 只重新评估已有指标、更新配置记录和报告，轨迹文件保持原样。证据缺失会重算。旧版本摘要仍可供查看器读取，但缺少新运行指纹时首次抽取会重新计算。

**标定**：`run` 仍按原规则保护已有输出，需要重算时显式 `--force`。只有验收规则变化时，可以单独重新验收：

```bash
./umi-calib recheck recordings/calibration/IAQEB26097M4CP/20261022/front/calibration_result --config /tmp/umi-calib-checks.yaml
```

这里的覆盖文件需自行准备。`recheck` 校验算法设置与原运行一致，仅修改验收项、结论和报告，保存完整先前验收及配置历史。若原运行用了算法覆盖，重验时需包含相同的算法覆盖。不支持的记录格式、缺少完整原始指标或未完成的执行不能重验；拒绝时保留原报告，需先完成计算。验收使用原始指标，`recheck` 不启用设备标定。

## 标定包与引擎边界

新标定结果（v3）顶层使用 `calibration.json` 和 `manifest.json`。相机 JSON 是唯一最终测量，清单记录相机身份、镜头、板配置、来源和参数哈希。运行证据位于 `run/record.json`。历史 v2 包的 `calibration_r1440.json` 和 `bundle.json` 继续支持。SLAM 参数和融合脚本不再复制到每台设备中。旧 v1 包可以只读加载：核对原始 JSON、元数据、外参和注册表哈希，忽略其中的算法实现，运行项目中的共享算法；不会改写已有设备档案。

内部逐段引擎的 dual 模式仍使用后镜头样例几何，样例身份记录在后镜头资产中。场景 `umi-scene` 的 `dual` 则要求各设备的实测后镜头标定，可通过 `umi-calib calibrate-rear` 求解并用 `install-rear` 登记；不能把前镜头验收解释为双镜头都已标定。

Basalt/ORB/OpenCV/GTSAM 内部尚未暴露的参数，通过当前引擎内容/依赖身份管理。浮点稳定性保护、矩阵维度、Jacobian 自检容差、格式/数学约定、PNG 无损压缩和执行超时策略保留在代码中。录像契约还包含 0.05 fps 的模式识别容差、1 ms 规则帧网格判据、1.5 倍中位帧间隔的丢帧判据，以及 50 ms 的 IMU 覆盖容差；这些集中在 `camera/x3/frame.py`，写入提取清单和轨迹设置，修改会改变代码指纹。当前 ORB 包装器使用纯单目模式，IMU 噪声由融合配置控制；已移除容易造成误解的 ORB YAML IMU 噪声默认项。


## 运行解释器

项目入口优先使用 `UMI_PY`；视频工具可由 `UMI_PY_TOOLS` 单独指定。旧安装生成的 `KIWI_PY`、`KIWI_PY_TOOLS` 在 `software/environment.py` 中兼容，同用途的新变量优先。已有 `.deps/env.sh` 无需为此重新安装。新安装生成 `UMI_PY`。这些变量选择解释器，不覆盖算法参数。

标定运行记录使用 schema 3，和 YAML 的 `schema_version: 1`、资产清单 schema 3 分别管理。`checks.minimum_board_extent_m` 不足使验收为 INCOMPLETE。D 的 `checks.gravity_residual_m_s2` 只影响 WARN，工具始终以 REPORT_ONLY 测量，因此只改该阈值可直接 recheck。
