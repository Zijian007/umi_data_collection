# 命令行、配置与测量资产

本页按设备、批次、用户选项和算法参数说明配置位置与覆盖顺序。[从设备到结果](../docs/01-安装与日常使用.md#从设备到结果)

## 配置心智模型

配置按“谁决定、决定什么”分成三层，全部随 Git 提交，所有机器一致；没有按机器区分的覆盖文件：

| 层级 | 权威入口 | 负责内容 | 典型修改者 |
|---|---|---|---|
| 采集声明 | 批次 `collection.yaml`、套件 `hardware/kits/<kit>/kit.yaml` | 场景、任务、使用手、后端、地图、音频同步、环境标签布局、设备和实测资产 | 数据采集者 |
| 项目默认值 | `collection_defaults.yaml`、`algorithms/*.yaml`、`algorithms/audio_sync/*.yaml` | 批次默认处理策略与算法参数 | 算法维护者 |
| 执行与诊断 | `operations/runtime.yaml`、`operations/viewer.yaml`、`operations/gripper_tag_detection.yaml` | 线程、超时、查看器监听与缓存、夹爪标签 ROI 和检测预算 | 工程维护者 |

批次/套件中的声明是具体数据的权威来源；`config/` 中的 YAML 是项目默认值；`--expert-config` 只覆盖本次算法实验；单次运行的 CPU、并发、端口用命令行参数。运行结果会冻结本次实际配置和来源，不能通过修改默认 YAML 追溯或改写历史结果。

## 用户日常配置

日常采集只维护批次 `collection.yaml`，设备或夹爪变化时维护 `kit.yaml`。批次省略的处理策略取自 [`collection_defaults.yaml`](collection_defaults.yaml)，它与 `collection.yaml` 结构相同：

```yaml
sync:
  mode: audio_auto
  algorithm: kiwi
map:
  layout: auto
  reference_arm: left
backend:
  mode: auto
  auto_order: [scene_map, poster_vio, poster_independent]
  poster_vio:
    base_tag: true
```

生效顺序为 **`collection_defaults.yaml` → 批次 `collection.yaml` → 任务 `tasks.<take>`**，后者优先。个别批次需要不同的后端、地图布局或同步方式时，直接写在该批次的 `collection.yaml`。修改 `collection_defaults.yaml` 会影响所有继承该项默认值的批次。Organizer 生成的 `collection.yaml` 记录采集信息，所需的默认环境海报布局由 UMI 解析。

内参来源只由 `kit.yaml` 的 `intrinsics.front/rear` 逐镜头选择。标定时用 `umi-calib run <目录> --intrinsics factory` 选择工厂内参流程，省略时为实测内参。

需要调整鲁棒惯性初始化或夹爪标签缺口补检时，在批次 `collection.yaml` 填写对应开关。两个开关接受 YAML 布尔值，按项目默认、批次、单次专家覆盖的顺序生效。用 `./umi-scene run <批次或任务> --show-config` 查看各项的值和来源。加 `--json` 可查看 `effective.processing_options`。

| 处理选项 | 项目默认与专家覆盖（`reconstruction.yaml` 结构） | 批次 `collection.yaml` |
|---|---|---|
| VIO 鲁棒惯性初始化 | `backends.poster_vio.initialization.robust_inertial_seed` | `backend.poster_vio.robust_inertial_seed` |
| 夹爪标签缺口补检 | `measurements.gripper_tags.verified_gap_recovery` | `measurements.gripper_tags.verified_gap_recovery` |

日常运行先选输入目录；算法调参写 YAML；设备和实测几何写入各自的资产声明。三个程序的分工如下：

| 入口 | 命令行输入 | YAML / 资产 |
|---|---|---|
| `umi-scene run` | 路径、`--dry-run`、`--force`、`--status`、`--show-config`、`--json`。单次 CPU 和并发选项见 `umi-scene run --help` | `algorithms/reconstruction.yaml` 提供重建默认值；批次日常配置统一写 `collection.yaml`（场景、kit、手和任务）；专家调参使用 `--expert-config`；`kit.yaml` 选择每镜头当前参数来源与测量文件 |
| `umi-calib run / verify` | 录像、输出路径、`--tag-size-mm`、`--intrinsics`（仅 run）、`--dry-run`、`--recompute`。CPU 和线程选项见 `umi-calib run --help` | `algorithms/calibration.yaml` 保存求解与验收参数；B 的 IMU 噪声默认值和 C 的重建参数来自 `algorithms/reconstruction.yaml`；板布局和版本化测量是独立资产 |
| `umi-view` | 结果根目录、`--host`、`--port` | `operations/viewer.yaml` 管理服务、预览和缓存；读取已有结果，不重新计算轨迹 |

`--expert-config` 仅用于本次专家算法实验；内参来源读取 `kit.yaml`，地图阶段读取 `collection.yaml.tasks` 和 `.pipeline/map_epochs.yaml`。逐个子命令的完整选项、默认值与互斥关系见[命令行参数参考](../docs/命令行参数.md)，也可用 `<入口> <子命令> --help` 查阅。

## 先查看本次配置

```bash
./umi-scene run scenes/test_shelf/20261120_002 --show-config
./umi-scene run scenes/test_shelf/20261120_002 --show-config --json
./umi-calib run /path/to/front --show-config
./umi-view scenes/test_shelf --port 9092 --show-config
```

`umi-scene` 的 `--show-config` 默认输出给人看的摘要：后端、任务覆盖、地图、音频同步、基座标签、处理开关、海报布局、专家覆盖文件、与项目默认不同的算法参数，以及执行资源；取自 `collection_defaults.yaml` 的项标注「项目默认」。加 `--json` 输出完整 JSON。`effective.reconstruction` 使用重建参数树，`effective.changed_from_defaults` 列出不同于项目默认的字段。标定和查看器的 `--show-config` 直接输出 JSON。

`--show-config` 展示后即退出；不读取视频内容、不启动求解环境、不扫描查看器结果或监听端口。场景与标定入口可以同时使用 `--expert-config`。场景的 `--show-config`、`--status` 和 `--dry-run` 互斥。需要检查录像与执行计划时，单独运行 `--dry-run`。标定 run / verify 仍需填写目录/录像位置参数，但展示时不访问它们。run 的输入目录末级名称必须为 front / rear，由目录名选择镜头；两者都使用 run，详见[标定目录约定](../docs/02-相机与夹爪标定.md#52-算标定)。

## 运行与查看器默认值

以下配置由 `umi/config/operational_settings.py` 读取。项目默认值只在 YAML 中定义；单次运行用命令行参数或环境变量调整，不需要改 YAML。修改 YAML 后重启程序生效。未知字段、错误类型、负数预算、重复键会在启动时报错。

| 配置 | 主要参数与初始值 | 是否影响结果缓存 |
|---|---|---|
| [runtime.yaml](operations/runtime.yaml) | 解码 8 线程、场景 OpenCV 4 线程、标定/宽度诊断 OpenCV 1 线程；标定默认预留 1 核、上限 24 worker，转 bag 上限 4 线程；双手并行每手至少 4 核；SLAM 超时 `floor(帧数/30) × 16 + 300` 秒，地图导出 180 秒 | 修改数值不使已完成的算法结果失效 |
| [viewer.yaml](operations/viewer.yaml) | `127.0.0.1:9092`；画面解码并发 2，预览并发 1、含运行中最多 2 个任务；画面内存缓存 64 张、地图缓存 4 份；预览 720 宽、1500 kbps、CRF 12、GOP 30；每图显示最多 50000 点 | 画质、尺寸、码率、GOP 只影响对应查看器派生缓存；线程、超时和缓存容量不改变内容身份 |
| [gripper_tag_detection.yaml](operations/gripper_tag_detection.yaml) | 工作镜头半幅 ROI、夹爪双标签全区域搜索间隔 30、候选上限 96、补检候选上限 16、每批补检 32 帧 | 检测缓存和使用夹爪测量的最终任务结果均重新计算，场景地图可复用 |

运行配置按 **项目 YAML → 支持的环境变量 → CLI 显式值** 解析。`UMI_OPENCV_THREADS` 覆盖场景 OpenCV 线程；`--workers` 受 `--cores` 与当前 CPU affinity 限制。`--jobs` 控制同时处理的任务数，默认 1。`jobs=1` 且 CPU 预算充足时可并行处理两手。

查看器按 **项目 YAML → CLI 显式值** 逐项解析。`--port 0` 表示自动分配端口。

例如，只调整这一次运行：

```bash
./umi-scene run scenes/test_shelf/20261120_002 --cores 0-7 --jobs 2
UMI_OPENCV_THREADS=2 ./umi-scene run scenes/test_shelf/20261120_002
./umi-view scenes/test_shelf --port 9093
```

`--show-config --json`（场景）、`--show-config`（标定）的 `operational` 和查看器的 `configuration` 列出全部生效值、每项来源和覆盖次序。场景/标定还显示实际可用 CPU 列表、自动/显式 worker 计算结果及双手并行资格；资格不代表输入一定包含两只手。启动终端打印精简资源摘要，JSON 结果仍单独输出到 stdout。

配置在进程内冻结，派生 Python 子进程继承同一份快照；不会在每帧重新读取文件。`--expert-config` 可用 `measurements.gripper_tags.roi_px` 覆盖本次场景的检测区域。检测预算、线程和查看器设置仍从对应的运行配置读取。`UMI_VIEW_CACHE` 与 `XDG_CACHE_HOME` 保持原有缓存目录规则，查看器 `--show-config` 会给出实际缓存路径，以及旧地图转换所用的 `runtime.map_export_timeout_s` 和来源。降低地图显示点数只抽样浏览器响应，不改写已发布地图及其哈希；提高预算最多显示已发布的 50000 点。

保留为代码常量的内容包括 X3 原片尺寸/镜头半幅/标定区域、实际标签字典与 ID、底层 ArUco 阈值组合、内部后台流水线的调度策略，以及页面轮询节奏。它们分别属于硬件约定、已验证检测实现或内部实现细节，暂不作为日常调参入口。诊断 ROI、对比度和采样频率由相关标定命令控制；轨迹视频编码位于 `reconstruction.yaml` 的 `video`。

## 算法参数归属

标定和单段轨迹抽取共用 [umi/config/loader.py](../umi/config/loader.py) 加载配置。场景流程由 `umi/scene/collection/configuration.py` 加载场景参数，并复用轨迹配置。数值默认值集中在下面的 YAML 中。

### 双手音频同步

批次通过 `collection.yaml` 的 `sync.algorithm` 选择算法。两个文件是项目级设置，作用于所有使用对应算法的批次。

- `kiwi.yaml` 包含指纹提取、投票门槛、GCC-PHAT 精对齐和长录制漂移参数。项目采用 48 kHz 音频、4096 点频谱窗、1024 点步长，每个锚峰最多配对 16 个指纹。 [Kiwi 参数](algorithms/audio_sync/kiwi.yaml)
- `specialist.yaml` 包含采样率、包络低通、降采样、搜索范围和峰值比参数。 [Specialist 参数](algorithms/audio_sync/specialist.yaml)

`umi-scene run` 从所选算法的 YAML 读取完整参数。这些参数会改变同步偏移，调整时直接修改对应 YAML 并提交，所有机器保持一致。用 `umi-scene run --show-config` 查看来源和实际参数。`sync.json` 的 `inputs.parameters` 保存实际参数，数值变化后会重新估计偏移。

### 轨迹与标定

| 内容 | 管理位置 | 生效范围 |
|---|---|---|
| [calibration.yaml](algorithms/calibration.yaml) | 标定算法、板检测、D 静止窗口、验收规则 | `umi-calib run / verify / recheck` |
| [reconstruction.yaml](algorithms/reconstruction.yaml) | 输入、视频、视觉、建图、定位、IMU 融合、环境海报、后端、重试、测量和报告 | `umi-scene`、内部逐段引擎与标定 C；标定 B 共用 IMU 噪声默认值 |
| 相机内参、实测相机–IMU 参数与时间差 | `hardware/.registry/` 中的版本化相机标定；含实测/工厂内参来源 | 按序列号及套件的前后镜头来源选择当前参数，通过验收后统一执行 `install` |
| TCP、宽度、遮手图 | 套件引用的测量文件；未自定义遮手图时用 `hardware/grippers/masks/front_hand_mask.png` | 按 `kit.yaml` 选择，更新后重新运行 |
| 输入/输出路径、CPU、并行数、保留中间文件 | 命令行请求 | 只影响执行方式，不写入算法配置 |
| 原片规格、标定区域、像素变换与派生帧率规则 | `umi/core/frame.py` | 相机图像规格；在结果中记录 |

`config/algorithms/reconstruction.yaml` 保存重建默认参数；批次 `collection.yaml` 的 `map` 保存地图布局、参考手、默认版本和建图海报尺度用途，`poster` 保存海报布局，`backend` 保存各后端的海报用法，顶层 `description` 保存描述；一次专家实验的算法覆盖写入 `--expert-config` 指定的文件。建图原片位于各批次 `mapping/`，设备与地图快照保存在该批次 `.pipeline/manifest.yaml`。

`input.mode` 和 `input.allow_irregular_frames` 只用于独立逐段引擎。场景使用批次声明的后端与地图布局，并始终校验原片帧时间规则。

每个批次只有一个 `.pipeline/`：`maps/` 保存本批次的地图版本、当前指针和建图缓存，`tasks/<任务>/` 保存任务原片清单、状态、缓存和运行历史。不同批次不自动选择彼此的地图；显式指定 `map.version` 时冻结该版本到本批次。

人工任务选项推荐写在批次 `collection.yaml` 的 `tasks.<任务名>` 下，无需编辑隐藏清单：

```yaml
instruction: 拿起盒子，放到托盘上
# 双手批次中这次只使用右手时声明；否则继承批次设置
arms: [right]
```

任务人工选项只能写在批次根目录 `collection.yaml.tasks.<take>`；`.pipeline/tasks/<take>/episode.yaml` 由程序生成并冻结原片身份。

同步配置见[双手同步配置](../docs/05-任务同步与夹爪测量.md#2-核验双手同步)。修改任务选项后重跑即可复用兼容地图和阶段缓存。

## 第一次阅读从哪里开始

| 要查看的流程 | 配置位置 |
|---|---|
| 相机标定 | `calibration.yaml` 的 `board`、`factory`、`trajectory_validation`、`gravity`、`checks`，以及 `reconstruction.yaml` 的 `fusion.imu` |
| 独立录像抽轨迹 | `reconstruction.yaml` 的 `input`、`video`、`vision`、`fusion`、`poster`、`retry` 和 `reporting.warnings` |
| 多任务共享场景坐标 | `reconstruction.yaml` 的 `input.pairing`、`input.preflight`、`mapping`、`localization`、`fusion.scene_map` 和 `measurements` |
| 海报后端 | `reconstruction.yaml` 的 `poster` 及 `backends.poster_vio`、`backends.poster_independent` |

## 先分清参数改变了什么

| 类型 | 例子 | 调整后的含义 |
|---|---|---|
| 采样与计算预算 | `video.frame_step`、`fusion.solver.node_stride`、`fusion.solver.lm_iterations` | 改变观测密度或计算量；更多迭代不保证更准确 |
| 优化权重与先验 | `fusion.visual.position_sigma_m`、`fusion.imu.acc_noise`、`fusion.priors.scale_sigma` | 改变观测与初值在求解中的影响；sigma 越小通常约束越强，不是误差上限 |
| 筛选、验收与选轮 | `localization.minimum_localized_fraction`、`checks`、`retry` | 决定保留哪些观测或结果；放宽门限不会修复观测，也不能证明精度提高 |
| 质量提醒 | `reporting.warnings` | 按阈值记录风险提醒 |

独立轨迹的 `retry.target_localized_fraction` 和 `maximum_placements` 是选轮目标，耗尽次数仍可采用最佳有效结果。`maximum_scale_change_ratio` 是硬性保护，超限轮次不能采用。`factory` 内分别列出了候选生成约束和 A 验收项。

注意单位和计数口径：`fusion.alignment.step_frames`、`fusion.solver.node_stride` 数的是处理后的帧；`fusion.alignment.minimum_pairs` 数的是运动区间对；`fusion.initial_gyro_bias.minimum_samples` 数的是 IMU 样本。`sigma` 是标准差，`integration_covariance` 是直接使用的协方差，不能按同一种量调节。

## 怎么使用

先只读检查，不创建输出或运行求解器：

```bash
./umi-calib run hardware/calibration_runs/CAMERA_SERIAL/YYYYMMDD/front --dry-run
./umi-scene run scenes/shelf/20261105_001 --dry-run
```

调整默认配置会影响后续使用这些默认值的运行。单次调参时，将需要修改的字段写入一个与 `reconstruction.yaml` 同结构的文件：

<!-- check: scene-expert -->
```yaml
# experiment.yaml
video:
  size_px: 960
fusion:
  solver:
    node_stride: 3
localization:
  minimum_localized_fraction: 0.6
```

```bash
./umi-scene run scenes/test_shelf/20261120_002 --expert-config experiment.yaml --show-config
./umi-scene run scenes/test_shelf/20261120_002 --expert-config experiment.yaml
```

显式覆盖优先于项目默认；支持批次和任务覆盖的门槛，优先级为项目默认、批次、任务、专家覆盖。设备、地图布局和海报用途写在批次或套件中。最终参数与覆盖文件的内容和哈希会记录到新地图或任务结果。数值相同的配置换路径或改注释不会触发重算；只改变任务参数时仍复用兼容地图。

场景专家配置、逐段引擎覆盖和标定 C 的轨迹覆盖均使用 `reconstruction.yaml` 的字段结构。已发布结果记录实际使用的覆盖内容。原片、资产和基础配置变化会触发失效检查。内参来源由 kit 解析，带建图原片的快照也会按显式 `--expert-config` 检查地图兼容性。

标定的 `--expert-config` 使用 `calibration.yaml` 中的 `board`、`checks` 等顶层字段，也可用 `imu` 单独覆盖本次标定 B 的四项噪声。C 的重建覆盖通过下文的 `trajectory_validation.config` 引用。

## 参数按流程组织

| 阶段 | 配置位置 | 主要内容 |
|---|---|---|
| A/C 板参考 | `calibration.yaml` 的 `board` | 板检测采样、PnP、精化和覆盖评估 |
| 工厂内参及 A 验证 | `calibration.yaml` 的 `factory` | MEI → KB4、原片一致性和留出验收 |
| B 相机–IMU | `reconstruction.yaml` 的 `fusion.imu` | 加速度/角速度噪声密度与零偏随机游走 |
| C 轨迹验证 | `calibration.yaml` 的 `trajectory_validation` | 重建覆盖文件路径；留空时使用重建默认值 |
| D 与标定验收 | `calibration.yaml` 的 `gravity`、`checks` | 静止窗口、朝向聚类和 A/B/C/D 验收条件 |
| 镜头视频 | `video` | 裁切、取帧、尺寸、编码和音轨 |
| 视觉 | `vision.orb`、`vision.initial_tag` | ORB 特征、金字塔、FAST 与初始化标签 |
| IMU 融合 | `fusion` | 尺度/重力对齐、断链、分段、观测权重、零速度约束和求解预算 |
| 选轮和提醒 | `retry`、`reporting.warnings` | 重试目标、尺度变化保护和轨迹质量提醒 |

标定 A/B 的图像准备固定每 2 个原始帧取 1 帧，`board` 的步长在准备后的序列上采样。`video` 控制重建与标定 C 的视频准备。标定板布局见 [标定板说明](../hardware/calibration_targets/README.md)，本次实测标签边长用 `--tag-size-mm` 传入。

在标定覆盖文件中用 `trajectory_validation.config` 指定 C 的重建覆盖：

```yaml
trajectory_validation:
  config: "trajectory-tuning.yaml"
```

路径相对于声明它的标定覆盖文件。重建覆盖使用 `reconstruction.yaml` 的结构，填写该验证使用的字段，例如 `video.size_px`。C 子进程读取解析后的参数；报告保存标定和重建的有效配置。

C 固定 `input.mode: handheld` 并关闭 `poster.standalone_fusion`。环境镜头 C 同时关闭场景的海报任务融合、建图诊断和尺度修正。标定 `run`、`verify` 和 `recheck` 以本次实测标签边长派生 `vision.initial_tag.size_m`，并将派生值写入配置快照、计算身份和子进程配置。日常建图和独立轨迹从重建配置读取该尺寸。

## 场景和单段轨迹的参数边界

固定地图任务默认启用 `localization.reverse_fill.enabled`。每个反向连续段至少与正向结果重叠 `minimum_overlap_frames` 帧（默认 3），全部重叠帧须通过 `maximum_position_difference_m`（默认 0.03 m）和 `maximum_rotation_difference_deg`（默认 3°）。位置差使用冻结的地图尺度。`localization.reverse_fill.context_s` 默认 1.0，指定缺口两侧的处理范围。达到全片的 75% 时直接处理全片，短窗未完全补齐时也会回退全片，各次均以原始正向观测核验。

标签中心距使用 `measurements.gripper_tags.maximum_reprojection_error_px`（默认 2.0 px RMS）和 `measurements.gripper_tags.maximum_center_ambiguity_mm`（默认 2.0 mm）。平面两解的重投影误差之差不超过 0.25 px、中心差超过后者时拒绝观测。夹爪标签边长在套件引用的测量 JSON 中填写。

| 计算位置 | `reconstruction.yaml` 中的生效配置 |
|---|---|
| 场景建图的尺度不确定度、重力和加速度零偏门限 | `mapping.alignment` |
| 场景建图复用的轨迹求解设置 | `fusion.alignment.step_frames`、`fusion.alignment.minimum_pairs`、`fusion.chains` 的断链/细分/最少区间对，以及 `fusion.solver` 的线性求解设置 |
| 场景固定地图任务的视觉权重 | `fusion.scene_map` |
| 场景任务复用的惯性设置 | `fusion.imu`、`fusion.initial_gyro_bias`、`fusion.priors` 的速度和零偏先验、`fusion.solver.lm_iterations`、`fusion.zupt` |
| 内部逐段引擎及工作镜头标定 C | `fusion` 的逐段尺度对齐、链/分段、双镜头一致性和 `fusion.visual` 观测权重 |

地图内容指纹记录场景门限与实际使用的轨迹参数。`fusion.imu` 的四项噪声默认值共用于 Basalt 标定和 GTSAM 融合；标定专家文件的 `imu` 可覆盖本次 B 的设置。`gravity` 中的标定 D 重力参考与 `fusion.gravity_m_s2` 各自生效。

## 固定精度和图像规格

以下数值由实现定义：

- 同一视频时间的匹配容差固定为 1000 ns，仅吸收文本精度差，不插值，定义于 `umi/reconstruction/estimation/trajectory_metrics.py`。
- 轨迹去中心 RMS 的数值退化保护固定为 `1e-6 m`，同样定义于 `trajectory_metrics.py`。实际移动范围的验收仍通过 `checks.minimum_board_extent_m` 配置。
- 覆盖归一化半径从相机原始镜头半幅尺寸推导，当前为 `1920 / 2 = 960 px`，不随 1440 裁切或轨迹输出尺寸变化；实际分母写入覆盖指标。

原配置快照与当前配置不同时，`recheck` 会拒绝，重新执行 `run` 或 `verify` 会用新配置生成独立的结果版本。

## 夹爪宽度处理参数

宽度处理默认值位于 `reconstruction.yaml` 的 `measurements.gripper_width`。单次调整沿用相同字段路径。测量 JSON 只保存 anchors、verified 等实测信息。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `measurements.gripper_width.maximum_interpolation_gap_s` | 0.2 | 宽度缺口两端最大间隔，秒；0 表示不填补 |
| `measurements.gripper_width.closed_threshold_mm` | 3.0 | 闭合判定阈值，毫米；非负且小于该手标定最大开口 |

这些参数参与任务结果身份，不参与建图身份。原片检测缓存仅由原片内容和检测器决定；测量、阈值和输出时间轴变化会复用检测。宽度报告统一保存在 episode.json，格式见[夹爪宽度标定与接入](../docs/05-任务同步与夹爪测量.md#夹爪宽度标定与接入)。

## 重试与配置校验

`retry.use_poster_position_support` 默认开启。完成海报融合且达到共享观测时刻门槛后，独立轨迹按 `retry.target_maximum_position_support_gap_s`（默认 2 s）判断是否重试。关闭海报支持或缺少统计证据时，使用 `retry.target_localized_fraction`。

场景海报后端共用 `poster.acceptance.maximum_position_support_gap_s`（默认 2 s），`backends.poster_independent` 和 `backends.poster_vio` 各自可用同名字段覆盖。独立后端将生效值同时用于重试和最终验收，固定地图的单手局部回退用 `localization.maximum_visual_gap_s`。对具体 take 执行 `run --show-config --json` 时，`effective.position_support` 显示候选后端的门槛和来源。

`attempts.json` 保留自然视觉的 `full_fraction`，另记 `coverage_basis`、`position_coverage_passed`、`position_support_fraction` 和 `maximum_position_support_gap_s`。多轮先比较放置数量，再比较覆盖是否通过及所用覆盖比例；原始视觉覆盖保留为后续比较依据。

`retry.maximum_scale_change_ratio` 控制重建的尺度发散保护：每个有位置约束的分段比较融合后尺度与自身初始化，使用 `max(后/前, 前/后)`，不同地图之间不直接比较。默认上限为 2 倍，超限轮次重试且不得选中，全部轮次不合格则失败。它不修改融合解，也不放宽 `calibration.checks`；当前上限是粗粒度保护，尚需多录像检验。C 的运行记录保存每轮视觉、对齐和尺度诊断，验收前保留子进程工作数据。

YAML 使用 1.2 规则，开关写 `true` / `false`，空覆盖写 `{}`；重复字段会报错。未知字段、类型错误、NaN/Inf、无效枚举、非正噪声/计数、越界比例、倒置的 WARN/FAIL 阈值都会在清理旧输出和执行算法之前报错。重建默认文件使用 `schema_version: 3`，专家文件可省略版本号。

## 视频处理参数

`config/algorithms/reconstruction.yaml` 的 `video:` 统一配置 INSV → 镜头视频的处理。默认仍为：**3840×1920 原片 → 工作/环境镜头的已标定 1440×1440 区域 → 720×720 H.264 视频，每两帧取一帧**。单次覆盖写在 `--expert-config` 的 `video` 下。

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

裁切和缩放会同步变换 SLAM 内参与遮手图：`f'=s·f`，`c'=s·(c−裁切起点+0.5)−0.5`。KB4 畸变系数、相机–IMU 外参、相机–TCP 几何保持原值。遮手图只生成本次运行的处理副本，注册表资产不改写。场景双路模式按套件选择环境镜头参数。内部逐段 `dual` 通过 `--rear-calibration` 选择本设备的登记参数，省略时尝试从 INSV 工厂参数推导。

取帧步长同步用于时间戳提取、MP4 编码、SLAM 帧率、独立融合时间网格、结果中的原片帧号。`fusion.alignment.step_frames`、`fusion.solver.node_stride` 等仍按处理后的帧数计数，改变取帧密度后需结合录像长度核验求解质量。

ORB 引擎要求 `Camera.fps` 为整数，因此自动使用四舍五入且至少为 1 的标称帧率（默认 30）；MP4 编码、时间戳和融合仍使用精确帧率（默认 `30000/1001`），无需额外配置。

原片尺寸、工作镜头与环境镜头的位置和鱼眼模型属于已登记标定的物理约定，不能靠视频配置改成另一种相机或录制模式。这里保留鱼眼画面，不提供未经标定验证的任意去畸变切换。结果视频使用 MP4。HEVC 或 yuv444p 需要对应播放器支持。

完整视频参数会写入镜头视频清单及处理配置快照；`frame_geometry` 记录实际裁切和帧率。参数变化会使对应缓存失效，重新运行后可用查看器检查轨迹。默认值之外的画面尺寸、视野或帧密度需用实际录像验证轨迹质量。

## 结果里记录什么

- 标定：`run/record.json` 的 `computation` 保存计算配置，`evaluation` 保存验收配置。
- 场景任务：统一保存在任务 `episode.json`；`inputs.trajectory_parameters.video` 保存视频参数，`arm_results.<手>.frame_geometry` 保存实际画面几何，`inputs` 保存处理身份与参数快照。
- 内部逐段轨迹：`details/run_settings.json` 的 `parameters`、`frame_geometry`，以及摘要中的运行指纹。
- TXT 报告：算法配置 ID、验收/提醒 ID、相对默认值的覆盖项和完整记录路径。

记录包含完整有效值、默认/覆盖文件来源、显式 CLI 覆盖、配置版本、实际相机和夹爪资产，以及 git 提交号（`code_revision`）。缓存复用不看代码和引擎版本；改了场景算法或重编引擎后用 `umi-scene run --force` 重算。轨迹使用的 ORB YAML 和融合 JSON 在运行时生成。

`configuration.computation_hash` 仅表示算法配置；内部逐段轨迹的外层 `computation_hash` 还包括原片与测量资产。`evaluation_hash` 仅表示验收或提醒规则。配置注释、键顺序、覆盖文件的位置不改变配置数值指纹。

## 改参数或改代码后如何处理已有结果

**轨迹**：改变重建数值、重试条件、输入或设备资产会重新计算；改变 `reporting.warnings` 只重新评估已有指标、更新配置记录和报告，轨迹文件保持原样。证据缺失会重算。改了算法代码或重编引擎不会自动失效，需要用 `umi-scene run --force` 重算。

**标定**在输入匹配时自动续跑或复用结果。用 `--recompute` 建立新版本。只有验收规则变化时，可以单独重新验收。

```bash
./umi-calib recheck hardware/calibration_runs/CAMERA_SERIAL/YYYYMMDD/front/calibration_result --expert-config umi-calib-checks.yaml
```

这里的覆盖文件需自行准备。`recheck` 校验算法设置与原运行一致，仅修改验收项、结论和报告，保存完整先前验收及配置历史。若原运行用了算法覆盖，重验时需包含相同的算法覆盖。不支持的记录格式、缺少完整原始指标或未完成的执行不能重验；拒绝时保留原报告，需先完成计算。验收使用原始指标，`recheck` 不启用设备标定。

## 标定包与引擎边界

标定结果（v3）顶层使用 `calibration.json` 和 `manifest.json`。相机 JSON 是唯一最终测量，清单记录相机身份、镜头、板配置、来源和参数哈希。运行证据位于 `run/record.json`。设备档案只保存测量，SLAM 参数与融合实现始终来自项目代码。

共享场景在 kit 的 `intrinsics.front/rear` 分别选择 measured / factory，默认 front measured、rear factory。每个镜头、每种来源使用一份当前参数，安装后自动更新；已开始的运行和已有结果使用自己的冻结参数。rear measured 需要已登记的独立标定；rear factory 优先使用已登记的工厂来源整包，缺少时以 X3 原片工厂 rear 内参/双镜头外参 + 本设备 front B 推导并沿用 front 时差，标记 `NOT_EVALUATED`。当前选择损坏或身份/来源错误时拒绝，不静默推导。两个镜头允许不同来源；完整参数包内的内参与 B 仍必须配套。配置位置见[套件说明](../hardware/kits/README.md#前后镜头独立选择来源)，不写入算法参数 YAML。

Basalt/ORB/OpenCV/GTSAM 内部尚未暴露的参数随引擎版本固定；引擎变化不会自动让缓存失效。浮点稳定性保护、矩阵维度、Jacobian 自检容差、格式/数学约定、PNG 无损压缩和执行超时策略保留在代码中。录像契约还包含 0.05 fps 的模式识别容差、1 ms 规则帧网格判据、1.5 倍中位帧间隔的丢帧判据，以及 50 ms 的 IMU 覆盖容差；这些集中在 `umi/core/frame.py`，写入提取清单和轨迹设置；修改后用 `umi-scene run --force` 重算已有场景缓存。当前 ORB 包装器使用纯单目模式，IMU 噪声由融合配置控制。


## 运行解释器

项目入口优先使用 `UMI_PY`；视频工具可由 `UMI_PY_TOOLS` 单独指定。已有 `.deps/env.sh` 无需为此重新安装。新安装生成 `UMI_PY`。这些变量选择解释器，不覆盖算法参数。

标定运行记录使用 schema 3，和 YAML 的 `schema_version`、资产清单 schema 3 分别管理。`checks.minimum_board_extent_m` 不足使验收为 INCOMPLETE。D 的 `checks.gravity_residual_m_s2` 只影响 WARN，工具始终以 REPORT_ONLY 测量，因此只改该阈值可直接 recheck。

标定配置的 `factory` 负责工厂来源的转换和独立 A 验证：转换 RMS/P99 与原片射线一致性属于候选生成约束；`minimum_holdout_frames`、`holdout_warn_rms_px` 属于可通过 `umi-calib recheck` 重新判定的 A 证据/提醒规则。后两项不会改变已生成的参数，缺 A 证据不能安装。

算法参数类型位于 [`umi/config/parameter_models.py`](../umi/config/parameter_models.py)；公开字段与内部值的映射见 [reconstruction.py](../umi/config/reconstruction.py)，覆盖合并、校验和快照由 [`umi/config/loader.py`](../umi/config/loader.py) 管理。规范 JSON、文件内容哈希与 git 提交号统一由 [`umi/infra/provenance.py`](../umi/infra/provenance.py) 计算，修改结构时保留序列化字段和单位。

## 维护布局的约定

新增参数放在实际消费它的流程小节，注释说明作用、单位、特殊值和调大/调小的影响；共享参数注明哪些模式使用、哪些模式另有配置。多个字段共同作用时，在相邻位置说明关系，避免让读者只凭字段名猜测。

重建 YAML 按处理流程分组。公开字段路径集中映射到原有内部参数名，供快照和缓存身份使用。场景参数的类型位于 `umi/store/models.py`，数值校验位于 `umi/scene/collection/configuration.py`。


### 环境海报参与任务融合

在批次 `collection.yaml` 的 `poster.layout` 中声明环境海报布局（路径相对场景根目录）；各阶段是否使用海报写在各自的位置：

<!-- check: collection -->
```yaml
kit: kit_001
arms: [left, right]
description: 海报采集
poster:
  layout: ../../hardware/environment_poster/environment_poster.json
backend:
  mode: scene_map
  scene_map:
    poster_constraint: staged
```

`poster.detection` 控制环境标签检测，省略时随用途开启，显式写 `false` 与已开启的用途冲突。`map.poster_scale` 声明建图阶段的海报尺度用途（`off` / `diagnose` / `correct`），`backend.scene_map.poster_constraint` 声明地图定位的海报融合与初值选择（`staged` / `joint` / `off`）。
省略 `poster` 时，会为需要海报的后端使用项目布局。纯地图任务可以省略海报，显式 `poster: null` 表示不提供布局。使用其他海报或只做建图尺度诊断时，在 `poster.layout` 声明实际布局。

算法默认文件 `reconstruction.yaml` 中的 `poster.detection` 设定标签最短投影边长和最少观测时刻数；`poster.corners` 设定像素噪声和 Huber 阈值；`poster.acceptance` 设定重投影 RMS 及位置、旋转的局部线性化标准差门槛。位置和旋转的默认上限分别是 0.05 m 和 5°。合格的海报时刻可用于评估共同位置支持缺口。输出字段与实测流程见[海报约束质量](../docs/03-场景处理.md#海报约束质量和双手实测评估)。

`poster.culling.maximum_fraction` 默认 0.15，数量上限向上取整，0 关闭；`poster.culling.minimum_retained_time_fraction` 默认 0.8。各轮按每手的原始观测集合累计验收数量、最少观测时刻和时间跨度。场景单次覆盖这些共享门槛也会影响本次建图尺度估计。记录中的 `culling` 保留阶段、预算及移除的观测。

`poster.rear_rotation_refinement` 默认启用，先验为 0.2 rad，修正角上限为 5°。关闭时严格固定标定外参；开启后超限则拒绝融合。结果的 `rear_rotation_refinement` 记录配置与实际修正角。

### 建图海报尺度

任务估计后端由 `collection.yaml.backend.mode` 选择。每个 take 可在 `tasks.<take>.backend` 覆盖个别字段。数值门槛的优先级由低到高为算法默认值、批次、任务和显式 `--expert-config`。后端顺序、字段与默认值见[估计后端说明](../docs/03-场景处理.md#选择轨迹估计后端)。

`backends.poster_vio` 的 `initialization` 管理整板/标签子集假设上限；`keyframes` 管理时间、位移和旋转选帧，间隔为 0 时保留逐帧对照；`frontend` 管理活动点、轨迹预算、长度、几何筛选及有界重关联；`culling` 管理短求解迭代、热启动重求解轮数和累计剔除预算；`holdout` 管理标签/自然像素留出及验收 RMS。专家选项 `backends.poster_vio.per_tag_huber: true` 在同帧批量投影时为每个标签的四个角点独立计算一个 Huber 权重，默认关闭。默认值见 `algorithms/reconstruction.yaml`，完整覆盖示例及证据边界见[场景处理](../docs/03-场景处理.md#选择轨迹估计后端)。

`map.poster_scale: correct` 开启建图阶段的标签尺度诊断和尺度融合（`diagnose` 只记录证据不修正），会同时开启环境标签检测，并要求建图设备可解析正式 rear 参数或工厂兜底。修正后固定新尺度重求重力与零偏，并复核整图惯性残差；地图形状保持原值。

`mapping.poster_scale` 的 `minimum_observation_times`、`minimum_motion_m`、`minimum_board_extent_m` 和 `maximum_relative_std` 控制海报尺度可用性。`imu_std_floor` 和 `poster_std_floor` 是相对标准差下限，`compatibility_sigma` 和 `maximum_scale_difference` 同时约束两种尺度的差异。共享几何门槛用于海报位姿，建图尺度门槛用于地图尺度修正，`backends.poster_vio.minimum_training_observation_times` 用于每手联合估计的训练观测。

批次后端使用 `backend.mode`、`backend.poster_vio.base_tag` 和 `backend.scene_map.poster_constraint`。单次算法覆盖使用 `reconstruction.yaml` 中的字段路径。


### 独立模式与回退的海报配置

内部独立引擎用 `--tag-layout` 指定布局，`--rear-calibration measured|factory` 选择本设备 rear 来源，省略时尝试工厂兜底；环境海报数值默认来自 `reconstruction.yaml` 的 `poster`。逐段引擎用 `--config` 文件覆盖该组中本引擎使用的字段，以 `poster.standalone_fusion: true` 启用海报优化。默认只诊断，启用需要本设备可用的 rear 参数（正式标定或工厂兜底）；这些参数不自动开启标定 C 验证或共享场景的标签约束。

地图定位的海报配置位于 `backend.scene_map.poster_constraint`。`staged` 先完成地图与 IMU 基线，再进行海报优化。`joint` 用视觉种子构建包含地图先验、IMU 和海报角点的最终图。默认采用 `staged`。共享海报几何和建图参数分别位于 `reconstruction.yaml` 的 `poster` 和 `mapping`。单手地图定位失败后的独立回退沿用地图任务的海报开关。
