# 命令行、配置与测量资产

日常运行先选输入目录；算法调参写 YAML；设备和实测几何写入各自的资产声明。三个程序的分工如下：

| 入口 | 命令行输入 | YAML / 资产 |
|---|---|---|
| `umi-scene run / build` | 路径、`--cores`、`--force`、`--dry-run`；run 另有 `--jobs`、`--skip-preflight`，build 另有 `--keep-work` | `trajectory.yaml`、`scene_parameters.yaml` 提供算法默认值；`scene.yaml` 保存场景模式、标签布局引用和长期参数覆盖；`batch.yaml` 选择 kit、手和地图；`kit.yaml` 固定设备版本与测量文件；`<任务>/task.yaml` 写指令、使用手和同步 |
| `umi-calib run / verify` | 录像、输出路径、`--tag-size-mm` 实测板边长、CPU/线程、继续/重算、保留中间文件 | `calibration.yaml` 保存求解与验收参数；C 验证读取 `trajectory.yaml`；板布局和版本化测量是独立资产 |
| `umi-view` | 结果根目录、`--host`、`--port` | `viewer.yaml` 管理服务、预览和缓存；读取已有结果，不重新计算轨迹 |

`--config` 选择本次算法覆盖文件；`--intrinsics-source` 选择已有标定来源（标定 run 中选择求解来源）；`--map-version` 为场景 run 显式选图。来源和地图的长期选择可以写在 batch / kit 声明里。逐个子命令的完整选项、默认值与互斥关系见[命令行参数参考](../docs/命令行参数.md)，也可用 `<入口> <子命令> --help` 查阅。

独立诊断工具的参数仍由该次命令传入：`umi-scene detect-tags` 的检测采样/投影尺寸、`umi-calib preview-width / detect-width` 的 ROI/对比度/检测频率、`umi-scene export` 的导出格式/采样频率。它们只作用于本次诊断或导出，不改正式轨迹流程的 YAML。

## 先查看本次配置

```bash
./umi-scene run scenes/test_shelf/20261109_002 --show-config
./umi-calib run /path/to/front --show-config
./umi-view scenes/test_shelf --port 9092 --show-config
```

`--show-config` 输出 JSON，展示配置值、来源和执行选项，然后退出；不读取视频内容、不启动求解环境、不扫描查看器结果或监听端口。场景与标定入口可以同时使用 `--config`。它优先于 `--dry-run`；配置展示成功不代表录像、标定资产和地图已满足运行条件，输入检查继续使用 `--dry-run`。标定 run / verify 仍需填写目录/录像位置参数，但展示时不访问它们。run 的输入目录末级名称必须为 front / rear，由目录名选择镜头；两者都使用 run，详见[标定目录约定](../docs/标定.md#52-算标定)。

## 运行与查看器默认值

以下配置由 `software/settings.py` 读取。项目默认值只在 YAML 中定义；机器配置写到同目录的 `*.local.yaml`，不提交 Git。文件不存在就使用项目默认值；修改后重启程序生效。未知字段、错误类型、负数预算、重复键会在启动时报错。

| 配置 | 主要参数与初始值 | 是否影响结果缓存 |
|---|---|---|
| [runtime.yaml](runtime.yaml) | 解码 4 线程、场景 OpenCV 4 线程、标定/宽度诊断 OpenCV 1 线程；标定默认预留 1 核、上限 24 worker，转 bag 上限 4 线程；双手并行每手至少 4 核；SLAM 超时 `floor(帧数/30) × 16 + 300` 秒，地图导出 180 秒 | 修改数值不使已完成的算法结果失效 |
| [viewer.yaml](viewer.yaml) | `127.0.0.1:9092`；画面解码并发 2，预览并发 1、含运行中最多 2 个任务；画面内存缓存 64 张、地图缓存 4 份；预览 720 宽、1500 kbps、CRF 12、GOP 30；每图显示最多 50000 点 | 画质、尺寸、码率、GOP 只影响对应查看器派生缓存；线程、超时和缓存容量不改变内容身份 |
| [tag_detection.yaml](tag_detection.yaml) | 夹爪双标签全区域搜索间隔 30、候选上限 96、补检候选上限 16、每批补检 32 帧 | 检测缓存和使用夹爪测量的最终任务结果均重新计算；不重建场景地图 |

覆盖顺序为 **项目 YAML → 同名 `.local.yaml` → 已支持的环境变量 → CLI 显式值**。目前运行参数的环境覆盖为 `UMI_OPENCV_THREADS`，只覆盖场景 OpenCV；CLI 的 `--workers` 覆盖标定自动 worker 预算，但仍受 `--cores` 与当前 CPU affinity 限制。`--jobs` 控制同时处理的任务数；仅 `jobs=1` 且满足双手 CPU 预算时并行处理两手。查看器的 `--host` / `--port` 覆盖 YAML，`--port 0` 仍表示自动分配。

例如，只调整当前机器：

```yaml
# config/runtime.local.yaml
calibration_max_workers: 12
decode_threads: 2
```

```yaml
# config/viewer.local.yaml
port: 9092
preview_workers: 2
preview_pending: 3  # 运行中 + 排队；不能小于 preview_workers
map_max_points: 20000
```

`--show-config` 的 `operational`（场景/标定）或 `configuration`（查看器）列出全部生效值、每项来源和覆盖次序。场景/标定还显示实际可用 CPU 列表、自动/显式 worker 计算结果及双手并行资格；资格不代表输入一定包含两只手。启动终端打印精简资源摘要，JSON 结果仍单独输出到 stdout。

配置在进程内冻结，派生 Python 子进程继承同一份快照；不会在每帧重新读取文件。`--config` 仍仅覆盖算法参数，不接收这三份操作配置。`UMI_VIEW_CACHE` 与 `XDG_CACHE_HOME` 保持原有缓存目录规则，查看器 `--show-config` 会给出实际缓存路径，以及旧地图转换所用的 `runtime.map_export_timeout_s` 和来源。降低地图显示点数只抽样浏览器响应，不改写已发布地图及其哈希；提高预算最多显示已发布的 50000 点。

保留为代码常量的内容包括 X3 原片尺寸/镜头半幅/标定区域、实际标签字典与 ID、底层 ArUco 阈值组合、内部后台流水线的调度策略，以及页面轮询节奏。它们分别属于硬件约定、已验证检测实现或内部实现细节，暂不作为日常调参入口。诊断 ROI、对比度、采样频率和导出采样率继续由已有 CLI 控制；轨迹视频编码继续放在 `trajectory.yaml`。

## 算法参数归属

标定和单段轨迹抽取共用 [software/config.py](../software/config.py) 加载配置。场景流程另由 `software/scene/management/inputs.py` 加载场景参数，并复用轨迹配置。数值默认值集中在下面的 YAML 中。

| 内容 | 管理位置 | 生效范围 |
|---|---|---|
| [calibration.yaml](calibration.yaml) | 标定算法、板检测、D 静止窗口、验收规则 | `umi-calib run / verify / recheck` |
| [trajectory.yaml](trajectory.yaml) | 视频尺寸与编码、ORB、重建重试、IMU 融合、ZUPT、质量提醒 | `umi-scene`、内部逐段轨迹引擎及标定 C 段验证 |
| [scene_parameters.yaml](scene_parameters.yaml) | 场景尺度/放置验收、固定坐标融合、原片配对、宽度处理与录制检查规则 | `umi-scene`；操作和覆盖方式见 [场景指南](../docs/场景处理.md) |
| 相机内参、实测相机–IMU 参数与时间差 | `registry/` 中的版本化相机标定；含实测/工厂内参来源 | 按序列号及套件的前后镜头来源选择当前参数，通过验收后统一执行 `install` |
| TCP、宽度、遮手图 | 套件引用的测量文件；未自定义遮手图时用项目通用图 | 按 `kit.yaml` 选择，更新后重新运行 |
| 输入/输出路径、CPU、并行数、保留中间文件 | 命令行请求 | 只影响执行方式，不写入算法配置 |
| 原片规格、标定区域、像素变换与派生帧率规则 | `software/camera/x3/frame.py` | 相机图像规格；在结果中记录 |

`config/scene_parameters.yaml` 保存算法默认参数；`scenes/<场景>/scene.yaml` 保存具体场景的地图模式和算法覆盖；建图原片位于各批次 `mapping/`，设备与地图快照保存在该批次 `.pipeline/manifest.yaml`。场景定义中的 `trajectory` 和 `parameters` 直接保存该场景的参数改动。

每个批次只有一个 `.pipeline/`：`maps/` 保存本批次的地图版本、当前指针和建图缓存，`tasks/<任务>/` 保存任务原片清单、状态、缓存和运行历史。不同批次不自动选择彼此的地图；显式指定 `map_version` 时冻结该版本到本批次。

人工任务选项写在外层 `<任务>/task.yaml`，无需编辑隐藏清单：

```yaml
instruction: 拿起盒子，放到托盘上
# 双手批次中这次只使用右手时声明；否则继承批次设置
expected_arms: [right]
```

同步配置见[训练数据指南](../docs/训练数据.md#2-核验双手同步)。修改任务选项后重跑即可复用兼容地图和阶段缓存；旧验收会失效。

## 第一次阅读从哪里开始

三份算法 YAML 都按处理流程编号。编号和小节标题是注释，字段路径保持稳定：`trajectory.yaml` 的融合参数仍直接放在 `fusion` 下；`scene_parameters.yaml` 的参数仍平铺，没有额外的分组映射。

| 当前要理解的流程 | 阅读顺序 |
|---|---|
| 相机标定 | `calibration.yaml`：1 板检测 → 2 工厂来源 A（实测来源可跳过）→ 3 相机–IMU B → 4 轨迹验证 C → 5 静止检查 D → 6 验收 |
| 独立录像抽轨迹 | `trajectory.yaml`：1 输入 → 2 视频 → 3 视觉建图/定位 → 4 融合 → 5 静止约束 → 6 重试选轮 → 7 提醒 |
| 多任务共享场景坐标 | `scene_parameters.yaml`：1 配对 → 2 预检 → 3 建图尺度 → 4 双镜头地图放置 → 5 任务定位 → 6 固定场景融合 → 7 夹爪测量 → 8 诊断输出；再查下方“场景和单段轨迹的参数边界” |

独立模式为每段录像建立自己的坐标参考；共享模式先建立米制场景地图，再把任务定位到该地图。两者复用部分视觉和惯性算法，但约束模型、验收门限和坐标含义有区别。**共享任务的定位覆盖不足，先看场景第 5 节；独立轨迹的选轮规则，先看轨迹第 6 节。**

`fusion` 较长，可按问题直接定位：4.1 重力与初始 bias；4.2 断链、尺度初始化和链筛选；4.3 地图放置与分段连续性；4.4 视觉权重和杆臂；4.5 IMU 噪声；4.6 初值先验；4.7 采样和求解预算。“链”是连续跟踪的视觉观测，“放置”是把地图变换到轨迹参考系，“杆臂”是相机与 IMU 之间的平移。

## 先分清参数改变了什么

| 类型 | 例子 | 调整后的含义 |
|---|---|---|
| 采样与计算预算 | `video.frame_step`、`fusion.node_stride`、`lm_iterations` | 改变观测密度或计算量；更多迭代不保证更准确 |
| 优化权重与先验 | `position_sigma_m`、`acc_noise`、`scale_prior_sigma` | 改变观测与初值在求解中的影响；sigma 越小通常约束越强，不是误差上限 |
| 筛选、验收与选轮 | `minimum_localization_fraction`、`checks`、`retry` | 决定保留哪些观测或结果；放宽门限不会修复观测，也不能证明精度提高 |
| 提醒与诊断输出 | `notes`、`export_auxiliary_trajectories` | 提醒风险或保留排障证据，不增加轨迹约束 |

以字段注释为准：独立轨迹的 `retry.minimum_full_fraction` 和 `maximum_placements` 是选轮目标，耗尽次数仍可采用最佳有效结果；`maximum_scale_change_ratio` 是硬性保护，超限轮次不能采用。`factory` 内也分开列出了候选生成约束和 A 验收项。

注意单位和计数口径：`fusion.step`、`node_stride` 数的是处理后的帧；`minimum_alignment_pairs` 数的是运动区间对；`initial_bias_min_samples` 数的是 IMU 样本。`sigma` 是标准差，`integration_covariance` 是直接使用的协方差，不能按同一种量调节。

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

场景算法优先级为 **项目默认 YAML → scene.yaml 内联覆盖 → 本次 --config 文件**。临时实验不必修改场景声明：把上面的 `trajectory` / `parameters` 两部分保存为 `experiment.yaml`，然后运行：

```bash
./umi-scene run scenes/test_shelf/20261109_002 --config experiment.yaml --show-config
./umi-scene run scenes/test_shelf/20261109_002 --config experiment.yaml
```

`build` 同样支持 `--config`。覆盖文件顶层只允许 `trajectory` 和 `parameters`；不接受设备、地图模式、标签布局或运行选项。最终参数与覆盖文件的路径、内容和哈希记录在新地图/任务结果中，不回写 scene / batch / kit 声明。数值相同的配置换路径或改注释不会触发重算；只改变任务参数时仍复用兼容地图。复用结果保留其原计算记录。

验收和导出使用结果冻结的覆盖内容，临时 YAML 删除后仍能进行；当前原片、资产和基础配置变化仍会触发失效检查。仅有批次快照时，`--intrinsics-source` 只能核对已有版本的来源；切换版本需要 batch / kit 中对应的完整声明。带建图原片的旧快照也会按显式 `--config` 检查地图兼容性。

标定的 `--config` 文件直接使用 `board`、`imu`、`checks` 等 calibration 字段，不包一层 `calibration:`，也不使用场景的 `trajectory:` / `parameters:` 外层。C 的轨迹实验配置通过下面的 `trajectory_validation.config` 引用。`--jobs`、`--cores`、`--force` 等执行选项不写入算法 YAML。

## 参数按流程组织

| 阶段 | 配置表 | 主要内容 |
|---|---|---|
| A/C：覆盖评估与板参考 | `calibration.board` | 检测采样步长、放大倍率、PnP 标签数、归一化重投影阈值、角点视线限制、精化次数、比较帧数、覆盖指标 |
| 工厂内参生成与 A 验证 | `calibration.factory` | MEI → KB4 转换误差、原片一致性、A 最低证据帧数与 RMS 提醒线 |
| B：相机–IMU 求解 | `calibration.imu` | 加速度/角速度噪声及 bias 随机游走先验 |
| C：轨迹验证 | `calibration.trajectory_validation` | 轨迹覆盖配置文件；留空使用默认 `trajectory.yaml` |
| D：多朝向静止检查 | `calibration.gravity` | 窗口时长、样本数、时间跨度、静止阈值、重力值、朝向聚类 |
| 标定验收 | `calibration.checks` | A/B/C/D 的质量门限和 C 证据完整性门槛，含和样例相机比较的提醒 |
| 镜头视频 | `trajectory.video` | 裁切范围、取帧步长、输出尺寸、缩放、编码器/像素格式、CRF/preset 和音轨 |
| 建图、逐图定位 | `trajectory.orb` | ORB 特征数量、金字塔、FAST、远点阈值、初始化标签参数 |
| IMU 融合 | `trajectory.fusion` | 对齐步长、图节点间隔、链拆分/分组/细分、噪声、连续性、图先验、Huber、求解迭代、静态初始 bias |
| 零速度约束 | `trajectory.zupt` | 开关、窗口、静止阈值与速度先验 |
| 重建与选轮 | `trajectory.retry` | 尝试总次数、地图位置覆盖要求、独立放置数上限、尺度变化倍率上限 |
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

C 固定覆盖 `input.mode: handheld` 和 `environment_tags.tag_fusion_enabled: false`，避免修改任务默认配置后，标定验证意外要求 rear/海报资产。环境镜头 C 后端同样关闭场景的海报任务融合、建图诊断和尺度修正开关。其余视频、ORB、融合参数仍按指定配置生效；派生后的配置参与快照与计算身份。

标定的 `run` 和 `verify` 会将本次实测板边长 `--tag-size-mm / 1000` 传给 C 段的 `orb.initial_tag_size_m`，优先于轨迹默认值和覆盖文件；候选与样例比较使用同一个尺寸。该派生覆盖会写入轨迹配置快照、内容指纹和子进程配置，`recheck` 从原运行的板尺寸恢复同一配置。切换 A4/A3 不用再手动修改 C 的初始化标签尺寸；日常场景建图和独立轨迹仍使用 `trajectory.yaml` 的 `orb.initial_tag_size_m`。

## 场景和单段轨迹的参数边界

固定地图任务的双向定位由 `scene_parameters.yaml` 控制：`localization_bidirectional` 默认 true，仅在正向有缺帧时反向处理；`localization_minimum_overlap_frames` 默认 3，要求每个反向连续段有足够的共同观测；`localization_max_position_difference_m` 默认 0.03，`localization_max_rotation_difference_deg` 默认 3.0，约束全部共同观测的最大差异。位置差使用冻结的地图尺度转换成米，不拟合新的坐标变换。参数、地图尺度和引擎身份参与任务定位缓存校验。设置 false 可恢复仅正向定位，原有覆盖率和最长视觉中断门限均保留。

`localization_reverse_context_s` 默认 1.0，必须为正数：优先处理覆盖全部缺口及两侧上下文的反向范围。范围达到全片的 75% 时直接执行全片；短范围未完全补齐时自动回退全片，仍以原始正向观测核验，不能用短范围恢复帧充当新的独立锚点。

标签中心三维间距使用 `tag_distance_max_reprojection_error_px`（默认 2.0 原始像素 RMS）和 `tag_distance_max_center_ambiguity_mm`（默认 2.0 mm），均为正数。平面两解的重投影误差相差 <=0.25 px、中心差超过后者时拒绝观测。标签实际边长属于套件资产，在 `gripper.tag_distance` 引用的 JSON 内分别填写 ID 0/1 的 `tag_sizes_mm`，不借用 AprilGrid 或 ORB 初始化标签尺寸。

场景复用轨迹算法中的部分计算步骤，但场景地图的验收门限和固定坐标融合模型由 `scene_parameters.yaml` 管理。下面按实际生效位置列出，不能只根据相似字段名选择要修改的文件。

| 计算位置 | 生效配置 |
|---|---|
| 场景建图：尺度相对标准差、重力一致性、加速度零偏门限 | `scene.maximum_scale_relative_std`、`scene.gravity_tolerance_m_s2`、`scene.maximum_acc_bias_m_s2`；通过统一的场景对齐参数转换覆盖轨迹中的对应字段 |
| 场景建图：对齐步长、转动断链、坏段细分、最少区间对、线性求解 | 复用 `trajectory.fusion.step / split_rotation_deg / refine_chain_pairs / minimum_alignment_pairs / minimum_chain_pairs / linear_iterations / linear_tolerance` |
| 场景任务：工作镜头与环境镜头视觉位置/旋转噪声、Huber | `scene.front_position_sigma_m / rear_position_sigma_m / visual_rotation_sigma_deg / huber`；不使用轨迹融合中的对应视觉参数 |
| 场景任务：IMU 噪声、bias 随机游走、初始 bias、速度/bias 先验、LM 迭代和 ZUPT | 复用 `trajectory.fusion` 中对应字段及 `trajectory.zupt` |
| 内部单段轨迹及工作镜头标定 C | 使用 `trajectory.fusion` 自己的链/分段、尺度/放置和视觉噪声参数；不读取场景门限 |

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

`config/trajectory.yaml` 的 `video:` 统一配置 INSV → 镜头视频的处理。默认仍为：**3840×1920 原片 → 工作/环境镜头的已标定 1440×1440 区域 → 720×720 H.264 视频，每两帧取一帧**。在 scene.yaml 的 trajectory.video 中覆盖时也使用相同字段。

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

裁切和缩放会同步变换 SLAM 内参与遮手图：`f'=s·f`，`c'=s·(c−裁切起点+0.5)−0.5`。KB4 畸变系数、相机–IMU 外参、相机–TCP 几何保持原值。遮手图只生成本次运行的处理副本，注册表资产不改写。场景双路模式使用各设备对应来源的完整环境镜头标定；内部独立 dual 验证仍明确使用样例环境镜头。

取帧步长同步用于时间戳提取、MP4 编码、SLAM 帧率、独立融合时间网格、结果中的原片帧号及训练对齐。`fusion.step`、`node_stride` 等仍按处理后的帧数计数，改变取帧密度后需结合录像长度核验求解质量。训练导出的 `--fps` 只设置训练样本时间轴，视频维持这里生成的帧率。

ORB 引擎要求 `Camera.fps` 为整数，因此自动使用四舍五入且至少为 1 的标称帧率（默认 30）；MP4 编码、时间戳和融合仍使用精确帧率（默认 `30000/1001`），无需额外配置。

原片尺寸、工作镜头与环境镜头的位置和鱼眼模型属于已登记标定的物理约定，不能靠视频配置改成另一种相机或录制模式。这里保留鱼眼画面，不提供未经标定验证的任意去畸变切换。MP4 是当前结果和训练包的文件协议。HEVC 或 yuv444p 需要对应播放器和训练解码器支持。

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

共享场景在 kit 的 `intrinsics.front/rear` 分别选择 measured / factory，默认 front measured、rear factory。每个镜头、每种来源使用一份当前参数，安装后自动更新；已开始运行和历史结果使用自己的冻结参数。rear measured 需要已登记的独立标定；rear factory 优先使用已登记的工厂来源整包，缺少时以 X3 原片工厂 rear 内参/双镜头外参 + 本设备 front B 推导并沿用 front 时差，标记 `NOT_EVALUATED`。当前选择损坏或身份/来源错误时拒绝，不静默推导。两个镜头允许不同来源；完整参数包内的内参与 B 仍必须配套。配置位置见[套件说明](../kits/README.md#前后镜头独立选择来源)，不写入算法参数 YAML。

Basalt/ORB/OpenCV/GTSAM 内部尚未暴露的参数，通过当前引擎内容/依赖身份管理。浮点稳定性保护、矩阵维度、Jacobian 自检容差、格式/数学约定、PNG 无损压缩和执行超时策略保留在代码中。录像契约还包含 0.05 fps 的模式识别容差、1 ms 规则帧网格判据、1.5 倍中位帧间隔的丢帧判据，以及 50 ms 的 IMU 覆盖容差；这些集中在 `camera/x3/frame.py`，写入提取清单和轨迹设置，修改会改变代码指纹。当前 ORB 包装器使用纯单目模式，IMU 噪声由融合配置控制；已移除容易造成误解的 ORB YAML IMU 噪声默认项。


## 运行解释器

项目入口优先使用 `UMI_PY`；视频工具可由 `UMI_PY_TOOLS` 单独指定。旧安装生成的 `KIWI_PY`、`KIWI_PY_TOOLS` 在 `software/environment.py` 中兼容，同用途的新变量优先。已有 `.deps/env.sh` 无需为此重新安装。新安装生成 `UMI_PY`。这些变量选择解释器，不覆盖算法参数。

标定运行记录使用 schema 3，和 YAML 的 `schema_version: 1`、资产清单 schema 3 分别管理。`checks.minimum_board_extent_m` 不足使验收为 INCOMPLETE。D 的 `checks.gravity_residual_m_s2` 只影响 WARN，工具始终以 REPORT_ONLY 测量，因此只改该阈值可直接 recheck。

标定配置的 `factory` 负责工厂来源的转换和独立 A 验证：转换 RMS/P99 与原片射线一致性属于候选生成约束；`minimum_holdout_frames`、`holdout_warn_rms_px` 属于可通过 `umi-calib recheck` 重新判定的 A 证据/提醒规则。后两项不会改变已生成的参数，缺 A 证据不能安装。

算法参数类型位于 [`software/parameters.py`](../software/parameters.py)；覆盖合并、校验和快照由 [`software/config.py`](../software/config.py) 管理。规范 JSON、文件、源代码及引擎身份统一由 [`software/provenance.py`](../software/provenance.py) 计算，修改结构时保留序列化字段和单位。

## 维护布局的约定

新增参数放在实际消费它的流程小节，注释说明作用、单位、特殊值和调大/调小的影响；共享参数注明哪些模式使用、哪些模式另有配置。多个字段共同作用时，在相邻位置说明关系，避免让读者只凭字段名猜测。

分区使用注释，不为展示分组新增 YAML 层级。修改字段路径需要同时考虑类型模型、覆盖文件、校验、结果快照和内容指纹；纯粹调整顺序与注释则应保持解析后的字段、类型、数值以及配置指纹不变。场景参数的类型与校验分别位于 `software/scene/models.py` 和 `software/scene/management/inputs.py`。


### 环境海报参与任务融合

在 `scene.yaml` 顶层添加 `tags: {layout: ../../hardware/calibration_targets/environment_poster.json}`（路径相对 scene.yaml）。仅声明布局时运行检测与位姿诊断，轨迹保持原值；在 `parameters` 内设置 `tag_fusion_enabled: true` 才加入角点因子，使用套件绑定的正式 rear 参数，未绑定时尝试上述工厂兜底；兜底不代表 rear 已独立验收。

`tag_minimum_side_px` / `tag_minimum_frames` 筛选像素尺寸和独立观测时刻；`tag_corner_sigma_px` / `tag_sigma_per_radius_px` 设置角点噪声；`tag_huber` 是每 ID 四角点共同的鲁棒核阈值；`tag_maximum_reprojection_rms_px` 检查初始化和最终像素残差。默认值只经合成数据验证。无足够观测保留基线，启用后的几何冲突明确失败。每段录像独立求解海报位姿，海报在录像内须静止、平整；任务融合本身不会改变已发布地图尺度；建图修正由下述开关单独控制。打印尺寸状态仍为 PDF 设计值，需核对成品尺寸后评估实拍效果。


### 建图海报尺度

同样需要 `tags.layout`。`tag_mapping_enabled: true` 开启建图尺度诊断；`tag_mapping_scale_enabled: true` 才将海报与 IMU 尺度在对数域按方差合并，且要求前者同时开启、建图设备可解析正式 rear 参数或工厂兜底。默认两项关闭。修正后固定新尺度重求重力与零偏，并复核整图惯性残差；地图形状保持原值。

`tag_mapping_minimum_frames`、`tag_mapping_minimum_motion_m`、`tag_mapping_minimum_board_extent_m` 和 `tag_mapping_maximum_relative_std` 控制海报尺度可用性；`tag_mapping_imu_std_floor` / `tag_mapping_tag_std_floor` 为相对标准差地板；`tag_mapping_compatibility_sigma` 与 `tag_mapping_maximum_scale_difference` 同时约束两尺度差异。这些默认值仍待实拍调参。几何不足保持原尺度；有效来源冲突或修正后惯性复核失败时拒绝发布新地图。布局、角点和诊断随地图版本保存；海报位姿不跨录像使用。


### 独立模式与回退的海报配置

独立命令用 `--tag-layout` 指定布局，可用 `--rear-calibration rear_<hash>` 绑定正式 rear，省略时尝试工厂兜底，在 `trajectory.yaml` 的 `environment_tags` 中设置 `tag_fusion_enabled`、`tag_minimum_side_px`、`tag_minimum_frames`、`tag_corner_sigma_px`、`tag_sigma_per_radius_px`、`tag_huber` 和 `tag_maximum_reprojection_rms_px`。默认只诊断，启用需要本设备可用的 rear 参数（正式标定或工厂兜底）；这些参数不自动开启标定 C 验证或共享场景的标签约束。

共享任务仍使用 `scene_parameters.yaml` 的扁平 `tag_*` 配置；单手质量失败后的独立回退显式将其映射到独立图的 `environment_tags`，沿用冻结资产，避免回退时悄悄丢掉已启用的海报约束。独立图联合优化分段尺度，不使用共享地图建图阶段的两个 `tag_mapping_*enabled` 开关。
