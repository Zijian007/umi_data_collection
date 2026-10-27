# 算法参数

标定和单段轨迹抽取共用 [software/config.py](../software/config.py) 加载配置。场景流程另由 `software/scene/management/inputs.py` 加载场景参数，并复用轨迹配置。数值默认值集中在下面的 YAML 中。

| 内容 | 管理位置 | 生效范围 |
|---|---|---|
| [calibration.yaml](calibration.yaml) | 标定算法、板检测、D 静止窗口、验收规则 | `umi-calib run / verify / recheck` |
| [trajectory.yaml](trajectory.yaml) | 视频尺寸与编码、ORB、重建重试、IMU 融合、ZUPT、质量提醒 | `umi-traj`，以及标定的 C 段轨迹验证 |
| [scene_parameters.yaml](scene_parameters.yaml) | 场景尺度/放置验收、固定坐标融合、原片配对与录制检查规则 | `umi-scene`；操作和覆盖方式见 [场景指南](../docs/场景.md) |
| 相机实测参数、时间差、TCP、遮手图 | `registry/` 中的相机与夹爪资产 | 按录像内部序列号选择，继续手动 `install` |
| 输入/输出路径、CPU、并行数、保留中间文件 | 命令行请求 | 只影响执行方式，不写入算法配置 |
| 原片尺寸、标定裁切、帧对应及像素缩放约定 | `software/camera/x3/frame.py` | 相机图像规格；在结果中记录 |

`config/scene_parameters.yaml` 保存算法默认参数；`scenes/<场景>/scene.yaml` 保存具体场景的地图模式、建图录像和设备版本。场景定义中的 `parameters` 可以引用算法参数覆盖文件。

## 怎么使用

先只读检查本次输入、设备选择和完整生效参数，不启动求解器、不创建输出、不安装组件。已有结果即使同时指定 `--force` 也保留。

```bash
./umi-calib run recordings/calibration/IAQEB26097M4CP/20261022/front --dry-run
./umi-traj recordings/scenes/supermarket_demo/demo_kit/20261022_001/IAQEB26097M4CP/VID_20261022_225713_00_005.insv --dry-run
```

调整默认配置会影响以后使用默认配置的运行。做实验时，另写一份只包含变更项的覆盖文件：

```bash
cat > /tmp/umi-traj-tuning.yaml <<'YAML'
retry:
  max_tries: 2

fusion:
  node_stride: 3

notes:
  minimum_full_fraction: 0.8
YAML

./umi-traj recordings/scenes/supermarket_demo/demo_kit/20261022_001/IAQEB26097M4CP/VID_20261022_225713_00_005.insv --config /tmp/umi-traj-tuning.yaml --dry-run
```

去掉 `--dry-run` 执行处理。优先级为 **项目默认 YAML → 显式 `--config` → 显式 CLI 选项**。例如 `--max-tries 1` 会覆盖文件中的 `retry.max_tries`；不传时保留文件的值。不会隐式搜索当前目录或用户主目录中的配置。默认文件和覆盖文件统一使用 YAML；此前自定义的 TOML 覆盖需改为 YAML，历史结果中的 JSON 快照仍可读取。

`--mode`、`--max-tries`、`--allow-irregular` 继续作为常用 CLI 入口。文件开启了丢帧处理时，可用 `--no-allow-irregular` 显式关闭。其他算法参数通过配置管理；`--jobs`、`--cores`、`--out`、`--force`、`--keep-work` 继续是执行选项。

## 参数按流程组织

| 阶段 | 配置表 | 主要内容 |
|---|---|---|
| A/B：检测板及评估覆盖 | `calibration.board` | 检测采样步长、放大倍率、PnP 标签数、归一化重投影阈值、角点视线限制、精化次数、比较帧数、覆盖指标 |
| B：相机–IMU 求解 | `calibration.imu` | 加速度/角速度噪声及 bias 随机游走先验 |
| D：多朝向静止检查 | `calibration.gravity` | 窗口时长、样本数、时间跨度、静止阈值、重力值、朝向聚类 |
| C：轨迹验证 | `calibration.trajectory_validation` | 轨迹覆盖配置文件；留空使用默认 `trajectory.yaml` |
| 标定验收 | `calibration.checks` | A/B/C/D 的 OK / WARN / FAIL 门限，含和样例相机比较的提醒 |
| 镜头视频 | `trajectory.video` | 正方形视频边长 `size_px`、x264 CRF 与 preset；每两帧取一帧固定 |
| 建图、逐图定位 | `trajectory.orb` | ORB 特征数量、金字塔、FAST、远点阈值、初始化标签参数 |
| 重建与选轮 | `trajectory.retry` | 尝试总次数、地图位置覆盖要求、独立放置数上限 |
| IMU 融合 | `trajectory.fusion` | 对齐步长、图节点间隔、链拆分/分组/细分、噪声、连续性、图先验、Huber、求解迭代、静态初始 bias |
| 零速度约束 | `trajectory.zupt` | 开关、窗口、静止阈值与速度先验 |
| 轨迹质量解释 | `trajectory.notes` | 覆盖率、起始 IMU 桥接时长和建图跟踪率提醒 |

表中的 `calibration.*`、`trajectory.*` 表示所属文件；文件内直接写 `board:`、`fusion:` 等映射，参数使用两个空格缩进。字段名尽量带单位：`_m`、`_mm`、`_px`、`_s`、`_deg`、`_rad_s`。IMU 噪声使用现有 Basalt/GTSAM 连续时间模型的单位，四个 noise/RW 值不能为零。

C 段可以独立指定轨迹实验配置。在标定覆盖文件中写：

```yaml
trajectory_validation:
  config: "trajectory-tuning.yaml"
```

相对路径以这份标定覆盖文件所在目录为基准。标定开始前会同时校验两份配置；C 子进程读取完整解析结果，最终标定报告记录两套配置。

YAML 使用 1.2 规则，开关写 `true` / `false`，空覆盖写 `{}`；重复字段会报错。未知字段、类型错误、NaN/Inf、无效枚举、非正噪声/计数、越界比例、倒置的 WARN/FAIL 阈值都会在清理旧输出和执行算法之前报错。配置结构版本为 `schema_version: 1`。

## 图像规格与轨迹分辨率

当前 X3 图像处理顺序是：**3840×1920 原片 → 单镜头 1440×1440 中心裁切 → 默认 720×720 轨迹视频**。标定使用 1440×1440 裁切图，轨迹建图、定位和查看器播放使用同一份缩放后的视频；画面保留鱼眼畸变。

原片模式、左右镜头位置、1440 裁切和取帧步长统一定义在 [frame.py](../software/camera/x3/frame.py)，属于现有标定的图像规格。每两帧取一帧，约 59.94 fps 变为约 29.97 fps；改变输出分辨率不改变帧时间或裁切视野。

调整轨迹分辨率只需覆盖一个参数，例如：

```yaml
video:
  size_px: 960
```

`size_px` 是正方形边长，必须是 2–1440 之间的偶数；偶数是 H.264/yuv420p 的编码要求，最大值避免放大超过标定裁切图。这个范围表示格式允许，过小的画面仍可能无法稳定提取特征。较大尺寸通常增加细节和计算量；默认以外的尺寸需用实际录像验证质量。

视频、前后镜头 SLAM 图像尺寸及内参、前镜头遮手图会同步调整。内参采用像素中心公式 `f' = s·f`、`c' = s·(c+0.5)-0.5`，KB4 畸变系数与相机–IMU 外参保持原值。遮手图用最近邻缩放，结果写入本次运行的 `debug/settings/`，设备档案中的原图只读。后镜头仍使用样例标定，调整尺寸不会改变其来源。

尺寸变化会使旧轨迹失去复用资格；完整生效参数、实际 SLAM 标量、遮手图来源/哈希和 `frame_geometry` 写入 `details/run_settings.json`，TXT 报告展示本次视频尺寸。已有标定和轨迹文件不会因编辑配置而自动改写。

## 结果里记录什么

- 标定：`diagnostics/report.json` 的 `parameters`。
- 轨迹：`details/run_settings.json` 的 `parameters`，以及摘要中的运行指纹。
- TXT 报告：算法配置 ID、验收/提醒 ID、相对默认值的覆盖项和完整记录路径。

记录包含完整有效值、默认/覆盖文件来源、显式 CLI 覆盖、配置版本、实际相机和夹爪资产、算法源文件内容哈希、引擎/依赖身份。未提交的代码修改也会进入代码哈希。轨迹使用的 ORB YAML 和融合 JSON 在运行时生成；完整中间输入用 `--keep-work` 保留。

`configuration.computation_hash` 仅表示算法配置；外层 `computation_hash` 还包括输入、测量资产、代码及引擎。`evaluation_hash` 仅表示验收或提醒规则。配置注释、键顺序、覆盖文件的位置不改变配置数值指纹。

## 改参数后如何处理旧结果

**轨迹**：改变重建数值、重试条件、输入、设备资产、算法代码或引擎会重新计算；改变 `notes` 只重新评估已有指标、更新配置记录和报告，轨迹文件保持原样。证据缺失会重算。旧版本摘要仍可供查看器读取，但缺少新运行指纹时首次抽取会重新计算。

**标定**：`run` 仍按原规则保护已有输出，需要重算时显式 `--force`。只有验收规则变化时，可以单独重新验收：

```bash
./umi-calib recheck recordings/calibration/IAQEB26097M4CP/20261022/front/umi_calib_out --config /tmp/umi-calib-checks.yaml
```

这里的覆盖文件需自行准备。`recheck` 校验算法设置与原运行一致，仅修改验收项、结论和报告，记录旧结论/规则 ID。若原运行用了算法覆盖，重验时需包含相同的算法覆盖。缺少完整原始指标、旧报告没有配置记录，或执行曾中断/失败时，会保留原报告并要求重跑；不会以舍入后的展示值替代原始指标。`recheck` 不启用设备标定。

## 标定包与引擎边界

新标定包（v2）只有 `calibration_r1440.json` 和 `bundle.json`。相机 JSON 是测量结果，元数据记录哈希与来源。SLAM 参数和融合脚本不再复制到每台设备中。旧 v1 包可以只读加载：核对原始 JSON、元数据、外参和注册表哈希，忽略其中的算法实现，运行项目中的共享算法；不会改写已有设备档案。

独立 `umi-traj --mode dual` 仍使用后镜头样例几何，样例身份记录在后镜头资产中。场景 `umi-scene` 的 `dual` 则要求各设备的实测后镜头标定，可通过 `calibrate-rear` 求解并用 `install-rear` 登记；不能把前镜头验收解释为双镜头都已标定。

Basalt/ORB/OpenCV/GTSAM 内部尚未暴露的参数，通过当前引擎内容/依赖身份管理。浮点稳定性保护、矩阵维度、Jacobian 自检容差、格式/数学约定、PNG 无损压缩和执行超时策略保留在代码中。录像契约还包含 0.05 fps 的模式识别容差、1 ms 规则帧网格判据、1.5 倍中位帧间隔的丢帧判据，以及 50 ms 的 IMU 覆盖容差；这些集中在 `camera/x3/frame.py`，写入提取清单和轨迹设置，修改会改变代码指纹。当前 ORB 包装器使用纯单目模式，IMU 噪声由融合配置控制；已移除容易造成误解的 ORB YAML IMU 噪声默认项。


## 运行解释器

项目入口优先使用 `UMI_PY`；视频工具可由 `UMI_PY_TOOLS` 单独指定。旧安装生成的 `KIWI_PY`、`KIWI_PY_TOOLS` 在 `software/environment.py` 中兼容，同用途的新变量优先。已有 `.deps/env.sh` 无需为此重新安装。新安装生成 `UMI_PY`。这些变量选择解释器，不覆盖算法参数。
