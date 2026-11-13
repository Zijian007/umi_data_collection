# UMI 相机标定与轨迹采集

从 Insta360 X3 原始 `.insv` 视频生成夹爪 TCP 轨迹。

统一称呼：**工作镜头**是安装后能看到夹爪的镜头，**环境镜头**是另一侧镜头。左右手相机各有这两个镜头。现有代码、命令和文件中的 `front` 对应工作镜头，`rear` 对应环境镜头，接口名称保持兼容。详见[镜头术语](docs/设备与坐标.md#镜头术语)。

使用三个程序：

| 程序 | 职责 |
|---|---|
| `./umi-calib` | 相机标定、验收与设备资产管理 |
| `./umi-scene` | 从场景批次发起建图和轨迹抽取，验收与导出 |
| `./umi-view` | 查看视频、轨迹与地图 |

三个入口的全部子命令、默认值和使用限制见[命令行参数参考](docs/命令行参数.md)。终端可用 `./umi-view --help`、`./umi-scene <子命令> --help`、`./umi-calib <子命令> --help` 查阅。

**准备设备 → 放好套件配置和场景视频 → 运行 → 查看。** 完整 WSL 操作见[日常使用](docs/安装与日常使用.md)。

设备线：每台相机分别标定，套件指定左右相机及前后镜头的实测/工厂来源，并引用套件内的夹爪几何和开口测量文件。相机标定可跨场景复用；TCP 几何还取决于夹爪和相机安装关系。

场景线：场景下每轮“建图＋几次操作”组成一个批次，每次操作是一个 take。批次只需声明使用哪个套件，每次 run 按当前 kit.yaml 自动读取所需测量。新采集轮次可新建批次；同一批次替换建图原片后，重跑会自动重建地图。

## 首次准备

```bash
bash install.sh
./umi-calib check
./umi-scene check

./umi-calib run recordings/calibration/CAMERA_SERIAL/20261105/front --tag-size-mm 24.0
# 阅读 front/calibration_result/report.txt 后登记
./umi-calib install recordings/calibration/CAMERA_SERIAL/20261105/front/calibration_result
```

`umi-calib run` 默认用 A 求实测 KB4，并自动对照工作镜头的工厂 MEI（`lens1`）。结果的 `factory/` 保存两个镜头的光学拟合、1440×1440 对照和 1920×1920 全镜头探索；这些自动诊断不能单独安装。

工作镜头 `run <front目录>` 和环境镜头 `run <rear目录>` 都可加 `--intrinsics-source factory`：从 B 原片的对应镜头工厂 MEI 拟合 KB4，A 保留独立图像验证，B 重新求相机–IMU 参数，C/D 继续验收。A 超标只提醒，缺少有效 A 证据不能安装。实测与工厂整包分开登记；选择方法见[标定指南](docs/标定.md#选择实测或工厂内参)。

每台相机单独在 `front/A/`、`B/`、`C/`、`D/` 放对应原片。登记结果保存在 `registry/`，由程序维护。然后填写 `kits/kit_001/kit.yaml`，字段见[套件配置说明](kits/README.md)。

## 日常处理

先准备：

```text
kits/kit_001/kit.yaml
scenes/shelf/20261105_001/
├── batch.yaml                       # kit: kit_001
├── mapping/原始建图文件名.insv
├── take_001/
│   ├── left/原始任务文件名.insv
│   └── right/原始任务文件名.insv
└── take_002/...
```

```bash
./umi-scene run scenes/shelf/20261105_001
./umi-view scenes/shelf
```

`run` 自动使用场景默认值、保存批次设备快照、登记原片、构建并固定本轮地图、抽取轨迹。手写配置并放好视频后直接执行 `run`。已有本地原片直接引用；保留相机文件名与 SD 卡源文件。

相机登记相应来源的完整参数后，可用 `./umi-scene run <批次> --intrinsics-source measured|factory` 选择来源，也可在 `batch.yaml` 中设置。来源配置见[套件说明](kits/README.md#前后镜头独立选择来源)。缺少对应版本会报错；来源或引擎变化需使用匹配的地图。

结果位于 `take_*/results/`，包括 `report.txt`、各手的位姿/宽度统一表 `trajectory.csv` 和 `front.mp4`，双手另有 `bimanual.csv`。重复运行复用有效结果，失败不会覆盖上次成功结果。每次运行按当前 kit.yaml 和批次原片校验复用条件；标定或建图输入变化会自动重算。自动清单、地图、缓存和运行记录统一收在每个批次唯一的 `.pipeline/`，由程序维护、无需查看；其中 `maps/` 属于本批次、`tasks/<任务>/` 保存任务内部数据。被取代的运行、地图版本和未用到的缓存会自动删除，见 [场景处理](docs/场景处理.md)。默认不生成空 `scene.yaml`。

`front.mp4` 从对应相机的原始 INSV 提取工作镜头画面，默认 720×720、约 29.97 fps。裁切、取帧、缩放、编码和音轨统一通过 [`trajectory.yaml` 的 `video:`](config/trajectory.yaml) 配置；范围、场景覆盖和联动规则见[视频处理参数](config/README.md#视频处理参数)。

`--dry-run` 只读预览；`run scenes --jobs 2` 可跨场景批处理。默认共享工作镜头地图；双路建图等高级配置见[场景指南](docs/场景处理.md)。不同地图版本不保证坐标一致。

## 验收与导出

```bash
./umi-scene review scenes/shelf/20261105_001 --accept --reviewer zijian
./umi-scene export scenes/shelf/20261105_001 --out exports/shelf
# 或导出训练中间包，无需先导出轨迹包
./umi-scene export scenes/shelf/20261105_001 --format training --out datasets/shelf
```

导出只接受当前有效且已人工验收的结果。训练包还要求操作指令、双手同步确认；夹爪宽度与训练就绪条件见[训练数据](docs/训练数据.md)。计算成功不等于已验收，查看器不自动验收。

[文档导航](docs/README.md) · [设备与目录](docs/设备与坐标.md) · [标定指南](docs/标定.md) · [参数管理](config/README.md)
