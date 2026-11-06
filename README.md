# UMI 相机标定与轨迹采集

从 Insta360 X3 原始 `.insv` 视频生成夹爪 TCP 轨迹。使用三个程序：

| 程序 | 职责 |
|---|---|
| `./umi-calib` | 相机标定、验收与设备资产管理 |
| `./umi-scene` | 从场景批次发起建图和轨迹抽取，验收与导出 |
| `./umi-view` | 查看视频、轨迹与地图 |

**准备设备 → 放好套件配置和场景视频 → 运行 → 查看。** 完整 WSL 操作见[日常使用](docs/安装与日常使用.md)。

设备线：每台相机分别标定，套件指定左右相机及其标定和夹爪几何版本。相机标定可跨场景复用；TCP 几何还取决于夹爪和相机安装关系。

场景线：场景下每轮“建图＋几次操作”组成一个批次，每次操作是一个 take。批次只需声明使用哪个套件，每次 run 按当前 kit.yaml 自动读取所需测量。新采集轮次可新建批次；同一批次替换建图原片后，重跑会自动重建地图。

## 首次准备

```bash
bash install.sh
./umi-calib check
./umi-scene check

./umi-calib run recordings/calibration/CAMERA_SERIAL/20261105/front --tag-size-mm 24.0
# 阅读 front/umi_calib_out/report.txt 后登记
./umi-calib install recordings/calibration/CAMERA_SERIAL/20261105/front/umi_calib_out
```

每台相机单独在 `front/A/`、`B/`、`C/`、`D/` 放对应原片。登记结果保存在 `registry/`，由程序维护。然后填写 `kits/kit_001/kit.yaml`，字段见[套件配置说明](kits/README.md)。

## 日常处理

先准备：

```text
kits/kit_001/kit.yaml
scenes/shelf/20261105_001/
├── batch.yaml                       # kit: kit_001
├── mapping/原始建图文件名.insv
├── take_001/videos/
│   ├── left/原始任务文件名.insv
│   └── right/原始任务文件名.insv
└── take_002/videos/...
```

```bash
./umi-scene run scenes/shelf/20261105_001
./umi-view scenes/shelf
```

`run` 自动使用场景默认值、保存批次设备快照、登记原片、构建并固定本轮地图、抽取轨迹。无需依次执行 `init/session/ingest/build`。已有本地原片直接引用；保留相机文件名与 SD 卡源文件。

结果位于 `take_*/results/`，包括 `report.txt`、各手 `traj_tcp.txt` 和 `front.mp4`，双手另有 `bimanual.csv`。重复运行复用有效结果，失败不会覆盖上次成功结果。每次运行按当前 kit.yaml 和批次原片校验复用条件；标定或建图输入变化会自动重算。自动清单、地图、缓存和运行历史统一收在各级 `.umi/`，默认不生成空 `scene.yaml`。

`front.mp4` 从对应相机的原始 INSV 提取前镜头画面，默认 720×720、约 29.97 fps。裁切、取帧、缩放、编码和音轨统一通过 [`trajectory.yaml` 的 `video:`](config/trajectory.yaml) 配置；范围、场景覆盖和联动规则见[视频处理参数](config/README.md#视频处理参数)。

`--dry-run` 只读预览；`run scenes --jobs 2` 可跨场景批处理。默认共享前镜头地图；双路建图等高级配置见[场景指南](docs/场景处理.md)。不同地图版本不保证坐标一致。

## 验收与导出

```bash
./umi-scene review scenes/shelf/20261105_001 --accept --reviewer zijian
./umi-scene export scenes/shelf/20261105_001 --out exports/shelf
# 或导出训练中间包，无需先导出轨迹包
./umi-scene export scenes/shelf/20261105_001 --format training --out datasets/shelf
```

导出只接受当前有效且已人工验收的结果。训练包还要求操作指令、双手同步确认；夹爪宽度与训练就绪条件见[训练数据](docs/训练数据.md)。计算成功不等于已验收，查看器不自动验收。

[文档导航](docs/README.md) · [设备与目录](docs/设备与坐标.md) · [标定指南](docs/标定.md) · [参数管理](config/README.md)
