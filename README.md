# UMI 相机标定与轨迹采集

从 Insta360 X3 原始 `.insv` 录像生成夹爪 TCP 位姿。场景采集的工作顺序是：

**安装 → 各设备标定并手动登记 → 场景建图 → 建立采集批次 → 导入任务、同步 → 抽取与查看。**

场景地图由多个批次、多个任务复用。设备档案全局管理，批次固定地图、标定和夹爪版本，同一设备可以在多个场景交错采集。[模拟商超示例](scenes/supermarket_demo/README.md) 已将现有录像迁入这套结构。

## 1. 安装

Linux x86_64，从项目根目录执行：

```bash
bash install.sh
./umi-traj check
./umi-calib check
```

机器要求与 Docker 用法见 [安装指南](docs/安装.md)。

## 2. 每台设备标定并手动登记

```text
recordings/calibration/<完整序列号>/<标定批次>/front/
├── A/原始文件名.insv
├── B/原始文件名.insv
├── C/原始文件名.insv
└── D/原始文件名.insv
```

```bash
./umi-calib run recordings/calibration/IAQEB26097M4CP/20261022/front --tag-size-mm 24.0
# 检查 umi_calib_out/report.txt 后手动登记：
./umi-calib install recordings/calibration/IAQEB26097M4CP/20261022/front/umi_calib_out
```

日期、序列号和板尺寸按实际替换。安装新标定保留旧版本。四段录法见 [标定指南](docs/标定/说明.md)，版本与目录规则见 [设备与目录](docs/设备与目录.md)。

## 3. 场景建图、分批采集

相机使用 **360 视频、4K60**，锁定曝光，快门 1/240 秒或更快；保留原始 INSV。建图需覆盖工作区并包含有效平移、加速度变化和旋转。具体录法见 [任务录制](docs/轨迹/录制.md)。

```bash
./umi-scene init scenes/supermarket_shelf \
  --mapping-session 20261025_001 --mapping-left /media/sd/VID_mapping.insv
./umi-scene build scenes/supermarket_shelf

./umi-scene session scenes/supermarket_shelf --name 20261025_001
./umi-scene episode scenes/supermarket_shelf/sessions/20261025_001 \
  --name take_001 --left /media/sd/VID_task.insv

./umi-scene run scenes/supermarket_shelf/sessions/20261025_001
./umi-view scenes/supermarket_shelf
```

导入命令读取实际序列号，保留文件名并校验复制；SD 卡原片保留。默认 `shared_front` 使用一张共享前地图；`dual` 支持左右前地图加共享后地图，需要各相机前后镜头实测标定。双手任务还需显式确认时间偏移。完整操作见 [场景与双手任务](docs/场景.md)。

## 4. 查看与使用结果

任务当前成功结果在 `scenes/<场景>/sessions/<批次>/episodes/<任务>/results/`。先读 `report.txt`，各手 `traj_tcp.txt` 为固定场景系中的 TCP 位姿，双手任务另有 `bimanual.csv`。

```bash
./umi-view
```

浏览器打开 **http://127.0.0.1:9091**，默认扫描 `scenes/`，同步查看轨迹、稀疏地图和前镜头视频。默认视频 720×720、保留鱼眼畸变。TCP 几何来自 CAD 名义值，内部一致性指标不能代替独立精度测量。

查看说明见 [三维查看](docs/查看.md)。原有 `umi-traj <录像.insv>` 保留用于单段独立实验，按当前默认标定处理、每段独立建图；操作与算法区别见 [轨迹抽取](docs/轨迹/使用.md) 和 [原理与验证](docs/轨迹/原理.md)。

[文档导航](docs/README.md) · [参数管理](config/README.md) · [代码维护](docs/dev/README.md)
