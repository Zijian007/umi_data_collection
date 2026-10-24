# UMI 相机标定与轨迹抽取

从 Insta360 X3 原始 `.insv` 录像生成夹爪 TCP 的位置和姿态。完整顺序是：

**安装 → 每台相机标定并手动启用 → 录制任务 → 抽取轨迹 → 同步查看视频与轨迹。**

## 1. 安装

在 Linux x86_64 上，从项目根目录执行：

```bash
bash install.sh
./umi-traj check
./umi-calib check
```

`umi-calib` 首次使用会补装标定组件。克隆、机器要求、安装日志和 Docker 用法见 [安装指南](docs/安装.md)。

## 2. 标定并启用相机

每台相机录制 A/B/C/D 四段，推荐按完整序列号整理：

```text
recordings/IAQEB26097M4CP/
├── calib_20261022/
│   ├── A/原始文件名.insv
│   ├── B/原始文件名.insv
│   ├── C/原始文件名.insv
│   └── D/原始文件名.insv
└── tasks/20261023/
    └── 原始文件名.insv
```

```bash
./umi-calib run recordings/IAQEB26097M4CP/calib_20261022 --tag-size-mm 24.0
# 查看 umi_calib_out/report.txt，验收通过后手动启用：
./umi-calib install recordings/IAQEB26097M4CP/calib_20261022/umi_calib_out
```

以上序列号、日期和板尺寸按实际替换。四段怎么录、报告怎么看，见 [标定指南](docs/标定/说明.md)。设备档案和目录规则见 [设备与目录](docs/设备与目录.md)。

## 3. 录制并抽取任务轨迹

相机使用 **360 视频、4K60**，锁定曝光，快门 1/240 秒或更快。每段按“静止 3–5 秒 → 环顾工作区并带平移 10–20 秒 → 正常动作 → 结尾静止”录制，保留原始 `.insv`。详见 [任务录制](docs/轨迹/录制.md)。

```bash
./umi-traj recordings/IAQEB26097M4CP/tasks/20261023
```

工具按录像内部序列号选择已启用标定。每段任务各自建图、定位和融合，标定结果不提供工作区地图。更多选项和排错见 [轨迹抽取](docs/轨迹/使用.md)。

## 4. 查看结果

结果默认在输入目录旁的 `umi_traj_out/<录像名>/`。先读 `report.txt`，使用 `traj_tcp.txt`，用 `trajectory.png` 预览。

```bash
UMI_VIEW_PYTHON=.deps/env/bin/python ./umi-view
```

浏览器打开 **http://127.0.0.1:9091**，同步查看三维轨迹、姿态、前镜头视频和速度。视频是保留鱼眼畸变的裁切，默认 720×720，可通过 [视频配置](config/README.md#图像规格与轨迹分辨率) 的 `video.size_px` 调整；每段录像的世界坐标系独立。TCP 几何来自 CAD 名义值，任务报告中的地图覆盖比例不代表实际误差。

操作见 [三维查看](docs/查看.md)，文件和字段见 [轨迹结果](docs/轨迹/结果.md)，精度证据见 [原理与验证](docs/轨迹/原理.md)。

完整入口：[文档导航](docs/README.md) · [代码维护](docs/dev/README.md)。

算法参数、配置覆盖、只读预检和重新验收见 [参数管理](config/README.md)。
