# UMI 相机标定与轨迹采集

从 Insta360 X3 原始 INSV 视频生成夹爪 TCP 轨迹，供查看、人工验收和数据导出。

**第一次采集**按[安装与日常使用](docs/01-安装与日常使用.md)完成环境安装、相机登记、套件配置和一次批次处理。已有套件时可从[准备批次和原片](docs/01-安装与日常使用.md#4-准备本次配置和视频)开始。

| 命令 | 用途 |
|---|---|
| `./umi-calib` | 求解和验收相机标定，登记设备参数 |
| `./umi-scene` | 建图、重建任务轨迹、验收和导出 |
| `./umi-view` | 对照原片画面查看轨迹、地图和结果报告 |

## 已有批次

```bash
./umi-scene run scenes/test_shelf/20261120_002 --show-config
./umi-scene run scenes/test_shelf/20261120_002
./umi-view scenes/test_shelf/20261120_002
```

用 `hardware/kits/<套件>/kit.yaml` 配置设备与夹爪，用批次的 `collection.yaml` 选择使用手、地图和轨迹后端。每次操作放在一个 `take_*` 目录中，结果从 `take_*/results/` 查看。后端选项见[场景处理](docs/03-场景处理.md#选择轨迹估计后端)，完整字段见[批次模板](scenes/example_scene/collection.example.yaml)。

查看结果后，使用 `umi-scene review` 记录人工验收，再导出[普通成果](docs/04-结果查看与验收.md#人工验收与导出)或[训练数据](docs/05-训练数据准备与导出.md)。

[文档导航](docs/README.md) · [命令行参考](docs/命令行参数.md) · [开发维护](docs/dev/README.md)
