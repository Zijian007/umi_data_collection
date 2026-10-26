# 模拟商超采集示例

用历史实验室录像演示“场景 → 套件 → 批次 → 任务”的组织与处理流程。现有一台真实相机，仅左手；同一录像用于建图和回放，验证固定坐标与处理流程，不构成商超实拍或跨任务精度验证。

```text
supermarket_demo/
├── scene.yaml
├── maps/                         各套件共享的场景地图
└── kits/demo_kit/
    ├── kit.yaml                 当前设备配置
    └── sessions/20261022_001/
        ├── session.yaml
        └── take_001/
            ├── episode.yaml
            └── results/          当前成功结果入口
```

| 配置 | 当前选择 |
|---|---|
| 地图模式 | `shared_front` |
| 地图版本 | `f4991ea43a3a-65d192ef` |
| 套件 | `demo_kit`，左手 |
| 相机 | `IAQEB26097M4CP` |
| 前镜头标定 | `839fd277d850` |
| 夹爪几何 | `handheld_v1 / b29b429c0fcc` |

从项目根目录执行：

```bash
./umi-scene kits scenes/supermarket_demo
./umi-scene inspect scenes/supermarket_demo
./umi-scene status scenes/supermarket_demo
./umi-scene run scenes/supermarket_demo
./umi-view scenes/supermarket_demo
```

配置入口：[场景](scene.yaml)、[批次](kits/demo_kit/sessions/20261022_001/session.yaml)、[任务](kits/demo_kit/sessions/20261022_001/take_001/episode.yaml)。当前任务共 1041 帧、约 34.7 秒，尚未人工验收。

当前套件配置见 [kit.yaml](kits/demo_kit/kit.yaml)。批次自动保存当时的设备与标定快照，修改套件不影响已有批次。任务使用手单独声明，状态文件只记录处理尝试、检查报告路径和人工验收。

展示地图只保存在场景地图版本中，任务通过 `details/summary.json` 的路径与哈希引用共享文件，不再保存地图副本。当前地图清单已去重，空导出日志已清理。

原片位置：

- 标定：`recordings/calibration/IAQEB26097M4CP/20261022/front/`。
- 任务：`recordings/scenes/supermarket_demo/demo_kit/20261022_001/IAQEB26097M4CP/`。

原片、标定、当前地图与当前轨迹结果保留，旧配置备份、迁移记录和旧实验结果已清理。当前流程见 [场景指南](../../docs/场景.md)，资产规则见 [设备与目录](../../docs/设备与目录.md)。
