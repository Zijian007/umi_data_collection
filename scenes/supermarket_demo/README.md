# 模拟商超：货架取放采集

这是场景管理与算法流程示例。现有实验室录像被迁入“商超货架取放”的组织结构；视频内容并非商超实拍。当前只有一台真实相机，不构造虚假的右手或同步数据。

| 项目 | 内容 |
|---|---|
| 场景 ID | `supermarket_demo` |
| 模拟业务 | 在货架区域取放商品 |
| 地图模式 | `shared_front` |
| 采集批次 | `20261022_001`，沿用原片日期 |
| 任务 | `take_001`，单手流程验证 |
| 设备 | `IAQEB26097M4CP` |
| 前镜头标定 | `839fd277d850` |
| 夹爪几何 | `handheld_v1 / b29b429c0fcc` |

建图和回放引用同一段历史录像，仅验证目录、资产选择、固定地图和处理流程；不构成跨任务定位或商超环境精度验证。

## 从这里开始

从项目根目录执行：

```bash
./umi-scene inspect scenes/supermarket_demo
./umi-scene run scenes/supermarket_demo/sessions/20261022_001 --dry-run
./umi-scene run scenes/supermarket_demo/sessions/20261022_001
./umi-view scenes/supermarket_demo
```

场景、批次和任务清单分别是 [scene.yaml](scene.yaml)、[session.yaml](sessions/20261022_001/session.yaml)、[episode.yaml](sessions/20261022_001/episodes/take_001/episode.yaml)。当前任务结果入口为 `sessions/20261022_001/episodes/take_001/results/`。

## 迁移位置

```text
recordings/calibration/IAQEB26097M4CP/20261022/front/
    A/、B/、C/、D/、umi_calib_out/

recordings/scenes/supermarket_demo/20261022_001/episodes/take_001/
    IAQEB26097M4CP/VID_20261022_225713_00_005.insv

scenes/supermarket_demo/
    scene.yaml
    maps/
    sessions/20261022_001/
    archive/independent/VID_20261022_225713_00_005/
    migration.json
```

迁移清单保存原路径、新路径和 SHA256；原片、原标定测量、原报告和旧独立轨迹均保留原内容。历史报告中的绝对路径代表当时运行位置，实际新位置以迁移清单为准。

旧独立轨迹在 `archive/independent/`，它的世界坐标仍然独立，不算新场景结果。默认查看器跳过归档；需要对照时可直接指定：

```bash
./umi-view scenes/supermarket_demo/archive/independent
```

新场景结果由迁移后的原片和固定地图重新计算。完整管理规则见 [设备与目录](../../docs/设备与目录.md)，新批次和双手任务操作见 [场景流程](../../docs/场景.md)。
