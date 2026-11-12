# 场景与采集

先在真实场景放好视频和 `batch.yaml`，再执行 `./umi-scene run scenes/<场景>/<批次>`：

```text
scenes/shelf/
├── scene.yaml                     # 可选：同类环境的处理参数
└── 20261105_001/                   # 每个批次独立持有地图
    ├── batch.yaml                 # 用户填写：kit: kit_001
    ├── mapping/原始建图文件名.insv
    ├── .umi/                      # 本批次唯一的内部存储
    │   ├── manifest.yaml          # 设备、原片和地图快照
    │   ├── import.json            # 最近一次有变化的导入报告
    │   ├── maps/                  # 本批次的版本、当前指针与建图缓存
    │   └── tasks/take_001/        # 任务原片清单、状态、缓存和运行历史
    └── take_001/
        ├── task.yaml              # 可选：指令、使用手和同步
        ├── left/                  # 左手操作原片
        ├── right/                 # 右手操作原片；单手只放对应一侧
        └── results                # 指向本批次 .umi/tasks/take_001/runs/ 的成功结果
```

每轮“建图＋几次操作”一个批次；各批次有自己的地图版本、当前指针和缓存，不自动引用其他批次的最新地图。明确配置 `map_version` 时，将指定版本冻结到本批次。换标定或替换建图原片可直接重跑当前批次，新采集轮次也可以建立下一批次。历史结果保留当时的设备和地图。单手套件只放对应侧；双手套件本批次只用一手时在 `batch.yaml` 明确写 `arms: [left]` 或 `[right]`。

预先分好的 take 保留名称与分组，本地原片直接引用。未分组原片可放批次 `pending/`，只有唯一配对后才归入 take；缺片或歧义会报告并保留。标定原片仍放 `recordings/calibration/`，套件放全局 `kits/`。

`example_scene/` 的 `.example.yaml` 是可复制的示例，不作为真实数据运行。复制 [batch.example.yaml](example_scene/20261104_001/batch.example.yaml) 为真实批次的 `batch.yaml` 后填写套件。

## 场景配置模板

新建场景可复制带注释的 [scene.example.yaml](example_scene/scene.example.yaml) 到场景根目录，命名为 `scene.yaml`，例如 `scenes/test_shelf/scene.yaml`。模板中的默认配置可直接使用；也允许完全省略此文件。

```yaml
mode: shared_front

# 如需调参，直接在同一文件填写，仅列改动项：
trajectory:
  video:
    size_px: 720
parameters:
  minimum_localization_fraction: 0.5
```

`shared_front` 支持左手、右手和双手采集；`dual` 表示工作/环境镜头双路处理，需要双路建图原片，优先使用套件绑定的正式环境镜头标定；未绑定时尝试 X3 INSV 工厂参数 + 本设备 front B 兜底，标记未经 rear 独立验收（见[标定指南](../docs/标定.md#双镜头标定与海报接入)）。场景名称由目录名确定；版本字段可省略。

标定来源由 `batch.yaml` 的 `intrinsics_source: measured|factory` 或 `build/run --intrinsics-source ...` 选择；套件须先绑定对应的完整版本，详见[双来源套件配置](../kits/README.md#同时保存实测与工厂来源)。这不属于 `scene.yaml` 的算法参数。

`trajectory` 对应 `config/trajectory.yaml`，`parameters` 对应 `config/scene_parameters.yaml`。两者直接填写映射，不用再创建或引用覆盖文件。新场景配置只管理选项；建图原片归批次 `mapping/`，设备版本归套件。

全部可选字段、单位和生效范围见 [参数说明](../config/README.md)。套件仍填在批次 `batch.yaml`，设备和标定版本仍填在 `kits/<套件>/kit.yaml`。

完整操作见[日常使用](../docs/安装与日常使用.md)。

日常只需关注配置、原片和 `results/`。无变化的导入不写新报告，相同输入和原因的失败记录去重；`--dry-run` 保持只读。`.umi/` 集中布局之前的旧批次不再自动迁移内部文件；需要继续处理时，请新建批次并放入原片重跑。指令与同步可通过任务目录中的 `task.yaml` 补充。

## 配置与命令迁移

kit.yaml、batch.yaml 手动填写；任务按 take 放好视频，由 `run <批次>` 自动登记和处理。补片后重跑；任务指令和同步填写在任务目录的 `task.yaml` 中。`build <批次>` 仅单独建图，和 run 读取同一份 batch.yaml。

历史 schema 2 的 scene.yaml 和原有结果仍可读取。新配置使用默认 schema 3，不再保存 mapping/devices，也不再引用参数文件；将旧参数文件中的改动项分别移入 trajectory、parameters 即可，不要将旧的场景建图元数据复制到新配置。
