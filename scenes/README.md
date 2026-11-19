# 场景与采集

先在真实批次放好视频和唯一的 `collection.yaml`，再执行 `./umi-scene run scenes/<场景>/<批次>`：

```text
scenes/shelf/
└── 20261105_001/                   # 每个批次独立持有地图
    ├── collection.yaml            # 用户填写：场景、kit、手和任务
    ├── mapping/原始建图文件名.insv
    ├── .pipeline/                 # 程序维护，无需查看；结果从 take_*/results 进入
    │   ├── manifest.yaml          # 设备、原片和绑定的地图版本
    │   ├── import.json            # 最近一次有变化的导入报告
    │   ├── maps/                  # 本批次的地图版本与建图缓存
    │   ├── map_epochs.yaml        # 可选：多张地图与 take 的采集顺序
    │   └── tasks/take_001/        # 任务原片清单、状态、缓存、当前结果与最近一次尝试
    └── take_001/
        ├── left/                  # 左手操作原片
        ├── right/                 # 右手操作原片；单手只放对应一侧
        └── results                # 指向本批次 .pipeline/tasks/take_001/runs/ 的成功结果
```

每轮“建图＋几次操作”一个批次；各批次有自己的地图版本和缓存，不自动引用其他批次的最新地图。批次内的某个 take 可在 `collection.yaml.tasks.<take>.map.version` 明确配置地图版本，结果会固定记录该版本。换标定或替换建图原片可直接重跑当前批次，新采集轮次也可以建立下一批次。每份结果自带当时的设备和地图副本；重跑成功后，被取代的旧结果会自动删除。单手套件只放对应侧；双手套件本批次只用一手时在 `collection.yaml` 明确写 `recording.arms: [left]` 或 `[right]`。

预先分好的 take 保留名称与分组，本地原片直接引用。未分组原片可放批次 `pending/`，只有唯一配对后才归入 take；缺片或歧义会报告并保留。标定原片仍放 `recordings/calibration/`，套件放全局 `kits/`。

`example_scene/` 的 `.example.yaml` 是可复制的示例，不作为真实数据运行。复制 [collection.example.yaml](example_scene/collection.example.yaml) 为真实批次的 `collection.yaml` 后填写场景、套件和任务。

## 场景配置模板

新建批次可复制带注释的 [collection.example.yaml](example_scene/collection.example.yaml) 到批次目录，命名为 `collection.yaml`，例如 `scenes/test_shelf/20261109_008/collection.yaml`。

```yaml
kit: kit_001
recording:
  arms: [left, right]
scene:
  description: shelf scene
map:
  layout: shared
  reference_arm: right
  selection:
    policy: latest_prior
reconstruction:
  backend: scene_map
```

`shared_front` 支持左手、右手和双手采集；`dual` 表示工作/环境镜头双路处理，需要双路建图原片，优先使用套件绑定的正式环境镜头标定；未绑定时尝试 X3 INSV 工厂参数 + 本设备 front B 兜底，标记未经 rear 独立验收（见[标定指南](../docs/标定.md#双镜头标定与海报接入)）。场景名称由目录名确定；版本字段可省略。

标定来源由根目录 `config/user.yaml` 的 `calibration.intrinsics_source` 选择；套件须先绑定对应的完整版本，详见[双来源套件配置](../kits/README.md#同时保存实测与工厂来源)。

算法参数默认来自 `config/algorithms/trajectory.yaml` 和 `config/algorithms/scene_parameters.yaml`；实验性覆盖只能写入专家文件，通过 `--expert-config experiment.yaml` 传入。新场景配置只管理选项；建图原片归批次 `mapping/`，设备版本归套件。

全部可选字段、单位和生效范围见 [参数说明](../config/README.md)。设备和标定版本填写在 `kits/<套件>/kit.yaml`。

完整操作见[日常使用](../docs/安装与日常使用.md)。

日常只需关注配置、原片和 `results/`。无变化的导入不写新报告，相同输入和原因的失败记录去重；`--dry-run` 保持只读。旧的拆分清单不再读取；需要继续处理时，请新建批次并放入原片重跑。指令与同步写入 `collection.yaml.tasks`。

## 日常流程

手动填写 `collection.yaml` 和 `kit.yaml`；任务按 take 放好视频，由 `run <批次>` 自动登记和处理。补片后重跑；任务指令和同步填写在 `collection.yaml.tasks` 中。`build <批次>` 仅单独建图，和 run 读取同一份 `collection.yaml`。
