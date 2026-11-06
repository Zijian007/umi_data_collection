# 场景与采集

先在真实场景放好视频和 `batch.yaml`，再执行 `./umi-scene run scenes/<场景>/<批次>`：

```text
scenes/shelf/
├── scene.yaml                     # 可选；默认不生成空配置
├── .umi/maps/                     # 自动保存的共享地图
└── 20261105_001/
    ├── batch.yaml                 # 用户填写：kit: kit_001
    ├── mapping/原始建图文件名.insv
    ├── .umi/                      # 自动保存的批次快照和导入报告
    └── take_001/
        ├── videos/left/、right/   # 本次操作原片
        ├── results/               # 成功后生成
        └── .umi/                  # 任务清单、缓存、状态和运行记录
```

每轮“建图＋几次操作”一个批次。换标定或替换建图原片可直接重跑当前批次，新采集轮次也可以建立下一批次。历史结果保留当时的设备和地图。单手套件只放对应侧；双手套件本批次只用一手时在 `batch.yaml` 明确写 `arms: [left]` 或 `[right]`。

预先分好的 take 保留名称与分组，本地原片直接引用。未分组原片可放批次 `pending/`，只有唯一配对后才归入 take；缺片或歧义会报告并保留。标定原片仍放 `recordings/calibration/`，套件放全局 `kits/`。

`example_scene/` 的 `.example.yaml` 仅展示输入与生成文件的区别，包含占位符，不作为真实数据运行。复制 [batch.example.yaml](example_scene/20261104_001/batch.example.yaml) 为真实批次的 `batch.yaml` 后填写套件即可。场景配置无需模板。

完整操作见[日常使用](../docs/安装与日常使用.md)。

日常只需关注配置、原片和 `results/`。无变化的导入不写新报告，相同输入和原因的失败记录去重；旧批次首次实际运行时自动迁移内部文件，`--dry-run` 保持只读。指令与同步可通过任务 `.umi/episode.yaml` 补充。
