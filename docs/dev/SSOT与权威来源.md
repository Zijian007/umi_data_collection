# SSOT 与权威来源

本项目按“每类事实一个权威来源”组织数据。`.pipeline/` 中的文件用于冻结快照、记录运行状态、保存结果和缓存；它们不能反过来成为新的用户配置入口。

## 权威来源表

| 数据 | 唯一权威来源 | 其他文件的角色 |
|---|---|---|
| 场景、批次和任务的用户声明 | 批次根目录 `collection.yaml` | `.pipeline/manifest.yaml` 是解析后的批次快照；`episode.yaml` 只登记原片身份 |
| 相机、套件、TCP 和夹爪资产 | `hardware/kits/<kit>/kit.yaml` 选择资产，`hardware/.registry/` 保存测量资产 | 运行目录中的 `devices/` 是本次运行冻结的副本 |
| 原始录像内容 | 批次目录中的 `.insv` 文件 | `episode.yaml`、`manifest.yaml` 只保存路径、设备身份和内容哈希 |
| Organizer 原片交接 | 批次根目录 `import_manifest.json`（只追加的导入输入） | UMI 校验路径和 SHA256 后把新条目追加进 `.pipeline/map_epochs.yaml`，已登记条目不可改；它不是第二份用户配置 |
| 本机通用设置 | `config/user.yaml` | 运行报告保存本次实际解析值 |
| 默认算法参数 | `config/algorithms/*.yaml` | 后端数值可在批次和任务中覆盖，`--expert-config` 的显式值优先用于本次运行 |
| 多地图采集顺序与 epoch 规划 | `.pipeline/map_epochs.yaml` | 结果 `data/episode.json` 记录最终选择及选择理由 |
| 当前批次绑定的默认地图 | `.pipeline/manifest.yaml.map` | 地图目录中的 `manifest.json` 是该不可变版本的完整身份和资产清单 |
| 任务最终使用的地图 | `episode_inputs.read_episode()` 解析出的 `map_selection` | 结果文件只保存解析后的不可变证据，不重新推断 |
| 任务运行状态和人工验收 | `.pipeline/tasks/<task>/state.json` | `episode.json` 是一次运行的不可变结果；`runs/` 保存运行现场 |
| 阶段缓存 | 各缓存目录中的 `ready.json` 与资产哈希 | 缓存只能加速计算，不能定义场景、地图或任务语义 |

## 使用手的解析

`kit.yaml.arms` 只定义左右手的设备与资产绑定，不充当当前采集使用手的隐式默认。
`collection.yaml.recording.arms` 必填，声明批次默认使用手；`tasks.<take>.arms`
可覆盖为其非空子集，未填写则继承批次。明确填写 `null`、空列表或批次之外的手均拒绝。

使用手校验与继承由 `umi/core/arms.py` 统一处理，入库规划和任务读取共用
`resolve_arms()`。下游消费 `Episode.expected_arms`，根据其数量判断单手/双手；
实际录像只用于检查完整性，不反过来修改声明。

新批次快照显式保存解析后的 `expected_arms`；旧快照省略该字段时，统一从其中冻结的
kit 绑定恢复默认，不查询当前 kit。任务 `episode.yaml` 仍只登记原片，不保存使用手配置。

## 视觉标签术语

各类标签按用途选择字典、尺寸与 ID。测距、基座安装、环境约束和相机标定使用各自的输入资产。

| 名称 | 载体与镜头 | 字典和 ID | 权威配置 | 代码与产物 |
|---|---|---|---|---|
| 夹爪标签 | 夹爪上的两枚小标签，工作镜头 `front` 观察 | `DICT_4X4_50`，固定 ID `0/1` | `hardware/kits/<kit>/kit.yaml` 的 `gripper.tag_distance`；开口宽度另用 `gripper.width` | `umi/perception/gripper/`；`tag_distance_*`、`gripper_width` |
| 夹爪基座标签 | 刚性安装在夹爪基座，供双手 `poster_vio` 使用 | `DICT_4X4_100`，各手 ID 唯一，避开环境海报 ID。配置宽度或测距时避开 `0/1` | kit 的 `gripper.base_tag` 声明资产，`backend.poster_vio.base_tag.enabled` 控制使用 | `umi/estimation/poster_vio/` |
| 环境标签 / 环境海报 | 场地中的刚性海报，环境镜头 `rear` 观察 | `DICT_4X4_100`，布局默认 ID `12–88` | `environment_poster` 声明布局和建图用途，`backend` 声明任务中的使用方式 | `umi/perception/poster/detection.py`、`umi/perception/poster/geometry.py` |
| 标定板标签 | 相机标定用 AprilGrid，不属于夹爪或环境海报 | `tag36h11` 等由标定板资产声明 | `hardware/calibration_targets/aprilgrid.json` 与 `umi-calib --tag-size-mm` | `umi/workflows/calibration/`；只服务相机标定和轨迹初始化 |

特别约束：`gripper.tag_distance` 的 ID 0/1 尺寸不能填环境海报的 100 mm，也不能填 AprilGrid 的 `--tag-size-mm`；环境海报布局不能作为夹爪测距资产；标定板尺寸不能参与任务标签融合。

## 当前配置与运行快照

kit 声明设备序列号、逐镜头 `intrinsics` 来源和 `gripper` 资产。海报用途和融合方式由 `collection.yaml` 声明。海报后端自动开启所需检测与融合，`scene_map` 的海报约束由 `backend.scene_map.poster_constraint` 单独控制。旧结果按冻结的专家覆盖读取。

`--show-config` 指定 take 时展示该 take 的使用手、后端与同步配置；来源报告区分 user.yaml 和 CLI。

地图规划快照由 Organizer 交接或 `run` 按录像时间戳登记，不是手工配置文件。

## 地图选择规则

地图选择只有一条解析路径：

```text
任务级 `map.version`
    > 任务级 `map.epoch`
    > map_epochs.yaml 按 latest_prior 选择
    > 批次 manifest.yaml 中已绑定的默认地图
```

使用手、后端、同步与地图决策由 `umi.workflows.scene.task_configuration.resolve_task_configuration()` 统一解析；正式读取和配置展示消费同一结果。地图决策保存到 `Episode.map_selection`，展示包含选中版本、epoch、原因和 ready/pending/unavailable 状态。未构建或无效选择可预览，但 scene_map 正式执行会拒绝，auto 则按既定回退策略处理。后续定位、报告和发布都消费这个结果，不在各阶段重新解释 `collection.yaml`、`map_epochs.yaml` 或批次地图指针。

`collection.yaml` 中的 `map.selection.version` 是用户声明；`.pipeline/manifest.yaml.map` 是已经发布并冻结的批次状态；地图版本目录中的 `manifest.json` 是地图资产本身的权威身份。三者不能互相覆盖：声明变化需要重新解析和校验，地图资产变化必须拒绝运行。

## 任务配置规则

任务人工选项只写在：

```yaml
collection.yaml:
  tasks:
    take_001:
      instruction: ...
      arms: [left, right]
      synchronization: assumed_simultaneous
```

`.pipeline/tasks/<task>/episode.yaml` 只记录自动发现的原片、序列号、路径和 SHA256，不保存另一份 `instruction`、`arms` 或 `synchronization`。读取任务时由 `layout.task_configuration()` 合并批次声明。

当前批次没有 `tasks.<take>` 时，任务会继承批次默认值。这是有意的默认规则；如果某个 take 有独立指令、同步方式或地图选择，应显式写入 `collection.yaml.tasks.<take>`。

## 快照、结果和缓存的边界

- 快照用于回答“本次运行当时解析到了什么”，不能替代当前用户配置。
- 结果用于回答“这次运行产出了什么”，不能作为下一次运行的输入配置。
- 状态用于回答“当前是否成功、是否已验收”，不能承载轨迹或地图语义。
- 缓存用于回答“某个阶段是否可以复用”，不能决定任务使用哪张地图。
- 报告用于展示和审计，不能成为算法阶段的第二个输入协议。

因此，手工修改 `.pipeline/manifest.yaml`、`episode.yaml`、`state.json`、结果文件或缓存清单，都不应被当作修改批次事实的方式；应修改 `collection.yaml`、`kit.yaml` 或对应的公共配置，然后重新运行。

