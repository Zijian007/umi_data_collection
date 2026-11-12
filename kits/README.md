# 设备套件

每个套件的 `kit.yaml` 指定左右相机序列号、相机标定版本和当前安装的夹爪文件，可跨场景复用。仅需手动维护，不需要安装或登记夹爪。

```yaml
arms:
  left:
    serial: LEFT_CAMERA_SERIAL
    calibration: '002'
    gripper:
      tcp: left_tcp_camera.json
      width: left_width.json
  right:
    serial: RIGHT_CAMERA_SERIAL
    calibration: '002'
    gripper:
      tcp: right_tcp_camera.json
      width: right_width.json
```

只保留实际使用的手。相机序列号和标定版本必须已在 registry 中登记，版本号保留引号。夹爪文件路径相对 `kit.yaml`，需位于套件目录内：

| 字段 | 要求 |
|---|---|
| `gripper.tcp` | 必填；相机与 TCP 的安装变换 JSON，平移单位为米 |
| `gripper.width` | 可选；当前安装的像素距离与实际开口毫米数的实测对应关系 JSON |
| `gripper.tag_distance` | 可选；ID 0/1 黑色外框边长 JSON，用于鱼眼三维标签中心间距 |
| `gripper.mask` | 可选；自定义遮手 PNG，省略时使用通用遮手图 |
| `rear_calibration` | 每手可选的正式 rear 标定版本。dual 或海报处理未绑定时尝试 X3 INSV 工厂参数 + 本设备 front B 兜底；显式绑定无效时拒绝。shared_front 不因此增加环境镜头地图定位 |
| `calibrations` | 与单版本字段互斥；按 measured/factory 固定完整镜头版本，见下文 |
| `intrinsics_source` | 可放在套件顶层或每手设备中；指定默认来源，CLI/批次可覆盖 |

左右安装几何相同时，可以共同引用一个 `tcp_camera.json`；不同时分别填写。没有实测开口关系时省略 `width`，不要填写示例数值。TCP 格式与方向见[设备与坐标](../docs/设备与坐标.md#几何资产)，宽度格式与测量方法见[训练数据指南](../docs/训练数据.md#夹爪宽度标定与接入)。

只测两个标签中心的实际距离时，可单独配置 `gripper.tag_distance: tag_distance.json`，无需 `gripper.width`。JSON 格式为 `{"tag_sizes_mm": {"0": 6.0, "1": 6.0}}`；数值是每枚标签黑色外框实测边长，单位 mm，不包括外侧白边，两枚可以不同。`kit_001` 和 `kit_002` 当前按用户给定尺寸使用各 6 mm；其他套件须填写各自尺寸。这与 AprilGrid 标定板尺寸是不同资产，不修改 `hardware/calibration_targets/aprilgrid.json`。

运行使用该相机已选定的工作镜头 KB4 内参，输出 `tag_distance_mm`。尺寸文件随运行快照保存并校验哈希，修改尺寸后重跑会重算三维值、复用原始角点缓存。标签中心间距与夹爪接触面开口分别输出；没有开口标定时不生成开合量或闭合状态。

```text
kits/kit_001/
├── kit.yaml
├── tcp_camera.json
├── left_width.json       # 有实测数据时创建
└── right_width.json      # 有实测数据时创建
```

批次在 `batch.yaml` 中声明 `kit: kit_001`。每次 `run` 读取当前套件，在开始计算前解析测量并复制到运行目录 `devices/<序列号>/`，结果记录实际文件的哈希。修改文件后重跑即可，历史运行保留自己的副本。配置变更适用于下一次运行，无需编辑 `.umi/manifest.yaml`。

`registry/` 仅保存相机档案和版本化标定。套件 ID 来自目录，不重复写入 YAML；套件与夹爪不再另设版本目录或默认指针。`.example.yaml` 不作为真实套件读取。

## 同时保存实测与工厂来源

需要切换时，以 `calibrations` 替代单个 `calibration` 字段。例如（版本号必须换成实际安装命令输出的版本，不代表仓库已有这些测量）：

```yaml
arms:
  right:
    serial: CAMERA_SERIAL
    calibrations:
      measured:
        front: '002'
        rear: rear_MEASURED_HASH
      factory:
        front: '003'
        rear: rear_FACTORY_HASH
    gripper:
      tcp: tcp_camera.json
      width: right_width.json
```

每个来源的 `front` 必填，`rear` 在使用环境镜头时填写。`calibrations` 不能与旧的 `calibration/rear_calibration` 并列。旧格式继续固定原版本；加开关只核对来源，不会自动替换旧字段中的版本，双来源切换应使用上面的 `calibrations` 写法。

选择优先级为 CLI `--intrinsics-source` → `batch.yaml` 的 `intrinsics_source` → 套件顶层 `intrinsics_source` → 每手 `intrinsics_source`；多来源配置均未指定时选 measured。未安装对应来源或镜头版本时明确报错，不会借用另一来源。

批次记录保存解析后的具体版本，每次从 `batch.yaml` 发起 `run` 仍会重新读取当前套件。切换来源会绑定匹配的独立地图/结果版本；旧、新批次记录归档在 `.umi/snapshots/`，已有成果保留运行时副本。地图来源、标定或引擎不匹配时需要重建，不能带着旧地图只换内参。


## 海报采集的完整绑定

先按[标定指南](../docs/标定.md#双镜头标定与海报接入)完成每台相机的 front/rear 验收与登记。下例版本仅为占位，必须替换成安装命令的实际输出：

```yaml
arms:
  right:
    serial: CAMERA_SERIAL
    calibration: '002'
    rear_calibration: rear_ACTUAL_HASH
    gripper:
      tcp: tcp_camera.json
```

front/rear 版本属于同一序列号，内参来源需匹配；多来源使用上面的 `calibrations` 格式。海报布局放在 `scene.yaml` 的 `tags.layout`，不写入 kit 或 registry。新安装 rear 不会自动修改此配置。用 `./umi-calib devices` 查看所有镜头版本，再用 `./umi-calib devices --kit kit_002 --require-rear` 检查所选套件的固定绑定；多来源可加 `--intrinsics-source measured|factory`，机器可读输出加 `--json`。检查只覆盖相机标定资产与绑定，不代表夹爪安装资产或双镜头组合实测已验收。

共享单手回退继承同一套冻结配置；直接运行独立轨迹时则显式传入 `--rear-calibration` 和 `--tag-layout`。更换版本对新运行生效，历史结果仍保留旧资产副本。
