# 设备套件

每个套件的 `kit.yaml` 指定左右相机序列号、前后镜头的内参来源和当前安装的夹爪文件，可跨场景复用。仅需手动维护，不需要安装或登记夹爪。

```yaml
arms:
  left:
    serial: LEFT_CAMERA_SERIAL
    intrinsics:
      front: measured
      rear: factory
    gripper:
      tcp: left_tcp_camera.json
      width: left_width.json
  right:
    serial: RIGHT_CAMERA_SERIAL
    intrinsics:
      front: measured
      rear: factory
    gripper:
      tcp: right_tcp_camera.json
      width: right_width.json
```

只保留实际使用的手。工作镜头参数必须已按相机序列号登记；kit 只选来源，不填写版本号。夹爪文件路径相对 `kit.yaml`，需位于套件目录内：

| 字段 | 要求 |
|---|---|
| `gripper.tcp` | 必填；相机与 TCP 的安装变换 JSON，平移单位为米 |
| `gripper.width` | 可选；当前安装的像素距离与实际开口毫米数的实测对应关系 JSON |
| `gripper.tag_distance` | 可选；ID 0/1 黑色外框边长 JSON，用于鱼眼三维标签中心间距 |
| `gripper.mask` | 可选；自定义遮手 PNG，省略时使用通用遮手图 |
| `intrinsics.front` | 工作镜头来源：`measured`（默认）或 `factory`；选择该设备此来源的当前完整参数 |
| `intrinsics.rear` | 环境镜头来源：`factory`（默认）或 `measured`；与 front 独立选择 |

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

批次在 `batch.yaml` 中声明 `kit: kit_001`。每次 `run` 读取当前套件，在开始计算前解析测量并复制到运行目录 `devices/<序列号>/`，结果记录实际文件的哈希。修改文件后重跑即可，历史运行保留自己的副本。配置变更适用于下一次运行，无需编辑 `.pipeline/manifest.yaml`。

`registry/` 管理相机档案及每个镜头、每种来源的当前参数，历史资产只供追溯。套件 ID 来自目录，不重复写入 YAML；套件与夹爪不再另设版本目录或默认指针。`.example.yaml` 不作为真实套件读取。

## 前后镜头独立选择来源

`measured` 使用已登记的实测内参及其配套相机–IMU 外参和时差；`factory` 使用工厂内参来源。可以组合 `front: measured`、`rear: factory`，也可以各自选择已准备好的另一来源。省略 `intrinsics` 时默认 front measured、rear factory。

| 选择 | 参数如何取得 |
|---|---|
| `front: measured` | 当前工作镜头实测整包，需先标定并 `install` |
| `front: factory` | 当前工作镜头工厂内参整包，仍须用该内参重新求解 B 并通过 C/D；需先以 factory 模式标定并 `install` |
| `rear: measured` | 当前已独立验收的环境镜头实测参数，需 `install <rear结果目录或rear_calibration.json>`；缺少时报错 |
| `rear: factory` | 优先使用当前已独立验收的工厂来源 rear；尚未登记时，使用原片工厂内参/双镜头几何与本设备 front B 推导 |

rear 工厂推导沿用 front 时差，标记 `NOT_EVALUATED`；它不等于通过 rear A/B/C/D。当前选择存在但损坏、身份或来源不符时直接报错，不会改用推导。只有实际需要环境镜头时才执行原片推导；`shared_front` 不因此增加环境镜头地图定位。

前后镜头都能选来源，不代表可以单独替换完整参数包中的内参：正式 factory 标定仍须使用与其配套的 B。详见[标定指南](../docs/标定.md#选择实测或工厂内参)。

## 更新当前参数与历史结果

通过统一的 `umi-calib install <结果目录>` 登记 front 或 rear 后，自动更新该相机、该镜头、该来源的当前选择。每个组合只有一份当前参数；更新一项不会改动另一镜头或另一来源。无需再编辑 kit 的版本号或执行版本激活命令。

每次从 `batch.yaml` 发起 run 都重新解析当前参数，计算开始后固定本次选择。参数变化时重新核对地图与结果复用条件；需要时自动重建。历史地图和运行结果保留自己的参数副本、哈希与内部身份，已有记录不会被新安装改写。registry 只保留每个镜头、每种来源的当前安装；被替代的固定版本目录不再作为日常配置项或历史读取入口。

旧 kit 的 `calibration` / `rear_calibration` / `calibrations` 写法仅在所指登记目录仍存在时兼容读取，便于迁移；新写法不能与这些字段混用。将旧 kit 改成 `intrinsics` 后，后续运行才跟随当前参数。项目内的 kit_001、kit_002 已迁移。

## 临时覆盖与检查

通常只在 kit 中分别选择来源。CLI 的 `--intrinsics-source` 可临时把两镜头统一为同一来源；它覆盖 batch.yaml 的同名选择，后者覆盖 kit。旧 kit 保留原来的固定版本核对规则，不会通过覆盖开关自动换成另一版本。

```bash
./umi-calib devices
./umi-calib devices --kit kit_002
./umi-calib devices --kit kit_002 --require-rear --json
./umi-scene run scenes/test_shelf/20261109_002 --show-config
```

`devices` 默认展示每个镜头、每种来源的当前参数，不展开历史版本。`--require-rear` 要求已独立验收的 rear，工厂推导不会使该检查通过；普通查询会标明“需原片推导、未独立验收”。JSON 保留实际参数身份供排障。检查只覆盖相机参数和选择，不代表夹爪安装资产或双镜头组合实测已验收。

## 海报采集

需要独立验收的环境镜头时，按[双镜头标定](../docs/标定.md#双镜头标定与海报接入)完成 rear 求解与登记，再将 `intrinsics.rear` 设为对应来源。海报布局写在 `scene.yaml` 的 `tags.layout`，不写入 kit 或 registry。共享单手回退继续使用本次冻结的相机参数与布局。
