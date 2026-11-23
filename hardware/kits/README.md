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
      base_tag:
        id: 2
        size_mm: 10.0
        dictionary: DICT_4X4_100
  right:
    serial: RIGHT_CAMERA_SERIAL
    intrinsics:
      front: measured
      rear: factory
    gripper:
      tcp: right_tcp_camera.json
      width: right_width.json
      base_tag:
        id: 3
        size_mm: 10.0
        dictionary: DICT_4X4_100
```

只保留实际使用的手。工作镜头参数须已按相机序列号登记，kit 只填写来源。夹爪文件路径相对 `kit.yaml`，可引用当前套件目录，也可用 `../shared/` 引用共享资产。

| 字段 | 要求 |
|---|---|
| `gripper.tcp` | 必填；相机与 TCP 的安装变换 JSON，平移单位为米 |
| `gripper.width` | 可选；当前安装的像素距离与实际开口毫米数的实测对应关系 JSON |
| `gripper.tag_distance` | 可选；ID 0/1 黑色外框边长 JSON，用于鱼眼三维标签中心间距 |
| `gripper.mask` | 可选；自定义遮手 PNG，省略时使用硬件通用遮手图（hardware/grippers/masks/front_hand_mask.png） |
| `gripper.base_tag` | 可选；poster_vio 用于跨相机约束的固定夹爪基座标签，ID 必须在两手间唯一，尺寸单位为 mm |
| `intrinsics.front` | 工作镜头来源：`measured`（默认）或 `factory`；选择该设备此来源的当前完整参数 |
| `intrinsics.rear` | 环境镜头来源：`factory`（默认）或 `measured`；与 front 独立选择 |

base tag 固定在不随开合运动的夹爪基座上，配置写在 kit 内，不是文件路径。
当前只对 poster_vio 的双手任务生效；单手跳过 base tag 检测和安装估计。双手的
自身和跨相机观测均来自工作镜头 front；仅为在双方共同时间范围内被对方看到的
标签估计并优化安装外参，该标签至少需要 3 个有效自身观测时刻提供安装初值。
只看到自己的标签时不拟合安装；只看到对方而对方看不到自身时，无法仅凭 ID 和
尺寸确定所需安装外参。没有有效跨手观测时不创建安装变量。

同一套件的 base tag ID 必须唯一，且不能与场景海报 ID 冲突；默认海报为 12–88，
示例使用 2/3。已配置宽度或测距时也不能占用 0/1。尺寸为黑框实测边长，白边不计。
安装变换会在任务图内带先验联合细化，修正量和自身/跨相机残差保存在质量证据中。
参数与验收边界见 [poster_vio](../../umi/estimation/poster_vio/README.md)。

左右安装几何相同时，可以共同引用 `../shared/tcp_camera.json`；不同时分别填写。没有实测开口关系时省略 `width`。TCP 格式与方向见[设备与坐标](../../docs/设备与坐标.md#几何资产)，宽度格式与测量方法见[训练数据指南](../../docs/训练数据.md#夹爪宽度标定与接入)。

只测两个标签中心的实际距离时，可单独配置 `gripper.tag_distance: tag_distance.json`，无需 `gripper.width`。JSON 格式为 `{"tag_sizes_mm": {"0": 6.0, "1": 6.0}}`；数值是每枚标签黑色外框实测边长，单位 mm，不包括外侧白边，两枚可以不同。`kit_001` 和 `kit_002` 当前按用户给定尺寸使用各 6 mm；其他套件须填写各自尺寸。这与 AprilGrid 标定板尺寸是不同资产，不修改 `hardware/calibration_targets/aprilgrid.json`。

运行使用该相机已选定的工作镜头 KB4 内参，输出 `tag_distance_mm`。尺寸文件随运行快照保存并校验哈希，修改尺寸后重跑会重算三维值、复用原始角点缓存。标签中心间距与夹爪接触面开口分别输出；没有开口标定时不生成开合量或闭合状态。

```text
hardware/kits/
├── shared/
│   ├── tcp_camera.json
│   └── tag_distance.json
└── kit_001/
    ├── kit.yaml
    ├── left_width.json   # 有实测数据时创建
    └── right_width.json  # 有实测数据时创建
```

批次在 `collection.yaml` 中声明 `kit: kit_001`。每次 `run` 读取当前套件，在开始计算前解析测量并复制到运行目录 `devices/<序列号>/`，结果记录实际文件的哈希。修改文件后重跑即可，历史运行保留自己的副本。配置变更适用于下一次运行，无需编辑 `.pipeline/manifest.yaml`。

`hardware/.registry/` 管理相机档案及每个镜头、每种来源的当前参数，套件 ID 来自目录。`.example.yaml` 可作填写参考，实际配置由 `kit.yaml` 读取。

## 前后镜头独立选择来源

`measured` 使用已登记的实测内参及其配套相机–IMU 外参和时差；`factory` 使用工厂内参来源。可以组合 `front: measured`、`rear: factory`，也可以各自选择已准备好的另一来源。省略 `intrinsics` 时默认 front measured、rear factory。

| 选择 | 参数如何取得 |
|---|---|
| `front: measured` | 当前工作镜头实测整包，需先标定并 `install` |
| `front: factory` | 当前工作镜头工厂内参整包，仍须用该内参重新求解 B 并通过 C/D；需先以 factory 模式标定并 `install` |
| `rear: measured` | 当前已独立验收的环境镜头实测参数，需 `install <rear结果目录或rear_calibration.json>`；缺少时报错 |
| `rear: factory` | 优先使用当前已独立验收的工厂来源 rear；尚未登记时，使用原片工厂内参/双镜头几何与本设备 front B 推导 |

rear 工厂推导沿用 front 时差，标记 `NOT_EVALUATED`；它不等于通过 rear A/B/C/D。当前选择存在但损坏、身份或来源不符时直接报错，不会改用推导。只有实际需要环境镜头时才执行原片推导；`shared_front` 不因此增加环境镜头地图定位。

前后镜头分别选来源。正式 factory 标定须使用与其配套的 B。详见[标定指南](../../docs/标定.md#选择实测或工厂内参)。

## 更新当前参数与历史结果

通过统一的 `umi-calib install <结果目录>` 登记 front 或 rear 后，自动更新该相机、该镜头、该来源的当前选择。每个组合只有一份当前参数；更新一项不会改动另一镜头或另一来源。无需再编辑 kit 的版本号或执行版本激活命令。

每次从批次 `collection.yaml` 发起 run 都重新解析当前参数，计算开始后固定本次选择。参数变化时重新核对地图与结果复用条件；需要时自动重建。历史地图和运行结果保留自己的参数副本、哈希与内部身份，已有记录不会被新安装改写。registry 只保留每个镜头、每种来源的当前安装；被替代的固定版本目录不再作为日常配置项或历史读取入口。

kit 使用 `intrinsics.front/rear` 选择参数来源，运行时读取各镜头当前登记的参数。

## 临时覆盖与检查

通常只在 kit 中分别选择来源。临时把所有 kit 的两个镜头统一切到 factory 时，在 `config/user.yaml` 设置 `features.factory_intrinsics.enabled: true`；省略或 false 时沿用 kit。该开关会影响后续所有批次，缺失或不匹配的登记资产直接报错。需要 measured 时在 kit 中逐镜头选择，并关闭全局 factory 覆盖。

```bash
./umi-calib devices
./umi-calib devices --kit kit_002
./umi-calib devices --kit kit_002 --require-rear --json
./umi-scene run scenes/test_shelf/20261120_002 --show-config
```

`devices` 默认展示每个镜头、每种来源的当前参数，不展开历史版本。`--require-rear` 要求已独立验收的 rear，工厂推导不会使该检查通过；普通查询会标明“需原片推导、未独立验收”。JSON 保留实际参数身份供排障。检查只覆盖相机参数和选择，不代表夹爪安装资产或双镜头组合实测已验收。

## 海报采集

需要独立验收的环境镜头时，按[双镜头标定](../../docs/标定.md#双镜头标定与海报接入)完成 rear 求解与登记，再将 `intrinsics.rear` 设为对应来源。布局填写在 `environment_poster.layout_path`，任务的海报开关写在批次的 `backend` 下。[场景后端配置](../../docs/场景处理.md#选择轨迹估计后端)
