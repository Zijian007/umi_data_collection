# 代码维护

[文档导航](../README.md) · [分层结构](#分层结构) · [缓存与复用](#缓存与复用) · [验证](#验证) · [当前待办](#当前待办)

配置和资产的职责见[权威来源](SSOT与权威来源.md)，变换方向、时钟与优化变量见[算法实现约定](算法实现约定.md)。

海报后端的地图融合开关、双板精度与货架回退结果见[地图融合交接](海报后端地图融合交接.md)。

## 从入口沿流程阅读

三个命令入口都是薄壳脚本，执行 `python -m umi.apps.<名称>`；`apps/` 只解析参数并调用对应流程。

| 流程 | 阅读入口 |
|---|---|
| 标定 | [标定代码指南](../../umi/calibration/README.md)：`umi/apps/calib.py` → `calibration/workflow.run()` / `verify()` |
| 场景任务 | [场景代码指南](../../umi/scene/README.md)：`umi/apps/scene.py` → `scene/run.run()` → `scene/run.run_episodes()` 与按需地图绑定 |
| 任务后端 | `scene/run.execute_episode()` → `scene/routing.execute()` → `scene_map` / `poster_independent` / `poster_vio`（`auto` 由 `_AutoRoute` 按顺序尝试） |
| 内部逐段引擎 | [逐段引擎指南](../../umi/reconstruction/segment/README.md)：`umi/apps/segment.py` → `segment/workflow.run_batch()` → `run_recording()` → `attempts.run_attempts()` |
| 无场景地图的 VIO | [poster_vio 代码指南](../../umi/reconstruction/backends/poster_vio/README.md)：`reconstruction/backends/poster_vio/preparation.prepare_task_inputs()` → `reconstruction/backends/poster_vio/solver.solve_joint_poster_imu()` |
| 三维查看 | [查看器代码指南](../../umi/apps/viewer/README.md)：`apps/viewer/server.main()` → `RecordingCatalog` → 浏览器播放状态 |
| 共用视觉 | [视觉代码指南](../../umi/perception/README.md)：录像准备、ORB 建图/定位、海报检测 |
| 相机数据 | [extraction.py](../../umi/devices/x3/extraction.py) 提取时间戳和 IMU；[frame.py](../../umi/core/frame.py) 定义录制几何、裁剪、帧率和时钟约定 |
| 设备选择 | [registry.py](../../umi/devices/registry.py) 读取内部序列号、选择标定与夹爪并核对哈希 |

`workflow.py` 展示顺序、阶段衔接与失败收尾；阶段函数返回具名结果。标定的 `acceptance.py` 决定验收，`result_io.py` 保存和校验记录，`reports.py` 只负责展示；第三方进程、依赖与日志在 `pipeline/runtime.py`，Basalt 适配在 `solve/basalt.py`。场景发布和报告在 `scene/publication.py`，运行审计在 `scene/task_state.py`，阶段缓存在 `umi/store/cache.py`；`umi/perception/decoding.py` 一次解码向多个检测器分发帧。标定的完整数据流、状态、恢复与登记契约见[标定架构与验证](标定架构与验证.md)。

## 分层结构

一次采集任务从 scene/run.py 进入。场景负责批次、地图版本、候选顺序与发布。重建代码集中在 reconstruction/，其中 backends/ 实现三个可选后端，segment/ 提供单段重建，estimation/ 提供共用的地图几何与融合。标定和任务流程通过这些入口获取轨迹。查看器读取已发布的轨迹结果。

`umi/` 按依赖方向分层，上层可以依赖下层，下层不得依赖上层。[`.importlinter`](../../.importlinter) 声明这些层，CI 用 `lint-imports` 检查；类型标注专用导入（`if TYPE_CHECKING:`）不计入。

| 层（上 → 下） | 目录 | 职责 |
|---|---|---|
| apps | [umi/apps](../../umi/apps) | 命令入口 `calib.py`、`scene.py`、`segment.py`、环境检查 `health.py` 和查看器 `viewer/`；只解析参数、调用流程、格式化输出 |
| calibration | [umi/calibration](../../umi/calibration) | 标定与验收流程 |
| scene | [umi/scene](../../umi/scene) | 批次准备、地图、任务后端、轨迹发布与固定夹具评估 |
| backends | [umi/reconstruction/backends](../../umi/reconstruction/backends) | scene_map、poster_independent、poster_vio 的观测到轨迹流程 |
| segment | [umi/reconstruction/segment](../../umi/reconstruction/segment) | 内部逐段重建引擎与融合工作进程 |
| estimation | [umi/reconstruction/estimation](../../umi/reconstruction/estimation) | 数值算法：`fusion/`（独立视觉惯性融合）、`fixed_map/`（固定地图融合与海报尺度）、`poster_factors.py`（共用海报因子）、`inertial.py`、地图米制化 `mapping.py` |
| store | [umi/store](../../umi/store) | 批次磁盘布局、阶段缓存 `StageCache`、结果读取，以及场景数据模型 `models.py` |
| perception | [umi/perception](../../umi/perception) | 解码、录像准备、ORB 建图/定位、海报检测与 PnP、音频同步 `audio/`、夹爪标签 `gripper/` |
| devices | [umi/devices](../../umi/devices) | X3 原片读取 `x3/`、相机登记表、标定资产与来源、轨迹用标定选择 `selection.py` |
| config | [umi/config](../../umi/config) | 配置加载、参数模型、`collection.yaml` 与场景参数 schema、运行与用户设置 |
| infra | [umi/infra](../../umi/infra) | 路径、终端、文件读写（含 TUM 轨迹）、缓存标记、内容哈希、ORB 安装布局、并发工具 |
| core | [umi/core](../../umi/core) | 位姿几何、时间轴、录制帧约定和使用手校验 |

流程之间也有方向：标定 → 场景 → 内部逐段引擎。环境镜头标定复用场景的录像准备与地图能力，场景的 `poster_independent` 复用逐段引擎的重建；反方向的依赖不允许。

安装资源（conda 环境锁、ORB-SLAM3 补丁与构建脚本、Basalt 预编译包）在顶层 [setup/](../../setup)，不属于 Python 包；KIWI 许可与改编说明在 [第三方来源](PROVENANCE.md#kiwi)。

### 新增代码放哪里

- 位姿几何、时钟换算和录制规格等公共约定放 `core/`。通用文件读写放 `infra/`，TUM 轨迹读写入口见 [file_store.py](../../umi/infra/file_store.py)。
- 设备或标定文件的读写放 `devices/`。提取图像和音频观测放 `perception/`，求解与验收数值放 `estimation/`。
- 一个算法需要的参数，在算法旁定义窄参数类（如 `PosterVioParameters`），由 `SceneParameters.for_algorithm()` 投影，算法不直接依赖配置文件结构。
- 场景编排和发布放 `scene/`，标定编排放 `calibration/`，后端专用求解放 `reconstruction/backends/`；命令行参数与输出格式放 `apps/`。
- 跨层引用前先看 `lint-imports` 是否通过；需要反向依赖时，通常说明该代码应下移一层。

### 融合工作进程

独立融合由 [`segment/fusion_runner.py`](../../umi/reconstruction/segment/fusion_runner.py) 执行：重建阶段调用 `execute(FusionRequest, ...)`，它在一次性的 spawn 工作进程里调用 `fuse()`。不用线程是因为 GTSAM 的 Python 绑定在优化期间持有 GIL，线程会让左右手融合串行；工作进程保留双手并行、`--cores` 的 CPU 绑定、nice、崩溃隔离，并把 stdout/stderr 写入 `logs/fuse.log`。输入、输出与失败诊断（`alignment_failure.json`、`poster_failure.json`）仍通过文件交换。

## 缓存与复用

缓存和结果复用**只依据数据输入**：原片内容、所选标定与资产、参数与配置。代码和外部引擎的版本不参与判断，也不需要维护任何源文件清单。

- 场景算法或引擎变化后，用 `umi-scene run --force` 重算本次实际执行的阶段。标定用 `umi-calib run --recompute` 建立新版本。阶段缓存的时间检查见 `umi/infra/file_store.py`。
- 结果、地图清单、逐段引擎设置和标定记录写入 `code_revision`（git 提交号及 `umi/` 是否有未提交改动），只用于追溯。
- 原片、设备资产、配置及结果文件的完整性用 `file_hash()` 的原始字节 SHA256 校验。数组缓存统一用 `file_store.read_npz_cache()` / `write_npz_cache()`：完成标记记录输入身份（按 JSON 规范形式比较）和数组文件哈希。

## 参数管理

见 [统一参数配置](../../config/README.md)。数值默认值写在 `config/*.yaml`，不可变模型位于 `umi/config/parameter_models.py`，加载、校验、覆盖和快照位于 `umi/config/loader.py`。文件内容哈希、规范 JSON 哈希与 git 提交号集中在 `umi/infra/provenance.py`；设备登记、场景和标定共用同一实现。workflow 加载一次，向阶段传递不可变参数。修改参数后同时核对复用、报告与默认数值回归。

## 命名和终端约定

外围代码使用 `request`、`camera_serial`、`keep_work`、`return_code`、`reason`、`error`。数学符号使用对应领域的惯例。场景日常流程统一为手写 kit.yaml/collection.yaml、放置视频，再执行 run；内部自动登记逻辑位于 ingest.py，无独立登记 CLI。历史 JSON 与第三方接口继续按各自协议读取，例如 `serial` / `why`、`subprocess.returncode`。

[terminal.py](../../umi/infra/terminal.py) 统一输出格式，调用为 `runtime.console.write(...)`。阶段负责进度，报告负责结果；标签依次表示工具、录像、步骤、重建次数和执行状态。标定验收 `PASS` / `FAIL` 与执行“完成 / 失败”分别展示。引擎原始日志保留原文。

## 验证

完成[安装](../01-安装与日常使用.md)与标定组件安装后，可直接通过根目录的统一脚本一键测试：

```bash
./test.sh        # 一键测试：主单元测试 (unittest，含宽度映射) + 前端测试 (node)
./test.sh --all  # 包含 CLI check 运行时检查与 KB4 C++ 数值回归
./test.sh --quick  # 快速内循环：数值、schema、时序与缓存契约测试
./test.sh --unit test_segment test_calibration  # 运行两个主题模块
./test.sh --unit test_segment.RearAssetsTests test_calibration.InventoryTests
./test.sh --unit test_configuration.ConfigTests.test_yaml_rejects_duplicate_keys_at_any_level
```

全量和快速档按主题模块并行执行，默认最多 8 个 Python worker。可用
`UMI_TEST_JOBS=1 ./test.sh --unit` 恢复串行，或设置更适合本机资源的 worker 数量。

也可分项执行各环境原生命令：

```bash
.deps/calib_env/bin/python -m unittest discover -s tests -q
node --test tests/js/viewer.test.mjs
./test.sh --checks
```

Node 只用于查看器前端测试，运行查看器不需要 Node。Python 测试覆盖输入、标定验收、流程、复用、重建协议、产物布局与查看器；涉及求解器和 SLAM 的单元测试使用替代引擎。真实录像端到端回归和独立误差评估需另外记录输入、设备标定、版本及结果。

`--quick` 从主题模块中选跑数值、配置、时序和接口契约类。`--unit` 默认运行全部 Python 测试，也可按 `test_模块`、`test_模块.测试类` 或 `test_模块.测试类.方法` 选跑。

仓库的轻量 CI 位于 `.github/workflows/quality.yml`：使用 Python、Ruff 和 import-linter 检查源码语法、代码规范与依赖分层，不依赖 GTSAM、OpenCV、SLAM 或真实录像。完整算法回归仍使用本机 `.deps` 环境执行 `./test.sh --quick`、`./test.sh --unit` 和必要的 `./test.sh --all`。

### 黄金回归

单元测试使用替代引擎，不能证明真实录像上的数值不变。改动求解、融合、检测或流程编排时，用 [tests/golden/golden.py](../../tests/golden/golden.py) 在保留批次 `scenes/test_shelf/20261120_002` 上对三个后端做端到端比对：

```bash
.deps/env/bin/python tests/golden/golden.py prepare            # 用硬链接生成 scenes/golden_<后端>/ 三个变体
.deps/env/bin/python tests/golden/golden.py run                # 依次重算，三个后端约 10 分钟
.deps/env/bin/python tests/golden/golden.py snapshot ~/umi_golden/new
.deps/env/bin/python tests/golden/golden.py compare ~/umi_golden/base ~/umi_golden/new
```

流水线是确定性的：同一代码从零运行两次，三个后端的轨迹逐位一致（ORB-SLAM3 使用离线同步模式）。因此默认容差为 0.1 mm / 0.01°，纯重构应得到 0.0；算法改动若有意改变结果，先确认差异再更新基线快照。`scenes/golden_*/` 已被 git 忽略。

复用测试夹具时导入模块或独立 fixture，避免把其他文件的 `TestCase` 类直接导入测试模块顶层：unittest 会再次收集该类。查重应比较源码文件、类和方法；仅按测试 ID 检查会漏掉 `tests.test_x` 与 `test_x` 的别名重复。

## 文档怎么维护

[用户文档导航](../README.md)面向采集和排障。每项现行行为在对应页面维护，其他页面用链接指向该说明。

| 内容 | 维护位置 |
|---|---|
| 首次安装与一次采集的主流程 | [01 安装与日常使用](../01-安装与日常使用.md) |
| 标定录制、求解、验收和登记 | [02 相机与夹爪标定](../02-相机与夹爪标定.md) |
| 场景/批次选项、配对、地图复用和排障 | [03 场景处理](../03-场景处理.md) |
| 查看器和结果字段 | [04 结果查看](../04-结果查看.md) |
| 任务指令、双手同步与夹爪宽度 | [05 任务同步与夹爪测量](../05-任务同步与夹爪测量.md) |
| 资产目录、版本、坐标约定 | [设备与坐标](../principles/设备与坐标.md) |
| 算法过程与当前验证边界 | [算法原理](../principles/算法原理.md) |
| 内部模块职责和引擎输出 | 对应 umi 模块 README |
| 第三方代码、预编译程序与改编来源 | PROVENANCE.md |
| 改动过程与验证证据 | git 提交说明（不另存文档） |

配置样例只在主流程保留必要示例，完整字段定义链接到对应参考。移动章节时同步本地链接与锚点；检查 shell 示例与当前 CLI 的参数一致。源码或功能变更后更新说明，避免在多个操作页面分别追加同一条兼容提示。

## 当前待办

- 扩展真实录像回归的录像覆盖，并记录标定 ID、引擎版本、耗时及重复运行波动。
- 独立测量相机到 TCP 的几何，验证目前使用的 CAD 名义值。
- 为需要输出夹爪开口的套件补做像素到毫米的宽度标定。
- 对 `poster_vio` 做实拍精度和资源基准验证，再评估动态物体筛除、滑窗边缘化和 GPU 加速。
- 明确仓库许可证及对外分发说明。

实时预览、任务卡和上传属于后续范围。软件回归结果和现场精度证据分别记录。
