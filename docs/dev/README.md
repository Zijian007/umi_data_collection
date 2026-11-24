# 代码维护

[文档导航](../README.md) · [分层结构](#分层结构) · [缓存与复用](#缓存与复用) · [验证](#验证) · [当前待办](#当前待办)

配置和资产的职责见[权威来源](SSOT与权威来源.md)，变换方向、时钟与优化变量见[算法实现约定](算法实现约定.md)。

## 从入口沿流程阅读

三个命令入口都是薄壳脚本，执行 `python -m umi.apps.<名称>`；`apps/` 只解析参数并调用 `workflows/`。

| 流程 | 阅读入口 |
|---|---|
| 标定 | [标定代码指南](../../umi/workflows/calibration/README.md)：`umi/apps/calib.py` → `workflows/calibration/workflow.run()` / `verify()` |
| 共享场景 | [场景代码指南](../../umi/workflows/scene/README.md)：`umi/apps/scene.py` 的 `COMMANDS` → `scene/batch_processing.run()` / `build()` → `scene/maps/repository.build_scene()` / `scene/workflow.run_episode()` |
| 任务后端 | `scene/workflow.execute_episode()` → `scene/tasks/backend_routing.execute()` → `scene_map` / `poster_independent` / `poster_vio`（`auto` 由 `_AutoRoute` 按顺序尝试） |
| 内部逐段引擎 | [逐段引擎指南](../../umi/workflows/segment/README.md)：`umi/apps/segment.py` → `segment/workflow.run_batch()` → `run_recording()` → `reconstruction.run_attempts()` |
| 无场景地图的 VIO | [poster_vio 代码指南](../../umi/estimation/poster_vio/README.md)：`scene/tasks/poster_vio_inputs.prepare_task_inputs()` → `estimation/poster_vio/solver.solve_joint_poster_imu()` |
| 三维查看 | [查看器代码指南](../../umi/apps/viewer/README.md)：`apps/viewer/server.main()` → `RecordingCatalog` → 浏览器播放状态 |
| 共用视觉 | [视觉代码指南](../../umi/perception/README.md)：录像准备、ORB 建图/定位、海报检测 |
| 相机数据 | [extraction.py](../../umi/devices/x3/extraction.py) 提取时间戳和 IMU；[frame.py](../../umi/core/frame.py) 定义录制几何、裁剪、帧率和时钟约定 |
| 设备选择 | [registry.py](../../umi/devices/registry.py) 读取内部序列号、选择标定与夹爪并核对哈希 |
| 训练导出 | [数据集代码指南](../../umi/dataset/README.md) |

`workflow.py` 展示顺序、阶段衔接与失败收尾；阶段函数返回具名结果。标定的 `acceptance.py` 决定验收，`result_io.py` 保存和校验记录，`reports.py` 只负责展示；第三方进程、依赖与日志在 `pipeline/runtime.py`，Basalt 适配在 `solve/basalt.py`。场景发布、报告和审计在 `scene/tasks/publication.py`，阶段缓存在 `umi/store/cache.py`；`umi/perception/decoding.py` 一次解码向多个检测器分发帧。标定的完整数据流、状态、恢复与登记契约见[标定架构与验证](标定架构与验证.md)。

## 分层结构

`umi/` 按依赖方向分层，上层可以依赖下层，下层不得依赖上层。[`.importlinter`](../../.importlinter) 声明这些层，CI 用 `lint-imports` 检查；类型标注专用导入（`if TYPE_CHECKING:`）不计入。

| 层（上 → 下） | 目录 | 职责 |
|---|---|---|
| apps | [umi/apps](../../umi/apps) | 命令入口 `calib.py`、`scene.py`、`segment.py`、环境检查 `health.py` 和查看器 `viewer/`；只解析参数、调用流程、格式化输出 |
| dataset | [umi/dataset](../../umi/dataset) | 已验收结果 → 训练中间包；固定夹具相对位姿评估 |
| workflows | [umi/workflows](../../umi/workflows) | 流程编排：`calibration/`、`scene/`（批次、地图、任务后端适配与发布）、`segment/`（内部逐段引擎与融合工作进程） |
| estimation | [umi/estimation](../../umi/estimation) | 数值算法：`fusion/`（独立视觉惯性融合）、`fixed_map/`（固定地图融合与海报尺度）、`poster_vio/`（任务内联合图）、`factors/`（共用海报因子）、`inertial.py`、地图米制化 `mapping.py` |
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
- 编排顺序、失败收尾、发布与报告放 `workflows/`；命令行参数与输出格式放 `apps/`。
- 跨层引用前先看 `lint-imports` 是否通过；需要反向依赖时，通常说明该代码应下移一层。

### 融合工作进程

独立融合由 [`segment/fusion_runner.py`](../../umi/workflows/segment/fusion_runner.py) 执行：重建阶段调用 `execute(FusionRequest, ...)`，它在一次性的 spawn 工作进程里调用 `fuse()`。不用线程是因为 GTSAM 的 Python 绑定在优化期间持有 GIL，线程会让左右手融合串行；工作进程保留双手并行、`--cores` 的 CPU 绑定、nice、崩溃隔离，并把 stdout/stderr 写入 `logs/fuse.log`。输入、输出与失败诊断（`alignment_failure.json`、`poster_failure.json`）仍通过文件交换。

## 缓存与复用

缓存和结果复用**只依据数据输入**：原片内容、所选标定与资产、参数与配置。代码和外部引擎的版本不参与判断，也不需要维护任何源文件清单。

- 改了算法、重编 ORB-SLAM3/Basalt 或升级依赖后，用 `--force` 重算。`umi-scene run --force` 让本次运行开始前完成的缓存全部失效（解码、定位、检测、任务结果），本次运行中新生成的仍可复用；已绑定的地图不变，连地图一起重建用 `build --force`。实现见 `umi/infra/file_store.py` 的 `recompute()` / `is_current()`。
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
./test.sh --unit test_rear_assets test_calibration_inventory  # 日常改动：只跑相关模块
./test.sh --unit test_config.ConfigTests.test_c_isolates_task_poster_and_dual_mode_but_keeps_solver_parameters
```

无指定测试目标时，`test.sh` 会按测试模块并行启动 8 个 Python worker。可用
`UMI_TEST_JOBS=1 ./test.sh --unit` 恢复串行，或设置更适合本机资源的 worker 数量。指定模块或单个用例时仍只执行指定目标。

也可分项执行各环境原生命令：

```bash
.deps/calib_env/bin/python -m unittest discover -s tests -q
node --test tests/js/viewer.test.mjs
./umi-calib check
./umi-scene check
```

Node 只用于查看器前端测试，运行查看器不需要 Node。Python 测试覆盖输入、标定验收、流程、复用、重建协议、产物布局与查看器；涉及求解器和 SLAM 的单元测试使用替代引擎。真实录像端到端回归和独立误差评估需另外记录输入、设备标定、版本及结果。

`--quick` 用于日常内循环，跳过建图、任务、场景和查看器等端到端工作流；`--unit` 不带模块时仍运行全部 Python 测试。`--unit` 带模块时接受 `test_模块` 或 `test_模块.测试类.方法`，从 `tests/` 解析。定位和迭代时运行快速档或相关模块，涉及公共配置、登记表或融合接口的改动收尾再跑全量。选跑不会自动推断依赖，也不会替代最后的跨模块回归。

仓库的轻量 CI 位于 `.github/workflows/quality.yml`：只安装 NumPy、Ruff 和 import-linter，检查全仓 Python 语法、可移植训练读取器边界、`ruff check umi` 与 `lint-imports` 分层合约，不依赖 GTSAM、OpenCV、SLAM 或真实录像。完整算法回归仍使用本机 `.deps` 环境执行 `./test.sh --quick`、`./test.sh --unit` 和必要的 `./test.sh --all`。

### 黄金回归

单元测试使用替代引擎，不能证明真实录像上的数值不变。改动求解、融合、检测或流程编排时，用 [tests/golden/golden.py](../../tests/golden/golden.py) 在保留批次 `scenes/test_shelf/20261120_002` 上对三个后端做端到端比对：

```bash
.deps/env/bin/python tests/golden/golden.py prepare            # 用硬链接生成 scenes/golden_<后端>/ 三个变体
.deps/env/bin/python tests/golden/golden.py run                # 依次 --force 运行，三个后端约 10 分钟
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
| 查看器、人工验收、成果包和结果字段 | [04 结果查看与验收](../04-结果查看与验收.md) |
| 训练输入、宽度测量、数组和就绪条件 | [05 训练数据准备与导出](../05-训练数据准备与导出.md) |
| 资产目录、版本、坐标约定 | [设备与坐标](../principles/设备与坐标.md) |
| 算法过程与当前验证边界 | [算法原理](../principles/算法原理.md) |
| 内部模块职责和引擎输出 | 对应 umi 模块 README |
| 第三方代码、预编译程序与改编来源 | PROVENANCE.md |
| 改动过程与验证证据 | git 提交说明（不另存文档） |

配置样例只在主流程保留必要示例，完整字段定义链接到对应参考。移动章节时同步本地链接与锚点；检查 shell 示例与当前 CLI 的参数一致。源码或功能变更后更新说明，避免在多个操作页面分别追加同一条兼容提示。

## 当前待办

- 扩展真实录像回归的录像覆盖，并记录标定 ID、引擎版本、耗时及重复运行波动。
- 独立测量相机到 TCP 的几何，验证目前使用的 CAD 名义值。
- 为需要训练开合通道的夹爪补做像素到毫米的宽度标定。仅有标签中心距的套件保持 `training_ready: false`。
- 对 `poster_vio` 做实拍精度和资源基准验证，再评估动态物体筛除、滑窗边缘化和 GPU 加速。
- 明确仓库许可证及对外分发说明。

实时预览、任务卡、上传、LeRobot 适配、模型训练和机械臂部署属于后续范围。软件回归结果和现场精度证据分别记录。
