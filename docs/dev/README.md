# 代码维护

[文档导航](../README.md) · [当前待办](#当前待办) · [历史归档](#历史归档)

配置和资产的职责见[权威来源](SSOT与权威来源.md)，变换方向、时钟与优化变量见[算法实现约定](算法实现约定.md)。

## 从入口沿流程阅读

| 流程 | 阅读入口 |
|---|---|
| 标定 | [标定代码指南](../../software/calibration/README.md)：`cli.main()` → `workflow.run()` / `verify()` |
| 共享场景 | [场景代码指南](../../software/scene/README.md)：`cli.main()` → `management.batch_processing.run()` / `build()` → `pipeline.maps.repository.build_scene()` / `workflow.run_episode()` |
| 轨迹抽取 | [轨迹代码指南](../../software/trajectory/README.md)：`cli.main()` → `workflow.run_batch()` → `run_recording()` |
| 无场景地图的 VIO | [poster_vio 代码指南](../../software/trajectory/algorithms/poster_vio/README.md) |
| 三维查看 | [查看器代码指南](../../software/viewer/README.md)：`server.main()` → `RecordingCatalog` → 浏览器播放状态 |
| 共用视觉 | [视觉代码指南](../../software/vision/README.md)：录像准备、ORB 建图/定位、视觉上下文与执行验收 |
| 相机数据 | [extraction.py](../../software/camera/x3/extraction.py) 提取时间戳和 IMU；[frame.py](../../software/camera/x3/frame.py) 定义录制几何、裁剪、帧率和时钟约定 |
| 设备选择 | [registry.py](../../software/camera/registry.py) 读取内部序列号、选择标定与夹爪并核对哈希 |
| 训练导出 | [数据集代码指南](../../software/dataset/README.md) |

`workflow.py` 展示顺序、阶段衔接与失败收尾；阶段函数返回具名结果。标定的 `acceptance.py` 决定验收，`result_io.py` 保存和校验记录，`reports.py` 只负责展示。标定第三方进程、依赖与日志放在 `pipeline/runtime.py`，Basalt 适配在 `solve/basalt.py`，数值计算在 `algorithms/`，独立命令在 `tools/`，安装资源在 `setup/`。

场景的任务读取、预检、调度和运行在 `scene/workflow.py`。发布、报告和审计在 `scene/pipeline/tasks/publication.py`，阶段指纹与缓存读写在 `scene/cache.py`。`vision/decoding.py` 分发帧并复用转换结果，由检测器提供自己的裁剪规则。`infrastructure/paths.py` 提供项目路径和运行解释器，`dataset/schema.py` 定义训练数组和采样策略。

相机测量的 Basalt 解析和版本化文件校验由 `camera/calibration_assets.py` 负责。X3 工厂内参拟合位于 `camera/x3/factory_kb4.py`。`scene/pipeline/maps/build.py` 负责建图，`repository.py` 负责版本读取、校验和批次绑定，场景建图和查看器共用 `vision/atlas.py` 的图集转换。

标定的完整数据流、状态、恢复与登记契约，以及实测缺口见[标定架构与验证](标定架构与验证.md)。

## Python 目录与命名检查

2026-11-23 核对了 software 下的 Python 源码和 tests 下的 105 个 Python 文件。按功能合并后，软件源码有 186 个 Python 文件、35 个目录。检查内容包括模块说明、公开函数、调用方、命令入口、动态导入和代码指纹引用。以下数量包含包初始化文件。

三套轨迹数值实现统一位于 trajectory/algorithms 下。固定地图应用适配仍由 scene/pipeline/tasks/scene_map.py 连接，环境镜头标定由 calibration/rear_workflow.py 调用共用地图和融合能力，轨迹格式转换由 trajectory/trajectory_export.py 完成。

| 目录 | Python 文件数 | 职责和本轮整理结果 |
|---|---|---|
| [software](../../software) | 1 | 包入口保持轻量，业务入口位于各应用包。 |
| [software/calibration](../../software/calibration) | 11 | 请求解析和配置预览集中到 pipeline/inputs.py，环境镜头编排在 rear_workflow.py。 |
| [software/calibration/algorithms](../../software/calibration/algorithms) | 3 | board_pnp 和 trajectory_metrics 分别对应位姿和误差计算。 |
| [software/calibration/pipeline](../../software/calibration/pipeline) | 6 | inputs 统一请求与配置预览，recordings、board_detection、steps、runtime 对应明确阶段。 |
| [software/calibration/solve](../../software/calibration/solve) | 3 | factory_intrinsics.py 表达工厂内参生成，与 basalt.py 的外部求解适配并列。 |
| [software/calibration/tools](../../software/calibration/tools) | 3 | euroc_to_bag.py 和 imu_physical_check.py 分别表达转换格式与诊断对象。 |
| [software/calibration/validation](../../software/calibration/validation) | 4 | factory_comparison.py 与 trajectory.py、gravity.py 按验证对象区分。 |
| [software/camera](../../software/camera) | 7 | 登记表、测量资产、来源、位姿几何和时钟各有独立公共入口。 |
| [software/camera/x3](../../software/camera/x3) | 7 | factory_geometry.py 表达工厂镜头模型，factory_kb4.py 专注光学拟合。 |
| [software/configuration](../../software/configuration) | 6 | loader、参数模型、操作设置、用户设置与音频参数按配置职责组织。 |
| [software/dataset](../../software/dataset) | 5 | export.py 负责已验收来源与发布，schema.py 统一数组、时间轴和坐标契约。 |
| [software/gripper_width](../../software/gripper_width) | 5 | width_estimation.py、calibration.py、series.py、video_tools.py 分别对应换算、测量资产、序列与调试。 |
| [software/gripper_width/detection](../../software/gripper_width/detection) | 4 | markers、frames、sampling 对应灰度标签、原片输入和时间采样。 |
| [software/gripper_width/distance](../../software/gripper_width/distance) | 3 | measurement 与 pose_estimation 区分测量编排和单标签位姿计算。 |
| [software/infrastructure](../../software/infrastructure) | 6 | paths、provenance、file_store、terminal、concurrency 是共用基础设施。 |
| [software/scene](../../software/scene) | 12 | workflow.py 负责预检与任务执行，cache.py 负责阶段复用，spatial.py 负责场景地图几何。 |
| [software/scene/management](../../software/scene/management) | 11 | batch_processing 编排批次，batch_snapshot 负责配置预览和冻结。登记、验收和清理单独处理。 |
| [software/scene/management/inputs](../../software/scene/management/inputs) | 3 | scene.py 读取场景和套件，batch.py 读取批次和任务。 |
| [software/scene/pipeline](../../software/scene/pipeline) | 3 | recordings 准备和预检录像，synchronization 计算双手偏移。 |
| [software/scene/pipeline/maps](../../software/scene/pipeline/maps) | 4 | build 构建地图，repository 负责读取、校验和批次绑定，epoch_selection 负责分批规划和选择。 |
| [software/scene/pipeline/tasks](../../software/scene/pipeline/tasks) | 10 | 三个后端与路由、轨迹写入、发布、测量和录像区间操作同层。 |
| [software/sync](../../software/sync) | 3 | specialist_audio.py、kiwi_audio.py 与两套音频同步算法名称对应。 |
| [software/trajectory](../../software/trajectory) | 10 | workflow 负责调度和复用，reconstruction 负责重建，environment_assets 负责环境输入，trajectory_export 负责产物与来源记录。 |
| [software/trajectory/algorithms](../../software/trajectory/algorithms) | 2 | inertial 共用预积分和惯性图，三套算法位于独立子包。 |
| [software/trajectory/algorithms/fixed_map](../../software/trajectory/algorithms/fixed_map) | 5 | geometry 同时负责固定地图对齐和重定位合并，fusion 和 poster 分别负责轨迹与海报融合。 |
| [software/trajectory/algorithms/fusion](../../software/trajectory/algorithms/fusion) | 7 | 六个实现模块负责 CLI、数据、对齐、求解、因子图和海报。solver 负责地图段放置和全帧率恢复。 |
| [software/trajectory/algorithms/poster_vio](../../software/trajectory/algorithms/poster_vio) | 7 | 六个实现模块负责模型与校验、初值、跟踪前端、基座标签、观测筛选和求解。 |
| [software/trajectory/reporting](../../software/trajectory/reporting) | 5 | artifacts 负责输出发布，reports 负责报告展示，constraint_quality 管质量证据，map_data 管展示地图。 |
| [software/trajectory/tools](../../software/trajectory/tools) | 2 | initialization_review.py 是基于保留数据的开发诊断入口。 |
| [software/viewer](../../software/viewer) | 3 | server 提供 HTTP 服务，ViewerDataError 直接定义在轻量包入口。 |
| [software/viewer/resources](../../software/viewer/resources) | 4 | video.py 集中视频来源、环境镜头裁切和预览，maps 与 cache 管地图及媒体缓存。 |
| [software/viewer/results](../../software/viewer/results) | 3 | trajectory_reader 解析轨迹，catalog 组织单手、双手结果及播放。 |
| [software/vision](../../software/vision) | 8 | decoding、recordings、slam、atlas、orb_environment 按共用视觉能力命名。 |
| [software/vision/poster](../../software/vision/poster) | 7 | layout、detection、pnp、geometry、factors、quality 对应布局、角点、初值、投影、因子和观测质量。 |
| [software/vision/tools](../../software/vision/tools) | 3 | make_lens_video.py 和 merge_frames.py 对应可独立运行的图像与定位数据工具。 |
| [tests](../../tests) | 105 | 保留 test_功能.py 与现有统一测试入口，夹具和并行运行器集中在本目录。 |

上一轮合并九组相邻职责，源码文件由 242 个减为 232 个，Python 目录由 43 个减为 41 个。后续继续合并批次、地图和算法模块。本轮将 scene 从 64 个文件收拢到 43 个，trajectory 从 44 个收拢到 38 个。

| 合并前 | 当前入口 |
|---|---|
| trajectory/calibration/assets.py、selection.py 与包入口 | [trajectory/calibration.py](../../software/trajectory/calibration.py) |
| trajectory/io/tum.py 与包入口 | [trajectory/tum.py](../../software/trajectory/tum.py) |
| dataset/coordinates.py 与 schema.py | [dataset/schema.py](../../software/dataset/schema.py) |
| calibration/configuration.py 与请求解析 | [calibration/pipeline/inputs.py](../../software/calibration/pipeline/inputs.py) |
| poster_vio_request.py 与准备过程 | [scene/pipeline/tasks/poster_vio.py](../../software/scene/pipeline/tasks/poster_vio.py) |
| viewer/errors.py 与包入口 | [viewer/__init__.py](../../software/viewer/__init__.py) |
| fusion/scale_checks.py 与轨迹质量指标 | [constraint_quality.py](../../software/trajectory/reporting/constraint_quality.py) |
| batch/discovery.py 与场景磁盘布局 | [scene/layout.py](../../software/scene/layout.py) |
| viewer/results/rear_video.py 与视频资源 | [viewer/resources/video.py](../../software/viewer/resources/video.py) |

本轮具体路径调整如下，目录移动包含其中的模型、初值、前端、求解器和来源声明。

| 原路径 | 当前路径 |
|---|---|
| `trajectory/poster_vio/` | [trajectory/algorithms/poster_vio/](../../software/trajectory/algorithms/poster_vio/) |
| `camera/x3/factory.py` | [camera/x3/factory_geometry.py](../../software/camera/x3/factory_geometry.py) |
| `calibration/solve/factory.py` | [calibration/solve/factory_intrinsics.py](../../software/calibration/solve/factory_intrinsics.py) |
| `calibration/validation/factory.py` | [calibration/validation/factory_comparison.py](../../software/calibration/validation/factory_comparison.py) |
| `calibration/tools/calib_folder_to_bag.py` | [calibration/tools/euroc_to_bag.py](../../software/calibration/tools/euroc_to_bag.py) |
| `calibration/tools/imu_calib_physical_check.py` | [calibration/tools/imu_physical_check.py](../../software/calibration/tools/imu_physical_check.py) |
| `scene/spatial/rear_calibration.py` | [calibration/rear_workflow.py](../../software/calibration/rear_workflow.py) |
| `dataset/source.py` | [dataset/export.py](../../software/dataset/export.py) |
| `scene/management/processing.py` | [scene/management/batch_processing.py](../../software/scene/management/batch_processing.py) |
| `scene/management/batch/execution.py` | [scene/management/batch_processing.py](../../software/scene/management/batch_processing.py) |
| `scene/pipeline/maps/lifecycle.py` | [scene/pipeline/maps/repository.py](../../software/scene/pipeline/maps/repository.py) |
| `scene/pipeline/tasks/estimation.py` | [scene/pipeline/tasks/backend_routing.py](../../software/scene/pipeline/tasks/backend_routing.py) |
| `scene/pipeline/tasks/policy.py` | [scene/management/collection_schema.py](../../software/scene/management/collection_schema.py) |
| `scene/pipeline/tasks/diagnostics.py` | [scene/pipeline/tasks/recording_intervals.py](../../software/scene/pipeline/tasks/recording_intervals.py) |
| `scene/pipeline/tasks/results/result_writer.py` | [scene/pipeline/tasks/publication.py](../../software/scene/pipeline/tasks/publication.py) |
| `scene/pipeline/tasks/results/reporting.py` | [scene/pipeline/tasks/publication.py](../../software/scene/pipeline/tasks/publication.py) |
| `scene/pipeline/tasks/backends/scene_reconstruction.py` | [scene/pipeline/tasks/scene_map.py](../../software/scene/pipeline/tasks/scene_map.py) |
| `scene/pipeline/tasks/backends/poster_vio_inputs.py` | [scene/pipeline/tasks/poster_vio.py](../../software/scene/pipeline/tasks/poster_vio.py) |
| `trajectory/pipeline/calibration_assets.py` | [trajectory/pipeline/environment_assets.py](../../software/trajectory/environment_assets.py) |
| `trajectory/reporting/postprocess.py` | [trajectory/trajectory_export.py](../../software/trajectory/trajectory_export.py) |
| `trajectory/reporting/metadata.py` | [trajectory/trajectory_export.py](../../software/trajectory/trajectory_export.py) |
| `trajectory/reporting/quality.py` | [trajectory/reporting/constraint_quality.py](../../software/trajectory/reporting/constraint_quality.py) |
| `trajectory/reporting/results.py` | [trajectory/reporting/artifacts.py](../../software/trajectory/reporting/artifacts.py) |
| `trajectory/algorithms/fusion/quality.py` | [trajectory/reporting/constraint_quality.py](../../software/trajectory/reporting/constraint_quality.py) |
| `viewer/results/rear.py` | [viewer/resources/video.py](../../software/viewer/resources/video.py) |
| `viewer/results/readers.py` | [viewer/results/trajectory_reader.py](../../software/viewer/results/trajectory_reader.py) |
| `sync/audio.py` | [sync/specialist_audio.py](../../software/sync/specialist_audio.py) |
| `sync/landmark_audio.py` | [sync/kiwi_audio.py](../../software/sync/kiwi_audio.py) |

名称使用所在目录的领域上下文。workflow.py 表达应用顺序，backend_routing.py 表达候选后端选择，measurement.py 表达三维中心距测量，pose_estimation.py 表达单标签位姿求解。models.py、cli.py、__init__.py 和 __main__.py 沿用 Python 项目的常见名称。

源文件移动会更新相应阶段的代码指纹。参数、资产与结果的字段名继续由各自 schema 定义，文件读取的兼容性通过现有结果及训练包测试检查。

## 参数管理

见 [统一参数配置](../../config/README.md)。数值默认值写在 `config/*.yaml`，不可变模型位于 `software/configuration/parameter_models.py`，加载、校验、覆盖和快照位于 `software/configuration/loader.py`。源代码、引擎、文件内容及规范 JSON 的身份计算集中在 `software/infrastructure/provenance.py`；设备登记、场景和标定共用同一实现。workflow 加载一次，向阶段传递不可变参数。修改参数后同时核对复用、报告与默认数值回归。

## 命名和终端约定

外围代码使用 `request`、`camera_serial`、`keep_work`、`return_code`、`reason`、`error`。数学符号使用对应领域的惯例。场景日常流程统一为手写 kit.yaml/collection.yaml、放置视频，再执行 run；内部自动登记逻辑位于 ingest.py，无独立登记 CLI。历史 JSON 与第三方接口继续按各自协议读取，例如 `serial` / `why`、`subprocess.returncode`；内部 Session/Episode 类型和 episode.json 文件名不表示还支持旧 CLI。

[terminal.py](../../software/infrastructure/terminal.py) 统一输出格式，调用为 `runtime.console.write(...)`。阶段负责进度，报告负责结果；标签依次表示工具、录像、步骤、重建次数和执行状态。标定验收 `PASS` / `FAIL` 与执行“完成 / 失败”分别展示。引擎原始日志保留原文。

## 验证

完成[安装](../安装与日常使用.md)与标定组件安装后，可直接通过根目录的统一脚本一键测试：

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

仓库的轻量 CI 位于 `.github/workflows/quality.yml`：它只安装 NumPy 和 Ruff，检查全仓 Python 语法、可移植训练读取器边界及本轮核心模块的静态错误，不依赖 GTSAM、OpenCV、SLAM 或真实录像。完整算法回归仍使用本机 `.deps` 环境执行 `./test.sh --quick`、`./test.sh --unit` 和必要的 `./test.sh --all`。

复用测试夹具时导入模块或独立 fixture，避免把其他文件的 `TestCase` 类直接导入测试模块顶层：unittest 会再次收集该类。查重应比较源码文件、类和方法；仅按测试 ID 检查会漏掉 `tests.test_x` 与 `test_x` 的别名重复。

## 代码与资产身份

`provenance.source_hash()` 对 Python 的 AST 计算版本化摘要，忽略注释、位置、排版和模块/类/函数文档字符串；常量、控制流、函数参数等语法变化仍使缓存失效。算法不得把文档字符串或源码行号用作数值输入。地图阶段、检测阶段和完整结果共用该规则。配置、原片、设备资产及结果文件继续使用 `file_hash()` 的原始字节 SHA256，完整性校验不经过语法归一化。切换指纹规则后旧计算缓存首次会重建；旧地图和结果文件仍保留。

## 文档怎么维护

[用户文档导航](../README.md)面向采集和排障。每项现行行为在对应页面维护，其他页面用链接指向该说明。

| 内容 | 维护位置 |
|---|---|
| 首次安装与一次采集的主流程 | 安装与日常使用.md |
| 标定录制、求解、验收和登记 | 标定.md |
| 场景/批次选项、配对、地图复用和排障 | 场景处理.md |
| 查看器、人工验收、成果包和结果字段 | 结果查看与验收.md |
| 训练输入、宽度测量、数组和就绪条件 | 训练数据.md |
| 资产目录、版本、坐标约定 | 设备与坐标.md |
| 算法过程与当前验证边界 | 算法原理.md |
| 内部模块职责和引擎输出 | 对应 software 模块 README |
| 变更过程、验证证据、阶段性方案和审计 | PROVENANCE.md 与 archive/ |

配置样例只在主流程保留必要示例，完整字段定义链接到对应参考。移动章节时同步本地链接与锚点；检查 shell 示例与当前 CLI 的参数一致。源码或功能变更后更新说明，避免在多个操作页面分别追加同一条兼容提示。

## 当前待办

- 建立可复现的真实录像端到端回归，记录输入、标定 ID、引擎版本、耗时、误差及重复运行波动。
- 独立测量相机到 TCP 的几何，验证目前使用的 CAD 名义值。
- 为需要训练开合通道的夹爪补做像素到毫米的宽度标定。仅有标签中心距的套件保持 `training_ready: false`。
- 对 `poster_vio` 做实拍精度和资源基准验证，再评估动态物体筛除、滑窗边缘化和 GPU 加速。
- 明确仓库许可证及对外分发说明。

实时预览、任务卡、上传、LeRobot 适配、模型训练和机械臂部署属于后续范围。软件回归结果和现场精度证据分别记录。

## 历史归档

[PROVENANCE](PROVENANCE.md) 记录来源和变更证据。归档保留当时的命令、配置和结果，现行流程以[用户指南](../README.md)为准。

| 主题 | 历史材料 |
|---|---|
| 场景实现 | [环境标签与自适应融合](archive/环境标签与自适应融合重构计划.md) · [无场景地图的双手轨迹重建](archive/无场景地图的双手轨迹重建方案.md) · [场景流水线重构会话](archive/本次重构会话总结.md) |
| 架构与清理 | [架构扫描](archive/架构扫描与重构.md) · [深度架构审计](archive/深度架构审计.md) · [清理记录](archive/清理遗留事项.md) |
| 环境与引擎 | [依赖目录调查](archive/依赖目录结构与清理方案.md) · [轨迹引擎验证](archive/轨迹引擎验证记录.md) |
| 初始范围 | [建仓计划](archive/建仓计划.md) · [实时预览方案](archive/QUICKSTART_X3_STREAM.md) |
