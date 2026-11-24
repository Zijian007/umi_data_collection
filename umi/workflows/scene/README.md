# 场景代码阅读指南

用户入口为 `umi-scene`。 硬核数学实现的统一变量、变换方向和优化边界见 [算法实现约定](../../../docs/dev/算法实现约定.md)。`run <批次>` 由 `umi/workflows/scene/batch_processing.py` 读取批次根目录唯一的 `collection.yaml`，自动保存设备快照和登记原片，再调用 workflow 的地图与任务流水线。场景代码按“每类事实一个权威来源”组织，具体边界见 [SSOT 与权威来源](../../../docs/dev/SSOT与权威来源.md)。`export --format training` 调用 umi.dataset。标定统一由 umi-calib 发起，查看由 umi-view 提供。

从 `umi/apps/scene.py` 找命令入口，再读 `umi/workflows/scene/maps/repository.py` 的 `build_scene()` 或 `workflow.py` 的 `run_episode()`。用户操作见[场景指南](../../../docs/03-场景处理.md)，完整选项见[命令行参数](../../../docs/命令行参数.md#umi-scene)。

模块通过真实目录直接导入；`__init__.py` 仅说明包职责，不注册别名或预先加载全部子模块。

三个任务后端与路由都位于 `umi/workflows/scene/tasks`。每个后端负责本模式的输入准备和求解，共用录像准备、测量、轨迹写入与发布。固定地图融合和 Poster VIO 共用 `umi/estimation/inertial.py` 的 IMU 因子，首尾裁剪与 VIO 初始化共用 `umi/perception/poster/pnp.py` 的位姿候选。

任务级估计路由位于 `umi/workflows/scene/tasks/backend_routing.py`，共享输入 `TaskInputs` 和输出 `ArmExecutionResult` 定义在 `umi/store/models.py`。 `TaskInputs.arm_cpu_sets` 保存各手 CPU 分配，`cache_audit` 为审计对象；输出的 `width_reports`、`tag_distance_reports` 为测量报告，`coordinate_record` 为实际输出坐标声明。三种数值后端直接接收同一个输入对象，包含所有使用手的冻结录像、时间偏移、输出目录、CPU 分配和缓存审计；场景及任务名称从其中的 `Episode` 获取。`poster_vio` 会从冻结录像加载标签观测和相机/IMU 资产，冻结数值可通过该后端的 `replay_task()` 回放，沿用已准备录像的时间轴。`auto` 在批次有 `mapping/` 原片时先构建或复用地图，再按 `scene_map → poster_vio → poster_independent` 执行；没有 mapping 原片且没有地图时按 `poster_vio → poster_independent` 执行，只对明确质量不足继续尝试。`MapSelection.available()` 表示目录与清单成对存在，`without_map()` 表示无地图或保留失败原因；无效组合在构造时拒绝。`scene_map` 保留地图布局、标签诊断/融合、`staged/joint` 初值和单双手分支；`poster_independent` 复用现有独立引擎持续融合海报；`poster_vio` 在任务海报坐标内联合优化共享海报位姿、每手 IMU 状态、标签角点和自然特征点，不读取旧场景地图。三者共用相机准备、TCP、测量、发布、验收和训练导出。场景适配器将已解析的 `SceneParameters` 投影为各算法声明的参数类型。

## 文件职责

```text
umi/apps/scene.py              umi-scene 子命令表 COMMANDS，每个子命令一个处理函数
umi/workflows/scene/
├── workflow.py               任务预检、输入解析、隔离运行（execute_episode）与调度
├── batch_processing.py       普通批次与多地图（epoch）批次的准备、建图和运行
├── batch_snapshot.py         配置预览与设备/建图资产冻结
├── ingest.py / import_manifest.py  原片登记、配对与 Organizer 交接
├── episodes.py / cleanup.py  任务状态、人工验收、成果导出与 .pipeline 保留规则
├── recordings.py             原片归档、录像准备与预检
├── synchronization.py  双手时间同步
├── camera_assets.py          按设备绑定选择并冻结镜头标定与夹爪资产
├── parameters.py             本次专家参数覆盖（SceneOverrides）
├── task_configuration.py     任务使用手、后端、同步与地图选择的统一解析
├── inputs/                   scene.py 场景与套件；batch.py 批次与任务清单
├── maps/                     build.py 建图；repository.py 版本读取、校验与批次绑定；epoch_selection.py
└── tasks/
    ├── backend_routing.py    显式后端分派；auto 由 _AutoRoute 按顺序尝试
    ├── scene_map.py          地图定位、恢复与固定场景融合
    ├── poster_vio_inputs.py  poster_vio 输入准备：检测、初始化、特征跟踪、冻结数值输入
    ├── poster_vio.py         poster_vio 求解、验收、回放
    ├── poster_independent.py 各手独立重建 + 海报坐标验收
    ├── recording_intervals.py 首尾裁剪与视觉缺口片段
    ├── measurements.py       环境标签与夹爪检测、宽度与中心距
    ├── trajectory_writers.py 单手轨迹表与双手对齐表
    └── publication.py        质量报告、发布、复用与运行审计
```

场景依赖的下层模块：数据模型 `TaskInputs`、`Episode`、`Scene` 等在 `umi/store/models.py`，批次磁盘布局在 `umi/store/layout.py`，阶段缓存在 `umi/store/cache.py`；地图米制化与放置在 `umi/estimation/mapping.py`；`collection.yaml` 与场景参数 schema 在 `umi/config/collection_schema.py`、`umi/config/scene_parameters.py`。

`inputs/scene.py` 读取场景和套件，`inputs/batch.py` 读取批次和任务清单；`camera_assets.py` 选择并冻结设备测量。`umi/store/models.py` 保存阶段数据，CalibratedCamera 按需计算并缓存小型资产哈希。`workflow.py` 和 `episodes.py` 直接使用这些实现；验收模块不导入 workflow。公共原子文件和锁由 `umi/infra/file_store.py` 提供，设备按镜头和参数来源保存的当前标定资产由 `umi/devices/registry.py` 管理。

## 采集管理

使用手的校验和继承统一在 `umi/core/arms.py`；kit 负责设备绑定，批次声明默认使用手，任务可覆盖为子集，下游消费 `Episode.expected_arms`。`umi/workflows/scene/task_configuration.py` 统一解析任务使用手、后端、同步与地图选择，执行和只读展示共用该结果。

输入解析位于 `umi/workflows/scene/inputs`：`scene.py` 负责场景和套件，`batch.py` 负责批次和任务。`umi/store/layout.py` 发现批次目录。`umi/workflows/scene/batch_snapshot.py` 解析配置并冻结资产。`umi/workflows/scene/batch_processing.py` 编排普通与多地图批次的准备、建图和任务运行。epoch 流程复用首次登记报告，建图后按更新的 epoch 清单解析任务地图。

批次入口和显式任务入口共用 `batch_processing._prepare_for_run()`，统一准备批次并阻断错误、待补齐或忽略的原片。失败摘要都包含具体原因和 `details`，预览也执行相同检查；一个批次失败不阻断其他批次。显式任务先按所属批次分组，每批只准备一次，最后按原输入顺序把可执行任务交给调度器。

```text
手写 kit.yaml → inputs.scene.read_kit()         解析相机版本与夹爪文件路径
手写 collection.yaml → batch_snapshot.read_inputs()   解析场景、批次、套件与建图来源
    batch_snapshot.make_record() / save_record()  构造快照、检查并发修改、按需写入
    inputs.scene.read_scene() / inputs.scene.read_kit()
    registry.pin() / version_assets()     读取全局当前标定资产

run → batch_processing.prepare_batch() → ingest.register_recordings()
                                         自动配对、登记与补片；预览共用规划
    recordings.inspect() / import_video() 原片身份、哈希、保留原名归档
    inputs.batch.read_episode()                 手、序列号与输入完整性
    ingest._write_episode()               有变化才更新清单和任务状态
```

各批次独立持有 `.pipeline/maps/` 及当前地图指针，任务内部数据集中在同一批次 `.pipeline/tasks/<任务>/`。目录层次为 `hardware/kits/<套件>/kit.yaml` 与 `scenes/<场景>/<批次>/.pipeline/manifest.yaml`，任务位于批次目录下。输入读取校验目录与清单身份一致；查看器从批次清单读取套件身份。

每个套件只有一个 `hardware/kits/<套件>/kit.yaml`；`inputs.scene.read_kit()` 读取当前配置，`validate_kit()` 校验当前配置与批次快照的共同结构。处理批次时核对实际测量并把配置保存到 `.pipeline/manifest.yaml`（schema 5），run 每次读取当前配置，记录不反向覆盖 kit.yaml。套件没有独立版本目录或指针文件，相机按前后镜头来源选择当前参数，内部历史身份与地图版本保留用于追溯。夹爪配置包含必填的 gripper.tcp、可选的 gripper.width、gripper.tag_distance 和 gripper.mask，路径相对 kit.yaml；输入入口统一解析测量，计算前复制到运行目录 devices/<序列号>/，后续直接使用解析值与副本。

场景名称和当前套件 ID 从目录推导，上级场景和批次清单由固定层级确定。生成清单省略空描述、默认模式、空覆盖和未绑定地图；当前套件省略 ID，批次快照仍保存 ID 和标定资产版本。批次快照不复制套件描述。

新任务原片清单 schema 5 仅保存原片身份与引用，集中在批次 `.pipeline/tasks/<任务>/episode.yaml`。人工选项 instruction、arms、synchronization 位于批次 `collection.yaml.tasks`。episode.yaml 中不接受这些字段。双手默认零偏移但未经确认；批次默认使用套件全部手，任务默认继承批次。缺片不会隐式降为单手。

指令进入结果 episode.json 和导出清单；修改指令或同步按有效任务选项的规范化哈希失效（仅修改注释不失效），需重新 run 和验收。训练格式由独立 [dataset 模块](../../dataset/README.md) 负责，scene 不承担训练采样。

`<批次>/.pipeline/tasks/<任务>/state.json` 只保存 attempt/review，身份与完整性从清单读取；attempt 只保存运行引用和输入哈希，错误原因及预检报告路径从运行目录读取，不重复落盘。录制检查在算法执行前落盘，算法失败仍有原始检查证据。

录像准备保留已解析清单，独立回退沿用同一份数据。任务发布从轨迹写入阶段取得逐帧证据和原片检查记录，供报告与运行清单使用。

## 算法执行

```text
build → map_repository.build_scene()
    map_build.build_version()              执行独立地图版本的构建
    inputs.scene.read_scene() / select_cameras() 固定建图输入与测量
    camera_assets.snapshot_cameras()             保存本次测量副本
    recordings.prepare_recording()        调用 perception 准备时间戳、IMU、镜头视频
    mapping.build_maps()                  umi/estimation/mapping.py
        mapping.build() / atlas_candidates() 纯视觉建图、逐图定位
        geometry.metric_alignment()       复用轨迹坏段筛选，共同尺度/重力与可观测性验收
        geometry.place_front()            工作镜头地图放入共享场景
    mapping.freeze_maps()                 地图资产冻结
    cleanup.prune_cache()                 建图成功后删除本次未用到的建图缓存；
                                          绑定由 build_scene 写入批次 manifest.yaml

run → workflow.run_episode()
    inputs.batch.read_episode()          批次、录像、预期使用手
    workflow.resolve_synchronization()   双手时间偏移
    workflow.preflight_recordings()         录制质量与完整性
    workflow.resolve_inputs()            绑定地图和相机，计算输入指纹
        backend_routing.select_map()          按 backend.mode 选择、构建或跳过地图
    publication.reuse_episode()   校验并复用已发布结果
    workflow.execute_episode()           冻结相机、TCP 与测量资产后开始隔离运行
    recordings.prepare_episode_recordings() 准备各手录像、检查公共时间区间
        recordings.prepare_with_cache()   复用地图与任务共用的录像准备缓存
    backend_routing.execute(TaskInputs, MapSelection)
        scene_map → scene_map.execute_task()
            scene_map.reconstruct_arm_in_scene() 地图定位、恢复与视觉/IMU 融合
            开启海报融合时按 staged/joint 初值构建海报约束图
        poster_independent → poster_independent.execute_task()
            poster_independent.reconstruct(require_poster=True) 各手录像内 SLAM/IMU 与持续海报约束
            转换到任务海报坐标，检查布局、重力与共同位置支持
        poster_vio → poster_vio.execute_task()
            poster_vio_inputs.prepare_task_inputs() 检测、初始化、特征跟踪并冻结数值输入
            在冻结的任务级观测上建立共享海报 + 多手 IMU + 自然特征联合图
            将求解结果重采样到原录像时间轴，再进入公共测量和发布流程
        auto → _AutoRoute：有 mapping 原片时先构建/复用地图，再按 scene_map → poster_vio → poster_independent；无 mapping 原片且无地图时 poster_vio → poster_independent
        三个后端共用测量、trajectory_writers.write_arm_result() 与任务结果汇总
    trajectory_writers.write_bimanual_trajectory() 公共时间轴与双手相对位姿
    result_io.recording_identity()   发布前仅核对原片；测量使用已冻结副本
    publication.write_episode_result()               报告、资产哈希与任务元数据
    file_store.publish_link(results)         全部成功后更新结果入口
    episodes.record_attempt()             更新状态与验收有效性
    cleanup.after_task()                  删除被取代的运行和本次未用到的任务缓存
```

`run_episodes()` 调度任务并汇总结果。以下 `BatchMap` 构建与绑定规则适用于显式 `scene_map` 及带 mapping 原片的 `auto`；`poster_independent` 和 `poster_vio` 跳过地图。场景地图流程每次调用按批次持有独立 `BatchMap`。同批首次任务在锁内构建/校验并固定地图，后续任务复用同一绑定；共同建图失败只执行一次。每个任务仍核对自身标定与地图匹配，复用前检查批次配置、建图原片及地图资产的文件身份，变化即拒绝继续；发布前复核原片与地图资产，已开始的任务仍使用冻结配置，不因外部配置编辑而中断。不同批次的地图准备互不阻塞，只读预检不落盘。

`--jobs 1` 且双手可用 CPU 至少 8 个时，按手拆分子进程的 CPU 集合，并行定位、融合、检测和结果写入；核心不足、单手或 `--jobs > 1` 时按手串行。任务级发布等待所有手成功；任一手失败时等其他线程退出后保存失败现场，不替换旧成功入口。

任务融合因视觉定位失败而保留失败现场时，若存在连续无视觉区间，运行目录会写入 `<arm>/no_visual/gap_*.mp4` 和 `<arm>/no_visual/manifest.json`。片段默认包含前后各 0.5 秒上下文；原始视频、视觉定位结果和 IMU 不会被修改。

多个已通过验收的地图候选会在 `metric_alignment.json` 中记录 `selection_score`，按固定尺度复核、对齐 RMS、尺度不确定度、条件数和覆盖证据选择；硬性验收门槛保持不变。

缓存只依据数据输入复用，不包含代码或引擎身份；`--recompute` 通过 `infra.file_store.recompute()` 让本次运行开始前的缓存全部失效，本次运行中新生成的仍可复用。完整地图输入保留原片路径、时间和 SHA256 以便追溯，`reuse_key` 按内容、相关配置和相机资产比较，不把文件路径或夹爪资产作为地图身份。工作镜头遮手图仍属于地图输入。

建图和任务通过 `cache.StageCache` 共用缓存存储机制，纯输入身份由 `cache.preparation_inputs()` / `visual_inputs()` 构造；地图专属镜头资产筛选与复用键也由 `cache` 生成。`umi/estimation/mapping.py` 的地图求解不导入地图流水线模块（由分层合约保证）。

场景写入锁内，`StageCache.stage()` 先验证缓存清单和资产哈希，再将产物复制到本次工作目录；读写不会共享可修改文件。准备缓存只保存数据集和镜头视频，当前相机设置、遮手图、时间偏移与算法上下文重新生成；视觉缓存使用实际镜头视频、相机设置、ORB 参数与遮手图（仅建图）；定位另包含地图内容和子地图索引。尺度与放置每次重新验收，不缓存最终数值解。

阶段正常完成后才发布 `ready.json`；缓存损坏重算。单地图批次的 `umi-maint scene build --recompute` 重算建图和定位，验收失败后的重试也刷新视觉建图和定位，避免重复读取同一失败候选。成功阶段缓存独立于版本 `work/`，后续阶段失败或清理工作目录不会丢失它。`manifest.json` / `error.json` 中的 `timings` 记录实际耗时与缓存状态。

`retry.max_tries` 是最大尝试次数。连续两轮若同一相机/镜头的全部被拒子地图，其帧索引、IMU 时间、位姿以及拒绝原因都相同，建图会提前停止，避免继续重建同一失败结果。轨迹变化、拒绝原因变化、没有候选轨迹证据或引擎暂时退出时不按重复轨迹提前停止，仍保留配置的重试次数。尺度门限保持原值；失败现场和已有成功地图均保留。

地图成功清单或失败 `error.json` 的 `retry` 是本次尝试记录的唯一汇总，包含每轮状态、拒绝证据摘要、`repeat_of_attempt` 和 `stop_reason`。`repeated_visual_failure` 表示因重复失败停止，`attempt_limit` 表示达到次数上限，`accepted` 表示通过。固定尺度复核失败时，日志和拒绝记录还标出对应的原片秒数；每轮完整验收数据继续保存在 `work/try_<次数>/<角色>/localize/m<子地图>/metric_alignment.json`。

场景与独立轨迹通过 `perception.runtime.visual_settings()` 共用镜头 YAML 写入、遮手图准备与视觉运行设置。场景显式选择本设备已验收的各镜头完整标定（measured/factory 内参均可，B 均实测）；独立 dual 使用本设备的登记参数或从 INSV 工厂数据推导环境镜头参数。

场景录像准备直接构造 `perception.models.VisualContext`，包含已解析参数、帧数与 CPU 设置；不构造 `TrajectoryRequest`。共用的视觉设置为 `VisualSettings`，内部逐段轨迹引擎的融合设置由 `ReconstructionSettings` 扩展；场景融合直接接收已解析参数，不生成单段融合脚本的配置文件。

场景后端用具名的手级求解结果传递轨迹、坐标声明和待联合求解的海报输入；未启用共享海报时仍在各手完成后立即写入，共享海报仍等待所有手后统一求解。`fixed_map_reconstruction.localize_with_cache()` 先尝试恢复已校验的观测，缺失或损坏时计算并发布完整缓存；引擎失败保留独立工作目录与日志，诊断复制仍在缓存命中或计算成功后执行。

任务准备和定位数据保存在 `<批次>/.pipeline/tasks/<任务>/cache/`；`run` 不提供 `--keep-work`。建图、环境镜头标定仍可保留求解中间文件。进度写 stderr，结构化 JSON 写 stdout。

每次真正执行的任务会在 `<批次>/.pipeline/tasks/<任务>/logs/<run-id>/` 持久化 `run.log`、`events.jsonl` 和 `run_manifest.json`。前者保留人读进度，第二个记录逐行的阶段/终端事件，最后一个记录输入指纹、状态、起止时间、耗时、结果路径及同一任务的 `runs/` 现场；成功和失败（包括原片预检拒绝）都写入。日志用于跟踪运行进度。质量摘要位于发布清单，完整融合证据位于内部运行目录的 `diagnostics.json`。

地图清单的建图定义只存于 `inputs.definition`。地图成功导出后移除原始展示 JSON 和空日志，保留有内容的日志及失败现场；已发布版本不因整理而改写。

`trajectory_writers.write_arm_result()` 将每手 TCP 位姿与已配置的夹爪测量写入 `trajectory.csv`，返回供双手对齐使用的 `ArmTrajectory`。发布阶段在内部运行目录组装 `published/`，写出 `report.txt` 和 schema 5 的 `data/episode.json`，校验值按发布文件和内部证据分别登记，再原子更新任务的 `results/`。逐帧信息、融合诊断与冻结设备文件留在同一次运行目录，由 `umi.store.result_io` 统一定位。schema 4 的发布目录同样可读。

有标签间距测量时，任务层 `tag_distance.<手>` 保存报告。`scene.frame_id` 记录坐标身份，`session` 记录批次名。输入记录保存地图与同步的内容摘要，`code_revision` 记录代码版本供追溯。查看器默认由 TCP 和固定外参重建相机和 IMU 的显示轨迹，`parameters.diagnostics.export_auxiliary_trajectories: true` 可在排障时写出 TUM 轨迹。

`umi.store.result_io` 提供只读的摘要和设置适配，查看器及训练导出共用；当前结果从任务层 `episode.json` 取值。查看器校验地图路径、版本、固定坐标与内容身份后直接加载共享地图。

地图资产错误由 `scene.errors.MapUnavailableError` 表示，兼容失败由其子类 `MapCompatibilityError` 表示；`PosterVioQualityError` 只表示 poster_vio 的质量不足。`auto` 只接收这些结构化地图/质量失败，普通输入、引擎和程序异常继续抛出；每次候选写入 `reconstruction_attempts/attempts.json`，并在独立目录中运行，不混用失败轨迹。无地图的支持缺口和双手重力夹角使用 `parameters.poster_independent` 独立阈值，见[配置说明](../../../config/README.md)。

`maps.repository.validate_manifest()` 只校验已读入清单的结构、版本、坐标、地图条目、尺度和放置变换，不访问文件或查询引擎。`load_published()` 负责指针解析、文件读取、哈希与当前场景兼容检查，各类错误仍在原边界转换。

## 命名和时间

- `SceneParameters` 是场景算法参数，`CalibratedCamera` 是已选定完整标定资产的相机。
- `recording_paths` 表示 INSV 路径；`PreparedRecording.lens_videos` 表示生成的镜头视频。
- `episode` 表示一次任务录制；`instruction` 表示操作描述，在批次 `collection.yaml.tasks.<take>` 中填写。
- `RecordingMetadata.started_at_unix_s` 是从文件名解析的 Unix 秒，仅用于配对候选；`duration_s` 是视频时长。
- `PreparedRecording.video_times_s` 使用原片视频时钟；`VisualObservations.imu_times_s` 使用本机 IMU 时钟，不混入跨相机偏移。
- `VisualObservations.T_reference_camera` 的参考系由 `reference_frame` 显式标明为 `visual_map`、`metric_map` 或 `scene`；只有原始视觉地图观测可施加尺度，融合只接受场景观测。
- `models.FusedTrajectory.T_reference_imu` 是调用方声明参考系中的 IMU 位姿；固定场景融合使用场景系，独立回退使用本次局部系。
- 新地图版本格式为 `YYYYMMDD_HHMMSS-随机8位`，前缀取本机建图开始时间，使用本地时区；复用地图时版本保持不变。

相机–IMU 和相机–TCP 外参、SE(3) 校验和位姿插值由 `umi/core/geometry.py` 提供。场景模块负责地图尺度与场景放置。TUM 轨迹读写使用 `umi/infra/file_store.py`。

地图尺度复用 `trajectory.algorithms.fusion.alignment.align_lenses()` 的运动段筛选，并由 `fusion.alignment.estimate_lens_alignment(shared_map=True)` 共用一个尺度、重力和零偏重新求解。场景层仅适配观测、检查可靠段一致性与可观测性、保存验收诊断，不复制另一套尺度求解器。分段端点速度隔离，地图变换始终只有一个。建图与任务融合共用 `inertial.estimate_gyro_bias()` 的初始陀螺仪零偏规则。

定标后由 `fusion.alignment.fixed_map_residuals()` 对所有运动链复核固定尺度下的位置方程；不重新拟合段尺度。每条链都会写入 `verified`、`inconsistent` 或 `insufficient_evidence` 状态，局部可靠但整图不一致的候选仍被拒绝，逐段结果保存于 `evidence.alignment.fixed_map_validation`。

外参函数遵守“目标系_from_来源系”：`imu_from_camera()` 返回 `T_imu_cam`，`camera_from_tcp()` 返回 `T_cam_tcp`。ORB 只使用图像，IMU 用于米制对齐和离线融合。地图放置为 `p_scene = R (s p_map) + t`，物理杆臂不乘视觉尺度。双手输出使用任务配置的固定偏移；固定场景不按任务重新置零。

## 双手坐标与时间的阅读入口

单手基线求解入口是 `scene_map.reconstruct_arm_in_scene()`。开启 `backend.scene_map.poster_constraint.enabled` 后，将环境镜头的海报角点与固定地图先验、IMU 放入同一张图优化。`staged` 先求地图与 IMU 基线，再以基线初始化海报图。`joint` 从视觉种子构建最终图。双手任务先分别求两手基线，再在 `poster.refine_shared_poster_pose()` 中共享一个刚性海报位姿。两手可以在不同时间看到不同 ID，已知布局负责建立约束。完全没有有效海报观测的手保留基线并记录 `not_observed_shared_board`。共享坐标来自建图时冻结的地图放置。

`umi/estimation/fixed_map/poster.py` 的单双手路径共用 `_tag_geometry()`、`_remap_map_observations()` 和 `_support_evidence()`。每只手在本轮求解内复用自己的镜头模型与标签几何，异常剔除后的重试重新准备观测；原地图先验只在原输出时刻采样，再重编号到含海报时刻的节点，不给新节点复制额外地图证据。公共组图因子及质量门槛继续使用原实现。

`umi/workflows/scene/tasks/publication.py` 从一次 `quality_summary()` 取坐标范围和公共指标；帧约束区间计算一次后用于分类及视觉/IMU 视图。逐帧来源仍保存在报告中，字段与 `frames.csv` 引用保持当前协议。

1. `Scene.role(arm, lens)` 选择固定地图：`shared_front` 两手使用同一工作镜头地图；`dual` 使用各自工作镜头地图与共享环境镜头地图。
2. `VisualObservations.placed(..., reference_frame='scene')` 把 `T_map_cam` 转入场景系：位置为 `R (s p_map) + t`，朝向为 `R R_map_cam`。
3. `fusion.fuse()` 使用各镜头 `T_cam_imu` 与原始 IMU 求 `T_scene_imu`；物理杆臂不乘地图尺度，不按任务重新置零。
4. `trajectory_writers.write_arm_result()` 输出 `T_scene_cam = T_scene_imu T_imu_cam` 和 `T_scene_tcp = T_scene_cam T_cam_tcp`。

求解节点时间为 `t_video + cam_time_offset`，处于该相机本机 IMU 时钟。输出时间为 `t_video + 跨相机偏移`。手动同步直接来自任务配置。双手 `audio_auto` 以第一路录像为参考，`recording.audio_sync.algorithm` 选择 `specialist` 的音量包络互相关或 `kiwi` 的频谱指纹投票与 GCC-PHAT 精对齐。两者都将同一常量偏移作用于视频、IMU 和曝光时间轴。`trajectory_writers.write_bimanual_trajectory()` 裁取已经应用偏移后的共同区间并插值。场景地图结果的 `frame_id` 来自固定地图版本。海报结果使用 `poster:<run_id>`，旧单手局部回退使用 `local:<run_id>`。

`umi/workflows/scene/synchronization.py` 负责 `run` 的音频同步：算法选择、原片和算法版本校验以及 `sync.json` 读写。

时间转换统一调用 `umi/core/timing.py`；EuRoC 文件的人工 +1 秒在读入准备数据时去掉，算法中不混入文件时钟。任务时间列统一为 `t_episode_s`。

任务锁、地图绑定、缓存复用、失败证据与发布顺序保留在 `run_episode()`；结果模块只写产物，不能更新成功指针或验收状态。第二只手求解或结果写入失败时，不发布部分成功结果。

宽度模块复用 `umi/perception/gripper/detection` 的检测器和现有毫米映射；原片像素检测只按原片和检测器缓存，换算、处理策略与固定时间偏移在缓存后应用。宽度资产不进入相机几何身份，另在 inputs.batch.episode_inputs 中记录内容哈希；episode.json 统一保留规范化测量快照和报告，不重复写各手 JSON；宽度数值与 TCP 统一保存在 trajectory.csv，逐帧诊断由 `parameters.diagnostics.gripper_width` 控制。参考 [宽度模块](../../perception/gripper/README.md)。

`gripper.tag_distance` 启用独立的夹爪标签中心三维间距。与宽度共用基础检测器，但按输出时间点独立采样、缓存角点，KB4 内参和各夹爪标签实测边长用于位姿解算；结果写入同一 CSV 的 `tag_distance_*` 三列，报告汇入 `episode.json`。尺寸资产不进入地图/相机几何身份，单独进入任务输入指纹；修改边长或质量门限重用角点，重算三维值。该测量不依赖开口锚点，不生成归一化开合量，缺失不插值填充。环境标签另由 `collection.yaml.environment_poster` 和 `environment_tags` 管理。

`umi/workflows/scene/tasks/measurements.py` 在各手重建前准备海报观测。配置了夹爪宽度或中心距时，未命中的检测共用一次原片顺序解码。只有一种夹爪测量时，由测量步骤按需检测。中心距补检仍按需执行。

## 单手独立回退

`backend_routing.execute()` 只把 `SceneFusionQualityError`、`PosterVioQualityError` 以及 poster_vio 前端明确标记的不可解码情况作为候选失败；预期手和实际准备录像都必须恰好为一只手。`umi/workflows/scene/tasks/poster_independent.py` 复用冻结的工作镜头标定、已裁切视频和原始 IMU，调用独立 `run_attempts()`；显式传入完整帧时间表，保持独立模式的尺度/初始化质量门限和重试策略，不启用实验性的初始化细化。普通输入、引擎及地图绑定错误不触发回退。

适配器返回 `models.FusedTrajectory`，其 `T_reference_imu` 明确表示声明参考系中的 IMU 位姿。`scene.scope=local`、唯一 `local:…` frame_id、`umi_local_episode` 和 `map=null` 声明局部坐标语义；所有下游读取均须遵守此声明。成功发布到 `local_results`，状态为 `local_ok/local_skipped`，不覆盖 `results`。宽度和标签间距由 `pipeline.measurements.measure()` 在发布前统一验收，返回 `ArmMeasurements`；共享场景验收/导出拒绝局部状态。查看器支持局部结果且不加载共享地图。

## 验证

`test_batch_parameters.py`、`test_ingest.py`、`test_recording_checks.py`、`test_episodes.py` 覆盖配置、入库、检查、验收与导出；`test_scene.py` 使用真实 GTSAM 验证尺度、重力、杆臂和固定坐标融合，ORB 在流程测试中使用替代程序。真实录像证据见 [PROVENANCE](../../../docs/dev/PROVENANCE.md)。

KIWI 方法来源和许可证见 [第三方来源](../../../docs/dev/PROVENANCE.md#kiwi)，运行时不依赖本机另一个 KIWI 仓库。

## 原片位置

建图原片位于 `scenes/<场景>/<批次>/mapping/`，任务原片直接位于 `scenes/<场景>/<批次>/<任务>/left/`、`right/`。`batch_inputs.task_video_sources()` 与 `batch_task_directories()` 统一发现任务原片目录，`batch_video_sources()` 再加入批次 `pending/`；不递归扫描任务的 `results/` 和 `.pipeline/`。配对依据序列号，预分组任务的左右手共用同一任务身份。

`import_video()` 直接引用本任务内已有原片，外部文件复制或显式硬链接到 `<任务>/<手>/`；没有全局场景原片根目录，其他目录布局明确拒绝。同一侧出现多个不同原片仍按歧义处理。向导和批次创建允许先准备左右手目录再登记清单。

`plan_tasks()` 保留已明确分组的 take；未分组原片通过时间与时长规则配对。`pending/` 仅暂存无确定任务归属的原片，任务清单写入成功后才删除已归位的 pending 副本。预览使用同一规划逻辑，不写文件。

## 批次地图

`inputs.batch.read_batch()` 每次读取当前 `collection.yaml`、kit.yaml 与 mapping/，返回具有明确字段的 `Session`；`save_record()` 更新最新解析记录，不让旧快照覆盖配置。单段 mapping 在 schema 5 中直接保存录像引用，per_arm 保留左右两段。run <任务> 也先刷新所属批次。

`build_scene(<批次>)` 校验完整建图输入身份后复用或构建地图；map 字段仅是复用候选。`auto` 在批次存在 `mapping/` 原片且尚未绑定地图时，会通过同一入口先构建或复用地图，再进入场景地图后端。成功结果保存当时的 kit 和完整输入身份；失败 error.json 保存 batch 与 inputs。计算期间 collection/kit 变化会拒绝绑定或发布，历史结果不改写。正式流程以 collection.yaml 和 kit.yaml 为准，不再读取旧位置的批次清单。

环境镜头 `umi-calib run <rear目录>` 的 A/B/D、恢复、验收和记录均复用 `calibration/workflow.py`。scene 只负责纯视觉建图、米制对齐和 IMU 融合，返回视频时间上的相机位置；板参考时间匹配和误差比较由标定模块统一计算。通过后生成的 `rear_calibration.json` 保持已有登记/消费接口，并绑定本次运行证据。

`config/user.yaml` 的 `features.factory_intrinsics.enabled` 先在 `inputs.batch.read_batch()` 中覆盖 kit 的来源选择，再由 `inputs.scene.validate_kit()` 将来源配置解析成具体工作/环境镜头版本。`camera_assets.select_camera()` 检查标定身份；factory 还核对 B 输入内参哈希、原片工厂射线一致性及真实引擎的前向 KB4 能力。复用检查匹配参数来源和标定，固定地图同样核对这两项；来源切换更新当前批次记录并清除不匹配的地图绑定；地图和任务成果保留各自输入与设备副本，不另存批次快照归档。配置格式与优先级见[套件说明](../../../hardware/kits/README.md#前后镜头独立选择来源)。

## 场景配置的唯一来源

`collection.yaml` 集中声明 `map`、`environment_poster`、`backend`、批次使用手和任务覆盖。`map.layout` 的用户取值为 `auto`、`shared` 或 `per_arm`，解析为 `shared_front` 或 `dual`。后端阈值按算法默认值、批次、任务、专家配置的顺序覆盖。设备配置来自 kit.yaml，建图录像来自批次 mapping/。

kit.yaml 和 collection.yaml 只通过手动创建和修改；原片按 take 分组或放入 pending，run 自动登记、补片和处理。run 与 build 共用 batch_snapshot.read_inputs/make_record/save_record 解析当前配置与生成内部快照。ingest.register_recordings() 只接收已解析的批次对象，不再接受旧命令的外部输入目录、手选择或硬链接选项；预览和实际运行在同一条路径规划任务，仅实际运行写入文件。build 仅接受批次。manifest.yaml、episode.yaml 仅从批次 .pipeline/ 当前布局读取；批次和任务清单仍支持 schema 4/5，人工选项优先从 collection.yaml.tasks 读取。已有 episode.json 结果协议保持可读。Session/Episode 是内部类型，不再作为用户命令名称。


环境海报的布局、检测和建图尺度修正配置在 `environment_poster`。地图定位是否融合海报以及融合初值的选择配置在 `backend.scene_map.poster_constraint`。融合默认随用途开启检测。实现位于 `umi/estimation/fixed_map/poster.py`，初值与投影位于 `umi/perception/poster/geometry.py`。单手每段录像求一个海报位姿，双手开启融合时求两手共享的海报位姿。布局随结果冻结，地图尺度保持固定。`staged` 先求地图与 IMU 基线再进行海报优化。`joint` 从视觉种子构建包含地图先验、IMU 和海报角点的最终图。环境检测按采样帧做周期性全图扫描，中间帧跟随上次标签窗口。[海报约束质量与实测评估](../../../docs/03-场景处理.md#海报约束质量和双手实测评估)

代码内部将 `tags.mapping.enabled` 和 `tags.mapping.scale.enabled` 映射为 `tag_mapping_enabled`、`tag_mapping_scale_enabled`；二者默认关闭。`umi/estimation/mapping.py` 保持视觉地图形状并处理米制镜头杆臂，合并后由 `geometry.metric_alignment(fixed_scale=...)` 精确固定尺度重做惯性验收。地图目录保留冻结布局及角点，任务不复用建图海报位姿。

位置支持由 `umi/perception/poster/quality.py` 复用 `geometry.py` 的解析 Jacobian、共享角点噪声和 Huber 权重计算，场景两种融合方法与独立图共享实现。`umi/estimation/constraint_quality.py` 处理精确支持事件缺口和输出时间单元关联；`frames.csv`、运行报告和训练导出沿现有通道携带来源 mask，未知历史证据不补为零。`umi.core.geometry.relative_pose_series()` 为双手表和训练采样共用；`umi/dataset/relative_pose_evaluation.py` 复用资产校验、TCP 读取、重采样及标定误差统计，提供实测固定夹具评估，不对普通录像假定相对位姿恒定。

## 镜头参数来源

`kit.yaml` 用 `intrinsics.front/rear` 选择 measured/factory，默认分别为 measured/factory。`registry.current_camera()` 将它解析为本次固定的 front/rear 资产身份；`validate_kit()` 读取批次快照时不查询当前选择。front/rear 使用独立来源字段，允许混用；同一设备参与建图和任务时仍要求实际参数一致，不同设备可选择不同来源。`register_rear()` 在档案锁内发布当前资产并更新该来源的当前选择；front 安装保留 rear 的选择。

后续 run 从 kit 重新解析，变化清除不匹配的地图绑定；已开始任务和已有成果继续使用冻结参数。设备查询默认只检查当前选择，已有结果校验自己的参数副本；kit 不填写版本号。

运行预算统一在 `config/operations/runtime.yaml`，机器覆盖用 `config/operations/runtime.local.yaml`；夹爪标签检测预算在 `config/operations/gripper_tag_detection.yaml`。`--show-config` 同时展示配置来源与实际 CPU/线程预算，启动终端输出精简摘要。覆盖顺序与缓存规则见 [配置说明](../../../config/README.md#运行与查看器默认值)。

### 海报后端首尾裁剪

`backend.poster_edge_trim` 默认关闭。开启时，`umi/workflows/scene/tasks/recording_intervals.py` 在首个适用的后端前计算同步后的保留区间，供两个海报后端和启用海报约束的 `scene_map` 共用。兼容读取的 `recording.trim` 作用于两个海报后端。边界由可靠多标签 PnP 和局部陀螺仪连续性确定，内部缺口保留。裁剪视图保留原片时钟与完整 IMU，按帧区间缓存镜头编码与地图定位。各候选复用任务内的环境标签观测，夹爪中心距补检和验收使用保留区间。裁剪与 VIO 初始化复用输入相同的 PnP 候选。`PreparedRecording.source_frame_start` 将前镜头特征和结果帧表映射回 INSV。每手结果的 `video_time_offset_s` 记录发布视频起点在任务时钟中的位置。裁剪规则纳入任务配置指纹。
