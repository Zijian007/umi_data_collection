# 场景代码阅读指南

`umi-scene run <路径>` 是场景处理入口。`umi/apps/scene.py` 解析参数，`run.py` 调度任务，`maps/binding.py` 按后端需要绑定或构建地图。用户操作见[场景指南](../../docs/03-场景处理.md)，配置与快照的归属见[SSOT 与权威来源](../../docs/dev/SSOT与权威来源.md)，求解和坐标约定见[算法实现约定](../../docs/dev/算法实现约定.md)。

## 文件职责

```text
umi/scene/
├── run.py                     批次和任务执行、并发调度
├── collection/
│   ├── configuration.py       采集声明、任务继承和专家覆盖
│   ├── config_report.py        生效配置与来源
│   ├── kit.py                  套件和设备声明
│   ├── batch.py                批次与任务清单、准备和登记编排
│   └── ingest.py               Organizer 交接、发现、配对和登记
├── inputs/
│   ├── recordings.py           录像读取、预检、时间戳、IMU 和标定冻结
│   ├── synchronization.py      双手同步
│   ├── recording_intervals.py  首尾裁剪与保留区间
│   └── identity.py             复用所需的输入指纹
├── routing.py                 候选顺序、回退和尝试记录
├── maps/
│   ├── binding.py              按需建图和版本绑定
│   ├── build.py                构建并发布地图版本
│   ├── versions.py             加载、校验和导入版本
│   └── epochs.py               读计划、选 epoch、构建后写回版本
├── measurements.py             环境标签、夹爪宽度和中心距
├── publication.py              TCP 轨迹、双手时间对齐、发布与复用
├── episodes.py                 任务发现与运行状态
├── relative_pose_evaluation.py   双手固定夹具实测评估
├── cleanup.py                   运行、缓存和地图保留策略
└── errors.py                    场景流程错误类型
```

数值算法在 `umi/reconstruction/`，磁盘布局、结果格式和阶段缓存在 `umi/store/`。三个后端位于 [reconstruction/backends](../reconstruction/backends)，各自从 `run.py` 进入。路由提供检测与轨迹写入函数，后端通过 `TaskInputs` 接收这些函数。后端共用 `TaskInputs` 和 `ArmExecutionResult`，后者的 `coordinate_record` 声明实际输出坐标。固定地图融合和 Poster VIO 共用 IMU 因子，`poster_independent` 调用内部逐段重建引擎。

## 采集管理

批次和显式任务共用 `collection/batch.py` 的 `prepare_for_run()`。`configuration.resolve_task_configuration()` 解析使用手、后端、同步和地图请求。批次准备通过 `read_inputs()`、`make_record()` 和 `save_record()` 冻结套件及批次快照，再由 `ingest.register_recordings()` 发现和登记原片。预览使用同一套规划并在内存里构造快照。错误、待补齐和忽略的原片会进入批次失败摘要，其他批次可继续处理。

各批次独立持有 `.pipeline/maps/` 及当前地图指针，任务内部数据集中在同一批次 `.pipeline/tasks/<任务>/`。目录层次为 `hardware/kits/<套件>/kit.yaml` 与 `scenes/<场景>/<批次>/.pipeline/manifest.yaml`，任务位于批次目录下。输入读取校验目录与清单身份一致；查看器从批次清单读取套件身份。

每个套件只有一个 `hardware/kits/<套件>/kit.yaml`；`collection.kit.read_kit()` 读取当前配置，`validate_kit()` 校验当前配置与批次快照的共同结构。处理批次时核对实际测量并把配置保存到 `.pipeline/manifest.yaml`（schema 5），run 每次读取当前配置，记录不反向覆盖 kit.yaml。套件没有独立版本目录或指针文件，相机按前后镜头来源选择当前参数，内部历史身份与地图版本保留用于追溯。夹爪配置包含必填的 gripper.tcp、可选的 gripper.width、gripper.tag_distance 和 gripper.mask，路径相对 kit.yaml；输入入口统一解析测量，计算前复制到运行目录 devices/<序列号>/，后续直接使用解析值与副本。

场景名称和当前套件 ID 从目录推导，上级场景和批次清单由固定层级确定。生成清单省略空描述、默认模式、空覆盖和未绑定地图；当前套件省略 ID，批次快照仍保存 ID 和标定资产版本。批次快照不复制套件描述。

新任务原片清单 schema 5 仅保存原片身份与引用，集中在批次 `.pipeline/tasks/<任务>/episode.yaml`。人工选项 instruction、arms、sync 位于批次 `collection.yaml.tasks`。episode.yaml 中不接受这些字段。批次必须明确填写 `arms`，任务默认继承批次，可声明其非空子集。双手默认用 `audio_auto` 估算偏移。

指令进入结果 episode.json；修改指令或同步按有效任务选项的规范化哈希失效（仅修改注释不失效），需重新运行。

`<批次>/.pipeline/tasks/<任务>/state.json` 分别记录最近尝试 `attempt`、最近选用的发布运行 `published_run`，身份与完整性从清单读取；attempt 只保存运行引用和输入哈希，错误原因及预检报告路径从运行目录读取，不重复落盘。录制检查在算法执行前落盘，算法失败仍有原始检查证据。

录像准备保留已解析清单，独立回退沿用同一份数据。任务发布从轨迹写入阶段取得逐帧证据和原片检查记录，供报告与运行清单使用。

## 算法执行

```text
umi-scene run → run.run()
    batch.prepare_for_run()             准备批次与登记原片
    run.run_episodes()                  按并发上限调度任务
    run.resolve_inputs()                相机、输入指纹；固定 scene_map 在此准备地图
    publication.reuse_episode()         校验并复用已发布结果
    run.execute_episode()               冻结资产，准备录像与任务时间轴
    routing.execute()          按 backend.auto_order 尝试候选
        scene_map → maps.binding.resolve_auto_map() → select_map()
                    在 auto 到达该候选时加载或构建所需地图
                    多地图批次由 maps.epochs.build_epoch() 构建所需 epoch
        scene_map.execute_task()              固定地图定位与融合
        poster_independent.execute_task()     逐手重建与海报整合
        poster_vio.execute_task()             冻结观测，解海报与双手联合图
        publication.write_arm_result()  生成各手 TCP 轨迹
    publication.write_bimanual_trajectory()  生成双手对齐轨迹
    publication.write_episode_result()  写报告、证据和任务元数据
    episodes.record_attempt()           更新运行状态
```


`run_episodes()` 调度任务并汇总结果。以下 `BatchMap` 构建与绑定规则适用于显式 `scene_map` 及带 mapping 原片的 `auto`；`poster_independent` 和 `poster_vio` 跳过地图。场景地图流程每次调用按批次持有独立 `BatchMap`。同批首次任务在锁内构建/校验并固定地图，后续任务复用同一绑定；共同建图失败只执行一次。每个任务仍核对自身标定与地图匹配，复用前检查批次配置、建图原片及地图资产的文件身份，变化即拒绝继续；发布前复核原片与地图资产，已开始的任务仍使用冻结配置，不因外部配置编辑而中断。不同批次的地图准备互不阻塞，只读预检不落盘。

`--jobs 1` 且双手可用 CPU 至少 8 个时，按手拆分子进程的 CPU 集合，并行定位、融合、检测和结果写入；核心不足、单手或 `--jobs > 1` 时按手串行。任务级发布等待所有手成功；任一手失败时等其他线程退出后保存失败现场，不替换旧成功入口。

任务融合因视觉定位失败而保留失败现场时，若存在连续无视觉区间，运行目录会写入 `<arm>/no_visual/gap_*.mp4` 和 `<arm>/no_visual/manifest.json`。片段默认包含前后各 0.5 秒上下文；原始视频、视觉定位结果和 IMU 不会被修改。

多个已通过验收的地图候选会在 `metric_alignment.json` 中记录 `selection_score`，按固定尺度复核、对齐 RMS、尺度不确定度、条件数和覆盖证据选择；硬性验收门槛保持不变。

缓存只依据数据输入复用，不包含代码或引擎身份；`--force` 通过 `infra.file_store.recompute()` 让本次运行开始前的缓存全部失效，本次运行中新生成的仍可复用。完整地图输入保留原片路径、时间和 SHA256 以便追溯，`reuse_key` 按内容、相关配置和相机资产比较，不把文件路径或夹爪资产作为地图身份。工作镜头遮手图仍属于地图输入。

建图和任务通过 `cache.StageCache` 共用缓存存储机制，纯输入身份由 `cache.preparation_inputs()` / `visual_inputs()` 构造；地图专属镜头资产筛选与复用键也由 `cache` 生成。`umi/reconstruction/estimation/mapping.py` 的地图求解不导入地图流水线模块（由分层合约保证）。

场景写入锁内，`StageCache.stage()` 先验证缓存清单和资产哈希，再将产物复制到本次工作目录；读写不会共享可修改文件。准备缓存只保存数据集和镜头视频，当前相机设置、遮手图、时间偏移与算法上下文重新生成；视觉缓存使用实际镜头视频、相机设置、ORB 参数与遮手图（仅建图）；定位另包含地图内容和子地图索引。尺度与放置每次重新验收，不缓存最终数值解。

阶段正常完成后才发布 `ready.json`；缓存损坏重算。`umi-scene run --force` 重算本次需要的地图和定位，验收失败后的重试也刷新视觉建图和定位，避免重复读取同一失败候选。成功阶段缓存独立于版本 `work/`，后续阶段失败或清理工作目录不会丢失它。`manifest.json` / `error.json` 中的 `timings` 记录实际耗时与缓存状态。

`retry.max_tries` 是最大尝试次数。连续两轮若同一相机/镜头的全部被拒子地图，其帧索引、IMU 时间、位姿以及拒绝原因都相同，建图会提前停止，避免继续重建同一失败结果。轨迹变化、拒绝原因变化、没有候选轨迹证据或引擎暂时退出时不按重复轨迹提前停止，仍保留配置的重试次数。尺度门限保持原值；失败现场和已有成功地图均保留。

地图成功清单或失败 `error.json` 的 `retry` 是本次尝试记录的唯一汇总，包含每轮状态、拒绝证据摘要、`repeat_of_attempt` 和 `stop_reason`。`repeated_visual_failure` 表示因重复失败停止，`attempt_limit` 表示达到次数上限，`accepted` 表示通过。固定尺度复核失败时，日志和拒绝记录还标出对应的原片秒数；每轮完整验收数据继续保存在 `work/try_<次数>/<角色>/localize/m<子地图>/metric_alignment.json`。

场景与独立轨迹通过 `perception.runtime.visual_settings()` 共用镜头 YAML 写入、遮手图准备与视觉运行设置。场景显式选择本设备已验收的各镜头完整标定（measured/factory 内参均可，B 均实测）；独立 dual 使用本设备的登记参数或从 INSV 工厂数据推导环境镜头参数。

场景录像准备直接构造 `perception.models.VisualContext`，包含已解析参数、帧数与 CPU 设置；不构造 `TrajectoryRequest`。共用的视觉设置为 `VisualSettings`，内部逐段轨迹引擎的融合设置由 `ReconstructionSettings` 扩展；场景融合直接接收已解析参数，不生成单段融合脚本的配置文件。

后端用具名的手级结果传递轨迹、坐标声明和待联合求解的海报输入。启用共享海报时，两手基线均完成后再联合求解。`scene_map.run.localize_with_cache()` 先尝试恢复已校验的观测，缺失或损坏时计算并发布完整缓存；引擎失败保留独立工作目录与日志，诊断复制仍在缓存命中或计算成功后执行。

任务准备和定位数据保存在 `<批次>/.pipeline/tasks/<任务>/cache/`；任务失败时保留最近一次失败现场。进度写 stderr，结构化 JSON 写 stdout。

每次真正执行的任务会在 `<批次>/.pipeline/tasks/<任务>/logs/<run-id>/` 持久化 `run.log`、`events.jsonl` 和 `run_manifest.json`。前者保留人读进度，第二个记录逐行的阶段/终端事件，最后一个记录输入指纹、状态、起止时间、耗时、结果路径及同一任务的 `runs/` 现场；成功和失败（包括原片预检拒绝）都写入。日志用于跟踪运行进度。质量摘要位于发布清单，完整融合证据位于内部运行目录的 `diagnostics.json`。

地图清单的建图定义只存于 `inputs.definition`。地图展示数据生成后移除转换中间文件和空日志，保留有内容的日志及失败现场；已发布版本不因整理而改写。

`publication.write_arm_result()` 将每手 TCP 位姿与已配置的夹爪测量写入 `trajectory.csv`，返回供双手对齐使用的 `ArmTrajectory`。发布阶段在内部运行目录组装 `published/`，写出 `report.txt` 和 `data/episode.json`（地图系和局部结果为 schema 5，海报系为 schema 6），校验值按发布文件和内部证据分别登记，再原子更新任务的 `results/`。逐帧信息、融合诊断与冻结设备文件留在同一次运行目录，由 `umi.store.result_io` 统一定位。

有标签间距测量时，任务层 `tag_distance.<手>` 保存报告。`scene.frame_id` 记录坐标身份，`session` 记录批次名。输入记录保存地图与同步的内容摘要，`code_revision` 记录代码版本供追溯。查看器由 TCP 和固定外参重建相机和 IMU 的显示轨迹。

`umi.store.result_io` 提供只读的摘要和设置适配，由查看器和场景流程共用；当前结果从任务层 `episode.json` 取值。查看器校验地图路径、版本、固定坐标与内容身份后直接加载共享地图。

地图资产错误由 `scene.errors.MapUnavailableError` 表示，兼容失败由其子类 `MapCompatibilityError` 表示；`PosterVioQualityError` 只表示 poster_vio 的质量不足。`auto` 只接收这些结构化地图/质量失败，普通输入、引擎和程序异常继续抛出；每次候选写入 `reconstruction_attempts/attempts.json`，并在独立目录中运行，不混用失败轨迹。无地图的支持缺口和双手重力夹角使用 `backends.poster_independent` 独立阈值，见[配置说明](../../config/README.md)。

`maps.versions.validate_manifest()` 只校验已读入清单的结构、版本、坐标、地图条目、尺度和放置变换，不访问文件或查询引擎。`load_published()` 负责指针解析、文件读取、哈希与当前场景兼容检查，各类错误仍在原边界转换。

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

地图尺度复用 `umi.reconstruction.estimation.fusion.alignment.align_lenses()` 的运动段筛选，并由 `fusion.alignment.estimate_lens_alignment(shared_map=True)` 共用一个尺度、重力和零偏重新求解。场景层仅适配观测、检查可靠段一致性与可观测性、保存验收诊断，不复制另一套尺度求解器。分段端点速度隔离，地图变换始终只有一个。建图与任务融合共用 `inertial.estimate_gyro_bias()` 的初始陀螺仪零偏规则。

定标后由 `fusion.alignment.fixed_map_residuals()` 对所有运动链复核固定尺度下的位置方程；不重新拟合段尺度。每条链都会写入 `verified`、`inconsistent` 或 `insufficient_evidence` 状态，局部可靠但整图不一致的候选仍被拒绝，逐段结果保存于 `evidence.alignment.fixed_map_validation`。

外参函数遵守“目标系_from_来源系”：`imu_from_camera()` 返回 `T_imu_cam`，`camera_from_tcp()` 返回 `T_cam_tcp`。ORB 只使用图像，IMU 用于米制对齐和离线融合。地图放置为 `p_scene = R (s p_map) + t`，物理杆臂不乘视觉尺度。双手输出使用任务配置的固定偏移；固定场景不按任务重新置零。

## 双手坐标与时间的阅读入口

单手基线求解入口是 `scene_map.reconstruct_arm_in_scene()`。`backend.scene_map.poster_constraint` 为 `staged` 或 `joint` 时，将环境镜头的海报角点与固定地图先验、IMU 放入同一张图优化。`staged` 先求地图与 IMU 基线，再以基线初始化海报图。`joint` 从视觉种子构建最终图。双手任务先分别求两手基线，再在 `poster.refine_shared_poster_pose()` 中共享一个刚性海报位姿。两手可以在不同时间看到不同 ID，已知布局负责建立约束。完全没有有效海报观测的手保留基线并记录 `not_observed_shared_board`。共享坐标来自建图时冻结的地图放置。

`umi/reconstruction/estimation/fixed_map/poster.py` 的单双手路径共用 `_tag_geometry()`、`_remap_map_observations()` 和 `_support_evidence()`。每只手在本轮求解内复用自己的镜头模型与标签几何，异常剔除后的重试重新准备观测；原地图先验只在原输出时刻采样，再重编号到含海报时刻的节点，不给新节点复制额外地图证据。公共组图因子及质量门槛继续使用原实现。

`umi/scene/publication.py` 从一次 `quality_summary()` 取坐标范围和公共指标；帧约束区间计算一次后用于分类及视觉/IMU 视图。逐帧来源仍保存在报告中，字段与 `frames.csv` 引用保持当前协议。

1. `Scene.role(arm, lens)` 选择固定地图：`shared_front` 两手使用同一工作镜头地图；`dual` 使用各自工作镜头地图与共享环境镜头地图。
2. `VisualObservations.placed(..., reference_frame='scene')` 把 `T_map_cam` 转入场景系：位置为 `R (s p_map) + t`，朝向为 `R R_map_cam`。
3. `fusion.fuse()` 使用各镜头 `T_cam_imu` 与原始 IMU 求 `T_scene_imu`；物理杆臂不乘地图尺度，不按任务重新置零。
4. `publication.write_arm_result()` 输出 `T_scene_cam = T_scene_imu T_imu_cam` 和 `T_scene_tcp = T_scene_cam T_cam_tcp`。

求解节点时间为 `t_video + cam_time_offset`，处于该相机本机 IMU 时钟。输出时间为 `t_video + 跨相机偏移`。手动同步直接来自任务配置。双手 `audio_auto` 以第一路录像为参考，`sync.algorithm` 选择 `specialist` 的音量包络互相关或 `kiwi` 的频谱指纹投票与 GCC-PHAT 精对齐。两者都将同一常量偏移作用于视频、IMU 和曝光时间轴。`publication.write_bimanual_trajectory()` 裁取已经应用偏移后的共同区间并插值。地图系结果的 `frame_id` 来自固定地图版本。三个后端的海报系结果使用 `poster:<原片与布局摘要>`，单手局部回退使用 `local:<run_id>`。

`umi/scene/inputs/synchronization.py` 负责 `run` 的音频同步：算法选择、原片和算法版本校验以及 `sync.json` 读写。

时间转换统一调用 `umi/core/timing.py`；EuRoC 文件的人工 +1 秒在读入准备数据时去掉，算法中不混入文件时钟。任务时间列统一为 `t_episode_s`。

`run.py` 负责锁、地图选择和执行顺序。`publication.py` 在所有使用手的轨迹和测量写入成功后，再更新 `results/` 或 `local_results/` 及运行状态。

宽度模块复用 `umi/perception/gripper/frame_detection.py` 的检测器和现有毫米映射；原片像素检测只按原片和检测器缓存，换算、处理策略与固定时间偏移在缓存后应用。宽度资产不进入相机几何身份，另在 inputs.identity.episode_inputs 中记录内容哈希；episode.json 统一保留规范化测量快照和报告，不重复写各手 JSON。宽度数值与 TCP 统一保存在 `trajectory.csv`。参考 [宽度模块](../perception/gripper/README.md)。

`gripper.tag_distance` 启用独立的夹爪标签中心三维间距。与宽度共用基础检测器，但按输出时间点独立采样、缓存角点，KB4 内参和各夹爪标签实测边长用于位姿解算；结果写入同一 CSV 的 `tag_distance_*` 三列，报告汇入 `episode.json`。尺寸资产不进入地图/相机几何身份，单独进入任务输入指纹；修改边长或质量门限重用角点，重算三维值。该测量不依赖开口锚点，不生成归一化开合量，缺失不插值填充。环境海报由批次 `poster` 与 `config/algorithms/reconstruction.yaml` 中的海报参数管理。

`umi/scene/measurements.py` 在各手重建前准备海报观测。配置了夹爪宽度或中心距时，未命中的检测共用一次原片顺序解码。只有一种夹爪测量时，由测量步骤按需检测。中心距补检仍按需执行。

## 单手独立回退

`routing.execute()` 在地图不可用或后端返回明确的质量拒绝时继续尝试下一个候选，`poster_vio` 前端无法解码也可触发回退。显式 `scene_map` 的局部回退要求预期手和实际准备录像都恰好为一只手。`umi/reconstruction/backends/poster_independent/run.py` 复用冻结的工作镜头标定、已裁切视频和原始 IMU，调用独立 `run_attempts()`；显式传入完整帧时间表，保持独立模式的尺度/初始化质量门限和重试策略，不启用实验性的初始化细化。输入和引擎错误会终止当前任务。

适配器返回 `models.FusedTrajectory`，其 `T_reference_imu` 明确表示声明参考系中的 IMU 位姿。`scene.scope=local`、唯一 `local:…` frame_id、`umi_local_episode` 和 `map=null` 声明局部坐标语义；所有下游读取均须遵守此声明。成功发布到 `local_results`，状态为 `local_ok/local_skipped`，不覆盖 `results`。宽度和标签间距由 `measurements.measure()` 在发布前统一验收，返回 `ArmMeasurements`；查看器支持局部结果且不加载共享地图。

## 验证

`test_scene_configuration.py` 覆盖批次和任务配置，`test_collection.py` 覆盖采集交接和原片入库，`test_scene_runs.py` 覆盖任务状态、运行和结果发布。`test_scene.py` 覆盖原片检查，并使用真实 GTSAM 验证尺度、重力、杆臂和固定坐标融合。流程测试使用 ORB 替代程序。真实录像证据见 [PROVENANCE](../../docs/dev/PROVENANCE.md)。

KIWI 方法来源和许可证见 [第三方来源](../../docs/dev/PROVENANCE.md#kiwi)，运行时不依赖本机另一个 KIWI 仓库。

## 原片位置

建图原片位于 `scenes/<场景>/<批次>/mapping/`，任务原片直接位于 `scenes/<场景>/<批次>/<任务>/left/`、`right/`。`collection.batch.task_video_sources()` 与 `batch_task_directories()` 统一发现任务原片目录，`batch_video_sources()` 再加入批次 `pending/`；不递归扫描任务的 `results/` 和 `.pipeline/`。配对依据序列号，预分组任务的左右手共用同一任务身份。

`import_video()` 直接引用本任务内已有原片，外部文件复制或显式硬链接到 `<任务>/<手>/`；没有全局场景原片根目录，其他目录布局明确拒绝。同一侧出现多个不同原片仍按歧义处理。先准备左右手目录，再用 `run` 登记并处理批次。

`plan_tasks()` 保留已明确分组的 take；未分组原片通过时间与时长规则配对。`pending/` 仅暂存无确定任务归属的原片，任务清单写入成功后才删除已归位的 pending 副本。预览使用同一规划逻辑，不写文件。

## 批次地图

`collection.batch.read_inputs()` 从当前 `collection.yaml`、`kit.yaml` 和建图原片准备本次批次，`save_record()` 写入快照。`read_batch()` 解析快照中的套件、建图原片和绑定地图，返回 `Session`。单段建图保存单个录像引用，`per_arm` 分别保存左右手引用。运行单个 take 时也先准备所属批次。

`maps.binding.build_scene()` 在 `run` 需要建图时校验本批次的建图输入，再复用或构建版本。`auto` 的地图候选通过同一入口绑定普通地图，待建 epoch 则按任务选择构建。成功结果保存当时的 kit 和完整输入身份；失败 error.json 保存 batch 与 inputs。计算期间 collection/kit 变化会拒绝绑定或发布，历史结果不改写。正式流程以 collection.yaml 和 kit.yaml 为准，不再读取旧位置的批次清单。

环境镜头 `umi-calib run <rear目录>` 的 A/B/D、恢复、验收和记录均复用 `calibration/workflow.py`。scene 只负责纯视觉建图、米制对齐和 IMU 融合，返回视频时间上的相机位置；板参考时间匹配和误差比较由标定模块统一计算。通过后生成的 `rear_calibration.json` 保持已有登记/消费接口，并绑定本次运行证据。

kit 的 `intrinsics.front/rear` 是唯一来源选择，由 `collection.kit.validate_kit()` 解析成具体工作/环境镜头版本。`recordings.select_camera()` 检查标定身份；factory 还核对 B 输入内参哈希、原片工厂射线一致性及真实引擎的前向 KB4 能力。复用检查匹配参数来源和标定，固定地图同样核对这两项；来源切换更新当前批次记录并清除不匹配的地图绑定；地图和任务成果保留各自输入与设备副本，不另存批次快照归档。配置格式与优先级见[套件说明](../../hardware/kits/README.md#前后镜头独立选择来源)。

## 场景配置的唯一来源

`collection.yaml` 集中声明 `arms`、`sync`、`poster`、`map`、`backend` 和任务覆盖。任务 `tasks.<take>` 可覆盖指令、使用手、同步、地图选择和部分后端选项，由 `umi/config/collection_schema.py` 的 `normalize_collection()` 校验。`map.layout` 的用户取值为 `auto`、`shared` 或 `per_arm`，解析为 `shared_front` 或 `dual`。后端阈值按算法默认值、批次、任务、专家配置的顺序覆盖。设备配置来自 kit.yaml，建图录像来自批次 mapping/。

`kit.yaml` 由采集者维护，`collection.yaml` 可手工填写或由 Organizer 生成。原片按 take 分组或放入 `pending/`。`run` 登记原片并处理；`--dry-run` 使用同一规划逻辑预览，实际运行写入 schema 5 的 `manifest.yaml` 和 `episode.yaml`。

环境海报的布局和检测开关配置在 `poster`，建图尺度用途在 `map.poster_scale`。地图定位是否融合海报以及融合初值的选择配置在 `backend.scene_map.poster_constraint`。融合默认随用途开启检测。实现位于 `umi/reconstruction/estimation/fixed_map/poster.py`，初值与投影位于 `umi/perception/poster/geometry.py`。单手每段录像求一个海报位姿，双手开启融合时求两手共享的海报位姿。布局随结果冻结，地图尺度保持固定。`staged` 先求地图与 IMU 基线再进行海报优化。`joint` 从视觉种子构建包含地图先验、IMU 和海报角点的最终图。环境检测按采样帧做周期性全图扫描，中间帧跟随上次标签窗口。[海报约束质量与实测评估](../../docs/03-场景处理.md#海报约束质量和双手实测评估)

`map.poster_scale` 默认 `off`。设为 `diagnose` 收集尺度证据，设为 `correct` 将海报尺度用于米制校正。`umi/reconstruction/estimation/mapping.py` 保持视觉地图形状并处理米制镜头杆臂，合并后由 `geometry.metric_alignment(fixed_scale=...)` 精确固定尺度重做惯性验收。地图目录保留冻结布局及角点，任务不复用建图海报位姿。

位置支持由 `umi/perception/poster/quality.py` 复用 `geometry.py` 的解析 Jacobian、共享角点噪声和 Huber 权重计算，场景两种融合方法与独立图共享实现。`umi/reconstruction/estimation/constraint_quality.py` 处理精确支持事件缺口和输出时间单元关联；`frames.csv` 和运行报告携带来源标记，未知历史证据不补为零。`umi.core.geometry.relative_pose_series()` 供双手表和 `relative_pose_evaluation.py` 中的实测固定夹具评估复用。

## 镜头参数来源

`kit.yaml` 用 `intrinsics.front/rear` 选择 measured/factory，默认分别为 measured/factory。`registry.current_camera()` 将它解析为本次固定的 front/rear 资产身份；`validate_kit()` 读取批次快照时不查询当前选择。front/rear 使用独立来源字段，允许混用；同一设备参与建图和任务时仍要求实际参数一致，不同设备可选择不同来源。`register_rear()` 在档案锁内发布当前资产并更新该来源的当前选择；front 安装保留 rear 的选择。

后续 run 从 kit 重新解析，变化清除不匹配的地图绑定；已开始任务和已有成果继续使用冻结参数。设备查询默认只检查当前选择，已有结果校验自己的参数副本；kit 不填写版本号。

运行预算统一在 `config/operations/runtime.yaml`，单次运行用 `--cores`、`--jobs` 或环境变量调整；夹爪标签检测预算在 `config/operations/gripper_tag_detection.yaml`。`--show-config` 同时展示配置来源与实际 CPU/线程预算，启动终端输出精简摘要。覆盖顺序与缓存规则见 [配置说明](../../config/README.md#运行与查看器默认值)。

### 海报后端首尾裁剪

`backend.edge_trim` 默认关闭。开启时，`umi/scene/inputs/recording_intervals.py` 在首个适用的后端前计算同步后的保留区间，供两个海报后端和启用海报约束的 `scene_map` 共用。边界由可靠多标签 PnP 和局部陀螺仪连续性确定，内部缺口保留。裁剪视图保留原片时钟与完整 IMU，按帧区间缓存镜头编码与地图定位。各候选复用任务内的环境标签观测，夹爪中心距补检和验收使用保留区间。裁剪与 VIO 初始化复用输入相同的 PnP 候选。`PreparedRecording.source_frame_start` 将前镜头特征和结果帧表映射回 INSV。每手结果的 `video_time_offset_s` 记录发布视频起点在任务时钟中的位置。裁剪规则纳入任务配置指纹。
