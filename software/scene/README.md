# 场景代码阅读指南

用户入口为 `umi-scene`。`run <批次>` 先由 `management/processing.py` 读取 batch.yaml，自动保存设备快照和登记原片，再调用 workflow 的地图与任务流水线。`export --format training` 调用 software.dataset。标定统一由 umi-calib 发起，查看由 umi-view 提供。

从 `cli.py` 找命令入口，再读 `pipeline/map_lifecycle.py` 的 `build_scene()` 或 `workflow.py` 的 `run_episode()`。用户操作见 [场景指南](../../docs/场景处理.md)。

模块通过真实目录直接导入；`__init__.py` 仅说明包职责，不注册历史别名或预先加载全部子模块。

## 文件职责

```text
software/scene/
├── cli.py                     参数与直接分发
├── workflow.py                任务编排、结果发布与失败收尾
├── models.py                  参数、清单和阶段数据结构
├── map_cache.py               建图内容身份、阶段缓存与计时
├── management/                采集配置与任务生命周期
│   ├── processing.py          从批次声明自动登记并发起处理
│   ├── batches.py             批次输入、资产校验与设备/建图快照
│   ├── episodes.py            任务状态、人工验收、导出
│   ├── ingest.py              原片配对、任务登记
│   ├── camera_assets.py       完整镜头标定选择、安装资产冻结与相机身份
│   └── inputs.py              清单读取、输入指纹与校验
├── spatial/                   多相机空间几何、视觉定位与图优化融合
│   ├── fusion.py              视觉与 IMU 融合、初始陀螺仪零偏
│   ├── poster.py              固定地图下的单海报角点联合融合
│   ├── poster_scale.py        建图海报尺度估计、方差合并和惯性复核
│   ├── geometry.py            坐标变换、惯性对齐与几何计算
│   ├── mapping.py             地图选择、米制对齐、场景放置与资产冻结
│   ├── rear_calibration.py    环境镜头轨迹生成，接共享标定 workflow
│   ├── relocalization.py      反向连续段与正向位姿一致性验收、缺帧补全
│   └── slam.py                ORB 建图、定位、图集导出
└── pipeline/                  原片预处理、质量预检与轨迹结果交付
    ├── recording_checks.py    原片和任务完整性检查
    ├── recordings.py          原片归档、时间戳/IMU/镜头视频准备
    ├── map_lifecycle.py       建图、地图发布、校验与批次绑定
    ├── reconstruction.py      任务准备、定位缓存与每手融合
    ├── fallback.py            单手质量失败后的独立局部重建
    ├── measurements.py        冻结配置下的夹爪宽度、标签中心距与拒绝证据
    └── results.py             轨迹、双手表、算法报告
```

`inputs.py` 负责清单读取与参数校验，`camera_assets.py` 选择并冻结设备测量；`models.py` 保存阶段数据，CalibratedCamera 按需计算并缓存小型资产哈希。`workflow.py` 和 `episodes.py` 直接使用它；验收模块不导入 workflow。公共原子文件和锁由 `software/storage.py` 提供，设备版本化标定资产由 `software/registry.py` 管理。

## 采集管理

```text
手写 kit.yaml → inputs.read_kit()         解析相机版本与夹爪文件路径
手写 batch.yaml → batches.read_inputs()   解析批次、套件与建图来源
    batches.make_record() / save_record()  构造快照、检查并发修改、按需写入
    inputs.read_scene() / read_kit()
    registry.pin() / version_assets()     读取全局版本化标定资产

run → processing.prepare_batch() → ingest.register_recordings()
                                         自动配对、登记与补片；预览共用规划
    recordings.inspect() / import_video() 原片身份、哈希、保留原名归档
    inputs.read_episode()                 手、序列号与输入完整性
    ingest._write_episode()               有变化才更新清单和任务状态
```

各批次独立持有 `.umi/maps/` 及当前地图指针，任务内部数据集中在同一批次 `.umi/tasks/<任务>/`。目录层次为 `kits/<套件>/kit.yaml` 与 `scenes/<场景>/<批次>/.umi/manifest.yaml`，任务位于批次目录下。输入读取校验目录与清单身份一致；查看器从批次清单读取套件身份。

每个套件只有一个 `kits/<套件>/kit.yaml`；`inputs.read_kit()` 读取当前配置，`validate_kit()` 校验当前配置与批次快照的共同结构。处理批次时核对实际测量并把配置保存到 `.umi/manifest.yaml`（schema 5；兼容读取历史 schema 4），run 每次读取当前配置，记录不反向覆盖 kit.yaml。套件没有独立版本目录或指针文件，相机标定与地图保留固定版本。夹爪配置包含必填的 gripper.tcp、可选的 gripper.width、gripper.tag_distance 和 gripper.mask，路径相对 kit.yaml；输入入口统一解析测量，计算前复制到运行目录 devices/<序列号>/，后续直接使用解析值与副本。

场景名称和当前套件 ID 从目录推导，上级场景和批次清单由固定层级确定。生成清单省略空描述、默认模式、空覆盖和未绑定地图；当前套件省略 ID，批次快照仍保存 ID 和标定资产版本。批次快照不复制套件描述。

新任务原片清单 schema 5 仅保存原片身份与引用，集中在批次 `.umi/tasks/<任务>/episode.yaml`。人工选项 instruction、expected_arms、synchronization 位于外层任务 `task.yaml`；读取历史清单时合并这些选项，重新登记时提取到 task.yaml。双手默认零偏移但未经确认；批次默认使用套件全部手，任务默认继承批次。缺片不会隐式降为单手。

指令进入结果 episode.json 和导出清单；修改指令或同步按有效任务选项的规范化哈希失效（仅修改注释不失效），需重新 run 和验收。训练格式由独立 [dataset 模块](../dataset/README.md) 负责，scene 不承担训练采样。

`<批次>/.umi/tasks/<任务>/state.json` 只保存 attempt/review，身份与完整性从清单读取；attempt 只保存运行引用和输入哈希，错误原因及预检报告路径从运行目录读取，不重复落盘。录制检查在算法执行前落盘，算法失败仍有原始检查证据。

## 算法执行

```text
build → map_lifecycle.build_scene()
    inputs.read_scene() / select_cameras() 固定建图输入与测量
    camera_assets.snapshot_cameras()             保存本次测量副本
    recordings.prepare_recording()        调用 vision 准备时间戳、IMU、镜头视频
    mapping.build_maps()
        slam.build() / atlas_candidates() 纯视觉建图、逐图定位
        geometry.metric_alignment()       复用轨迹坏段筛选，共同尺度/重力与可观测性验收
        geometry.place_front()            工作镜头地图放入共享场景
    mapping.freeze_maps()                 地图资产冻结
    storage.write_json(current.json)      全部成功后发布地图

run → workflow.run_episode()
    inputs.read_episode()                 批次、录像、预期使用手
    recording_checks.check_episode()      录制质量与完整性
    map_lifecycle.bind_map()                   固定本批次地图
    camera_assets.snapshot_cameras()             保存所选相机、TCP、宽度与遮手图副本
    reconstruction.prepare_episode_recordings() 准备各手录像、检查时间重叠
        reconstruction.prepare_cached()
            recordings.prepare_recording()
    每只手独立执行：
        reconstruction.reconstruct_arm_in_scene()
            reconstruction.localize_cached() → slam.localize_with_recovery()
                slam.localize_with_recovery()  缺口及上下文反向补全；不足时回退整段
                relocalization.merge()  同图同尺度检查连续段，只补缺帧
            VisualObservations.placed()  显式转入固定场景系
            fusion.fuse()                输出 T_scene_imu
        gripper_width.pipeline.process() 有标定时检测指尖、返回对齐后的宽度和报告
        results.write_arm()              换算相机/TCP；位姿与宽度合写 trajectory.csv
    results.write_bimanual()              校验同一场景、共同时间区间与双手表
    inputs.recording_identity()           发布前仅核对原片；测量使用已冻结副本
    results.write_episode()               报告、资产哈希与任务元数据
    storage.publish_link(results)         全部成功后更新结果入口
    episodes.record_attempt()             更新状态与验收有效性
```

`run_episodes()` 只负责任务并发、逐任务失败隔离与汇总。地图绑定在批次锁内直接读取和固定，不递归；只读预检走同一输入准备路径，不落盘。

任务融合因视觉定位失败而保留失败现场时，若存在连续无视觉区间，运行目录会写入 `<arm>/no_visual/gap_*.mp4` 和 `<arm>/no_visual/manifest.json`。片段默认包含前后各 0.5 秒上下文；原始视频、视觉定位结果和 IMU 不会被修改。

多个已通过验收的地图候选会在 `metric_alignment.json` 中记录 `selection_score`，按固定尺度复核、对齐 RMS、尺度不确定度、条件数和覆盖证据选择；硬性验收门槛保持不变。

`map_cache.source_identity()` 显式列出准备、视觉与完整建图的代码依赖；新增算法依赖时必须更新对应集合。完整地图输入保留原片路径、时间和 SHA256 以便追溯，`reuse_key` 按内容、相关配置、相机资产和引擎比较，不把文件路径或夹爪资产作为地图身份。工作镜头遮手图仍属于地图输入。

场景写入锁内，`MapBuildCache.stage()` 先验证缓存清单和资产哈希，再将产物复制到本次工作目录；读写不会共享可修改文件。准备缓存只保存数据集和镜头视频，当前相机设置、遮手图、时间偏移与算法上下文重新生成；视觉缓存使用实际镜头视频、相机设置、ORB 参数、遮手图（仅建图）及引擎身份；定位另包含地图内容和子地图索引。尺度与放置每次重新验收，不缓存最终数值解。

阶段正常完成后才发布 `ready.json`；缓存损坏重算。`build --force` 全部重算，验收失败后的重试也刷新视觉建图和定位，避免重复读取同一失败候选。成功阶段缓存独立于版本 `work/`，后续阶段失败或清理工作目录不会丢失它。`manifest.json` / `error.json` 中的 `timings` 记录实际耗时与缓存状态。

`retry.max_tries` 是最大尝试次数。连续两轮若同一相机/镜头的全部被拒子地图，其帧索引、IMU 时间、位姿以及拒绝原因都相同，建图会提前停止，避免继续重建同一失败结果。轨迹变化、拒绝原因变化、没有候选轨迹证据或引擎暂时退出时不按重复轨迹提前停止，仍保留配置的重试次数。尺度门限保持原值；失败现场和已有成功地图均保留。

地图成功清单或失败 `error.json` 的 `retry` 是本次尝试记录的唯一汇总，包含每轮状态、拒绝证据摘要、`repeat_of_attempt` 和 `stop_reason`。`repeated_visual_failure` 表示因重复失败停止，`attempt_limit` 表示达到次数上限，`accepted` 表示通过。固定尺度复核失败时，日志和拒绝记录还标出对应的原片秒数；每轮完整验收数据继续保存在 `work/try_<次数>/<角色>/localize/m<子地图>/metric_alignment.json`。

场景与独立轨迹通过 `vision.runtime.visual_settings()` 共用镜头 YAML 写入、遮手图准备与视觉运行设置。场景显式选择本设备已验收的各镜头完整标定（measured/factory 内参均可，B 均实测）；独立 dual 仍使用历史环境镜头样例，不接受显式来源切换。

场景录像准备直接构造 `vision.models.VisualContext`，包含已解析参数、帧数与 CPU 设置；不构造 `TrajectoryRequest`。共用的视觉设置为 `VisualSettings`，内部逐段轨迹引擎的融合设置由 `ReconstructionSettings` 扩展；场景融合直接接收已解析参数，不生成单段融合脚本的配置文件。

任务准备和定位数据保存在 `<批次>/.umi/tasks/<任务>/cache/`；`run` 不提供 `--keep-work`。建图、环境镜头标定仍可保留求解中间文件。进度写 stderr，结构化 JSON 写 stdout。

地图清单的建图定义只存于 `inputs.definition`，读取旧清单也使用此字段。地图成功导出后移除原始展示 JSON 和空日志，保留有内容的日志及失败现场；已发布版本不因整理而改写。

`results.write_arm()` 将同帧 TCP 位姿与已配置的测量写入各手唯一的主表 `trajectory.csv`，未配置开口标定时不生成宽度列，随后返回 `ArmTrajectory`，携带任务时间、`T_reference_tcp`、参考坐标 ID、TCP 轴向定义和各手运行元数据。`write_episode()` 将各手指标/测量合并到 schema 4 的 `episode.json.arm_results`，共享地图引用仅保存在任务层 `map`；预检全文收入 `preflight` 后移除临时检查文件，人工报告只保留根 `report.txt`。双手表与任务元数据直接消费内存对象，不重新读取刚写出的文件。

标签间距报告只保存在任务层 `tag_distance.<手>`，没有测量时不写空报告。坐标身份仅保存在 `scene.frame_id`，批次名仅保存在 `session`；输入记录使用地图/同步内容摘要，`inputs.code` 合并全部源码哈希。默认不保存可重建的相机/IMU TUM 轨迹；`parameters.export_auxiliary_trajectories: true` 可开启排障导出，查看器默认由 TCP 和固定外参重建显示。

`software.scene.result_io` 提供只读的摘要和设置适配，查看器及训练导出共用；新结果从任务清单取值，旧结果继续读取各手 `details/summary.json` 和 `details/run_settings.json`，不改写历史结果。查看器校验地图路径、版本、固定坐标与内容身份后直接加载共享地图。

## 命名和时间

- `SceneParameters` 是场景算法参数，`CalibratedCamera` 是已选定完整标定资产的相机。
- `recording_paths` 表示 INSV 路径；`PreparedRecording.lens_videos` 表示生成的镜头视频。
- `episode` 表示一次任务录制；`instruction` 表示操作描述，在任务目录的 `task.yaml` 中填写。
- `RecordingMetadata.started_at_unix_s` 是从文件名解析的 Unix 秒，仅用于配对候选；`duration_s` 是视频时长。
- `PreparedRecording.video_times_s` 使用原片视频时钟；`VisualObservations.imu_times_s` 使用本机 IMU 时钟，不混入跨相机偏移。
- `VisualObservations.T_reference_camera` 的参考系由 `reference_frame` 显式标明为 `visual_map`、`metric_map` 或 `scene`；只有原始视觉地图观测可施加尺度，融合只接受场景观测。
- `models.FusedTrajectory.T_reference_imu` 是调用方声明参考系中的 IMU 位姿；固定场景融合使用场景系，独立回退使用本次局部系。
- 新地图版本格式为 `YYYYMMDD_HHMMSS-随机8位`，前缀取本机建图开始时间，使用本地时区；复用地图时版本保持不变。

相机–IMU 和相机–TCP 外参、SE(3) 校验和位姿插值由 `software/camera/geometry.py` 提供；场景模块保留地图尺度与场景放置。通用 TUM 读写直接使用 `software/pose_io.py`，不依赖独立轨迹后处理模块。

地图尺度复用 `trajectory.algorithms.fusion_alignment.align_lenses()` 的运动段筛选，并由 `fusion_alignment_solver.estimate_lens_alignment(shared_map=True)` 共用一个尺度、重力和零偏重新求解。场景层仅适配观测、检查可靠段一致性与可观测性、保存验收诊断，不复制另一套尺度求解器。分段端点速度隔离，地图变换始终只有一个。建图与任务融合共用 `inertial.estimate_gyro_bias()` 的初始陀螺仪零偏规则。

定标后由 `fusion_alignment_solver.fixed_map_residuals()` 对所有运动链复核固定尺度下的位置方程；不重新拟合段尺度。每条链都会写入 `verified`、`inconsistent` 或 `insufficient_evidence` 状态，局部可靠但整图不一致的候选仍被拒绝，逐段结果保存于 `evidence.alignment.fixed_map_validation`。

外参函数遵守“目标系_from_来源系”：`imu_from_camera()` 返回 `T_imu_cam`，`camera_from_tcp()` 返回 `T_cam_tcp`。ORB 只使用图像，IMU 用于米制对齐和离线融合。地图放置为 `p_scene = R (s p_map) + t`，物理杆臂不乘视觉尺度。双手输出使用任务配置的固定偏移；固定场景不按任务重新置零。

## 双手坐标与时间的阅读入口

从 `reconstruction.reconstruct_arm_in_scene()` 看单手求解。两手各调用一次，没有双手联合优化；共享坐标来自建图时冻结的地图放置，而不是把两段轨迹事后各自平移到原点。

1. `Scene.role(arm, lens)` 选择固定地图：`shared_front` 两手使用同一工作镜头地图；`dual` 使用各自工作镜头地图与共享环境镜头地图。
2. `VisualObservations.placed(..., reference_frame='scene')` 把 `T_map_cam` 转入场景系：位置为 `R (s p_map) + t`，朝向为 `R R_map_cam`。
3. `fusion.fuse()` 使用各镜头 `T_cam_imu` 与原始 IMU 求 `T_scene_imu`；物理杆臂不乘地图尺度，不按任务重新置零。
4. `results.write_arm()` 输出 `T_scene_cam = T_scene_imu T_imu_cam` 和 `T_scene_tcp = T_scene_cam T_cam_tcp`。

求解节点时间为 `t_video + cam_time_offset`，处于该相机本机 IMU 时钟；输出时间为 `t_video + 跨相机偏移`，后者来自任务同步配置，旧任务默认零偏移且未经确认。`write_bimanual()` 只裁取已经应用偏移后的共同区间并插值，不自动估计双机时差。所有输出的 `frame_id` 来自同一固定地图版本。

时间转换统一调用 `software/camera/timing.py`；EuRoC 文件的人工 +1 秒在读入准备数据时去掉，算法中不混入文件时钟。双手 CSV 的历史字段 `t_scene_s` 表示任务时间，不是另一个物理时钟。

任务锁、地图绑定、缓存复用、失败证据与发布顺序保留在 `run_episode()`；结果模块只写产物，不能更新成功指针或验收状态。第二只手求解或结果写入失败时，不发布部分成功结果。

宽度模块复用 main 的检测器和毫米映射；原片像素检测只按原片和检测器缓存，换算、处理策略与固定时间偏移在缓存后应用。宽度资产不进入相机几何身份，另在 episode_inputs.gripper_width 中记录内容哈希与检测代码身份；episode.json 统一保留规范化测量快照和报告，不重复写各手 JSON；宽度数值与 TCP 统一保存在 trajectory.csv，逐帧诊断由 gripper_width_diagnostics 控制。参考 [宽度模块](../gripper_width/README.md)。

`gripper.tag_distance` 启用独立的标签中心三维间距。逐帧角点与宽度检测共享，KB4 内参和各标签实测边长用于位姿解算；结果写入同一 CSV 的 `tag_distance_*` 三列，报告汇入 `episode.json`。尺寸资产不进入地图/相机几何身份，单独进入任务输入指纹；修改边长或质量门限重用角点，重算三维值。该测量不依赖开口锚点，不生成归一化开合量，缺失不插值填充。

## 单手独立回退

`workflow.run_episode()` 只捕获 `spatial.fusion.SceneFusionQualityError`；预期手和实际准备录像都必须恰好为一只手。`pipeline/fallback.py` 复用冻结的工作镜头标定、已裁切视频和原始 IMU，调用独立 `run_attempts()`；显式传入完整帧时间表，保持独立模式的尺度/初始化质量门限和重试策略，不启用实验性的初始化细化。普通输入、引擎及地图绑定错误不触发回退。

适配器返回 `models.FusedTrajectory`，其 `T_reference_imu` 明确表示声明参考系中的 IMU 位姿。`scene.scope=local`、唯一 `local:…` frame_id、`umi_local_episode` 和 `map=null` 声明局部坐标语义；所有下游读取均须遵守此声明。成功发布到 `local_results`，状态为 `local_ok/local_skipped`，不覆盖 `results`。宽度和标签间距由 `pipeline.measurements.measure()` 在发布前统一验收，返回 `ArmMeasurements`；共享场景验收/导出拒绝局部状态。查看器支持局部结果且不加载共享地图。

## 验证

`test_batch_configuration.py`、`test_ingest.py`、`test_recording_checks.py`、`test_episodes.py` 覆盖配置、入库、检查、验收与导出；`test_scene.py` 使用真实 GTSAM 验证尺度、重力、杆臂和固定坐标融合，ORB 在流程测试中使用替代程序。真实录像证据见 [PROVENANCE](../../docs/dev/PROVENANCE.md)。

KIWI 方法来源和许可证见 [NOTICE.md](NOTICE.md)，运行时不依赖本机另一个 KIWI 仓库。

## 原片位置

建图原片位于 `scenes/<场景>/<批次>/mapping/`，任务原片直接位于 `scenes/<场景>/<批次>/<任务>/left/`、`right/`。`inputs.task_video_sources()` 与 `batch_task_directories()` 统一发现任务原片目录，`batch_video_sources()` 再加入批次 `pending/`；不递归扫描任务的 `results/` 和 `.umi/`。配对依据序列号，预分组任务的左右手共用同一任务身份。

`import_video()` 直接引用本任务内已有原片，外部文件复制或显式硬链接到 `<任务>/<手>/`；没有全局场景原片根目录。旧 `<任务>/videos/` 作为兼容输入继续原地读取，已有清单不因布局变化改写，新导入或补齐的原片采用扁平布局。两种布局同一侧出现多个不同原片仍按歧义处理。向导和批次创建允许先准备左右手目录再登记清单。

`plan_tasks()` 保留已明确分组的 take；未分组原片通过时间与时长规则配对。`pending/` 仅暂存无确定任务归属的原片，任务清单写入成功后才删除已归位的 pending 副本。预览使用同一规划逻辑，不写文件。

## 批次地图

`batches.read_inputs()` 每次读取当前 batch.yaml、kit.yaml 与 mapping/，返回具有明确字段的 `BatchInputs`；`save_record()` 更新最新解析记录，不让旧快照覆盖配置。单段 mapping 在 schema 5 中直接保存录像引用，dual 保留左右两段；read_batch 兼容旧 schema 4，算法内部展开参考槽位。run <任务> 也先刷新所属批次。

`build_scene(<批次>)` 校验完整建图输入身份后复用或构建地图；map 字段仅是复用候选。成功结果保存当时的 kit 和完整输入身份；失败 error.json 保存 batch 与 inputs。计算期间 batch/kit 变化会拒绝绑定或发布，历史结果不改写。旧的无 batch.yaml 清单仍可作为显式历史输入读取；日常配置驱动流程以 batch.yaml 和 kit.yaml 为准。

环境镜头 `umi-calib run <rear目录>` 的 A/B/D、恢复、验收和记录均复用 `calibration/workflow.py`。scene 只负责纯视觉建图、米制对齐和 IMU 融合，返回视频时间上的相机位置；板参考时间匹配和误差比较由标定模块统一计算。通过后生成的 `rear_calibration.json` 保持已有登记/消费接口，并绑定本次运行证据。

`build/run --intrinsics-source measured|factory` 先在 `batches.read_inputs()` 中覆盖批次选择，再由 `inputs.validate_kit()` 将来源配置解析成具体工作/环境镜头版本。旧单版本配置保留绑定并检查来源。`inputs.select_camera()` 检查标定身份；factory 还核对 B 输入内参哈希、原片工厂射线一致性及真实引擎的前向 KB4 能力。来源、标定和引擎纳入复用身份，固定地图也须匹配；来源切换归档 `.umi/snapshots/` 并清除不匹配的地图绑定。配置格式与优先级见[套件说明](../../kits/README.md#同时保存实测与工厂来源)。

## 场景配置的唯一来源

新 scene.yaml 默认 schema 3，可省略版本字段，只保存 mode、reference_arm、description、trajectory 和 parameters。两组参数以内联映射覆盖默认值，设备归 kit.yaml，建图录像归批次 mapping/。schema 2 仅用于读取历史配置，不再有 init 或 build --mapping 写入入口。

kit.yaml 和 batch.yaml 只通过手动创建和修改；原片按 take 分组或放入 pending，run 自动登记、补片和处理。run 与 build 共用 batches.read_inputs/make_record/save_record 解析当前配置与生成内部快照。ingest.register_recordings() 只接收已解析的批次对象，不再接受旧命令的外部输入目录、手选择或硬链接选项；预览和实际运行在同一条路径规划任务，仅实际运行写入文件。build 仅接受批次。已有 manifest.yaml、episode.yaml、episode.json 的字段和版本协议保持可读，Session/Episode 是这些历史记录的内部类型，不再作为用户命令名称。


环境海报实验入口：`scene.yaml` 的 `tags.layout` 声明刚性布局，默认只诊断；`parameters.tag_fusion_enabled: true` 将环境镜头角点加入固定地图与 IMU 图。实现位于 `spatial/poster.py`，初值与投影位于 `vision/poster_pose.py`。每段录像重新估计一个海报位姿；布局随结果冻结，地图尺度保持固定。使用方法与当前验证边界见[重构计划](../../docs/dev/环境标签与自适应融合重构计划.md#当前代码的使用与边界2026-11-11)。

建图阶段另用 `tag_mapping_enabled` 开启海报尺度诊断、`tag_mapping_scale_enabled` 显式应用修正；二者默认关闭。`spatial/poster_scale.py` 保持视觉地图形状并处理米制镜头杆臂，合并后由 `geometry.metric_alignment(fixed_scale=...)` 精确固定尺度重做惯性验收。地图目录保留冻结布局及角点，任务不复用建图海报位姿。
