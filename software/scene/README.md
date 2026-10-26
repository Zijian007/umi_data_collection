# 场景代码阅读指南

从 `cli.py` 找命令入口，再读 `workflow.py` 的 `build_scene()` 或 `run_episode()`。用户操作见 [场景指南](../../docs/场景.md)。

## 文件职责

```text
software/scene/
├── cli.py                 参数与直接分发
├── workflow.py            建图、抽取、缓存与失败收尾
├── models.py              参数、清单和阶段数据结构
├── inputs.py              清单读取、资产选择、输入指纹与校验
├── capture_config.py      场景下的套件与批次配置
├── ingest.py              原片配对、任务登记
├── recordings.py          原片归档、时间戳/IMU/镜头视频准备
├── recording_checks.py    原片和任务完整性检查
├── slam.py                ORB 建图、定位、图集导出
├── mapping.py             地图选择、米制对齐、场景放置与资产冻结
├── geometry.py            坐标变换、惯性对齐与几何计算
├── fusion.py              视觉与 IMU 融合、初始陀螺仪零偏
├── episodes.py            任务状态、人工验收、导出
├── results.py             轨迹、双手表、算法报告
└── rear_calibration.py    后镜头标定与独立 C 验收
```

`models.py` 不读取 YAML、设备资产或文件哈希。读取逻辑统一在 `inputs.py`，`workflow.py` 和 `episodes.py` 直接使用它；验收模块不导入 workflow。公共原子文件和锁由 `software/storage.py` 提供，设备实测资产由 `software/registry.py` 管理。

## 采集管理

```text
kit / session → capture_config.py          更新当前套件、保存批次设备快照
    inputs.read_scene() / read_kit()
    registry.pin() / version_assets()     读取全局实测资产

ingest / episode → ingest.py               唯一配对或显式登记
    recordings.inspect() / import_video() 原片身份、哈希、保留原名归档
    inputs.read_episode()                 手、序列号与输入完整性
    ingest._write_episode()               有变化才更新清单和任务状态
```

目录层次为 `scenes/<场景>/kits/<套件>/sessions/<批次>/<任务>/`。输入读取校验目录与清单身份一致；查看器标签和导出路径都包含套件。

每个套件只有一个 kit.yaml；`inputs.read_kit()` 读取当前配置，`validate_kit()` 校验当前配置与批次快照的共同结构。创建批次时核对实际测量并把配置保存到 session.yaml（schema 4），处理已有批次只读快照。套件没有独立版本目录或指针文件，相机标定、夹爪与地图继续使用各自的固定资产版本。

新任务清单 schema 5 保存 instruction 和 synchronization；兼容读取 schema 4，双手默认零偏移但未经确认。单手不要求跨设备同步确认。批次与任务分别保留 expected_arms，缺片任务不会隐式降为单手。

指令进入结果 episode.json 和导出清单；修改指令或同步沿用清单哈希失效机制，需重新 run 和验收。训练格式由独立 [dataset 模块](../dataset/README.md) 负责，scene 不承担训练采样。

`state.json` 只保存 attempt/review，身份与完整性从清单读取；attempt.preflight_report 指向运行目录中的报告。录制检查在算法执行前落盘，算法失败仍有原始检查证据。

## 算法执行

```text
build → workflow.build_scene()
    inputs.read_scene() / select_cameras() 固定建图输入与测量
    recordings.prepare_recording()        调用 vision 准备时间戳、IMU、镜头视频
    mapping.build_maps()
        slam.build() / atlas_candidates() 纯视觉建图、逐图定位
        geometry.metric_alignment()       米制尺度、重力与零偏
        geometry.place_front()            前地图放入共享场景
    mapping.freeze_maps()                 地图与测量资产冻结
    storage.write_json(current.json)      全部成功后发布地图

run → workflow.run_episode()
    inputs.read_episode()                 批次、录像、预期使用手
    recording_checks.check_episode()      录制质量与完整性
    workflow.bind_map()                   固定本批次地图
    workflow.prepare_episode_recordings() 准备各手录像、检查时间重叠
        workflow.prepare_cached()
            recordings.prepare_recording()
    每只手独立执行：
        workflow.reconstruct_arm_in_scene()
            workflow.localize_cached() → slam.localize()
            VisualObservations.placed()  显式转入固定场景系
            fusion.fuse()                输出 T_scene_imu
        results.write_arm()              换算相机/TCP、写该手结果
    results.write_bimanual()              校验同一场景、共同时间区间与双手表
    inputs.episode_inputs()               发布前重新核对输入
    results.write_episode()               报告、资产哈希与任务元数据
    storage.publish_link(results)         全部成功后更新结果入口
    episodes.record_attempt()             更新状态与验收有效性
```

`run_episodes()` 只负责任务并发、逐任务失败隔离与汇总。地图绑定在批次锁内直接读取和固定，不递归；只读预检走同一输入准备路径，不落盘。

场景与独立轨迹通过 `vision.runtime.visual_settings()` 共用镜头 YAML 写入、遮手图准备与视觉运行设置。调用方先明确选择各镜头标定：场景只传实测镜头，独立模式仍显式使用其样例后镜头配置。

场景录像准备直接构造 `vision.models.VisualContext`，包含已解析参数、帧数与 CPU 设置；不构造 `TrajectoryRequest`。共用的视觉设置为 `VisualSettings`，独立 `umi-traj` 的融合设置由 `ReconstructionSettings` 扩展；场景融合直接接收已解析参数，不生成单段融合脚本的配置文件。

任务准备和定位数据保存在 `cache/`；`run` 不提供 `--keep-work`。建图、后镜头标定仍可保留求解中间文件。进度写 stderr，结构化 JSON 写 stdout。

地图清单的建图定义只存于 `inputs.definition`，读取旧清单也使用此字段。地图成功导出后移除原始展示 JSON 和空日志，保留有内容的日志及失败现场；已发布版本不因整理而改写。

`results.write_arm()` 接收共享展示地图路径，在每只手的 summary.map 中记录相对路径与内容哈希；不读取或写入地图点副本。它返回 `ArmTrajectory`，直接携带任务时间、`T_scene_tcp`、场景坐标 ID 和 TCP 轴向定义。双手表与任务元数据消费这个对象，不重新读取刚写出的摘要。查看器校验地图路径、版本、固定坐标与内容身份后直接加载共享文件。

## 命名和时间

- `SceneParameters` 是场景算法参数，`CalibratedCamera` 是已选择实测资产的相机。
- `recording_paths` 表示 INSV 路径；`PreparedRecording.lens_videos` 表示生成的镜头视频。
- `episode` 表示一次任务录制；`instruction` 表示操作描述，CLI 使用 `--instruction`。
- `RawVideoInfo.started_at_unix_s` 是从文件名解析的 Unix 秒，仅用于配对候选；`duration_s` 是视频时长。
- `PreparedRecording.video_times_s` 使用原片视频时钟；`VisualObservations.imu_times_s` 使用本机 IMU 时钟，不混入跨相机偏移。
- `VisualObservations.T_reference_camera` 的参考系由 `reference_frame` 显式标明为 `visual_map`、`metric_map` 或 `scene`；只有原始视觉地图观测可施加尺度，融合只接受场景观测。
- `FusionResult.T_scene_imu` 是固定场景中的 IMU 位姿，不是相机或 TCP 位姿。
- 新地图版本格式为 `YYYYMMDD_HHMMSS-随机8位`，前缀取本机建图开始时间，使用本地时区；复用地图时版本保持不变。

相机–IMU 和相机–TCP 外参、SE(3) 校验和位姿插值由 `software/camera/geometry.py` 提供；场景模块保留地图尺度与场景放置。通用 TUM 读写直接使用 `software/pose_io.py`，不依赖独立轨迹后处理模块。

外参函数遵守“目标系_from_来源系”：`imu_from_camera()` 返回 `T_imu_cam`，`camera_from_tcp()` 返回 `T_cam_tcp`。ORB 只使用图像，IMU 用于米制对齐和离线融合。地图放置为 `p_scene = R (s p_map) + t`，物理杆臂不乘视觉尺度。双手输出使用任务配置的固定偏移；固定场景不按任务重新置零。

## 双手坐标与时间的阅读入口

从 `workflow.reconstruct_arm_in_scene()` 看单手求解。两手各调用一次，没有双手联合优化；共享坐标来自建图时冻结的地图放置，而不是把两段轨迹事后各自平移到原点。

1. `Scene.role(arm, lens)` 选择固定地图：`shared_front` 两手使用同一前地图；`dual` 使用各自前地图与共享后地图。
2. `VisualObservations.placed(..., reference_frame='scene')` 把 `T_map_cam` 转入场景系：位置为 `R (s p_map) + t`，朝向为 `R R_map_cam`。
3. `fusion.fuse()` 使用各镜头 `T_cam_imu` 与原始 IMU 求 `T_scene_imu`；物理杆臂不乘地图尺度，不按任务重新置零。
4. `results.write_arm()` 输出 `T_scene_cam = T_scene_imu T_imu_cam` 和 `T_scene_tcp = T_scene_cam T_cam_tcp`。

求解节点时间为 `t_video + cam_time_offset`，处于该相机本机 IMU 时钟；输出时间为 `t_video + 跨相机偏移`，后者来自任务同步配置，旧任务默认零偏移且未经确认。`write_bimanual()` 只裁取已经应用偏移后的共同区间并插值，不自动估计双机时差。所有输出的 `frame_id` 来自同一固定地图版本。

时间转换统一调用 `software/camera/timing.py`；EuRoC 文件的人工 +1 秒在读入准备数据时去掉，算法中不混入文件时钟。双手 CSV 的历史字段 `t_scene_s` 表示任务时间，不是另一个物理时钟。

任务锁、地图绑定、缓存复用、失败证据与发布顺序保留在 `run_episode()`；结果模块只写产物，不能更新成功指针或验收状态。第二只手求解或结果写入失败时，不发布部分成功结果。

## 验证

`test_capture_config.py`、`test_ingest.py`、`test_recording_checks.py`、`test_episodes.py` 覆盖配置、入库、检查、验收与导出；`test_scene.py` 使用真实 GTSAM 验证尺度、重力、杆臂和固定坐标融合，ORB 在流程测试中使用替代程序。真实录像证据见 [PROVENANCE](../../docs/dev/PROVENANCE.md)。

KIWI 方法来源和许可证见 [NOTICE.md](NOTICE.md)，运行时不依赖本机另一个 KIWI 仓库。
