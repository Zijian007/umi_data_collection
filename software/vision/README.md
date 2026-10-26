# 共用视觉执行

`scene` 与独立 `trajectory` 都依赖本模块。本模块不读取采集清单、不选择设备、不绑定地图版本，也不融合 IMU。

| 文件 | 职责 |
|---|---|
| `models.py` | 视觉路径、设置、上下文、录像数据与地图阶段结果 |
| `recordings.py` | 原片提取、丢帧策略、镜头视频准备 |
| `runtime.py` | 显式镜头设置写入、遮手图准备、外部进程与引擎检查 |
| `paths.py` | ORB 可执行文件及地图导出器的安装路径 |
| `slam.py` | 纯视觉建图、固定地图定位、逐图定位合并及产物验收 |
| `tools/` | 可独立执行的镜头裁切和定位合并工具 |

`VisualContext` 直接包含已解析参数、核心设置、路径、帧数及 CPU 设置，不携带 `TrajectoryRequest`。独立流程通过 `ReconstructionContext.visual` 提供视觉上下文；场景直接构造 `VisualContext`。

调用方明确选择镜头及其参数文本，`visual_settings()` 只负责生成运行资产，不自动回退样例标定。相机–IMU–TCP 共用几何位于 `software/camera/geometry.py`，遮手图缩放位于 `software/camera/images.py`。

`build_maps()` 允许更新地图并保存图集；`localize_map()` 加载固定图集，使用 `localization_only`，不扩建地图。二者共用底层可执行文件，分次调用。

每次 ORB 执行写 `execution.json`，包含退出码、预期帧数和验收结论。正常结束也需要完整的帧索引、有限时间、有效定位姿态及结束统计。历史引擎在 Shutdown/析构阶段的 SIGSEGV（139/-11）仅在这些完整性检查通过后允许进入后续地图/定位产物检查。超时、强制终止及其他退出码不接受。此兼容策略保留了历史运行能力，不把异常退出等同于正常结束。

`logs/tracking.json` 记录各镜头各阶段的覆盖统计；文本日志供排障，结构化记录供结果生成。独立重建另写 `logs/run.json` 记录运行身份和完成状态。

第三方仍安装在其原有 `Examples/Monocular-Inertial` 路径，实际使用纯视觉模式；Python 代码通过 `paths.py` 访问，不要求重编译或移动已有引擎。
