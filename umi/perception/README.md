# 共用视觉执行

场景与内部逐段引擎 都依赖本模块。本模块不读取采集清单、不选择设备、不绑定地图版本，也不融合 IMU。

| 文件 | 职责 |
|---|---|
| `models.py` | 视觉路径、设置、上下文、录像数据与地图阶段结果 |
| `recordings.py` | 原片提取、丢帧策略、镜头视频准备 |
| `runtime.py` | 显式镜头设置写入、遮手图准备、外部进程与引擎检查 |
| `umi/infra/orb_install.py` | ORB 可执行文件和地图导出器路径、工厂内参所需的前向 KB4 探测 |
| `slam.py` | 纯视觉建图、固定地图定位、逐图定位合并及产物验收 |
| `decoding.py` | 多检测器共用一次 PyAV 顺序解码，全部检测完成校验后各自保存缓存 |
| `poster/detection.py` | 环境镜头 ArUco `DICT_4X4_100` 海报角点、真实 PTS、检测缓存与采样可见性报告；不求海报位姿 |
| `poster/geometry.py` | KB4 海报角点投影、解析导数与多帧板位姿初值 |
| `poster/quality.py` | 共享角点噪声、异常剔除预算、证据记录及条件几何质量 |
| `umi/estimation/factors/poster.py` | 场景与独立图共用的鲁棒角点因子、安装旋转修正及验收 |
| `poster/layout.py` | 已核对 PDF 的海报米制布局、角点顺序校验与几何身份 |
| `tools/` | 可独立执行的镜头裁切和定位合并工具 |

海报实现集中在 `poster/`，直接从实际子模块导入。`geometry.py` 保留投影、残差和初始化，`quality.py` 依赖几何计算噪声与筛查指标，`umi/estimation/factors/poster.py` 复用两者构建因子；检测和布局不依赖场景或独立轨迹业务。包初始化只说明职责。

`VisualContext` 直接包含已解析参数、核心设置、路径、帧数及 CPU 设置，不携带 `TrajectoryRequest`。独立流程通过 `ReconstructionContext.visual` 提供视觉上下文；场景直接构造 `VisualContext`。

调用方明确选择镜头及其参数文本，`visual_settings()` 只负责生成运行资产，不自动回退样例标定。相机–IMU–TCP 共用几何位于 `umi/core/geometry.py`，遮手图裁切和缩放由 `umi/perception/runtime.py` 使用 X3 的镜头裁切规则完成。

环境海报目前可独立检测，无需环境镜头标定。默认使用已从用户 PDF 核对的 [77 标签布局](../../hardware/environment_poster/environment_poster.json)：

```bash
./umi-scene detect-tags /path/to/recording.insv --out /tmp/environment-tags
```

仅接受 X3 3840×1920、约 59.94 fps 的双镜头原片，检测左半幅 `rear`。字典为 `DICT_4X4_100`，默认登记 ID 12–88、黑边 1 格；同帧重复 ID 全部拒绝，其他 ID 计为未知。默认每两个源帧取一个，可用 `--frame-step` 调节；始终保存真实 PTS，不用帧率推算时间。`--minimum-side-px` 可增加像素边长筛选，默认 0 仅统计。可用 `--layout` 指定同格式布局；`--border-bits` 若指定，必须与布局一致。

输出 `observations.npz`（采样源帧号、视频时刻、ID、原始半幅角点及筛选标记）、`detection.json`（原片和检测设置及角点文件哈希）和 `report.json`。`TagObservations.corners_px` 减去 240 px 得到标定裁切坐标，`imu_times_s(offset)` 显式接收环境镜头时间偏移；默认观测只含视频时间。标定裁切之外的检测保留用于统计，但 `usable=False`。

`poster.detection.load_observations(path)` 是原始角点 NPZ 的统一读取入口：禁用 pickle，恢复全部数组并验证时间、索引、角点与有效性契约。检测缓存、建图快照、场景融合和独立引擎共用该入口；来源身份与哈希仍由各自调用方核对。

可见率按采样帧计算，同一帧多个 ID 不重复计为多个时刻；最长缺口也是采样统计，不能推断跳过的帧。`usable` 仅表示像素与标定裁切筛选通过，不是海报位姿可观测性或轨迹验收。报告记录当前布局和 PDF 来源哈希，角点缓存与物理几何分开；更正尺寸时可复用像素观测。布局是 `pdf_design`，不代表已测量打印成品。独立检测命令输出角点和可见性报告。场景的海报布局、裁剪和融合开关见[后端配置](../../docs/03-场景处理.md#选择轨迹估计后端)。

`VideoDecode` 向宽度、采样中心距和环境海报检测器交付同一原始帧，各自保留 ROI、采样规则和检测状态。场景同时启用两类以上检测时预先填充各自缓存；缓存命中的检测器不注册，全部命中时不打开视频。像素宽度逐源帧检测，中心距按最近真实 PTS 选择输出帧，海报按源帧号采样。各检测器按原片和自身设置独立判断角点缓存是否可复用；检测器专用参数变化只重跑受影响的检测。ffmpeg 镜头视频生成与按需补检仍单独解码。

视频参数来自父流程已解析的 `TrajectoryParameters.video`，字段说明见[视频处理参数](../../config/README.md#视频处理参数)。`prepare()` 将 `frame_step` 传给时间戳提取；`prepare_lens_videos()` 写出完整参数 YAML，并通过 `--config` 传给子工具，子工具不再接收单独的尺寸、CRF 或 preset 覆盖。裁切范围、内参、遮手图与 SLAM 帧率按同一配置生成，注册表中的原始测量资产保持只读。

`build_maps()` 允许更新地图并保存图集；`localize_map()` 加载固定图集，使用 `localization_only`，不扩建地图。二者共用底层可执行文件，分次调用。

`localize_map(..., reverse=True)` 使用引擎的 `--reverse`，按约 64 MiB 的帧块逆序解码，不生成中间视频。CSV 按逆序写原始帧号和原始视频时间，视觉引擎内部使用递增时间；只允许加载固定地图的定位，不允许反向建图或导出建图轨迹。场景层负责排序、转换到相机–IMU 时间并复核两方向位姿。仅开头缺失时，短窗倒放未补齐会尝试 `--reverse --warm_reverse`。引擎从原片起点正向跟踪到折返点，再保留同一跟踪器状态回读；反向 CSV 排除折返点，只记录回读的帧。能力字段为 `warm_reverse_localization`，旧驱动仍可使用普通倒放。旧安装需要重编译 `x3_mono_slam`；`--capabilities` 的 `reverse_localization` 表明已支持。

`frame_range=(start, end)` 可限制原始帧区间 `[start, end)`；运行验收按区间长度检查结束统计，CSV 必须保留绝对帧号。场景优先反向重放缺口及上下文；未完全补齐时回退整段反向，建图和逐图尺度求解继续使用全片。

每次 ORB 执行写 `execution.json`，包含退出码、预期帧数和验收结论。正常结束也需要完整的帧索引、有限时间、有效定位姿态及结束统计。历史引擎在 Shutdown/析构阶段的 SIGSEGV（139/-11）仅在这些完整性检查通过后允许进入后续地图/定位产物检查。超时、强制终止及其他退出码不接受。此兼容策略保留了历史运行能力，不把异常退出等同于正常结束。

`logs/tracking.json` 记录各镜头各阶段的覆盖统计；文本日志供排障，结构化记录供结果生成。独立重建另写 `logs/run.json` 记录运行身份和完成状态。

第三方仍安装在其原有 `Examples/Monocular-Inertial` 路径，实际使用纯视觉模式，Python 代码通过 `umi/infra/orb_install.py` 访问。当前构建默认启用修复后的前向 KB4，限制单射线角度 ≤85°，不支持完整 >180° 视野。`umi.infra.orb_install.require_forward_kb4()` 检查实际二进制的 `--capabilities`；工厂来源遇到旧引擎时明确报错，需要重编译。实测与工厂来源共用同一几何，重编引擎后用 `umi-scene build --force` 重建地图。
