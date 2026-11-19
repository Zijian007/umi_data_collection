# 引擎来源与变更历史

[代码维护](README.md) · [当前文档](../README.md)

本文件记录来源与历史改动；各日期下的旧路径、旧函数名描述当时状态。当前调用链以代码目录的阅读指南为准。

轨迹引擎最初从实验室仓库原样拷贝，不带该仓库的提交历史。

- 仓库：`github.com/Torbjorn-s-Apple-Americano/umi`
- 分支：`dev_Zijian`
- 提交：`71a124d640b998dbbfcf35504dea830e179eddfc`（短号 `71a124d`）

拷贝的路径：

- `software/camera/`
- `software/trajectory/`
- `software/calibration/`
- `software/__init__.py`
- `install.sh`、`umi-traj`、`umi-calib`、`Dockerfile`、`.dockerignore`
- `docs/安装与日常使用.md`、`docs/场景处理.md`、`docs/算法原理.md`、`docs/标定.md`（当初拷来时是四份平铺的说明，后来按两块功能收进这两个目录）
- `docs/dev/QUICKSTART_X3_STREAM.md`

初次导入时，上述路径里的算法文件与该提交逐字节相同；此后的修改按下文记录。本仓库另外加了 `docs/dev/todo.md`、本文件、`.gitignore`，以及 `software/trajectory/calib/README.md`（说明 `calibration_r1440.json` 是实验室给某一台 X3 标定的样例，不是厂家出厂标定，也不是众包默认标定）。

此后如果改融合、ORB-SLAM3 封装、X3 读取或 Basalt 标定，在本文件追加一笔：日期、改了什么、为什么改。

## 2026-11-18：按 specialist 参考补齐 poster-VIO 初始化、前端与关键帧恢复

本轮以本机 `/home/zijianwang/specialist` 的 `da29719cdbaa366df0599984068ad4553d9cc86d` 为参考。对应 MIT 来源、路径和本地适配边界写入 `software/trajectory/poster_vio/NOTICE.md`，完整许可证保留为 `LICENSE.specialist`；没有引入 specialist 包、MEI 模型、GPU 内核或其记录格式。

初始化增加整板/单标签/标签对子集的确定性 KB4 PnP 假设、完整标签多数一致性、平面多解拒绝和原始训练预算；保存逐标签 `tag_initialization_mask`。清洁完整板观测走快速路径，异常帧才进入有限子集搜索。前端增加 KB4 视线极几何、纯旋转分支、确定性本质矩阵 RANSAC，分批三角化检查视差/条件数/深度/重投影，并将前端筛选计数写入诊断。

同镜头轨迹增加有界 ORB 描述子重关联：休眠库按时间和数量限额保留，候选须通过 Hamming 比值与当前米制初值重投影，ID 不复用。自然点留出按冻结图节点索引划分；留出像素可用于关联和前端几何筛选，但不参与三角化初值、联合图或后验剔除，前后镜头分别记录残差和覆盖。

联合求解先使用 `precull_iterations` 短求解判断异常；若没有异常则同一轮继续完整优化，不消耗剔除轮数。剔除重求解复用上轮 GTSAM 状态和 LM 阻尼，固定图节点时刻与累计预算，显式按三维点优先、海报/状态时间顺序构造消元顺序。关键帧按时间/平移/旋转选取；标签和自然观测时刻强制并入主图。非主图输出节点使用固定端点的局部条件惯性图恢复，端点位姿/速度/零偏固定；零残差区间直接传播，其他区间优化，不使用普通插值或端点误差摊分。

参数增加 `initialization`、`keyframes`、`frontend.revisit`、`culling.precull_iterations` 与 `holdout.natural_stride` 分组；配置快照、验证、轻量 CI、测试入口和操作说明同步更新。动态语义分割、滑窗边缘化和 GPU 加速仍未接入。

验证：specialist 对照与新前端/初始化/热启动/自然留出/关键帧回归 45 项通过（16.055 s）；综合配置、文档和 VIO 定向回归 121 项通过（21.114 s）；快速档 **209 项通过（40.002 s）**；最终 `PYTHONDONTWRITEBYTECODE=1 ./test.sh --all` 的 **1030 项 Python 测试通过（239.073 s）**，查看器前端、CLI 自检和 KB4 C++ 检查通过。Ruff、语法、Shell 语法、文档链接与差异空白检查通过。合成加速双手对照结果见 `/tmp/umi-vio-specialist-keyframes-benchmark.json`：关键帧主图 72 个状态节点、恢复 200 个输出节点、217 个变量/223 个因子；逐帧对照 272 个状态节点、817 个变量/623 个因子。关键帧模式的条件恢复会增加小规模合成耗时，不能据此宣称整体加速；生成帧前端对照另记录于 `/tmp/umi-vio-specialist-frontend-benchmark.json`，其只测内存合成帧，不含真实解码和实机精度。完整与快速日志分别为 `/tmp/umi-vio-specialist-final-all.log`、`/tmp/umi-vio-specialist-final-quick.log`。

## 2026-11-18：poster-VIO 持续补点、有限轮剔除与标签留出

修复自然特征前端在累计创建 2000 条轨迹后停止补点的问题：活动点及其短历史独立维护，退役时三角化，通过后按整个录像的时间区间分配保留预算。各区间优先保留较长轨迹，活动点和归档都有数量上限，特征 ID 单调递增、不循环复用。低纹理区间的预算不向其他区间挪用，避免前段候选挤掉后段；这属于有界的轨迹保留策略，尚不是关键帧/分窗优化。

联合求解增加最多两轮后验异常剔除与重求解。标签按整次四角点观测处理，复用共享剔除门槛及数量/时间覆盖规则，预算始终相对原始训练标签；自然像素预算按每手每镜头的原始观测数分别计算，累计比例向下取整。移除后不足两次观测或重新三角化失败的轨迹整体移除，其剩余观测同样计入预算。提案超预算、破坏训练标签时间覆盖或耗尽轮数后仍有异常均拒绝整任务，不删到恰好过关。证据保存逐轮原始观测索引、ID/镜头/时间、残差、提案是否实际应用及前后分布；初始严格 PnP 失败仍拒绝，不宣称后验剔除解决了鲁棒初始化。

生产输入在 PnP、惯性初始化和自然点三角化前，按完整标签时刻划分内部留出，首尾保留训练；划分必须保持训练所需的时刻数、跨度及最大间隔。对应 rear 帧不参与自然特征跟踪与标签区域掩膜构造，留出像素不参与联合图或异常剔除。冻结数值输入增加逐标签 `tag_holdout_mask`，输入诊断记录训练/留出时刻数；后验记录留出误差与时刻。至少两个留出时刻通过才记为 `passed`，超限或非正深度则拒绝；证据不足、配置关闭和数值调用未准备留出分别明确标记。这是同一录像、标定下的标签观测一致性评估，不能替代独立 TCP/双手精度参考；自然点留出尚未实现。

前端活动点/保留轨迹/长度、后验轮数/自然点预算/门槛和留出步长/RMS 统一进入 `parameters.poster_vio.frontend/culling/holdout`，同步类型、schema、校验、专家覆盖、快照和操作说明。快速档增加后验数值回归，轻量 CI 增加新策略模块。合成回归覆盖小预算下的末段补点、孤立异常恢复、异常超预算拒绝、跨轮累计预算、轮数上限、留出像素破坏不改变初值或训练拟合、留出拒绝不写轨迹，以及冻结输入、分镜头来源和配置校验。

留出划分按唯一时刻批量计算局部间隔与训练容量，避免逐候选重复扫描整组观测；数值回放同时校验首尾保留和剩余训练时间覆盖，非有限留出投影明确拒绝。单次本机合成测量：10000 个时刻、30000 次标签观测的划分从 0.610814 s 降为 0.002176 s，100000 个时刻、300000 次观测为 0.023441 s；这是划分步骤耗时，不是整条求解性能基准。

验证：最终 `PYTHONDONTWRITEBYTECODE=1 ./test.sh --all` 的 **1016 项 Python 测试通过（230.495 s）**，查看器前端、CLI 自检和 KB4 C++ 数值检查通过；快速质量档 **195 项通过（34.907 s）**。算法、配置与文档定向回归 104 项通过（14.402 s）；最后的划分优化与输入边界加固后，后验/留出模块 12 项回归通过（5.596 s）。Ruff、语法、Shell 语法、文档本地链接及差异空白检查通过。最终完整与快速回归日志为 `/tmp/umi-vio-refinement-20261118-final-all.log`、`/tmp/umi-vio-refinement-20261118-final-quick.log`。

## 2026-11-18：poster-VIO 接入工作镜头自然特征

自然特征前端改为一次顺序解码原始 INSV，同时提取右半幅 front 工作镜头和左半幅 rear 环境镜头各自的 1440 标定裁切。front 使用冻结相机遮手图，按最近邻缩放，显式将资产中“非零表示遮挡”转换为 OpenCV 的允许区域；该限制同时作用于新建特征和光流延续，避免手部轨迹进入静态环境图。rear 继续排除检测到的标签区域。原始时间戳不一致、录像截断和遮手图缺失/损坏均拒绝输入。

联合图分别使用两镜头的固定内外参和时间偏移，自然点按手、镜头和特征 ID 分配独立键。同一镜头的合格自然点才能组成该镜头的视觉支持；记录各镜头轨迹数、像素残差和精确支持时刻。发布的 `tracked` 是两镜头支持的并集，`front_tracked` 仅表示 front，逐帧 `visual_lenses` 保留两镜头来源。冻结数值输入保存分镜头标定与轨迹，回放核对实际使用镜头的标定身份；仍共用现有任务原子发布与资产哈希协议。

合成回归覆盖两镜头不同内外参/时差、按手和镜头重名的三维点、原片半幅与一次解码、遮手图反向语义、已有特征进入遮挡区后的停止、rear 支持不冒充 front，以及完整任务发布后的逐帧来源、数值输入、复用与失败保持旧结果。操作说明、模块指南、方案当前实施范围与待办已同步；描述子重关联、动态物体筛除、有限轮联合剔除、留出评估和大规模优化仍未交付，真实录像及 TCP 精度未在本轮验收。

验证：VIO 合约、初始化、前端与任务模块定向回归 30 项通过（6.903 s）；`PYTHONDONTWRITEBYTECODE=1 ./test.sh --all` 的 **1001 项 Python 测试通过（225.739 s）**，查看器前端、CLI 自检和 KB4 C++ 数值检查通过；快速质量档 **180 项通过（28.279 s）**。Ruff、语法、文档本地链接与差异空白检查通过。完整及快速回归日志分别保存在 `/tmp/umi-poster-vio-dual-20261118-all.log` 与 `/tmp/umi-poster-vio-dual-20261118-quick.log`。

## 2026-11-18：接入第一版任务级 poster-VIO 数值后端

新增 `poster_vio` 初始化和任务路由：从冻结的环境海报角点按真实 IMU 时间建立逐帧平面 PnP 种子，拒绝非正深度和不足的观测；任务图固定海报参考系，联合优化每手的 IMU 位姿/速度/偏置、海报角点和可用自然特征点。新增稀疏光流、三角化前端，输出原始像素轨迹与任务海报坐标下的三维点初值；有效但没有自然纹理的录像可返回零自然轨迹；缺失、截断或时间戳变化的录像明确拒绝，不静默退化为仅标签求解。

`poster_vio` 路由不读取预建场景地图，不要求 ORB 可执行文件或词典。标签旋转与陀螺仪积分先拟合陀螺仪零偏，不把低角速度当作静止；标签 PnP 与 IMU 预积分再在任意海报坐标中初始化重力和加速度零偏；双手方向一致后固定共享重力和海报规范。图节点包含全部工作镜头曝光与 rear 观测时刻，最终轨迹取真实优化状态。冻结布局、相机、IMU 和角点 ID 在路由边界交叉校验，部分注入的回放输入会拒绝。

门槛集中于 `scene_parameters.yaml` 的独立 `poster_vio` 分组：最小独立标签帧数、跨度、观测间隔、共同位置支持缺口和双手重力差。求解完成后还检查标签/自然点残差、正深度、零偏和海报条件几何；不将所有输出帧标成视觉支持。`run_report.json` 记录实际后端与变换链，运行目录保留原始角点、数值输入和质量证据，并将这些文件纳入发布资产哈希以检测复用时的篡改。纯海报后端不读取地图 epoch，也不因无关地图计划阻止发布。

当前作为实验性显式后端，`auto` 在可用地图的 `scene_map` 质量不足后尝试本后端，无地图时直接从本后端开始，失败再回退到 `poster_independent`。rear 光流采用顺序解码、前后向检查、遮掉标签区域、视差/正深度三角化与轨迹数量/长度限制；front 自然点、描述子重关联、动态筛除、有限轮联合剔除、留出评估和大规模关键帧/分窗尚未交付。真实录像、TCP 和独立参考精度尚未由本变更证明。

验证：最终 `PYTHONDONTWRITEBYTECODE=1 ./test.sh --all` 的 **995 项 Python 测试通过（222.725 s）**，查看器前端、CLI 环境自检和 KB4 C++ 数值检查通过；快速质量档 **177 项通过**，新增 VIO 合约/初始化/前端已加入快速档。新增测试覆盖慢转与陀螺仪零偏的区分、任意海报朝向、双手不同运动/外参及非零镜头时差、同名自然点隔离、自然点解析 Jacobian、原始帧裁切/光流/时间戳校验、异常输入和像素质量拒绝、无地图完整发布/复用/查看器、失败保留旧结果及冻结数值输入篡改拒绝。Ruff、语法编译和 `git diff --check` 通过。日志：`/tmp/umi-poster-vio-20261118-final.log`、`/tmp/umi-poster-vio-20261118-quick.log`；回归用例保存在 `tests/test_poster_vio_*.py`，不依赖临时日志运行。

## 2026-11-18：独立训练包读取、人工采样窗口与海报准备阶段

训练包读取从 `dataset/export.py` 分离到 `dataset/reader.py`；格式版本、时间轴、手级字段和刚体数组名称集中在 `dataset/schema.py`。读取器只依赖 NumPy 与项目基础身份函数，导入时不加载场景管理、OpenCV、SciPy 或 GTSAM；哈希、数组形状、时间轴、视频登记、坐标声明和训练就绪条件仍在读取时校验。导出器只负责筛选已验收结果、生成文件和原子发布，磁盘协议保持兼容。

训练导出新增 `--start-s/--end-s` 人工确认窗口，窗口外样本不进入时间轴；最终窗口写入包清单和每集元数据。采样器记录实际窗口、保留 `segment_id` 断点，并继续在质量策略之后对宽度与视频索引使用同一时间轴。无效窗口、窗口无共同有效区间和不完整 CLI 参数都会明确失败。

共享海报融合把每只手的可用观测、初始板位姿和初始化剔除计划提取为 `_prepare_shared_poster_arms()`；图构建、优化、后验剔除和证据发布顺序不变。这个阶段边界与 specialist 的观测准备 / 联合图 / 证据分层一致，后续新增观测质量诊断不再混入求解器主体。

验证：训练包、读取边界、窗口及海报组合回归共 24 项通过；海报融合全模块 33 项通过；快速质量档 162 项通过；完整 Python 回归 **969 项通过（211.194 秒）**。`py_compile` 和 `git diff --check` 通过。独立读取子进程确认 `software.dataset.reader` 不加载 `gtsam`、`cv2`、`scipy`、场景 workflow 或导出模块。

融合图中的 `segof`、`n_obs`、`n_rot`、`n_cont` 改为 `segment_for_frame`、`observation_counts`、`rotation_only_counts`、`continuity_factor_counts`，只改善局部语义，不改变因子插入顺序或发布字段。稳定性样例新增固定数值门槛：重力、尺度不确定度、位置 RMS、错位窗口尺度跨度、固定地图残差和三组相位证据均需落在物理合理范围；对应 5 项测试通过。

`optimize_fused_trajectory()` 的节点值、视觉因子和 IMU 边插入收拢到局部 `add_nodes_and_factors()` 阶段函数；主流程现在明确区分参数准备、图构建、优化和证据发布。30 项融合定向回归通过，未改变 GTSAM 因子插入顺序或固定观测结果。

新增 `.github/workflows/quality.yml` 轻量 CI：只安装 NumPy 和 Ruff，执行全仓语法编译、可移植训练读取器边界测试以及核心改动文件的 F/E9 静态检查；完整 GTSAM/SLAM 回归仍由本机 `test.sh` 负责。

标定 workflow 中原本名为 `measurement_record` 的对象实际保存各阶段的延迟准备函数，已改为 `recording_preparers`；调用链和阶段复用行为保持不变。标定 workflow 与交接边界定向回归 55 项通过。

`umi-scene` CLI 将运行覆盖、专家配置、`show-config` 和 OpenCV 线程初始化提取为 `_prepare_execution()`；命令分发只处理业务命令，38 项处理/文档契约测试通过。

viewer server 将录像媒体处理和静态资源处理从 `do_GET()` 分离为 `serve_recording()` / `serve_static()`，Range、ETag、预览版本和错误响应保持原协议；viewer 定向测试通过。

epoch 规划中的地图/任务事件从位置元组改为具名 `_MapEvent` / `_TaskEvent`，排序、序号和 needs_review 原因保持一致；地图 epoch、批次映射和场景管理定向回归 31 项通过。

新增 `software/trajectory/poster_vio/` 契约层，定义任务级海报视觉惯性后端的观测、资产、`task_poster` 坐标、质量证据和结果校验；明确禁止把场景地图先验或现有 `poster_independent` 实现伪装成联合 BA。当前仅交付软件边界与 2 项契约测试，数值求解器仍需独立算法验证。

poster_vio 契约增加独立观测质量门槛：最小标签帧数、最大观测间隔和最小时间跨度在优化器启动前统一筛选；契约测试扩展为 4 项，结果质量状态区分 `ready`、`solved` 和 `insufficient_observations`。新增第一版 `solve_joint_poster_imu()`：每手建立独立 IMU 位姿/速度/偏置状态，共享板位姿并加入海报角点因子，合成观测回归验证联合图、正深度和重投影证据；自然特征点因子尚未接入任务路由。

## 2026-11-18：统一内部语义命名、收紧阶段结果并分离共用缓存

任务输入的 CPU 分配及审计对象改名为 `arm_cpu_sets`、`cache_audit`，任务输出的测量报告改名为 `width_reports`、`tag_distance_reports`。输出坐标声明统一使用 `coordinate_record`，单手分支明确为 `arm_coordinate_record`；身份构造中的测量资产分别使用 `width_assets`、`tag_distance_assets`。重力单位方向、地图尝试目录及请求版本使用明确名称。内部 Python 字段和参数同步迁移，磁盘 JSON/CSV、原有状态、坐标含义和协议键保持原值。

场景后端入口统一为 `execute_task()`，注册适配器明确为 `_run_scene_map()` / `_run_poster_independent()`。录像准备与定位的缓存入口使用 `prepare_with_cache()` / `localize_with_cache()`，`defer_poster_fusion` 明确表示先交出基线和观测、延后共享海报求解。`MapSelection` 校验目录和清单必须成对存在、已选地图不能同时携带失败原因，并提供 `available()` / `without_map()` 构造入口。手级求解结果使用冻结的具名对象，删除四元组的位置解包，立即写入及联合求解后写入的时序保持不变。

定位缓存入口分别调用恢复观测、计算并发布两个步骤；损坏条目重算，引擎工作目录仍独立于缓存，失败保留日志，不发布缺失或不完整的结果。显式任务先按所属批次分组，每批只准备一次；处理后一次性筛除已报告或失败的任务，保留原始任务顺序，避免反复解析所属目录及重建列表。

原 `pipeline/maps/cache.py` 分为共用 `scene/cache/storage.py` 的 `StageCache`、纯内容身份 `scene/cache/identity.py` 和地图专属 `pipeline/maps/identity.py`。准备与视觉身份由独立函数生成，缓存对象只承担存储、完整性、审计和计时及运行上下文。`spatial` 不再导入地图流水线，共用缓存也不导入流水线；rear 标定对共用录像准备的调用继续保留。源码身份引用真实的新路径，准备缓存继续只追踪解码依赖，相关算法缓存按实际源码变化失效。

对照本轮起点 `/tmp/umi-scene-contracts-20261118-before`：20 种批次/显式任务、预览/执行及问题/异常组合的摘要、准备和调度调用完全一致；共用缓存存储的五个方法及函数语法树一致，证据 `/tmp/umi-scene-contracts-equivalence.json`。首次计算、复用、观测损坏、恢复证据损坏、引擎失败五种定位场景的观测、审计、诊断、引擎调用和失败日志行为完全一致，证据 `/tmp/umi-scene-contracts-localization-equivalence.json`。独立差异 `/tmp/umi-scene-contracts-only.patch`，均非运行依赖。

工作盘曾耗尽空间，本轮只将项目生成的 41 个字节码目录（14,304,269 字节）移到 `/tmp/umi-scene-contracts-bytecode`，后续测试关闭字节码写入；原始录像、标定、业务配置及历史成果未改写。

验证：`PYTHONDONTWRITEBYTECODE=1 ./test.sh --all` 的 965 项 Python 测试（210.591 s）、查看器前端、CLI 自检和 KB4 C++ 数值检查全部通过，日志 `/tmp/umi-scene-contracts-all.log`。新增五项回归覆盖地图状态约束、观测/恢复证据损坏仅重算一个条目且轨迹表一致、引擎失败保留日志及旧成功结果、跨批次显式任务顺序，以及数值算法/共用缓存对地图流水线的依赖边界。此前命名 37 项、结构 76 项及缓存 82 项定向回归通过，日志 `/tmp/umi-scene-contracts-names-final.log`、`/tmp/umi-scene-contracts-structure.log`、`/tmp/umi-scene-contracts-cache-final.log`；文档和边界检查 7 项通过，日志 `/tmp/umi-scene-contracts-docs.log`。同级 Organizer 27 项交接测试全部执行通过，日志 `/tmp/umi-scene-contracts-organizer.log`。场景目录无未引用顶层导入，无旧接口或旧缓存模块引用，`git diff --check` 通过。

## 2026-11-18：按职责命名场景模块并收拢任务后端与结果

先完成四项职责命名：`scene/configuration.py` 改为 `overrides.py`，`scheduling.py` 改为 `cpu_budget.py`，任务的 `evaluation.py` 改为 `relative_pose_evaluation.py`，`fallback.py` 改为 `independent_reconstruction.py`。最后一项同时服务显式海报后端及单手局部回退，不再以失败路径命名共用引擎适配。

随后将任务估计实现收拢到 `pipeline/tasks/backends/`：场景执行 `execution.py` 改为 `scene_map.py`，地图定位和融合 `reconstruction.py` 改为 `scene_reconstruction.py`，海报后端及独立引擎适配同置其中。共用的 `prepare_cached()` 和 `prepare_episode_recordings()` 原样移到 `tasks/recording_preparation.py`，workflow 直接调用，不经场景地图后端准备输入。

轨迹表、报告、任务清单与发布移到 `tasks/results/`；公共测量及手级轨迹写入 `write_arm_result()` 从原执行模块原样移到 `results/arm_writer.py`，两种后端直接调用，消除海报后端为写结果而依赖场景执行模块的关系。`estimation.py` 保留后端注册与整任务回退。任务目录直接平铺的实现文件由 13 个减为 6 个；新增包初始化只有说明，不转发旧模块或预加载求解器。

所有源码导入、测试拦截、动态调用路径、定位缓存的显式源码依赖和当前代码指南同步更新。历史记录保留当时的旧路径。源码与缓存身份按真实路径和内容更新，未伪造旧身份；结果协议、求解顺序、回退条件、并行预算与发布边界保持原值，未改业务配置、原始录像、注册资产或历史结果。

以起点快照 `/tmp/umi-scene-names-20261118-before` 对照场景下 305 个函数与类，归一化模块引用和依赖路径后，定义语法树全部一致；核对 `/tmp/umi-scene-names-semantic-audit.json`，本轮独立差异 `/tmp/umi-scene-names-only.patch`，均非运行依赖。

验证：`./test.sh --all` 的 960 项 Python 测试（209.981 s）、查看器前端、CLI 自检与 KB4 C++ 数值检查全部通过，日志 `/tmp/umi-scene-names-all.log`。四项命名调整的 42 项、后端与目录边界的 51 项定向回归通过，日志 `/tmp/umi-scene-names-stage1.log`、`/tmp/umi-scene-names-stage2.log`；文档、源码身份与轻量导入的 12 项检查通过，日志 `/tmp/umi-scene-names-contracts.log`。同级 Organizer 27 项交接测试全部执行通过，日志 `/tmp/umi-scene-names-organizer.log`。场景目录无未引用顶层导入，`git diff --check` 通过。

## 2026-11-18：统一任务后端契约与批次准备，分离地图声明校验

共享输入 `TaskInputs` 和输出 `ArmExecutionResult` 统一定义在 `scene/models.py`。场景地图和任务海报后端直接接收同一个输入对象，手级测量与轨迹写入也共用此对象，删除适配层展开后再传递的重复参数。场景和任务名称从 `Episode` 读取；CPU 分配明确为 `dict[str, str] | None`，轨迹输出明确为 `ArmTrajectory`，替换原先的 `bool`、`object` 标注。后端选择、整任务回退、并行预算、求解与发布规则保持不变。

地图仓库的 `validate_manifest()` 只处理已读入的声明，集中检查结构、版本、坐标、地图条目、尺度与刚体变换。指针解析、文件读取、哈希、当前场景兼容和引擎查询保留在读取流程，异常边界和类别保持原值。抽取前后校验块的归一化语法树完全相同；新增回归检查纯校验不访问文件或引擎、不修改输入，并覆盖损坏的声明。

批次和显式任务入口统一使用 `_prepare_for_run()`，共用原片准备及错误、待补齐、忽略项检查。`_BatchInputError` 携带结构化问题，两个入口共用失败摘要生成。显式任务入口的错误信息现在增加具体原片路径、原因及 `details`；批次已有输出保持一致。问题批次仍整批阻断，其他批次继续处理，预览也遵守同样规则。

以本轮起点对照两种入口、预览/执行以及正常、三类原片问题和准备异常共 20 种组合：准备与任务调度调用完全一致，除显式任务增加问题明细外，摘要完全一致。证据 `/tmp/umi-flow-processing-equivalence.json`，起点快照 `/tmp/umi-flow-cleanup-20261118-before`，独立源码补丁 `/tmp/umi-flow-cleanup-only.patch`，均非运行依赖。源码身份按既有规则更新，业务配置、原片、设备资产及历史结果没有改写。

验证：`./test.sh --all` 的 960 项 Python 测试（212.041 s）、查看器前端、CLI 自检与 KB4 C++ 数值检查全部通过，日志 `/tmp/umi-flow-full.log`。后端 27 项、地图 25 项、处理/布局/配置 49 项定向回归通过，日志分别为 `/tmp/umi-flow-contracts.log`、`/tmp/umi-flow-maps.log`、`/tmp/umi-flow-processing-final.log`。文档更新后 3 项契约测试通过，同级 Organizer 27 项交接测试全部执行通过，日志 `/tmp/umi-flow-docs.log`、`/tmp/umi-flow-organizer.log`。七个实现模块无未引用顶层导入，`git diff --check` 通过。

## 2026-11-18：收敛海报融合和任务报告的重复计算

场景单双手海报路径共用标签几何准备、地图观测重编号和位置支持证据组装。双手每只手在本轮初始化与组图间复用自身镜头模型、板角点及像素观测，避免重复构造；初始剔除及优化后剔除的重试仍重新准备过滤后的观测，不跨轮缓存。每只手的后镜头模型和时间偏移保持独立。节点时刻、变量编号、因子插入顺序、单轮剔除预算及最终验收顺序保持原值。

任务报告复用一次质量摘要中的坐标范围、行数、最长视觉缺口和地图一致性指标；约束区间只计算一次，同时提供分类及视觉/IMU 视图。逐帧来源字段和 CSV 引用继续保留，报告及结果 schema 未改变。没有新增配置项、业务层级或外部依赖。

清理前后在同一进程内使用真实 GTSAM 对照 12 组固定观测：单双手 staged/joint、仅诊断、无观测、单手缺海报、固定 rear 外参、初始化剔除、优化后剔除，以及不同 rear 内参/时偏和部分标签集。输出位姿、跟踪标记及全部质量证据逐值相同，位姿最大绝对差均为 0；双手正常求解的相机模型构造次数由 4 降为 2，剔除重求解由 8 降为 4。新增永久回归覆盖双手使用不同相机模型、时偏和标签子集。

用同一份运行输入分别生成场景、任务海报与单手局部三类报告，JSON 文件逐字节相同；缺失、空和分段帧表的区间摘要也完全相同。对照记录为 `/tmp/umi-cleanup-poster-equivalence.json`、`/tmp/umi-cleanup-report-equivalence.json`。起点快照 `/tmp/umi-cleanup-poster-20261118-before`，源码补丁和公共签名复核为 `/tmp/umi-cleanup-poster-only.patch`、`/tmp/umi-cleanup-poster-review.json`，均非运行依赖。源码身份仍按既有规则变化，不改写原始录像、业务 YAML、设备资产或历史成果。

验证：`./test.sh --all` 的 956 项 Python 测试（209.735 s）、查看器前端、CLI 自检与 KB4 C++ 数值检查全部通过，日志 `/tmp/umi-cleanup-poster-all.log`。此前海报/位置支持/场景 73 项定向回归及新增不同相机参数用例通过，文档契约 3 项通过；日志 `/tmp/umi-cleanup-poster-targeted.log`、`/tmp/umi-cleanup-poster-cameras.log`、`/tmp/umi-cleanup-poster-docs.log`。两个修改模块无未引用顶层导入，`git diff --check` 通过。

## 2026-11-18：删除遗留死代码并收敛观测读取与训练导出

删除相机注册表中无调用的 `_current_directory()`、`_remove_previous_versions()` 及其唯一专用依赖 `_installed_source()`，三者共 37 行函数实现；清理未使用的 `parent`、图优化 `node_set`、物理检查工具 `sys` 和统一读取后不再使用的两处 NumPy 导入。版本查找直接选择镜头对应的来源指针并按键查询，仍保留不可变历史版本及独立 rear 资产的查找规则。未修改设备索引或已安装的资产。

四处原始海报 NPZ 恢复逻辑统一到 `vision/poster/detection.load_observations()`，共用 `allow_pickle=False`、字段恢复与 `TagObservations.validate()`。缓存键、文件哈希和冻结快照的来源校验继续位于原有调用边界；不放宽损坏数据的处理。

训练 `export.build()` 从 78 行收敛为 42 行，单集采样、视频复制、元数据与哈希写入由 `_write_episode()` 承担，外层保留输出锁、临时目录、最终源结果复核与整包原子发布。没有增加导出层级或改变数据协议。相对于本轮起点，8 个实现文件净减少 51 行；原有公共函数签名不变。

报告逐帧来源仍属于当前消费契约（已有场景回归明确检查），本轮保留字段；融合求解和因子插入顺序未改动。`parse_gyro()` 的别名调用和公共入口导出继续保留。缓存按现有真实源码身份规则失效，不伪造旧身份，也不改写原始录像、场景 YAML、设备资产及历史成果。

使用同一份冻结结果分别运行清理前后的导出代码：`manifest.json`、`export_report.json`、每集 `episode.json`、`samples.npz` 和两路 MP4 共 6 个文件逐字节一致，核对为 `/tmp/umi-cleanup-export-equivalence.json`。本轮起点快照为 `/tmp/umi-cleanup-20261118-before`，独立源码补丁与公共签名核对为 `/tmp/umi-cleanup-only.patch`、`/tmp/umi-cleanup-review.json`；均非运行依赖。

验证：中断后补跑的 `./test.sh --all` 共 955 项 Python 测试（207.336 s），查看器前端、CLI 自检与 KB4 C++ 数值检查全部通过，日志 `/tmp/umi-cleanup-final-all.log`。新增回归覆盖原始观测数组与类型保真、缺字段/越界/对象数组拒绝，以及第二集视频失败时清理已写出的第一集且不发布部分数据。此前 80 项定向回归通过；文档契约 3 项、同级 Organizer 27 项测试均通过，日志分别为 `/tmp/umi-cleanup-targeted.log`、`/tmp/umi-cleanup-docs.log`、`/tmp/umi-cleanup-organizer.log`。静态复核未发现未引用的顶层导入（排除显式公共导出），`git diff --check` 通过。

## 2026-11-17：收敛坐标读取、自动回退与训练质量策略

修正场景代码指南中的旧模块名、地图专用调用链和坐标声明，区分任务级估计路由与地图内融合初值。训练模块与 exports 入口增加格式对照、明确参考系的读取示例；无地图设计方案更新已交付的海报结果契约，完整 `poster_vio` 仍保留为未来能力。

新增 `dataset/coordinates.py`，从同一坐标声明生成导出元数据，`read_pose_series()` 返回 `T_reference_tcp` 视图及参考系/范围。磁盘 `<arm>_T_scene_tcp` 字段和 schema 1/2 读取保持兼容；缺少显式坐标字段时从 scene 推导，矛盾声明、非米制单位、无效刚体变换拒绝读取。

地图读取显式识别缺失、损坏及声明无效（`MapUnavailableError`）与不兼容（`MapCompatibilityError`）。`auto` 仅捕获这些地图错误及既有 `SceneFusionQualityError`，普通程序异常和执行错误不再被宽泛捕获为回退理由。持久记录 `category/reason/error_type`，数据读取失败保留异常链；引擎查询失败与地图文件读取边界分离。无地图共同位置支持缺口与双手重力夹角迁到 `parameters.poster_independent`，默认仍为 2 s、5°；此前通过地图参数调整这两项实验门槛的覆盖需要迁移，旧日常 YAML 继续有效，旧单手回退规则保留。

训练导出增加 `--quality-policy all|position_supported`、`--unknown-support drop|error`。默认保留时间有效样本；严格模式要求所有手插值两端均有位置支持，历史未知证据不会冒充无标签，仍接受已知的自然视觉正证据。时间缺口与质量删除分开统计，策略及各手拒绝/未知数量写入清单与报告。剔除后重建连续片段；即使低帧率未采到源坏段或时间缺口，也切断 `segment_id`。宽度采样复用筛选后的时间轴，失败不发布部分数据集。未更改求解器、原始录像或既有成果。

新增文档契约检查，使用真实配置及 CLI 解析器验证标记示例、检查当前文档链接及指南函数入口；小视频真实导出检查字段表、读取参考系、筛选报告与失败原子性。文档检查加入 `test.sh --quick`。

验证：`./test.sh --all` 的 952 项 Python 测试（207.567 s）、查看器前端、CLI 自检与 KB4 C++ 数值检查全部通过，日志 `/tmp/umi-contract-final-all.log`。最终补充的地图清单内部条目校验另以 26 项地图/海报后端回归通过，日志 `/tmp/umi-contract-map-final.log`；文档示例、真实导出字段和低频分段边界 13 项回归通过，日志 `/tmp/umi-contract-doc-final.log`。同级 Organizer 的 27 项测试全部执行通过，日志 `/tmp/umi-contract-organizer.log`。`git diff --check` 通过。

## 2026-11-17：查看器按结果和资源分组，核验 Organizer 交接

查看器迁移 7 个模块：目录发现、任务结果、协议读取和显示位姿位于 `viewer/results/`；视频、地图及可再生缓存位于 `viewer/resources/`。HTTP 入口、静态资源、公共错误继续留在原位置。所有调用直接导入实际模块，没有旧路径转发，两个子包初始化不加载求解器或创建执行资源。标定代码已有明确分层，本轮没有继续拆分。

迁移前后浏览器目录列表和轨迹响应逐字段一致，场景/轨迹/标定三类代码身份也完全一致，因此此次 viewer 调整不导致数值算法缓存失效。地图转换缓存仍记录实际地图/缓存实现文件，移动后按资源身份重建。970 个函数/类的归一化 AST 核对无差异，报告 `/tmp/umi-layout-viewer-semantic-audit.json`；浏览器协议及身份对照为 `/tmp/umi-layout-viewer-contract-before.json` 与 `-after.json`。

对同级 `insv_organizer` 的检查发现：业务代码通过原片目录、`collection.yaml`、`import_manifest.json` 与 UMI 交接，不导入内部模块，启动器使用的 `.deps` 路径也未改变；其交接测试仍导入前几轮迁移的两处旧路径，被宽泛的 ImportError 跳过规则隐藏。已更新测试到 `management.inputs.scene` 和 `pipeline.maps.epoch_selection`，改为仅仓库不存在时跳过，仓库存在但导入失败则报错；增加两项跳过条件回归。Organizer 27 项测试全部执行通过（包括真实双边增量交接），启动器 `--help` 正常，业务源代码、配置、原片和提交目录未改写。日志 `/tmp/insv-organizer-umi-final.log`。

本轮 UMI 验证：查看器及相关流程 70 项定向回归通过，`./test.sh --all` 的 930 个 Python 测试（204.349 s）、前端、CLI 自检与 KB4 C++ 回归全部通过，日志 `/tmp/umi-layout-viewer-all.log`。新子包轻量导入、旧模块入口、YAML 对照和当前文档链接检查通过；viewer 根目录实现文件从 9 减为 2（HTTP 入口和公共错误）。

## 2026-11-17：夹爪测量与采集管理分组

继续迁移 16 个模块：夹爪共用检测、源像素处理和按时刻采样集中到 `gripper_width/detection/`，三维标签中心距集中到 `distance/`，预览与监测集中到 `tools/`；开口标定、换算和时间序列保留顶层。场景声明解析集中到 `scene/management/inputs/`，批次发现、配置/资产快照、执行与 epoch 编排集中到 `batch/`；公共调度、任务验收、登记、资产和清理保留管理层。夹爪根目录实现文件从 11 减为 3，管理层从 15 减为 7，统计不含包初始化文件。

导入、工具入口、缓存依赖和代码指南同步更新，没有新增旧模块转发。检测、测距与采样的代码身份使用迁移后的真实文件，跨目录解码/时钟依赖改为项目根路径定位。任务代码身份的八项编排排除保持原集合，改用精确路径表达以避免文件重名影响策略。业务 YAML、原始录像、注册资产和历史成果没有改写；相关缓存按新的代码身份失效。

验证：夹爪 86 项、场景管理 137 项定向回归通过；`./test.sh --all` 的 930 个 Python 测试（205.108 s）、前端、CLI 自检和 KB4 C++ 回归全部通过，日志 `/tmp/umi-layout-measurement-all.log`。归一化 AST 核对 970 个函数/类，仅五处身份/寻址调整符合预期，没有计算逻辑差异，报告 `/tmp/umi-layout-measurement-semantic-audit.json`。合成 KB4 测距、像素宽度映射及时间采样的六组数组与迁移前逐元素一致，样本为 `/tmp/umi-layout-measurement-numeric-before.npz` 与 `-after.npz`。新子包保持轻量导入，旧模块入口、YAML 对照及当前文档链接检查通过。

## 2026-11-17：按职责收拢海报、融合与场景流水线

迁移 35 个模块到真实子包：共享海报位于 `vision/poster/`，独立融合位于 `trajectory/algorithms/fusion/`，地图生命周期及缓存位于 `scene/pipeline/maps/`，任务估计、测量、评估和发布位于 `scene/pipeline/tasks/`。海报原几何模块的七个噪声、剔除和条件质量函数原样移到 `poster/quality.py`，几何、质量、因子保持单向依赖。入口、公共模型与通用原片准备继续保留原层级；包初始化只记录职责，没有旧路径转发或预加载求解器。同步更新导入、动态调用字符串、融合 `python -m` 入口、脚本文件位置、缓存依赖及文档。海报检测缓存原先按相邻文件查找解码实现，改为查找父级视觉目录。

代码身份继续记录真实路径，相关旧缓存按新身份失效；本次没有修改业务 YAML、原始录像、注册资产或历史成果。归一化 AST 核对覆盖 970 个函数/类，除检测缓存文件寻址外未发现计算逻辑变化；报告 `/tmp/umi-layout-semantic-audit.json`。新子包导入不加载 NumPy/OpenCV/GTSAM，旧模块入口检查通过；地图缓存测试验证新质量模块纳入依赖。

验证：`./test.sh --all` 的 930 个 Python 测试（202.883 s）、前端测试、CLI 自检和 KB4 C++ 回归全部通过，日志 `/tmp/umi-layout-all.log`。008 使用原冻结输入在新入口重放融合，左右手的 `traj_tum.txt`、`state.csv`、`seg_placement.csv` 与迁移前逐字节一致；核对报告 `/tmp/umi-layout-008/layout_validation.json`。这是代码组织回归验证，不代替外部物理精度实验。

## 2026-11-17：任务级估计接口与双手海报独立重建

继续收敛剔除、安装修正和选轮规则：将既有场景角点/seat 因子提取到 `vision/poster_factors.py`，场景和独立图共同使用；共享参数保留原 0.2 rad 先验，新增严格关闭修正及默认 5° 上限。剔除数量、剩余时间覆盖与原始观测记录集中到 `vision/poster_pose.py`；修复单手最终剔除绕过比例预算、共享图剔除前提前 RMS 拒绝、建图尺度递归重置预算的问题。独立图增加优化后的单轮重求解，复用对齐与基线。重试复用精确位置支持统计，稀疏合格标签可补足位置缺口，原视觉覆盖、尺度硬保护及旧证据路径保留。建图身份纳入实际消费的两项剔除参数。

本轮验证：`./test.sh --all` 的 929 个 Python 测试（203.474 s）、前端、CLI 自检和 KB4 C++ 回归全部通过，日志 `/tmp/umi-polish-budget-all.log`。008 使用原冻结 SLAM、IMU、海报角点重放到 `/tmp/umi-polish-budget-008`，左右 RMS 仍为 1.75409/1.32351 px，rear 修正 0.39755°/0.59358°，均通过新增上限；左手单轮剔除 24 个观测，右手 0 个。对真实重放产物执行新旧重试判定：右手自然视觉覆盖 0.7824，旧策略判为可疑，新策略在共同位置支持缺口 0.10010 s 时不再仅因覆盖率重试；这是保存产物的融合重放与策略验证，未重新运行整条 SLAM 流水线或宣称测得物理精度。报告 `/tmp/umi-polish-budget-008/retry_validation.json`，原录像、collection 配置和原成功结果未改写。

同日继续补齐质量证据：把场景角点噪声模型收敛到 `vision/poster_pose.py`，独立图和场景 staged/joint 复用现有 Jacobian 与鲁棒权重，输出固定海报/标定下的逐时刻局部精度诊断。无地图后端只用几何合格的观测检查位置支持缺口；原始弱观测仍保留鲁棒因子。逐帧视觉/海报/共同来源沿既有 CSV、报告、训练采样传递，未知历史证据保留未知。双手相对位姿复用统一几何函数；新增 `evaluate-relative`，使用外部实测固定夹具参考，复用 TCP 读取、重采样、资产哈希和标定统计，不做拟合对齐或修改验收状态。

008 复用保存的局部 SLAM、IMU 和海报角点重放融合：左手 489/472、右手 384/379 个海报观测/几何合格时刻，共同位置支持最大缺口分别 0.03337/0.10010 s；重投影 RMS 仍为 1.75409/1.32351 px。仅是内部证据和回归验证，未获得独立夹具真值，不能据此认证双手相对精度。重放输出位于 `/tmp/umi-polish-008`，原始 008 及原成功结果未改写。

上一轮验证：`./test.sh --all` 的 918 个 Python 测试、前端测试、CLI 自检及 KB4 C++ 回归全部通过；日志 `/tmp/umi-polish-all.log`。008 重放输出进一步经过现有轨迹写入器、真实 MP4 读取和训练采样器，386 个共同采样点的来源 mask 与相对位姿通过检查；结果在 `/tmp/umi-polish-008/output/validation.json`。与原 TCP 文件的位置差异最大约 0.8 μm，处于文件数值精度量级。该诊断采样未变更原录像尚未确认的同步状态，也未发布为训练就绪数据。

新增 `scene/pipeline/estimation.py` 的任务级估计输入及后端注册，将 `scene_map` 与 `poster_independent` 放在同一条场景流水线。批次及任务可选 `reconstruction.backend: scene_map|poster_independent|auto`；旧配置保持场景地图路径。新 `auto` 只加载已有地图，在地图不可用、标定/视角不兼容或结构化场景融合质量失败时，整任务切换，不混用一只手的场景解和另一只手的局部解。输入错误、海报几何冲突和执行错误仍失败。完整标签初始化与原始视觉点 BA 的 `poster_vio` 保留为后续设计，未作为本次已完成能力。

`poster_independent` 复用现有独立引擎，在各手本录像局部视觉地图、分段尺度和 IMU 图中持续融合海报，之后以 `inverse(T_world_board)` 转换到共同板坐标。两手必须实际融合成功、布局一致，并通过板坐标重力方向及自然视觉/海报联合支持缺口检查。海报坐标不假定 z 向上；时间同步不由静态海报推断。结果新增 `umi_poster_episode`、`task_poster` 范围、`poster:<run_id>` 和空地图引用，仍原子发布到 results；查看器、复用、人工验收和训练中间包已接入，训练位姿声明为 `T_board_tcp`。

独立海报初始化先用鲁棒板拟合获得残差，再有限地剔除孤立异常，避免总体 RMS 在剔除前提前拒绝；剔除比例限制 15%，实际最终 RMS、正深度和优化下降仍在原生图验收。修复剔除重算后证据错误嵌套，并保存实际参与约束的时间。任务已有环境角点缓存经原身份校验后冻结给独立引擎，避免两次解码检测。本次没有直接移植新的 specialist 源码，复用的是 UMI 已有独立引擎；其后续可移植模块见设计方案。

验证：`./test.sh --all` 的 **903 项 Python 测试（192.168 秒）**、查看器前端、标定/场景 CLI 自检和 KB4 C++ 数值回归通过，日志 `/tmp/umi-poster-independent-all.log`。随后补充缺失显式地图版本及布局推断边界，相关 **82 项测试、12 项 subtests** 通过；覆盖批次/任务配置优先级、无地图、损坏地图、双手整体回退、禁止输入错误回退、旧结果保护、共同约束缺口、重力不一致、查看器及训练读取。`git diff --check` 与新增文档链接检查通过。

真实 008 采用复制原片的隔离批次 `/tmp/umi-poster-independent-008-aih16sre/test_shelf/20261109_008`，未修改原批次配置、原片或原成功结果。成功运行 `397221b6c5ec4963b09d55c8aa77f9b9`：左手 490 帧、海报 RMS 1.7541 px、剔除 24 次观测；右手 386 帧、RMS 1.3235 px、未剔除。两手重力方向差 0.3612°；右手自然视觉缺口 2.8028 s 时仍有海报约束，两类位置支持的最大时间间隔均约 0.033 s。隔离批次没有 `.pipeline/maps`，结果 `map: null`，两手均记录 `scene_map_priors_used: false`。发布资产哈希和查看器双手加载通过；相同输入重复运行返回 skipped。日志 `/tmp/umi-poster-independent-008-rerun.log`、`/tmp/umi-poster-independent-008-reuse.log`，复核记录 `/tmp/umi-poster-independent-008-validation.json`。

首轮实跑暴露了直接套用场景地图“自然视觉缺口”门槛的问题，曾拒绝右手；修正为按真实自然视觉与海报时刻并集判断位置支持缺口，并增加两类约束同时丢失时拒绝的测试。未降低最大缺口阈值或最终重投影阈值。008 的同步仍为未确认的 assumed_simultaneous，海报资产仍为 pdf_design，真实轨迹误差未测量，因此此试跑不代表双手训练精度已经通过独立验收。

## 2026-10-15 结构

轨迹实际走的路径和另一套没人调用的相机接口叠在一起，画面裁切也写了多份。这次只改调用关系。

- 删掉未接入轨迹的 canonical capture：`software/camera/contracts.py`、`capture_io.py`，以及 `software/camera/x3/` 里的 `cli.py`、`export_capture.py`、`sdk_export.py`、`projection.py`、`profile_guard.py`、`lens_geometry.py`、`video_frames.py`、`timestamps.py`。
- 4K60 的画面（3840x1920、左右镜头、中心 1440、每两帧取一帧）收到 `software/trajectory/frame.py`。`x3_extract`、`make_lens_video.py`、`calibration/dataset.py`、`postprocess.py` 用这一处。裁切数字与原来相同：前镜头 `(2160, 240, 1440, 1440)`，后镜头 `(240, 240, 1440, 1440)`。
- `umi-traj` 对一段录像的入口是 `software.trajectory.cli.run_recording`。命令行参数先收成 `RunRequest`。
- `profiles.py` 的说明改成和这条工具一致：轨迹用 4K60 的 1440 裁切，不用表里的默认 960，也不跑 5.7K30。

没有改融合计算、ORB-SLAM3、IMU 读数、Basalt 解算，也没有改标定包复制 `kiwi_fuse3s.py` 的方式。

## 2026-10-15 说明

给人看的入口和用词改过。算法、`kiwi_fuse3s.py` 这个文件名、`KIWI_*` 环境变量、结果里的 `source: factory` 都没改。

- 根目录加了 `README.md`。使用说明里的克隆地址改成这个仓库。
- 文档和 `umi-calib` 的中文输出把「出厂标定」改成「样例标定」。
- 模块说明里不再指向本仓库没有的实验记录和旧路径。融合文件开头的改版日记收成对当前行为的说明。

## 2026-10-15 标定所需文件

从同一份 `umi@71a124d` 补拷了 `umi-calib` 要用、当初整树没带的文件。硬件 CAD 仍不在这个仓库。

- `hardware/calibration_targets/`：A4 / A3 标定板的 json、pdf、png、yaml。程序读 `aprilgrid_6x6_24mm_A4.json`。
- `software/calibration/setup/basalt/umi-basalt-0.1.7-fsoffset-linux-x86_64.tar.xz`：sha256 `c531b065fde3d51b4a188168eb3867cef3ec17338ae11f9397d0a5236c5aa4c2`，与 `basalt/PROVENANCE.md` 一致。

## 2026-10-15 入口

根目录只留 `README.md` 作为使用者的五步说明。`todo.md`、本文件、`QUICKSTART_X3_STREAM.md` 挪到 `docs/dev/`。实时预览程序仍不在仓库里，录制说明不再指向它。

## 2026-10-18 入口按两块功能

给人看的文档收成 `docs/标定/` 和 `docs/轨迹/`。根目录 `README.md` 先写标定、轨迹两条命令，再写同一台相机的五步。`software/trajectory/calib/` 的说明改成：这里是轨迹要读的样例标定和夹爪 CAD，不是 `umi-calib` 的产出。算法路径、`kiwi_fuse3s.py`、`KIWI_*`、`source: factory` 都没改。


## 2026-10-22 标定外围流程整理

标定入口按职责拆分，便于分别维护命令、验收和报告：

- `software/calibration/cli.py`：参数解析与命令分发。
- `workflow.py`：A/B 求解、D 检查、C 验证的执行顺序。
- `runtime.py`：依赖检查、子进程执行、CPU 设置与运行路径。
- `checks.py`：检查项状态、C 验收规则和最终结论。
- `reports.py`：JSON / 文本报告写出。
- `devices.py`：标定包安装与设备档案管理。

保留原有命令选项、求解器参数、验收阈值、图像提取及融合算法。
`verify` 仅对未登记序列号回退到样例标定；已停用设备或损坏档案明确报错。
`software.camera.registry.UnknownCameraError` 是 `RegistryError` 的子类，调用者可捕获父类。后续清理已删除 `software.registry` 兼容模块别名，调用方统一使用 `software.camera.registry`。
验证报告的 `calibration` 对象增加显示用的 `name` 字段，原字段保留。

外围回归检查（不运行真实视频求解）：

```bash
.deps/calib_env/bin/python -m unittest discover -s tests -v
./umi-calib check
```

## 2026-10-22 标定目录自动识别

`umi-calib run <目录>` 支持 A/B/C/D 子文件夹各放一个原始 `.insv`，保留平铺四段按文件名排序的输入方式。布局混用、缺段或同段多个视频时明确报错，输入发现逻辑当时位于 `software/calibration/inputs.py`（当前为 `software/calibration/pipeline/inputs.py`），求解与验收流程不变。

## 2026-10-22 标定产物精简

结果顶层保留 `report.txt`、`bundle/`、`diagnostics/`；详细 JSON、板配置、A 阶段结果、D 检查、日志与可选工作目录集中到 `diagnostics/`。
B 阶段成功后将最终参数移入 `bundle/calibration_r1440.json`，不再保存顶层和 B 阶段的重复副本；保留求解器、阈值、标定包格式和手动启用方式。
报告优先展示失败项、测量或覆盖提醒，再展示样例参数差异；加入生成时的设备档案核对快照。安装兼容旧 `report.json` 位置，并刷新本地和设备档案中的文本报告。
当前 M4CP 的既有输出也按相同布局整理；原始录像和标定包不变。

## 2026-10-22 轨迹外围流程与结果复用

`software/trajectory/cli.py` 负责参数和批量调度，单视频流程移入 `workflow.py`。
`models.py` 定义运行选项，`selection.py` 将设备档案转换为有明确字段的标定选择及资产路径，`reuse.py` 集中判断旧结果是否匹配。
重试和评分位于 `runner.py`，诊断发布与清理位于 `artifacts.py`，输入发现位于 `inputs.py`，环境诊断位于 `runtime.py`。
标定选择优先级保持为 `UMI_TRAJ_CALIB`、`--device`、视频内部序列号；`software.camera.registry.resolve_insv` 的原有字典 API 保留供标定模块使用。

成功结果新增 `run_fingerprint`，比较录像路径/大小/修改时间、小型标定及夹爪资产完整哈希、处理模式/掉帧选项/重试次数。
缺少记录信息的旧结果、设置或资产变化、必要输出缺失时重新计算；设备档案校验仍发生在结果复用之前。
大型录像不额外完整哈希；轨迹算法或外部二进制升级后仍需 `--force`。
重试排序、求解器命令、融合算法、后处理算法及现有产物结构保持原样。

外围回归测试使用模拟执行器覆盖重试、成功、失败、跳过和更换标定后的重算，不启动真实视频求解：

```bash
.deps/env/bin/python -m unittest discover -s tests -v
./umi-traj check
```

## 2026-10-22 TXT 展示 C 段轨迹误差

全套标定和单段 verify 的文本报告增加 C 段相机位置误差专节，展示已计算的 RMSE、95 分位、尺度比、尺度对应的距离偏差、缩放诊断误差和参考数据覆盖。缺失或失败时明确说明无法评估，不改变计算和验收规则。

## 2026-10-22 轨迹结果目录整理

轨迹输出顶层只保留 `traj_tcp.txt`、可选的 `trajectory.png`、人读的 `report.txt` 和 `diagnostics/`。
相机/IMU 轨迹、逐帧 CSV、配置、完整摘要、日志和保留的工作数据集中到 `diagnostics/`。
`software/trajectory/results.py` 管理布局、旧路径兼容和报告生成；不修改后处理数值算法，不删除独有中间数据。
`reuse.py` 和标定 C 检查都通过兼容读取函数访问结果，目录整理不改变输入指纹，不导致单纯因移动目录而重算。
已存在的成功匹配结果在跳过计算时也会整理；本次任务 `VID_20261022_225713_00_005` 的既有产物已整理并核对数据完整性。

## 2026-10-22 UMI Trace 本地三维查看器

新增 `umi-view` 和 `software/viewer/`，参考 Wholebody TraceLab 的轨迹、视频、速度和时间轴联动交互，读取 UMI 新旧结果布局。
使用 Python 标准库 HTTP 服务和本地 Canvas 三维透视投影，不引入 Wholebody、外网 CDN 或前端构建依赖。
支持 TCP/相机/IMU 切换、姿态轴、参考轨迹、视角控制、逐帧定位、倍速和循环播放；缺少视频时独立回放轨迹。
查看器只读已有结果，不修改标定、求解算法和录像；默认监听 127.0.0.1:9091。

## 2026-10-24 浏览器视频解码兼容

UMI Trace 在原生视频解码失败时自动切换为服务器 JPEG 画面预览，也提供手动“兼容播放”入口。
新增 `software/viewer/media.py` 与 `static/media.mjs`：复用已安装的 FFmpeg，限并发、限超时；前端合并待处理请求，并按实际显示画面更新对应轨迹点。
兼容画面帧率低于原生视频，但不写出额外视频或抽帧数据，不修改轨迹结果、原始录像或标定配置。

## 2026-10-24 标定流程按业务阶段拆分

`software/calibration/workflow.py` 直接展示读取录像 → A 内参 → B 相机–IMU → D 重力检查 → 生成标定包 → C 轨迹验证。
新增 `models.py` 定义请求、统一路径、阶段结果和阶段异常；`inputs.py` 只负责录像发现与身份检查。
原 `dataset.py` 改名为 `recording_data.py`，保留原 INSV 提取和解码算法。
A/B 求解分别放入 `intrinsics.py`、`camera_imu.py`，C/D 验证和单段 verify 放入 `validation.py`。
`checks.py` 集中验收条件，`reports.py` 将阶段结果写为原有 TXT/JSON 报告，`runtime.py` 负责进程与运行环境，`devices.py` 保留显式安装与设备管理。

求解参数、噪声先验、检查阈值、输出布局和默认手动启用方式保持原样。
失败时新增 `failed_stage` 和 `diagnostics/logs/stage_error.log`，统一清理工作目录，保留已完成阶段证据；`--keep-work` 仍可保留工作数据。
相机身份检查提前到输出目录清理之前，输入不合法时不会因 `--force` 删除已有输出。
重构前后以模拟外部求解器比较命令、报告、返回码和标定包内容；实际录像、既有标定和设备档案不作改动。
开发阅读顺序见 `software/calibration/README.md`。

## 2026-10-24 标定辅助文件归组

标定目录顶层保留 12 个主流程模块、README 和 Python 包入口。
`board_pnp.py`、`aprilgrid_detect.py` 移入 `algorithms/`；转 bag 和重力检查两个独立工具移入 `tools/`。
安装脚本改为 `setup/install.sh`，依赖清单改为 `setup/requirements.txt`，Basalt 压缩包、补丁与来源记录移入 `setup/basalt/`。
同步更新 Python 导入、子进程脚本路径、`umi-calib` 自动安装入口和文档链接；安装产物仍位于原来的 `.deps/`，标定命令、计算方法和结果布局保持原样。

## 2026-10-24 标定调用链简化

CLI 显式调用 `workflow.run(request)` / `workflow.verify(request)`，移除原 `cmd_run` → `run_calibration` 转发层。
命令行值转换移入 `inputs.py`，`models.py` 只定义请求与阶段数据；运行环境准备移入 `runtime.py`。
主流程使用完整模块名调用阶段，样例参数和录像名显式传递，报告不再作为求解与验收的数据来源；最终结论与退出码由 workflow 决定，报告只记录和展示。
单段 verify 的读取、检测、轨迹计算直接列在 workflow 中；内参 JSON 读取移入共享板几何模块，B/C 不再依赖 A 的求解模块。
移除录像提取模块无效的 `sys.path` 修改，保留必要的 Python 包启动入口。
对比重构前后的正常、失败、误差超限等 13 种模拟场景，核对返回码、求解命令、报告和标定包一致；命令行选项、算法、阈值和默认手动启用方式保持原样。

## 2026-10-24 标定死代码与默认对照精简

删除无调用的 AprilTag 渲染辅助函数 `tag_bits`、相关常量和未读取的检测器状态；保留实际检测和角点顺序。
录像提取保留磁盘输出，删除无人使用的进度回调、返回值及仅附加在返回值中的图像统计；源提取清单仍由 `x3_extract` 正常写入。
移除 `Refused` 与其父类 `RuntimeError` 的重复捕获；设备安装不再先复制启用前的报告，只在刷新启用状态后发布最终报告。

C 段默认只运行新标定轨迹。新增显式 `--compare-sample`，按需额外运行样例轨迹，两者共用同一份板检测结果。
JSON 报告记录 `compare_sample`，未运行时不生成 `C_factory`；TXT 区分对照未运行、已运行和失败。样例对照的指标或执行失败均不改变新标定验收。
A/B 求解、D 物理检查、新标定 C 验证和全部验收阈值保留，旧结果读取与旧命令兼容保留，实际录像、既有标定和设备档案不作修改。

## 2026-10-24 标定 workflow 参数、报告与异常收尾统一

`verify` 读取、板检测和轨迹验证分别记录失败阶段，运行异常保留原始原因和堆栈，不再一律称为“录像读不了”；TXT 与 JSON 均展示失败阶段。
`validation.trajectory_check` 改为接收 `StageContext`，其余标定选择和比较名称使用命名参数，原有子进程参数和误差计算保持原样。
`reports.py` 提供 `CalibrationReport` 和 `VerificationReport`，统一负责组装、计时、写入和展示。两个 workflow 自行维护 `check_results` 并判定，再将检查项交给报告；不再从报告对象读取验收状态。
全套标定中的阶段外异常（例如验收函数报错）也生成含已完成测量的 FAIL 报告。
两个入口的成功和失败均在写完报告后由外层 `finally` 清理临时目录，中断时也清理，`--keep-work` 行为保持不变。
阶段顺序、A/B 求解、D/C 验证、样例对照开关、检查阈值、旧结果读取和显式安装保持原样。

## 2026-10-24 轨迹模块按流程整理

`cli.main()` 显式调用 `workflow.run_batch()`，批处理再调用 `run_recording()`。单段流程依次写出标定选择、输入复用、时间戳与 IMU 提取、重建重试、TCP 轨迹生成、报告发布和重复产物清理。
`models.py` 定义请求、路径和阶段结果，`recording_data.py` 负责轨迹数据准备，`reconstruction.py` 负责执行重建，原重试指标与排序移入 `checks.py`。
日志归档与清理收进 `results.py`，文字和终端展示移入 `reports.py`，移除重叠的 `artifacts.py`、旧 `runner.py` 和未调用的标定环境适配函数。
失败新增阶段与堆栈记录，中断保留诊断数据；成功先发布报告再清理重复文件，清理失败只在成功报告中补充提醒。同名录像在批处理启动前拒绝，避免并发覆盖。

共享 `frame.py` 与 `x3_extract.py` 移至 `software/camera/x3/frame.py`、`extraction.py`；标定和轨迹直接依赖相机读取模块。提取清单的 `tool` 路径随源文件位置更新，数据内容不变。
原 `trajectory/calibration.py` 更名为 `calibration_assets.py`，区分标定封装与参数求解。数值算法、独立工具、资产、安装资源分别放入 `algorithms/`、`tools/`、`assets/`、`setup/`；根安装脚本与各调用方同步更新。
融合脚本、定位合并脚本、画面定义、原始配置/标定资产和安装资源逐字节保留。资产内的旧路径注释保留，以免改变现有标定包生成与校验所依赖的文件内容。
标定选择优先级、CLI 参数、重试阈值、数值参数、结果布局、旧结果读取和输入指纹保持兼容；已登记设备、原片和已落盘轨迹未作修改。

验证包括 84 项 Python 测试、轨迹与标定环境自检、Shell 语法检查。用现有录像实际提取的时间戳和 IMU 与重构前逐字节比较；用保存的 IMU 位姿和融合证据重跑前后处理，核对指标及六份输出（含预览图）逐字节相同；用短 4K59.94 合成片验证独立镜头工具的真实 FFmpeg 输出与清单相同。另用替代外部程序执行实际 Bash 编排，核对 handheld/dual 两种模式下的各阶段调用参数仅有路径搬迁差异。本次未重新执行完整 ORB-SLAM3 建图，也未重装依赖或构建容器。
阅读入口见 `software/trajectory/README.md`。

## 2026-10-24 合并轨迹外围小模块

将 `trajectory/checks.py` 中的建图质量评估、阈值和排序移入 `reconstruction.py`，与重试循环集中阅读；将 `trajectory/inputs.py` 中的录像发现、去重和默认输出目录移入 `recording_data.py`，统一管理输入发现和数据准备。删除两个旧文件，更新 workflow、测试调用与阅读指南，顶层业务模块由 13 个减为 11 个。
`SelectedCalibration.as_postprocess_dict()` 删除后处理未读取的 `gripper_dir`、`tcp_sha256`、`mask_sha256` 三个字段；注册表的校验字段保留。
验证搬移函数语法树与重试控制流等价，算法、工具、资产与安装资源逐字节一致，删除适配字段前后的完整运行配置一致。84 项 Python 测试与轨迹、标定环境自检通过。

## 2026-10-24 轨迹落盘布局分为交付结果、详情与排查数据

根目录发布 `traj_tcp.txt`、`front.mp4`、`trajectory.png` 和 `report.txt`；相机/IMU 轨迹、逐帧信息、配置与摘要归入 `details/`，日志、数据集和各轮重建归入 `debug/`。录像时间、位姿转换、重建算法、质量阈值和输入指纹保持不变。
新运行成功后默认保留结果、根视频和 `debug/logs/`，清理 `debug/dataset/` 与 `debug/reconstruction/`；`--keep-work`、失败或中断时保留完整中间数据。视频在报告发布前独立保存到根目录，同文件系统优先硬链接，无法硬链接时原子复制，后续清理不会使 viewer 视频失联。
`results.py` 统一读取当前 `details/`、旧 `diagnostics/` 和平铺布局，也统一查找新旧视频位置；viewer 与标定 C 验证复用该入口。迁移前检查全部目标冲突，迁移保留已有独有文件并修复重试视频的软链接；复用旧结果只迁移布局，不清理原有中间数据。摘要记录 `details-v2`、新的配置相对路径与中间数据保留状态。
已迁移 `recordings/M4CP/tasks/umi_traj_out/VID_20261022_225713_00_005`，迁移前在 `/tmp/` 保存完整副本并演练；原有轨迹、视频、配置和中间数据逐项校验哈希一致，轨迹指标不变，已有成功结果仍可复用。没有重新执行 SLAM。
验证包括 91 项 Python 测试、viewer JavaScript 测试、轨迹/标定环境检查，覆盖三种布局读取、冲突拒绝、迁移幂等、软链接修复、复制回退、发布失败保留数据、两种保留模式下的重试视频播放。实际结果能加载全部 1041 个 TCP/相机/IMU 轨迹点，并用 FFmpeg 解码根视频画面。

## 2026-10-24 按“建图、定位、IMU 融合”显式编排轨迹重建

`workflow.run_recording()` 依次读取时间戳与 IMU、准备镜头视频、重建并选择尝试、将 IMU 位姿转换为相机/TCP 轨迹。`recording_data.prepare_lens_videos()` 在重试循环外生成 `debug/lens_videos/` 中的共用视频，各轮通过引用使用同一份画面。视频准备失败直接归因到对应镜头，不进入建图重试。
`reconstruction.run_attempt()` 明确调用 `build_maps()`、`relocalize_all_maps()`、`fuse_visual_imu()`；`run_attempts()` 只负责整轮重试、质量评估和最佳尝试选择。双镜头先各自建图，再分别逐图定位并合并，最后共同融合。地图、地图元数据、定位结果和融合结果使用具名数据结构传递；既有 ORB 日志接口集中在 `read_atlas_maps()` 解析。
外部程序启动、环境变量、nice/taskset/timeout 与 stdout/stderr 由 `runtime.py` 统一管理。删除旧的 `tools/run_kiwi_segment.sh` 和无人使用的路径常量，业务执行顺序由 Python 维护。`postprocess.postprocess()` 更名为 `export_trajectories()`，明确表达 IMU → 相机 → TCP 转换及结果写入。
失败归因细分到镜头视频准备、建图、逐图定位、定位合并、IMU 融合与质量检查；每轮摘要保留失败阶段与原始日志位置。沿用 SLAM 已写出有效产物时可继续处理的既有策略。终端主进度改为四步，并显示各轮内部阶段。
融合脚本、视频裁切工具、定位合并工具保持逐字节一致；位姿换算仅更名及更新注释，数值逻辑不变。选用的标定、原始录像、已经落盘的结果与输入指纹未作修改。
验证包括 100 项 Python 测试、轨迹/标定环境自检，以及旧 Shell 与新 Python 编排的 handheld/dual 对照：使用替代 SLAM/GTSAM 引擎、真实 FFmpeg 和原定位合并工具，核对外部参数、裁切视频、合并定位、全部指标和六份后处理文件一致。覆盖多图、遮手图仅用于建图、共享视频、整轮重试、CPU 与时间参数、各阶段失败。现有 M4CP 标定包仍通过校验，已有轨迹仍可复用；本轮未重新执行完整真实 SLAM 建图。

## 2026-10-24 外围命名与终端

后续精简：删除未调用的 MP4 遍历、旧清单说明函数和通用 profile 表；录像几何识别与固定裁剪集中到 `camera/x3/frame.py`。移除设备注册表中未使用的跳过哈希检查开关、报告复制参数及重复字典合并。复用摘要只读取一次，强制重算跳过复用判断；质量检查异常保留原始原因并统一重试。IMU 解码、数值算法与标定验收阈值保持原样。

标定和轨迹使用一致的请求、序列号、保留中间文件及失败信息命名。当前单段轨迹入口为 `software.trajectory.workflow.run_recording`，请求类型为 `TrajectoryRequest`。`software/infrastructure/terminal.py` 统一终端格式，各阶段与报告模块负责业务文案；移除旧打印函数、设备状态转发函数及数据对象中的终端描述方法。CLI 参数与 JSON 字段保持兼容，数值算法和引擎日志协议保持原样。

## 2026-10-24 文档整理

README 按安装、标定并手动启用、任务录制、抽取、查看排列。公共安装与设备规则集中到 docs/；轨迹命令与结果字段分开，查看器操作移至 docs/结果查看与验收.md。代码命名、终端约定和测试入口集中到 docs/dev/README.md。旧建仓计划和尚未迁入的实时预览教程移入 archive/，当前待办只列尚未完成事项。修正过时的三步进度、仅重建图的重试描述和旧终端报错；运行代码、设备档案和既有产物未改动。

## 2026-10-25 轨迹原理与 KIWI 来源说明

对照 KIWI 官方公开参考实现 0.1.0（`TToTMooN/KIWI_software`，提交 `22860a30dc1c2e9e5f9a2e55c4955a4fba84646e`）的场景估计文档、ORB C++ 入口和 GTSAM 融合代码，更新“代码尚未公开”的过时说明。本项目借鉴纯视觉建图、地图定位、离线 IMU 融合的阶段划分；融合程序沿用实验室仓库导入的独立实现，并未切换为官方程序。

`docs/算法原理.md` 按数据准备、建图、定位、融合组织，补充流程图、IMU 参与边界、地图单位与世界坐标，以及官方共享场景地图和本项目逐段建图的区别。保留已有精度与历史回归记录，修复旧节号引用。项目入口、录制、使用和代码指南通过链接指向详细原理，统一“纯视觉 → 定位 → 离线融合”的表述；修正 C 段地图被直接丢弃的旧描述，明确其不会被后续任务载入。

本次仅修改文档，不改变算法、参数、设备档案或录像产物。验证包括 63 个本地文档链接与章节锚点、Markdown 代码块闭合、差异空白检查，以及历史精度和回归表格未变。

## 2026-10-25 共享场景、多段任务与双手坐标

在 `feat/shared-scene-bimanual` 分支新增 `umi-scene` 和 `software/scene/`，保留原有未提交工作。场景和任务分别用 YAML 清单管理，原始录像继续按相机序列号保存。`shared_front` 复用参考前地图；`dual` 建立左右前地图与共享后地图。建图阶段估计尺度/重力、验收前后地图放置并固定版本；任务只加载该版本，分别融合每只手的 IMU，不重新定义原点或水平朝向。

新增独立 GTSAM 场景融合路径，其度量对齐和前后地图放置参考 KIWI 公开参考实现 0.1.0，修改和 Apache-2.0 许可随场景模块的 `NOTICE.md`、`LICENSE` 保存。原有 `kiwi_fuse3s.py` 保持不变。场景尺度与重力退化、定位覆盖不足、长时间无视觉、资产变化和未确认同步均明确失败；不会以任意坐标或样例后镜头参数填补。

新增显式手动/音频同步，定义 `t_episode = t_video + offset_s`；音频同步要求各相机实测音视频延迟，并记录相关性、峰值比与录像身份。固定偏移模型尚不校正时钟漂移。输出包括同场景 TUM、双手共同时间上的 `bimanual.csv`、各手标定与时间证据、报告和地图展示；新版本成功后原子发布，保留旧成功结果和失败证据。

`calibrate-rear` 复用现有 A/B/D 阶段，增加明确的后镜头裁切；C 用后镜头地图与 IMU 求轨迹后，对照独立标定板位置验收。通过后才生成带序列号、镜头身份和验收记录的后镜头 JSON，不改变前镜头默认标定。共用验收允许后镜头不传前镜头样例比较；原前镜头调用保持原有比较规则。

查看器新增场景任务适配和第二路视频控制：校验左右手场景/地图身份，裁取共同时间，对另一只手插值；三维轨迹和双视频共用任务时钟，各视频保留自身时间偏移。Python 读取侧仍只依赖标准库。

验证：184 项 Python 测试、15 项直接执行的 JavaScript 测试、命令入口/Shell 语法与文档链接检查。数值测试使用真实 GTSAM，覆盖已知 2.3 倍尺度、物理杆臂、三地图双手、非零同步偏移和失败发布；后镜头裁切使用真实 FFmpeg/PyAV，后镜头 C 的尺度错误会阻止测量发布。无界面 Chrome 验证双视频播放、跳转和 0.15 秒时间偏移。

真实录像 `VID_20261022_225713_00_005.insv` 在 `/tmp/umi-scene-probes/` 完成单相机建图、米制对齐、固定地图重复定位与融合；1041 帧均有地图位置约束，同一原片作为两个任务时坐标逐点一致。这验证固定坐标与复用，不是独立真值精度测试。尚无真实双机/后镜头新录制数据做完整物理回归；原始录像、设备档案和原有落盘轨迹保持不变。

## 2026-10-25 场景采集批次、设备版本与商超组织示例

- 目录按场景 → 采集批次 → 任务 → 完整设备序列号组织；标定原片独立保存在 `recordings/calibration/<serial>/<batch>/<front|rear>/`。
- `scene.yaml`、`episode.yaml` 使用 schema 2，新增 `session.yaml` 固定地图版本、地图清单 SHA256、相机标定和夹爪几何版本；未保留旧场景清单兼容读取。任务预检校验原片 SHA256 和 INSV 序列号。
- 全局设备索引 version 2；前后镜头标定和夹爪几何按版本保存。新默认值用于显式创建新配置，已有批次继续引用旧版本。设备登记和停用使用写入锁，新版本完整复制、校验后发布。
- 新增 `umi-scene session`；`init`、`episode` 自动读取设备身份、保留原名并复制原片到规范目录，同名不同内容拒绝覆盖。`run` 支持按场景、批次、任务批量执行；固定地图后更换当前地图不影响旧批次。
- `umi-calib install-rear`、`install-gripper` 登记独立资产版本；场景初始化可导入通过验收的后镜头文件。双镜头任务检查前地图与实际左右设备一致，防止换设备或换手后误套固定变换。
- 原子文件、写入锁与结果指针移至公共 `software/infrastructure/file_store.py`，设备档案不依赖场景模块。运行指纹包括公共存储与设备选择代码。
- 查看器默认扫描 `scenes/`，按“场景/批次/任务”区分同名任务；跳过历史归档，直接指定归档目录仍可查看。
- 已将现有 `recordings/M4CP/` 原片、标定求解产物和独立轨迹迁入 `supermarket_demo` 示例结构；5 段原始视频均核对设备序列号。58 个原有文件共 4,818,534,083 字节全部通过迁移前后 SHA256 核对，清单在 `scenes/supermarket_demo/migration.json`；原设备索引和旧独立轨迹保留在归档中。
- 商超仅为目录与流程模拟：视频仍为原实验室录像，单台真实相机，同一段录像用于建图和回放。不将旧独立世界坐标结果标为共享场景结果，不生成虚假的右手。真实商超、多任务泛化和双机精度仍需实拍验收。
- 验证：192 项 Python 测试、15 项 JavaScript 测试通过；新增跨场景交错采集、地图版本固定、默认标定/几何更新、同名原片冲突、序列号与哈希错误测试。迁移后的真实视频完成新路径建图和固定地图回放，1041 帧、约 34.7 秒；标定与独立轨迹入口在迁移路径下只读预检通过。
- 最终实录场景地图版本为 `f4991ea43a3a-65d192ef`，采集批次显式固定此版本；再次 build/run 均复用成功结果。查看器默认目录的 HTTP 检查通过：场景/批次/任务标签、1041 帧轨迹、固定地图以及视频 Range 请求（206）；临时服务器测试后已关闭。

## 2026-10-25 套装领用与离线采集流程

新增版本化套装、人员领用、归还与批次快照。套装允许左手、右手或双手，任务只能显式选择批次手的非空子集。相机跨套装别名也不能重复领用；已归还批次仍可离线处理。历史处理按固定测量读取已停用设备，不回退当前默认值。批次可先建立，首次实际处理固定地图版本。

按用户决定移除跨相机同步估计：双手假设同时开始、结束，输出偏移为零。删除 synchronization.py 和旧 sync 命令；保留各设备标定中的相机–IMU 时间差。任务 schema 3，批次 schema 2，场景 schema 2。

入库按套装身份和唯一时间/时长候选配对，重复内容不重复建任务，缺片与歧义保留待确认。拒绝用文件修改时间或虚构时长配对。录制检查统一 PASS/WARN/FAIL/UNKNOWN，初始运动只提醒，曝光信息缺失明确未知，完整规则入报告。准备与定位分阶段缓存，失败不覆盖已有成功入口；只读预检不落盘。进度写 stderr，JSON stdout 可直接解析。

新增任务状态、绑定当前成功结果的人工验收与导出。换原片、产生新结果或最近尝试失败会使验收失效；导出只接受当前有效且人工通过的结果，保留地图坐标身份。state.json 包含验收记录，应持久保留。

验证：215 项 Python 测试、15 项 JavaScript 测试通过，覆盖单手/双手套装、跨场景领用、漏片补齐、导入幂等、只读检查、缓存重试、失败保护、人工验收、导出和结构化输出。现有真实单相机任务沿用地图 f4991ea43a3a-65d192ef，成功重跑 1041 帧、约 34.7 秒；没有真实双机新采数据或独立真值精度结论。

历史样例登记为仅左手 demo_kit，未知人员保留 unknown，不补造历史领用；原批次/任务清单归档后显式迁移，历史录像、测量、地图与旧运行产物保留。

## 2026-10-25 简化批次登记

按用户当前范围移除操作员、领用和归还：删除 checkout/return 命令、--operator/--checkout 参数、领用记录与冲突检查；批次创建仅要求场景、名称和套装，可选使用手。Session、状态、运行元数据及导出清单不再生成 operator/checkout 字段。当前示例清单同步清理；历史归档和已发布运行保留原内容。套装版本固定、跨场景复用、单手完整性、验收和导出继续保留。

本轮验证：216 项 Python 测试通过；真实样例只读预检通过，未重新运行数值求解或改写历史成功结果。

## 2026-10-25 场景下的套件层次

采集组织收敛为场景 → 套件 → 批次 → 任务。套件配置由 registry/kits 移至 scenes/<场景>/kits/<套件>/，批次随套件组织；kit/kits 命令必须指定场景，session 只选择当前场景已有套件。测量仍按设备保存在全局 registry，地图仍在场景层共享。原片路径、任务目录校验、查看器标签与导出路径加入套件身份，避免多套件同名批次冲突。

模拟商超现有套件、批次和原片已显式迁移，原清单保存在 archive/scene_kit_hierarchy。原片 SHA256 核对一致，86 个既有测量、地图、缓存及运行文件的大小和修改时间核对一致；未改写历史结果内容。清单变化使原任务状态重新等待处理，历史成功结果仍可查看。

本轮验证：218 项 Python 测试、15 项 JavaScript 测试通过，覆盖跨场景同名套件隔离、同场景不同套件的同名批次、目录与套件身份约束、查看器和导出无冲突。迁移后真实任务使用已有准备与定位缓存完成 1041 帧融合，沿用地图 f4991ea43a3a-65d192ef。

## 2026-10-25 场景流程冗余清理

移除迁移后 6 个空目录，包括旧 scenes/supermarket_demo/sessions、registry/kits 和旧原片目录空层级。保留场景地图、设备测量、原片与历史归档。

删除场景 CLI 未使用的 ROOT、PreparedRecording 未使用的 data 字段、地图冻结未用的 scene 参数、定位缓存未用的 cores 参数，以及测试中的未使用导入。任务抽取已改为缓存准备/定位，删除不再生效的 run --keep-work、对应参数传递和始终不存在的 work 路径清理；build 与后镜头标定的 keep-work 保留。预检与实际处理共用地图/设备/输入准备代码，预检继续只读。失败收尾复用状态读取入口，移除经 workflow 间接导入 models 工具的绕行。修正已停用同步功能的旧提示和代码指南函数名，压缩商超示例中的重复迁移说明。

验证：218 项 Python 测试通过；software 与 tests 的 AST 扫描无未使用导入。真实单手任务完成 1041 帧回归，再次运行复用成功；查看器可读取完整轨迹并保持地图坐标身份。文档链接和差异空白检查通过。

## 2026-10-25 收拢场景管理入口

删除 CLI 对批次创建和任务登记的转发函数，直接调用业务入口；套件查询接收已读取的场景对象，避免为每个套件重新加载场景与算法参数；设备快照直接验证字段，移除构造单元素 devices 再校验的绕行。任务发现改为遍历前剪枝，跳过地图、缓存、运行与归档，并拒绝把无关文件解释为任务入口。

地图绑定在批次锁内读取当前清单、选择地图并按需写入一次，移除递归回读。手动登记和自动入库共用清单写入，保留补片前的操作描述，相同登记不再改写清单或清除验收。任务使用手限制到批次所声明的范围。查看器同时核对任务和批次清单身份，配置变化后显示待处理并隐藏失效验收，已有轨迹仍可查看。

验证：224 项 Python 测试、15 项 JavaScript 测试通过。新增并发首次绑定、任务发现剪枝、重复登记不改清单/验收、自动补片保留描述、批次使用手约束和查看器失效验收测试。真实任务完成 1041 帧回归与结果复用，实际 episode 命令重复登记不改变清单和状态文件，查看器读取与文档链接检查通过。

## 2026-10-25 场景命名与职责边界重构

按领域语义统一 SceneParameters、CalibratedCamera、VisualObservations，固定资产引用使用 device_bindings，原片引用使用 recording_refs，已发布地图使用 map_manifest；缓存目录、地图目录、运行目录、导出暂存目录采用明确角色名称。原始录像路径与镜头视频分开命名，绝对开始时间为 started_at_unix_s，操作描述为 instruction，CLI 对应 --instruction。当前示例清单的描述字段同步迁移，数据目录和测量文件保持位置不变。

collection/lifecycle/preflight/engine 分别整理为 capture_config/episodes/recording_checks/slam。新增 inputs 集中读取清单、选择固定资产和计算输入身份；models 从 350 行收敛到数据结构，验收模块直接使用 inputs，不再反向导入 workflow。原片准备移至 recordings，惯性初始零偏移至 fusion，slam 仅负责视觉进程边界。移除旧模块，无旧名称转发层。测试按套件配置、任务管理和录制检查整理。

对重构前后 19 个几何、融合、数据准备、SLAM 函数/结构做语法树对照，仅归一化本轮标识符与接口改名，全部等价；没有调整数值公式、阈值或取帧规格。

验证：225 项 Python 测试、15 项 JavaScript 测试通过；场景模块依赖图无环，无旧模块转发或未使用导入。真实单相机录像完成 1041 帧、约 34.7 秒回归，沿用固定地图 f4991ea43a3a-65d192ef；重复运行复用成功。实际 CLI 的建图/轨迹只读预检和重复任务登记均不改写现有清单、状态或地图指针；查看器读取、文档链接和差异空白检查通过。

## 2026-10-25 地图版本加入时间

新建地图版本使用 `YYYYMMDD_HHMMSS-输入指纹12位-随机8位`，时间取运行机器本地时区的建图开始时间。版本读取允许时间前缀中的下划线，保留路径字符检查。已有地图目录与批次绑定保持不变，入口注释和场景文档同步说明格式。

验证：228 项 Python 测试通过，新增时间格式、同秒强制构建区分、旧格式地图复用与固定版本、非法路径字符回归。实际 `umi-scene inspect scenes/supermarket_demo` 成功读取既有地图；未重建或改名现有地图。

## 2026-10-25 精简地图版本产物

新地图清单移除顶层重复的 definition，读取与生成统一使用 inputs.definition，旧清单无需改写。成功地图导出删除空 export.log，保留非空诊断和失败日志；原始展示 JSON 仍在成功转换后清理。补全场景地图导出路径缺失的 NumPy 导入，避免未被流程替代程序覆盖的实际导出报错。入口注释与场景文档同步更新。

验证：231 项 Python 测试通过，覆盖清单去重、已有清单读取与复用、空日志删除、非空日志和失败现场保留。实际读取已有 ORB 图集，在临时目录成功导出 2740 个地图点，无临时 JSON 或空日志残留；既有地图资产、地图指针和批次绑定的 22 个文件哈希均保持不变。已发布地图不修改，以保持历史批次固定清单及结果的可追溯性。

## 2026-10-25 批次引用套件固定版本，精简任务状态

批次清单升级到 schema 3，kit 仅保存 id/version；设备和测量版本统一从场景下不可变 kit.yaml 读取并校验，不回退 current。套件读取移入 inputs，移除 capture_config 的读取实现，无循环依赖或旧入口转发。任务清单升级到 schema 4，移除恒定 timing，程序继续统一按双手同时录制、零输出偏移处理；批次与任务各自的 arms 保留。

state.json 只保存 attempt/review，身份、使用手和完整性从清单读取；attempt.preflight_report 引用运行目录的报告，移除完整报告副本。检查报告在算法前写入，融合失败仍保留检查证据。查看器通过 ruamel.yaml 直接读取采集清单，无需算法环境，支持无状态快照的待处理、缺片任务，并在清单改变后清除当前验收显示。

当前商超示例的批次、任务和状态已迁移，原配置归档到 archive/kit_references；既有套件版本、地图、原片及历史运行保持内容不变。因清单和代码身份改变，任务重新抽取，不将旧验收迁为新结果的验收。

验证：235 项 Python 测试、15 项 JavaScript 测试通过，覆盖不可变套件引用、更新默认版本后历史批次不切换、缺失或篡改固定资产拒绝回退、禁止批次内嵌设备副本、单手/双手、报告引用、失败证据和人工验收导出。真实单手录像沿用 f4991ea43a3a-65d192ef 地图完成 1041 帧、约 34.7 秒抽取，重复运行复用成功；查看器读取正常，状态文件约 380 字节，地图和套件资产哈希保持不变。

## 2026-10-25 套件简化为当前配置与批次快照

取消套件自身的版本号、current.json、versions/ 和专用资产清单，登记套件直接原子写入唯一 kit.yaml。当前配置包含 ID、描述、各手序列号及实际标定和夹爪版本。创建批次校验实测资产并自动保存设备快照到 session.yaml（schema 4）；已有批次直接读取自己的快照，不依赖当前 kit.yaml，修改或移除当前配置不改变历史批次。相机标定、夹爪和地图的资产版本及完整性校验继续保留。

移除套件版本存储、查找、指针读取和路径转发函数，无旧格式兼容或入口转发。配置与快照共用 validate_kit，套件列表和 CLI 文案更新。商超示例迁移为 kit.yaml + sessions，旧套件版本目录和指针移至 archive/simple_kit_layout；原片、测量、地图和历史运行保持内容不变。用户文档、代码阅读说明与可执行入口注释同步更新，前一节描述的套件版本引用方案由本节取代。

验证：237 项 Python 测试、15 项 JavaScript 测试通过。覆盖单手/双手、场景隔离、当前配置更新或缺失时旧批次快照保持独立、新批次拒绝无效实测资产、套件目录身份一致、验收与导出。真实单手录像使用原地图完成 1041 帧、约 34.7 秒抽取，重复运行复用成功；查看器读取正常，地图资产哈希保持不变。

## 2026-10-26 删除冗余场景产物

按用户要求实际删除过期配置和迁移备份（collection_schema、scene_kit_hierarchy、kit_references、simple_kit_layout、migration）、场景根 migration.json、8 份被当前成功结果替代的运行、1 个无引用旧地图版本、2 份旧准备缓存、3 份旧定位缓存，以及有效定位缓存中的进程中间目录。上文的迁移归档路径描述保留当时的操作历史，不再代表当前落盘文件。

清理前核对当前结果、处理状态、批次固定地图、有效缓存键及完整资产哈希；持有地图与任务写锁后删除，共 162 个文件，释放 280,678,400 字节（约 268 MiB）。场景占用从约 426 MiB 降至 158 MiB，保留当前地图 f4991ea43a3a-65d192ef、当前运行 be88bc43879f4acc9e3e731c5b29f638 和各一份有效准备/定位缓存。原片、实测资产、当前配置与结果内容保持不变；独立坐标系旧轨迹仍保留在 archive/independent。

验证：受保护文件清理前后哈希一致；再次 run 复用成功结果，未重新建图或定位；查看器成功加载 1041 帧及固定场景身份，地图资产校验和文档差异检查通过。本轮未修改算法代码。

用户随后明确要求删除剩余旧独立实验结果，已删除 archive/ 全目录的 22 个文件，另释放 84,557,824 字节（约 81 MiB），场景目录剩余约 78 MiB。清理时当前配置、地图、结果和缓存共 41 个文件的内容哈希保持一致；再次 run 复用成功，查看器读取 1041 帧、地图校验和文档差异检查通过。示例 README 已移除归档目录说明。

## 2026-10-26 场景任务引用共享展示地图

任务写入不再解析和复制地图点数据，每只手的 summary.map 只保存场景展示地图的相对路径与 SHA256；移除 left/right/details/map.json。查看器直接读取共享版本，校验布局、本场景范围、版本、坐标系和内容哈希，支持单独打开原位置的任务结果；独立 umi-traj 结果继续使用自己的轻量地图与原有图集转换路径。文档说明移动数据时需保留共享地图和任务的相对结构。

按用户明确要求对现有地图做一次元数据迁移：删除重复 definition 和空 export.log，原子更新地图清单、当前指针与批次固定清单哈希。地图版本与 frame_id 保持原值，ORB 图集、展示点和固定变换的内容哈希不变。真实任务重新融合并生成精简结果，再删除被替代的旧运行，当前仍只有一份成功运行，无新增备份或归档产物。

验证：242 项 Python 测试、15 项 JavaScript 测试通过；新增共享地图读取、跨场景/软链接越界、哈希变化和坐标不匹配回归，双手共用一份地图并验证直接打开任务结果。额外的 18 项地图测试在最终输入类型校验调整后通过。真实结果 1041 帧，TCP 轨迹和视频与整理前逐字节一致；场景目录和单个结果目录均成功读取 2740 个地图点，重复 run 复用成功结果，差异空白检查通过。

## 2026-10-26 任务目录直接放在采集批次下

移除 episodes/ 容器层，任务现在位于 scenes/<场景>/kits/<套件>/sessions/<批次>/<任务>/。输入校验、手动登记、自动入库、任务发现和查看器直接使用这一结构，不保留旧路径兼容。任务名拒绝内部目录名称，发现和入库统一跳过内部目录，避免任务与批次入库报告或处理产物混淆。文档、示例和可执行入口注释同步改为直接路径。

现有 take_001 连同缓存、运行与结果指针整体移到批次下，修复任务清单中的批次和原片相对路径，删除空容器。真实任务重新生成正确的共享地图引用，再删除被替代的旧运行；未新建归档或数据副本。批次设备快照、原片、地图版本和坐标保持不变。

验证：244 项 Python 测试、15 项 JavaScript 测试通过，覆盖直接任务目录、内部名称冲突与发现剪枝、配对补片、批次地图绑定、单手/双手、验收导出和查看器。真实任务 1041 帧，轨迹和视频与移动前逐字节一致，地图文件哈希不变；场景和独立任务结果均成功读取 2740 个地图点。重复手动登记不改清单/状态的内容和修改时间，入库只读预览识别已有任务且不创建报告，重复 run 复用成功结果；当前只有一份成功运行。


## 2026-10-26 冗余与历史接口清理

- 将共用视觉执行设置抽成 `VisualSettings`；独立轨迹的 `ReconstructionSettings` 继承它。场景准备移除未使用的融合脚本配置与 `parameters.json` 写入，数值融合仍直接使用原参数。
- 视频和地图通过 `results.attempt_paths()` 共用当前与历史重建目录解析；非法轮次拒绝读取，避免静默回退到另一轮。旧标定包验证、旧结果迁移与旧地图读取保留。
- 删除未使用的 import 和测试中的失效环境变量；更新融合脚本配置说明、相机轨迹标定来源文案，以及查看器对场景/独立轨迹坐标的说明。
- 合并移动端重复 CSS，清除被覆盖声明及 grid 布局下无效的 flex 属性；保持最终样式。
- 验证包含 Python 全套回归、前端测试，以及七个视口宽度的 CSS 声明等价检查；未重新运行真实原片端到端精度评估。


## 2026-10-26 双手场景流程可读性

- `run_episode()` 保留预检、地图绑定、复用、锁、输入复核与发布；提取 `prepare_episode_recordings()` 管理录像准备和共同时间检查，`reconstruct_arm_in_scene()` 集中地图定位、固定坐标变换及单手 IMU 融合。两手仍分别求解，不添加联合优化。
- 任务报告、资产哈希与 `episode.json` 写入移至 `results.write_episode()`；成功指针与状态仍由 workflow 更新，输出字段、算法参数及坐标约定保持原样。
- 场景代码指南集中说明地图角色、地图到场景、IMU 到 TCP 的变换链，以及本机标定时差与跨相机偏移的区别。
- 回归新增双镜头角色和固定变换、本机时钟、第二只手失败及报告写入失败不发布部分结果的检查；真实原片与物理精度未重新评估。


## 2026-10-26 场景与独立轨迹的共用能力

- 新增 `software/camera/geometry.py`，统一外参矩阵构造与逐帧位姿右乘。场景与独立轨迹复用相机–IMU、相机–TCP 变换，保留各自原有的矩阵乘法顺序、四元数输出约定、尺度和坐标放置。公共模块由已有 camera 目录扫描纳入两条流程的代码指纹。
- 提取 `trajectory.runtime.visual_settings()`，统一显式镜头配置写入、遮手图缩放和视觉环境构造；独立轨迹另外生成融合配置。场景的后镜头实测要求不变，公共准备不自动回退样例。
- 两套数值融合及采集管理保持独立；验证覆盖非零杆臂与旋转的位姿链、公共代码指纹、实测前后镜头配置与工具解释器选择。未重跑真实原片端到端精度评估。


## 2026-10-26 简化新地图版本号

新建地图版本改为 `YYYYMMDD_HHMMSS-随机8位`，不再把输入指纹重复放入目录名。清单 `inputs` 的完整输入身份及复用判断不变，同秒强制重建使用不同随机后缀。旧版带指纹或无时间前缀的目录继续按原版本读取，不迁移或改写已有地图及批次绑定。验证覆盖新格式、同秒强制重建、输入未变复用、历史两种格式读取和固定批次版本。


## 2026-10-26 按批次完成接口、公共视觉与融合引擎整理

1. 共享前地图新增 `--mapping`；保留单个旧左右参数，拒绝混用、重复原片及未使用的配置。`migrate-legacy` 明确替代早期 uninstall 含义。内部样例常量不再使用 FACTORY 命名；UMI_PY/UMI_PY_TOOLS 为新解释器变量，旧安装通过 environment.py 和入口兼容。删除过期查看器待办及机器名注释。
2. 共用录像提取、镜头视频、视觉上下文、ORB 建图/定位/图集合并与运行环境迁至 `software/vision`，图像遮罩准备迁至 `camera/images.py`。场景直接使用 VisualContext，不再构造独立重建请求；公共视觉不导入 scene/trajectory。第三方构建布局由 vision/paths.py 集中管理，保留现有二进制目录。代码指纹覆盖新增公共模块。
3. 原 kiwi_fuse3s.py 拆分为 fusion_cli（解析）、fusion_data（显式输入/结果及 I/O）、fusion（局部状态求解）、fusion_factors（因子与自检）；旧脚本仅为兼容入口。求解不读写文件、无可变模块全局状态，可重复调用。短定位合并工具也改为显式 main。
4. 每轮独立重建写 run.json，各视觉阶段写 execution.json 和 tracking.json；结果生成优先使用结构化记录，历史文本仍可读取。超时或其他异常退出不再因存在残片而成功；仅兼容历史 Shutdown/析构 SIGSEGV 139/-11，且要求完整帧表、有效姿态与结束统计，再检查地图/定位产物。
5. 新增 upgrade-device --dry-run 与显式升级命令，生成新的 <测量ID>-v2 不可变版本，保留来源和旧版本，不自动改默认值。activate-calibration 单独启用默认版本，历史套件/批次绑定保持原样。样例借用设备不能通过升级变为实测设备。未对真实设备执行升级或切换。

验证：最终 Python 全套回归 262 项通过，查看器 Node 测试通过，解析 Jacobian 自检 PASS（最大误差约 5.99e-7）。共用模块依赖边界、无执行副作用的导入、连续求解状态隔离、带负号向量的旧入口、异常退出/残片、迁移只读预览、版本保留和显式启用均有覆盖。

真实验证使用既有 IAQEB26097M4CP 单相机原片 VID_20261022_225713_00_005.insv：公共视觉迁移后和完整融合拆分后各完成一次独立全流程，均为 1041 帧、约 34.7 秒、TCP 路程约 2.95 m、位置约束覆盖 100%。同一段真实视觉观测与 IMU 分别送入优化前脚本与新求解器，traj_tum.txt、state.csv、seg_placement.csv 逐字节一致，除耗时外的融合证据一致。另在 /tmp 中隔离复制已发布场景地图与清单，完成场景任务处理及共享地图查看验证。现有原片、已发布地图与注册表资产不迁移、不覆盖；回归产物在 /tmp，既有任务状态不变。以上是流程与一致性验证，没有新增真实双机数据，也没有独立物理精度结论。

## 2026-10-27 场景与独立轨迹架构模块化及代码规范化

1. **场景模块架构分层与子包化**：消除 `software/scene/` 原平铺的 14 个文件，顶层仅保留 `cli.py`（命令分发）、`workflow.py`（端到端编排）、`models.py`（强类型数据契约）。下设三大高内聚子包：
   - `management/`：负责场景套件与批次管理、原片配对入库、任务状态与输入指纹（`capture_config.py`, `episodes.py`, `ingest.py`, `inputs.py`）。
   - `spatial/`：负责视觉建图、多相机几何变换、前后地图放置与图优化融合（`mapping.py`, `geometry.py`, `fusion.py`, `slam.py`, `rear_calibration.py`）。
   - `pipeline/`：负责原片归档解码、质量完整性预检与轨迹产物发布（`recordings.py`, `recording_checks.py`, `results.py`）。
2. **独立轨迹模块架构分层与碎片整合**：
   - 将原 29 行的 `recording_data.py` 升格重构为更规范的 `pipeline/inputs.py`；原 45 行的 `plots.py` 归集进 `reporting/` 产物交付子包；
   - 划分为 `calibration/`（资产与设备选择）、`pipeline/`（多轮重建调度、复用与运行环境）、`reporting/`（结果布局、质量分析、元数据与展示地图）及底层的 `algorithms/`（图优化求解器内核）；
   - 根目录下保留极简的门面入口 `cli.py`、`workflow.py`、`models.py`；内部兼容垫片已在后续清理中删除。
3. **历史阶段的循环依赖解耦**：
   - 当时曾在 `software/scene/__init__.py` 与 `software/trajectory/__init__.py` 中使用门面导出与 `sys.modules` 别名映射；当前版本已删除这些内部兼容层，测试和生产代码均直接导入 canonical 模块；
   - 将 `selection.py` 与 `runtime.py` 对 `software.registry` 的依赖改为函数级延迟导入，彻底消除包初始化阶段的潜在循环依赖。
4. **代码可读性与 PEP 8 命名规范化**：
   - 解密了 `metadata.py`、`assets.py`、`fusion.py`、`workflow.py`、`dataset/` 等核心文件中的单字母（`p`, `s`, `c`, `d`, `ic`, `cad`, `cand`, `z`, `dur`）与压缩分号语句，代码符合 PEP 8 规范。
5. **验证**：全量执行 `./test.sh --all`，主单元测试 293 项全部通过（Ran in 28.6s OK），夹爪历史测试 5 项全部通过，查看器前端 JS 测试 15 项全部通过，`umi-calib check` 与 `umi-traj check` 自检全部通过。

## 2026-10-27 深度清理过度设计、过度防御与文件树优化

1. **彻底移除动态 `git_commit` 与脆弱外部依赖**：移除 `trajectory` 与 `calibration` 运行时中直接拉起 `git rev-parse HEAD` 子进程的逻辑；清理测试套件中的相关 mock。
2. **精简旧运行记录解析与超时收敛**：移除 `metadata.py` 中通过空格切分匹配旧版 `run.txt` 的过渡解析逻辑，收敛为读取规范结构化的 `run.json`；将 `--help` 与 `ffmpeg` 探测命令 60s/30s 超长超时收敛至 10s。
3. **文件树瘦身与平台边界清晰化**：
   - 清理未跟踪的本地 100MB+ 临时测试视频及缓存产物；
   - 彻底移除 `Dockerfile` 与 `.dockerignore`，明确原生 Linux / WSL2 主机计算边界，消除在非 x86_64 / 轻薄设备上通过容器运行密集数值求解的性能误导；
   - 更新 `install.sh` 与 `docs/安装.md`，为 Windows 用户提供标准的 WSL2 (Ubuntu 22.04) 指南。
4. **工作流冗余消除与实操效率提升**：
   - `umi-scene review` 增加多路径与采集批次支持，支持对 `sessions/<批次>` 一键批量验收所有成功任务，彻底省去逐个任务敲几十次命令；
   - `umi-scene ingest` / `episode` 增加 `--link` 硬链接支持，同一文件系统下原片秒级归档且零额外磁盘占用（20GB 视频从几分钟物理复制缩至 0.01 秒完成）；
   - `umi-scene build` 增加场景未初始化时的静默自动初始化支持，消除必须先手动敲 `init` 的繁琐；
   - 更新操作手册与场景指南，明确标定 `--install`、批次验收与硬链接的最高效用法。
5. **验证**：主单元测试增加至 296 项全部通过，夹爪历史测试 5 项全部通过，查看器前端 JS 测试 15 项全部通过。


## 2026-11-08 标定板资产与布局统一

- `hardware/calibration_targets/aprilgrid.json` 成为唯一布局定义，记录 tag36h11、2 格黑边、6×6 行列和 0.3 间距比例。物理尺寸只由本次 `--tag-size-mm` 输入；生成给 Basalt 的 JSON 仍只包含原有四个几何字段。
- A4/A3 PDF、PNG 和生成自检记录移到 `prints/`，打印图案内容不变；移除各纸张重复的 JSON/YAML 和无人读取的通用 YAML。
- `umi-calib check` 改为检查打印记录与统一布局的一致性、打印资产存在性；打印记录不再提示编辑无效的 YAML。标定板目录增加 README，操作说明与 CLI 帮助统一为实测外黑框边长。
- 验证：标定相关 98 项单元测试与 `./umi-calib check` 通过；A4/A3 PNG 均由实际检测器识别全部 36 个标签，A4/A3/实测尺寸生成的 Basalt 几何字段与原约定一致。未重跑实物录像求解。

## 2026-11-08 配置项收敛与标定板尺寸传递

- 标定 C 的 ORB 初始化尺寸从本次板几何派生，覆盖轨迹预设；前镜头 run、verify、样例比较、后镜头 Scene 和 recheck 共用有效配置，快照、检查点与子进程 YAML 均保存实际尺寸。
- 从 calibration.yaml 和 BoardParameters 移除 time_match_tolerance_ns、minimum_trajectory_spread_m、coverage_radius_px。前两项保留为原数值的数学保护常量，覆盖半径由原始镜头图像规格推导，实际分母写入覆盖指标。
- 场景对齐的三项门限覆盖集中为 alignment_parameters()，建图定义复用同一转换；明确共享场景与独立轨迹的参数边界，保留各模型有独立用途的噪声和验收门限。
- 旧自定义覆盖中若包含移除的三项，需删除；历史结果不自动改写，旧快照与当前配置不同时按已有规则拒绝 recheck，可通过 resume 更新计算。
- 验证：全部 Python 单元测试 443 项通过（90.756 s），`./umi-calib check` 与 `git diff --check` 通过。新增实测 A3 尺寸的前后镜头传递、候选/样例/verify/recheck、子进程 YAML 以及场景参数覆盖和建图指纹回归；未重跑实物录像求解。

## 2026-11-08 任务原片目录去掉 videos 层

- 新任务原片直接使用 `<任务>/left/`、`right/`，手动登记、自动导入和缺片补齐均采用相同路径；任务分组和设备身份继续分别取自任务目录和 INSV 序列号。
- `inputs.task_video_sources()` 与 `batch_task_directories()` 统一发现左右手原片目录，默认扫描只包含原片目录及 pending，跳过 results、.umi、建图和其他内部目录。
- 旧 `<任务>/videos/` 与已有清单继续原地读取，不自动搬动原片或改写路径；旧新布局可共同构成同一任务，多段同侧不同原片仍保留为待确认。
- 更新日常目录示例、开发说明、批次预检和缺原片提示；训练导出包中的 videos/ 与生成的 lens_videos/ 保持原用途。

- 验证：全量执行 447 项 Python 测试，唯一失败是端到端布局测试仍期望旧 videos 路径；同步该断言后，其所在的 6 项测试定向复跑全部通过。新增/更新覆盖新旧布局、扫描边界、单手、同名原片、补片、幂等、原片 inode 与只读预览；`./umi-scene check` 和 `git diff --check` 通过。

## 2026-11-09 场景配置与采集入口收敛

- 新 scene.yaml 默认 schema 3，可省略版本字段，仅管理模式、参考来源、描述和内联 trajectory/parameters 覆盖；不再保存设备、建图录像或引用额外参数文件。历史 schema 2 仍可读取，原片、测量和历史结果不自动迁移或改写。
- 移除 init、场景级 build --mapping/--mode 和 CLI 中重复的场景配置写入逻辑；用户命令统一为 batch、task、ingest --batch。batch 创建 batch.yaml，build 只接受批次；日常仍由 run 完成登记、建图与轨迹处理。
- create_batch、run、build、task、ingest 共用 batch_request、batch_record、refresh_record 解析当前配置及生成内部快照。只读预览从同一解析结果构造内存对象，不写快照；新批次不自动采用当前地图，复用必须明确写 map_version。
- 内联参数与 CLI 覆盖分别记录，保留参数严格校验、计算指纹与历史快照读取；前后地图参考选择从场景设置统一保存到批次记录。Session/Episode 及 manifest.yaml、episode.yaml、episode.json 作为内部记录协议继续读取，不再作为用户命令名称。
- 更新场景模板、配置说明、操作文档和相关测试；覆盖内联默认值/错误值、历史文件读取、命令迁移、只读预览和批次配置来源。

- 验证：456 项 Python 单元测试全部通过（91.743 s），`./umi-scene check`、`./umi-scene --help` 与 `git diff --check` 通过；文档本地链接检查通过。未运行真实原片的整批轨迹重建。

## 2026-11-09 现行文档核对

对照当前配置解析、批次处理、地图输入指纹和训练导出实现，修正现行说明中的 build --mapping 残留、套件更新与旧批次的关系、地图自动复用承诺、逐段引擎与共享场景流程混淆，以及“宽度/训练包尚未接入”和“实测 TCP”的不准确表述。区分当前指南与带日期的历史实验/临时证据，保留原始历史结论和归档内容。

验证：核对 11 份现行文档，35 条 umi-scene 和 19 条 umi-calib shell 示例均通过当前参数解析，112 个本地链接目标存在，git diff --check 通过。本次只改文档，未运行实际录像求解或重建。


## 2026-11-09 套件配置仅保留手写入口

- 移除 `umi-scene kit`、`kits` 及 `capture_config.register_kit()`、`list_kits()`；套件只通过手动创建和修改 `kits/<套件>/kit.yaml` 配置。保留读取、明确版本校验、批次引用和历史快照行为。
- 原套件命令附带的宽度资产登记迁移为 `umi-calib install-width SERIAL MEASUREMENT --gripper ID --gripper-version VERSION`；仅登记不可变测量并输出版本，由操作者手动填入对应手的 `width_calibration`，不修改套件。
- 同步日常操作、场景处理、设备、标定、宽度接入和套件说明；测试改为显式写入完整 kit.yaml，覆盖旧命令拒绝、单/双手、套件修改、历史快照和宽度登记不创建套件。

验证：457 项 Python 测试全部通过（93.702 s），`./umi-scene check`、两个 CLI 帮助和 `git diff --check` 通过；49 条文档命令通过参数解析，95 个本地链接目标存在。未运行真实原片重建，未修改实际套件或设备资产。

## 2026-11-09 批次配置仅保留手写入口

- 移除 `umi-scene batch` 及未被运行流程使用的 `capture_config.create_batch()`；用户手动创建批次目录和 batch.yaml，常规配置只需 kit。单手选择和历史地图复用分别通过 arms、map_version 配置。
- 日常流程统一为手写配置、放视频、run、查看和验收；run、task、ingest、build 继续共用配置解析和自动快照，不要求提前登记批次。保留资产校验、首次批次目录检查和历史记录读取。
- 测试通过临时 YAML 配置走实际解析流程；验证首次 run/task 无需提前登记、不改写用户配置，以及快照、原片、单/双手、地图复用和只读预览。同步 CLI 帮助、场景处理与开发说明。

验证：最终全量 457 项 Python 测试通过（91.139 s）；56 条文档命令通过参数解析，110 个本地链接存在，CLI 帮助与 git diff --check 通过。首次全量中一项测试夹具在导入清理时漏传 mock 目标，修复后该模块 6 项与最终全量均通过。未运行真实原片重建，未修改实际套件、批次或设备资产。

## 2026-11-09 移除独立原片导入和任务登记命令

- 移除 `umi-scene ingest`、`task` 的参数解析、分发、相关 CLI 辅助函数和导入摘要格式化；日常只按目录放好原片，再由 run 自动登记、配对、补片和处理。内部 ingest.py 及已有记录协议继续供自动流程使用。
- 人工配对通过同一 take 的 left/right 目录表达；单手批次通过 batch.yaml 的 arms 声明。操作指令、单任务使用手和同步继续编辑自动生成的任务 .umi/episode.yaml，不新增配置文件；修改后重新 run 和验收。
- 同步首页、文档导航、日常使用、场景处理、训练指南、开发说明和示例注释；移除旧 CLI 操作步骤，改为目录组织与自动发现。测试覆盖已删除命令拒绝、自动登记保持原片和配置、手填任务指令校验、当前套件的只读预览及设备不匹配报告。

验证：457 项 Python 测试全部通过（92.525 s）；自动登记/布局/处理的定向回归分别 13、6、16 项通过。50 条文档命令通过参数解析，111 个本地链接存在；CLI 帮助和 git diff --check 通过。未运行真实原片重建，未修改实际套件、批次或设备资产。

## 2026-11-09 场景登记与批次准备架构收敛

- 按 `.cursor/skills/deep-architecture-refactoring/SKILL.md` 梳理场景管理链路。删除无生产调用的 register_task、resolve_batch、capture_config.py、未使用的控制台实例，以及 episodes.read_state 纯转发层；状态读取直接使用 layout.read_state，清单修改的副作用明确命名为 invalidate_task_state。
- 新增 batches.py，集中批次声明读取、资产校验、建图引用和快照并发检查；BatchInputs 明确 root/declaration/previous_record/kit/arms/scene/mapping_paths，删除重复 parameters 字段和字符串下标式请求传递。移除旧失败记录临时回填分支及无需复制的批次内建图原片导入。
- processing.py 从 233 行收敛到 128 行，负责发现批次和调度。预览构造内存快照，实际运行保存快照，二者统一调用 ingest.register_recordings 规划原片；不再维护第二套预览过滤与配对逻辑。
- 自动登记只接收已解析的 Session；移除旧 CLI 的外部目录、使用手和硬链接参数以及 Path/Session 双类型适配。RawVideoInfo 改为 RecordingMetadata，extract_video_info 改为 inspect_recording；登记局部变量使用任务、原片、设备身份、报告等领域名称，补齐入口类型标注。
- 测试通过配置和记录夹具准备初始状态，实际补片通过自动登记验证，指令修改通过 YAML 输入与验收失效验证。test_capture_config.py 更名为 test_batch_configuration.py；新增预览/实际复用一致性和重复原片归属均拒绝的回归。已有资产及持久化记录的读取协议仍由 inputs/layout/migration 负责，不在本次登记链路重构中重写。

验证：原有 457 项全量回归通过（90.957 s）；新增两项后，处理模块 18 项定向回归通过（5.916 s），最终登记模块 13 项通过。当前测试共 459 项。`./umi-scene check`、50 条文档命令解析、111 个本地链接与 git diff --check 通过。未重建真实原片，未修改实际套件、批次或设备资产。

## 2026-11-09 现行文档按任务与参考职责重组

- 重排八份现行文档。安装与日常使用按环境、标定、套件、配置与录像、run、查看验收导出六步展开，安装参数单列；删除重复的采完之后流程和已删除命令的措辞。
- 场景处理作为高级配置与排障参考，集中场景/批次字段、配对补片、地图复用、批处理与故障检查，套件字段链接到 kits/README。设备与坐标集中目录、文件职责、身份版本、TCP 和历史资产迁移。
- 结果查看与验收明确查看、人工验收、普通导出、输出字段、播放与远程访问的阅读顺序。内部逐段引擎输出与日志移入既有 software/trajectory/README.md；训练指南按指令、同步、重跑验收、导出组织，宽度测量作为后置的可选参考。
- 算法原理首先说明日常共享场景、米制尺度与发布条件，再说明逐段引擎和来源。原有误差表、安装依赖和重跑证据原意保留，集中归档到 docs/dev/archive/轨迹引擎验证记录.md；现行算法文档概述适用边界并链接历史数据。
- 文档导航区分操作、字段原理、开发历史；开发指南明确每类内容的维护位置，避免按每次变更向多个用户页面追加重复说明。保留现有主要文件路径，更新迁移内容的相对链接与章节锚点。

验证：检查 16 份当前入口与参考文档，152 个本地链接及锚点、24 个跨文档入站锚点、48 条场景/标定命令参数解析和 8 个 YAML 示例均通过；git diff --check 通过。本次仅重组文档，未修改业务代码，未运行真实标定、重建或算法回归。

## 2026-11-09 宽度测量改为套件内直接引用

- width_calibration 改为相对 kit.yaml 的本地 JSON 文件路径，推荐 left_width.json / right_width.json。用户填写路径需位于套件目录内，解析后在批次记录中保存明确文件路径；run 直接校验和读取测量，不再走宽度资产登记。
- 删除 umi-calib install-width、devices.install_width、registry.register_width/width_path，移除宽度版本号与专用 registry/gripper_width 存储逻辑。相机标定、TCP 几何和后镜头继续使用原有资产管理。
- 宽度内容哈希参与任务和检测缓存身份，不进入建图几何身份；修改文件后重跑使相关结果与验收失效，地图仍可复用。宽度结果报告新增 calibration_source，保存完整源 JSON；已有 calibration 字段继续保存规范化测量，inputs 保存源文件哈希，历史运行内容保持原值。
- 同步套件、训练、设备、标定和模块文档。测试覆盖套件内相对路径、实际相机选择、缺文件与无效锚点、绝对路径/目录越界/符号链接越界拒绝、旧 CLI 移除、同名文件更新、地图复用，以及旧结果与原测量保留。未创建或启用示例数值的真实宽度测量。

验证：463 项 Python 测试全部通过（93.232 s）；宽度模块 15 项和当前配置 6 项定向回归通过。16 份文档中的 152 个本地链接及锚点、46 条命令示例与 git diff --check 通过；umi-calib 帮助已无 install-width。未运行真实原片重建，未改动现有 kit_001 的设备选择或真实测量资产。

## 2026-11-09 宽度原始检测缓存与报告收敛

- 新增 gripper_width/detection.py，以 DetectionSeries 保存原片 PTS 和标记中心像素距离。缓存身份仅包含原片哈希、检测器代码/OpenCV 和原片几何；测量、核验状态、闭合阈值、插值策略与同步变化不再触发重复解码和标记检测。检测参数保持既有双 ArUco 配置，毫米换算继续使用既有分段线性映射。
- pipeline.py 负责缓存后的毫米换算、三帧中位数、短缺口插值与输出。宽度报告只返回任务 episode.json，不再写 details/gripper_width.json；删除重复的 calibration_source，保留规范化实测快照、原测量文件哈希和检测代码身份。
- 插值间隔、闭合阈值、诊断开关统一进入现有 scene_parameters.yaml，支持 scene.yaml.parameters 内联覆盖。两个数值允许零，诊断字段严格校验布尔类型；宽度参数不进入建图身份。测量 JSON 若仍含 max_gap_s/closed_threshold_mm 会提示迁移，避免静默忽略旧的自定义值。
- 逐帧 gripper_width_frames.csv 默认关闭，开启诊断时附加原始像素距离；普通成果导出始终包含宽度 CSV，并按结果资产清单携带可选逐帧诊断，不再要求重复 JSON。训练采样继续使用规范化测量、有效性与来源，保持单独的训练时间重采样约束。
- 更新参数与训练指南、模块说明。新增/更新回归覆盖修改锚点/verified/阈值/间隔/同步均复用检测、原片或检测器变化使缓存失效、缓存篡改拒绝、可选诊断及导出、规范参数验证和宽度策略修改复用地图。

验证：468 项 Python 测试全部通过（93.098 s）；宽度模块 18 项、场景配置 5 项、当前配置 7 项定向测试通过。16 份文档的 153 个本地链接及锚点、46 条命令示例、9 个 YAML 示例和 1 个宽度 JSON 示例通过检查；git diff --check 通过。未重建真实原片，未修改真实测量文件或套件设备绑定。

## 2026-11-09 原片宽度检测叠加预览

- 新增 `umi-calib preview-width VIDEO.insv [--out PREVIEW.mp4]`，使用基础环境直接处理原片，无需 Basalt、相机资产登记或宽度测量 JSON。默认在输入旁生成 `<原片名>_width_preview.mp4`，拒绝覆盖已有输出，失败时清理临时视频。
- 宽度提取与预览共用 ArUco 检测器。预览逐帧标出标签框、ID、中心及连线，以原始前镜头 1920×1920 像素计算 ID 0/1 中心距离；缺失或重复目标 ID 时显示 N/A。输出缩放至 960×960 H.264 MP4，保留原片帧间时间，叠加原始时间戳。
- 更新标定和训练指南，说明已知物理开口与像素读数的配对方法；预览不自动生成毫米测量关系。新增真实编码/解码测试，覆盖可见叠加、缺失及重复标签、不规则帧时间、自定义输出、覆盖保护、错误清理和损坏原片提示。

验证：473 项 Python 测试全部通过（91.486 s），其中新增预览专项 5 项；实际 shell 入口已对合成 INSV 成功生成并解码 MP4，人工检查首帧叠加画面。16 份文档的 153 个本地链接及锚点、48 条命令示例通过检查，CLI 帮助、shell 语法与 git diff --check 通过。未使用真实相机录像验证，未改动实际测量或套件设备绑定。

## 2026-11-09 暗小指尖标签的同帧补检

- 检查用户提供的 `VID_20261109_134802_00_041.insv`：430 帧中原预览双标签覆盖 64 帧（14.9%），ID 0 唯一检出 73 帧、ID 1 检出 336 帧。抽查原始半幅，标签约 30×20 像素、对比度低且有透视压缩；比较放大、对比度、阈值及锐化处理，选用同帧锐化补检。
- MarkerDetector 保留原图检测，目标 ID 缺失时增加一次锐化图检测，自适应阈值常量改为 3；只补入原图缺失的目标，保留已有坐标和重复 ID 歧义。没有跨帧跟踪或预览插值。预览和宽度提取统一使用原片解码灰度，避免颜色转换差异导致读数不同；检测代码身份变化会使旧原始检测缓存自然失效。
- 实际 CLI 生成独立的 `VID_20261109_134802_00_041_width_preview_v2.mp4`，保留用户旧预览。双标签覆盖提高至 183/430 帧（42.6%），完整解码验证 430 帧、960×960、H.264 和递增时间戳，并抽查叠加框位于实际标签；仍有漏检，文档补充照明、尺寸、安装角度和重新核验锚点的要求。

验证：478 项全量 Python 测试通过（95.455 s），宽度及预览定向 28 项通过；新增暗小标签原始坐标、补检重复 ID 拒绝、不沿用前帧、原检测坐标保留、预览/提取测量一致性测试。16 份文档的 153 个本地链接及锚点、48 条命令示例和 git diff --check 通过。未覆盖原始录像或旧预览，未修改实际套件、锚点或设备资产；覆盖率提升不代表全部恢复帧均已有人工真值验证。

## 2026-11-09 候选区域局部放大补检

- 在现有原图和同帧锐化之后，利用两次检测返回的未解码四边形建立局部候选区域。合并嵌套候选、保留边距，依次尝试 2 倍、3 倍和锐化 3 倍图；单帧最多处理 96 个有界候选，不增加 ROI 配置或命令参数，也不依赖前帧位置。
- 补检仍使用原字典和纠错设置，只补入缺失目标。按像素中心关系还原裁剪偏移和缩放坐标，要求结果中心位于对应候选附近；重复发现同一物理标签时合并，拒绝给已有物理标签分配另一个 ID。分离的同 ID 标签继续作为歧义，不产生有效双标签距离。
- 预览与 run 共用改进检测器，检测代码哈希自动更新缓存身份。更新训练指南与模块说明。合成测试覆盖原图及锐化均失败后的恢复、亚像素坐标精度、边界裁剪、局部重复标签、错误重赋 ID、非目标标签和噪声输入。
- 对用户同一段 430 帧原片生成独立的 v3 预览，最终双标签检测覆盖为 212 帧（49.3%），比 v2 的 183 帧多 29 帧。完整解码验证 H.264、960×960、430 帧与递增 PTS，抽查六幅叠加画面；另外抽查八帧原片，其中四帧新旧均有效，原有中心和距离逐值相同，第 180 帧由缺失恢复为 281.61 原始像素。此次两次整片生成分别约 139 秒和 83 秒，机器同时运行验证任务，未作为严格性能基准；候选补检增加计算成本。

验证：483 项全量 Python 测试通过（110.546 s），宽度/预览定向 33 项通过；16 份文档的 153 个本地链接及锚点、48 条命令示例和 git diff --check 通过。保留原片及原始、v2 预览，未改动真实套件或测量锚点；未对全部恢复帧标注人工真值，检测覆盖率仍不足以承诺每帧可测。

## 2026-11-09 标签活动区域自动收敛与检测提速

- 用户确认标签在相机中的活动范围较小。性能抽查八帧触发 636 次 ArUco 调用，主要开销来自全幅检测及无关候选补检；线程数与拼图批处理实验未采用为正式策略，程序不修改进程级 OpenCV 线程设置。
- MarkerDetector 首先全图定位，在双标签唯一解码后建立有余量的活动区域；逐帧在区域内重新解码，成功观测只扩展区域。每 30 帧全图重定位，尺寸变化重置区域。前帧只提供搜索范围，不提供本帧坐标或距离；移出区域时最多等待 29 帧重新定位，期间保留缺失。区域内重复 ID 继续拒绝，区域外重复标签由定期全图搜索发现。
- 原有同帧锐化、候选区域放大和原始像素坐标继续使用。裁剪坐标统一加活动区域偏移；run 和预览共用状态与检测代码，原始检测缓存身份随代码哈希更新。同步模块及训练指南中的搜索范围和重定位延迟说明。
- 采用同一批已解码的前 100 帧顺序测量：v3 全图检测 21.375 秒、自动区域检测 2.673 秒，二者均有 12 帧双标签。完整生成 v4 预览用时 25.018 秒，430 帧中双标签 212 帧（49.3%），与 v3 覆盖数相同。解码检查 H.264、960×960、430 帧及递增 PTS，并抽查六幅叠加画面。性能数字仅代表本机及该段原片。

验证：487 项全量 Python 测试通过（95.140 s），宽度/预览定向 37 项通过。新增活动区域搜索面积缩小和坐标还原、移出区域后的周期重定位、区域内重复 ID 拒绝、输入尺寸变化重置测试；保留缺失不沿用前帧与预览/提取一致性验证。16 份文档的 153 个本地链接及锚点、48 条命令示例和 git diff --check 通过。未覆盖原片和先前预览，未改动实际套件或宽度锚点。

## 2026-11-09 固定 ROI 与无界面 30 Hz 检测

- preview-width 增加 `--roi LEFT TOP RIGHT BOTTOM`，使用原始 1920×1920 前镜头半幅坐标固定搜索范围。固定范围不自动扩展或全图重定位，检测后恢复裁剪偏移；更新坐标与越界校验、区域内唯一性和预览测试。
- 新增 `umi-calib detect-width VIDEO.insv --roi ... [--hz 30] [--out FILE.csv] [--benchmark]`，基础环境运行，不启动浏览器或编码视频。默认依据原片 PTS 按原速采样输出，落后时跳过过期采样；测速模式取消等待和丢样。CSV 保存标签中心、像素距离、时间及裁剪/转换/检测耗时，缺失或重复 ID 留空；标准错误保存进度和统计。已有 CSV 拒绝覆盖，失败清理临时文件。
- monitor 对 8 位 YUV420 先按色度边界裁剪，再转换 ROI 灰度，保留颜色空间与量程，支持奇数坐标；其他像素格式使用标准灰度转换。实测本段 ROI 灰度与全幅转换后裁剪逐像素相同。解码使用 4 线程，检测期间 OpenCV 使用单线程并在结束后恢复；快速路径保留原图和同帧锐化，跳过候选放大，按原始半幅的最小像素周长限制过滤噪声，避免 ROI 归一化阈值过低带来的额外开销。
- 同一段原片、ROI 600/980/1350/1250、目标 30 Hz：测速处理全部 430 个源帧、215 次采样，用时 2.789 秒，包含解码和输出的吞吐为 77.09 Hz，裁剪/转换/检测中位数 9.919 ms、P95 10.819 ms。原速模式用时 7.247 秒，平均 29.67 次/秒，215 次采样、无过期跳样；两种模式均有 92 次唯一双标签测量。原速测量同时运行了定向测试，性能仅代表本机和该段原片，不代表所有硬件的时限保证。
- 保留 MP4 预览功能，未保留浏览器播放相关新增代码或修改 viewer。更新训练与模块文档，说明固定坐标、快速路径的覆盖差异和测速方法；未创建实测锚点或修改套件。

验证：499 项全量 Python 测试通过（101.943 s），检测/预览/标定入口定向 39 项通过。新增时间采样、原速节拍、落后丢样、原始坐标与缺失、奇数 YUV 裁剪和 RGB 回退、CSV 原子输出及线程恢复测试；launcher 验证 detect-width 不安装 Basalt。16 份文档的 153 个本地链接及锚点、52 条命令示例、CLI 帮助、shell 语法和 git diff --check 通过。真实 CLI 在 /tmp 生成测速与原速 CSV，未覆盖原片、既有预览、设备资产或真实测量文件。

## 2026-11-09 快速检测同步保存标签与 ROI 叠加视频

- detect-width 的 CLI 默认在原片旁保存 `<原名>_width_detect.mp4`；新增互斥的 --video-out / --no-video，CSV 的 --out 保持原义。Python monitor.run 保留可选 video_output，供仅需要 CSV 的调用者使用。
- PreviewWriter 集中前镜头缩小、标签框和 ID、中心连线、原始像素距离、ROI 框及 H.264 编码，供逐帧预览和快速采样检测共用。监测只对已检测采样帧编码，与 CSV 一一对应；原片 PTS 间隔保持不变，丢样不缩短播放时间，缺失或重复 ID 仍显示 N/A。显示分辨率 960×960，距离按原始 1920×1920 半幅计算。
- 图像转换限制线程并在转换时直接缩小，快速监测使用 veryfast 编码预设和两个编码线程。CSV 与 MP4 在计算完成后发布，编码中断不发布半成品，已有输出拒绝覆盖；更新命令帮助和训练指南。
- 实际运行用户原命令：430 个源帧产生 215 次检测和 215 帧视频，92 次唯一双标签，无过期跳样；原速模式用时 7.298 秒，整体平均 29.46 次/秒，检测 P95 11.258 ms。含编码测速用时 5.798 秒，整体吞吐 37.08 Hz。完整解码检查 H.264、960×960、215 帧及递增时间戳，末帧时间 7.140467 秒，并人工核对距离、标签框与蓝色 ROI 框。性能数字仅代表本机与该原片。

验证：501 项全量 Python 测试通过（98.179 s），检测/预览/标记定向 33 项通过；覆盖默认 MP4、ROI 叠加像素与文本、CSV/视频帧对应、缺失/重复 ID、丢样后保留 PTS、关闭视频、既有视频保护和编码失败清理 CSV/MP4。16 份文档的 153 个本地链接及锚点、54 条命令示例、CLI 帮助、shell 语法及 git diff --check 通过。已生成用户原片旁的 width_detect MP4；未覆盖原片、width.csv、既有预览或设备资产。

## 2026-11-09 暗光 ROI 对比度增强

- 对用户 VID_20261109_155248_00_042.insv 的固定范围 500/980/1550/1250 比较原图、线性增益、Gamma 和 CLAHE；该段抽样中线性 2 倍增益比其他候选更有效。新增 preview-width / detect-width 的 --contrast GAIN，默认 1 保持未增强流程；建议从 2 尝试，不改动原始录像、套件或场景参数。
- 在当前检测 ROI 内预先进行线性灰度增益及饱和裁剪，再执行既有检测和同帧锐化，未增加检测轮次。快速增强模式以 24 原始像素作为最小标记周长，减少增强噪声的候选开销；未增强时门限不变。显示 MP4 对同一 ROI 增亮并标注 contrast 倍率，几何坐标仍为原始半幅像素。未识别及歧义继续输出缺失，增强可改变检测结果。
- 为给增强检测预留时间，监测在增强模式使用 ultrafast 编码预设，未增强模式继续使用 veryfast，编码仍为两个线程；增强视频文件可能更大。原片和用户旧检测视频均保留，实验中间成片移至 /tmp，最终增强版另存 contrast2_30hz.mp4。
- 相同原片和 212 个 30 Hz 采样，未增强双标签 5 次（2.36%），2 倍增强 15 次（7.08%）。最终含编码测速用时 6.016 秒，35.24 Hz；原速用时 7.158 秒，29.62 次/秒，无过期跳样，检测 P95 21.710 ms。最终视频 H.264、960×960、212 帧，与旧视频逐帧 PTS 一致；人工抽查恢复帧的标签框及 ROI 亮度。数据仍严重欠曝，软件增益未使大部分帧恢复，不以检测覆盖提升代替人工真值验证。

验证：506 项全量 Python 测试通过（98.308 s），相关定向 38 项通过；新增欠曝标签恢复与原始坐标、歧义/噪声缺失、非法增益拒绝、仅增强显示 ROI 且不修改源图、CLI 传递至视频及摘要的测试。16 份文档的 153 个本地链接及锚点、55 条命令示例和 git diff --check 通过。实际命令生成增强 MP4/CSV，未改动用户原片或旧检测成果。

## 2026-11-09 AprilTag 原生 ArUco 检测对照

- 核对 AprilRobotics/apriltag 当前官方实现：已包含 tagAruco4x4_50 家族及 Python 绑定，可读取现有 OpenCV DICT_4X4_50 标签。将源码 b7c0ebe9aa20f82ec7a828579004f9e706bfecd9 临时下载至 /tmp/umi-apriltag-aruco-eval，Release 编译 C 库及 Python 绑定，通过临时 PYTHONPATH 运行；未安装进项目环境或更改默认后端。
- 先以 OpenCV 生成的 ID 0、1 验证兼容性，两个 ID 的四种 90 度旋转均能零纠错解码。再对 042 原片相同 ROI 的 212 个 30 Hz 灰度采样进行对照；统一保留唯一 ID 0/1，距离中心采用四角均值，避免原生投影中心和既有度量定义混用。
- OpenCV 未增强为 5 次双标签，2 倍增强为 15 次；显式将 errorCorrectionRate 设为 0 后仍为 15 次。AprilTag 使用 tagAruco4x4_50、threads=1、decimate=1、maxhamming=0，未增强及 2 倍增强均为 0 次双标签。另试增强后锐化仍为 0 次，2 倍放大为 6 次，放大并锐化为 5 次，CLAHE 为 2 次。单纯允许 1 位纠错但不放大也未检出双标签。统计为检出数量，不代表已逐帧人工标注的准确率。
- 该对照只说明当前暗光小标签录像在上述参数下未见 AprilTag 优势，不能推及其他标签尺寸、光照或参数。实验数据保存在 /tmp/umi-width-apriltag-compare.json 及相关日志，当前生产路径继续使用 OpenCV。官方能力依据见 [AprilTag README](https://github.com/AprilRobotics/apriltag#choosing-a-tag-family) 与 [Python 家族注册源码](https://github.com/AprilRobotics/apriltag/blob/b7c0ebe9aa20f82ec7a828579004f9e706bfecd9/apriltag_pywrap.c)。

## 2026-11-09 小标签快速阈值与解码采样优化

- 对去噪假设进行抽样与整片验证。双边、中值和较强高斯会损伤小格子；轻度高斯 sigma=0.5 在暗片整段由 15 次降为 14 次，较亮对照片由 94 次降为 77 次，因此未接入去噪选项。
- 快速检测改用 3～23 像素、步长 4 的自适应阈值窗口，以及每格 8×8、边缘忽略比例 0.25 的规范解码图。阈值轮数保持六次，不追加源图放大、平滑或额外检测，不增加纠错位数。仅 detect-width 的 fast 模式采用新设置，完整预览与场景宽度检测设置保持原值，不新增用户参数。
- 固定 ROI 500/980/1550/1250、相同 30 Hz 采样对比：042 暗片未增强从 5/212 提升至 12/212，2 倍对比度从 15/212 提升至 36/212；041 较亮片未增强从 92/215 提升至 165/215，2 倍对比度从 94/215 提升至 170/215。两段原片共同检出帧的距离差 P95 分别约 1.30、1.04 原始像素，不将检出增加视为绝对测量精度提升；锚点应使用一致的处理方式复核。
- 暗片实际 CLI 含编码测速 6.053 秒、35.03 Hz；原速用时 7.157 秒，212 次采样、36 次双标签、无落后丢样，检测 P95 24.414 ms。另存 optimized MP4，完整解码确认 H.264、960×960、212 帧与 CSV 时间逐条一致。新增检出中心集中在两指标签活动区，并抽查 12 幅成片定位；未进行全量人工真值标注，暗片有效比例仍只有约 17%。

验证：507 项全量 Python 测试通过（96.631 s），标记/检测/预览/场景宽度定向 57 项通过；新增含噪欠曝小标签四种旋转的 ID 与原始中心坐标回归。16 份文档的 153 个本地链接及锚点、55 条命令示例和 git diff --check 通过。实验数据记录于 /tmp/umi-width-denoise-full.json、/tmp/umi-width-window-full.json，保留原片、旧视频、测量锚点和设备资产。

## 2026-11-09 暗标签补检阈值与噪声开销优化

- 对低对比度解码门限、格子采样密度、格子边缘忽略比例和二值化阈值进行抽样及两段整片对照。降低 minOtsuStdDev、增加规范格子采样没有稳定收益，未接入新参数。快速路径最终保留每格 8×8、忽略比例 0.25，将锐化补检的 adaptiveThreshConstant 从 3 改为 2，窗口范围从 3～23 收紧至 7～23、步长 4。跳过最小窗口减少噪声候选，不追加检测轮次，不增加纠错位数；完整预览和场景提取的阈值保持原值。
- 固定 ROI 500/980/1550/1250、相同 30 Hz 采样：042 暗片未增强从 12/212 变为 30/212，2 倍增强从 36/212 变为 51/212；041 较亮片未增强从 165/215 变为 167/215，2 倍增强从 170/215 变为 173/215。暗片增强模式新增 16 次、丢失原有 1 次；共同检出帧距离差 P95 约 0.55 原始像素，较亮片约 0.46 像素。检出总数不代表绝对测量精度，仍需一致处理方式核验锚点。
- 两种模式的锐化补检均增加同一物理标签检查，避免给已有标签追加另一个 ID 后产生虚假的双标签距离。新增暗片 16 次检出的局部定位框均经人工抽查；136 个合成噪声及无关 ID 负样本未检出目标 ID，不能据此保证所有实际画面的误检率。另补充更强噪声、四种旋转下的标签中心回归，以及同帧重编码歧义拒绝测试。
- 实际 CLI 含 MP4 编码测速用时 6.433 秒、32.96 Hz；原速用时 7.162 秒，212 次采样、51 次双标签、无落后丢样，检测 P95 25.123 ms。另存 lowlight MP4，完整解码确认 H.264、960×960、212 帧，时间与 CSV 逐条一致；旧视频和原片均保留。暗片有效检出仍约 24%，不能代替补光或清晰成像。

实验记录：/tmp/umi-width-decoding-coarse.json、/tmp/umi-width-decoding-full.json、/tmp/umi-width-decoding-trim.json、/tmp/umi-width-decoding-gain1.json、/tmp/umi-width-threshold2-synthetic.json；未修改实际设备资产或测量锚点。

验证：510 项全量 Python 测试通过（97.220 s），包括新增欠曝强噪声标签四种旋转的原始中心坐标、无关编码与暗噪声拒绝、同一物理标签不能补为另一个 ID 的回归。16 份文档的 153 个本地链接及锚点、55 条命令示例和 git diff --check 通过。

## 2026-11-09 快速补检局部对比度范围调整

- 检查真实漏检画面中的 rejectedCandidates：部分实际标签已有候选四边形但未通过编码识别，也有候选偏到外侧白边。OpenCV 的候选提取与编码识别确为独立阶段，参见 [官方检测流程](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)。抽样放宽多边形近似阈值未带来收益；contrast=6 使较亮片抽样双标签降至 0/54，不采用统一提高增益。
- 比较补检锐化 sigma=0.6、0.8、1、1.4、2。快速路径最终将差分锐化中的 GaussianBlur sigma 从 1 调为 2，保留 `2*gray-blurred` 的权重；主检测、阈值、纠错设置和检测轮次不变。完整预览与场景检测继续使用 sigma=1，未增加 CLI 配置。
- 同一 ROI 500/980/1550/1250、30 Hz 采样、contrast=2，042 暗片双标签由 51/212 增至 65/212（新增 18 次，丢失 4 次）；041 较亮片由 173/215 增至 196/215（新增 25 次，丢失 2 次）。未增强时分别由 30/212 增至 40/212、167/215 增至 192/215。共有检出帧距离变化 P95 分别约 1.01、0.28 原始像素；检出覆盖不等同于绝对测量精度。
- 人工检查两段录像新增检出帧的局部标签框，位置均落在实际标签上；204 个合成噪声及无关编码负样本未检出目标 ID，36 组弱光含噪正样本对照无原有双标签丢失。该验证没有全量人工坐标真值，不能作为普遍误检率保证。暗片仍只有约 31% 的采样能够得到双标签距离。
- 实际 CLI 含编码吞吐 32.01 Hz（6.622 秒）；原速 7.161 秒、212 次采样、65 次双标签、无落后丢样，检测 P95 24.800 ms。另存 sharp2 MP4；完整解码确认 H.264、960×960、212 帧，与 CSV 时间逐条一致。保留原片、旧视频和实际测量资产。

验证：标记、无界面检测、预览和场景宽度相关 60 项测试通过（12.615 秒）；扩展欠曝强噪声测试至无需增益的暗标签及四种旋转后，23 项标记测试通过（1.521 秒）。实验保存在 /tmp/umi-width-success-coarse.json、/tmp/umi-width-success-full.json、/tmp/umi-width-sharp2-check.log，候选诊断及新增定位抽查图片也保存在 /tmp。

## 2026-11-09 夹爪安装配置归入 kit

- registry 收敛为相机档案与标定版本，移除夹爪注册、默认夹爪、夹爪版本选择及 install-gripper 入口。kit.yaml 每手明确填写 serial、calibration 和 gripper；gripper.tcp 必填，width 与 mask 可选，文件路径相对套件目录。
- kit_001 两台相机继续固定 002；将原 handheld_v1/003 的几何原样复制到 kits/kit_001/tcp_camera.json。未添加不存在的实测宽度关系。原 001、002、003 及清单、遮手图、旧指针完整迁入 hardware/grippers/archive，仅供历史追溯，运行不读取旧指针。
- 场景入口解析 TCP 与宽度测量，复用 registry 已解析的相机测量；计算前将实际资产复制到运行目录 devices/<序列号>/。后续换算直接使用已解析的数据和副本，发布时不重新选择或校验整套配置。小型资产哈希按选择缓存，逐帧检测流程不增加配置读取。宽度文件及处理策略变化不触发地图重建。
- 相机标定验收解除 TCP 依赖；内部逐段引擎默认输出相机/IMU 轨迹，显式 --tcp 才输出 TCP。相应报告、产物复用和测试同步调整。日常文档与配置示例统一使用 kit 文件引用。

验证：339 项相关 Python 回归通过（83.264 s），覆盖相机独立验收、套件解析、源 TCP 在计算中变化时的快照隔离、宽度与地图复用、后镜头标定和训练导出。实际 kit_001 读取与临时快照检查通过，两台相机保持 002、TCP 与原 003 字节一致；umi-calib devices 和帮助正常。16 份文档的 154 个链接及锚点、55 条命令示例和 git diff --check 通过。本次未重跑真实视频建图或标定求解。

## 2026-11-09 场景位姿与夹爪宽度合并输出

- 每手主结果统一为 trajectory.csv，同一行保存原片/任务时间、TCP 位置（米）、四元数和夹爪开口（毫米）、闭合量、有效性及来源。新场景运行不再生成独立 traj_tcp.txt 和 gripper_width.csv；未配置或无有效宽度时数值列留空。TCP 定义继续保存在 episode.json。
- 宽度流水线返回 WidthResult，结果写入器直接合写位姿和宽度，不读回中间宽度文件。训练读取一次统一表，由 ArmSamples 传递宽度序列，共用解析后的时间轴；旧分离结果只在读取边界保留支持。查看器和普通成果导出同步选择统一表，独立标定轨迹引擎继续使用 TUM。
- 相机/IMU 辅助轨迹及逐帧诊断仍在 details/；现有历史结果不改写。重跑批次后使用新布局。同步更新目录、时间、单位、宽度缺失值及有效性的文档说明。

验证：131 项相关回归通过（54.646 s）；整理一次解析/校验的读取路径后，35 项宽度、训练导出与查看器测试通过（11.037 s）。覆盖 TCP 数值不变、任务时间偏移、宽度缺失与长缺口、旧分离格式读取、导出不重复读取宽度、查看器以及失败不替换上次成功结果。文档 154 个本地链接及锚点、55 条命令示例与 git diff --check 通过；未重跑真实视频建图。

## 2026-11-10 实测/工厂内参来源与说明同步

- 工作镜头 `run`、环境镜头 `calibrate-rear` 和场景 `build/run` 支持 `--intrinsics-source measured|factory`。工厂模式从 B 原片拟合固定 KB4，保留 A 独立图像验证，重新求 B，再验收 C/D；两来源分别保存完整标定。历史 `UMI_TRAJ_CALIB=factory` 仍表示实验室样例。
- 套件以 `calibrations.<来源>.{front,rear}` 绑定版本；选择优先级为 CLI、批次、套件顶层、每手配置，多来源配置均未指定时选 measured。旧单版本配置不自动换版本。来源、参数和引擎都参与地图复用身份，切换保留历史成果及批次快照。
- 工厂 A 留出 RMS 超标只提醒，缺有效证据不能安装；C 的 30 mm / 10% 失败门槛不变。构建默认启用 ≤85° 前向 KB4，工厂入口检查真实二进制能力，1920 全镜头仍仅诊断。数值、真实工作镜头结果及环境镜头待验证范围见[标定架构与验证第 14 节](标定架构与验证.md#14-实测出厂内参来源开关2026-11-10)。此前功能回归共 542 项通过；M4CP 正式工厂流程经 recheck 为 PASS（有提醒），结果未安装，实际套件仍固定原版本。
- 同步根 README、日常指南、标定/场景/参数参考、模块指南、批次模板、入口注释和命令帮助。明确区分自动光学诊断与正式候选，修正“只支持实测”“环境镜头仅数值拟合”等旧描述；历史实验保留原数据并标明当时的默认几何，重构计划也区分后续标定进展与尚未实施的标签融合。

本次说明核对：14 份 Markdown 的 196 个本地链接及锚点通过；来源选择 13 项回归通过；五个 CLI 帮助入口正常，Shell 语法与 Python 编译检查通过。此次未重跑实拍标定或建图，也未修改实际套件、设备标定及验收阈值。

## 2026-11-10 固定地图任务的反向定位补全

- `test_shelf/20261109_001/take_001` 原先正向定位覆盖 270/380 帧，开头连续 3.4368 秒无有效地图位姿。相同地图、参数和逐像素无损画面的倒序对照得到 114 帧有效位姿，恢复了全部开头 103 帧，证明这一样例存在初次重定位与连续跟踪能力的差异。
- 引擎新增仅用于固定地图定位的 `--reverse`，使用约 64 MiB 的前向解码帧块反向送入纯视觉跟踪，不转码、不反向处理 IMU。引擎内部时间递增，CSV 保存原始帧号和视频时间；Python 恢复原始排序与相机–IMU 时差。真实 380 帧的状态和位姿与无损倒序视频对照逐项一致。
- 场景任务正向缺帧时运行反向定位；`relocalization.py` 按反向连续跟踪段检查同图重叠位姿，默认至少 3 帧、全部重叠帧位置差不超过 3 cm、旋转差不超过 3°。只补缺帧，保持正向位姿；拒绝无重叠、证据不足和不一致的段。原有建图尺度路径及覆盖率/最长中断门限保持原值。
- 补全参数、地图尺度、算法及引擎身份进入定位缓存；正反向证据和观测一起校验哈希。失败现场保留逐镜头诊断，成功结果汇入 `episode.json` 的 `arm_results.<手>.localization`，`report.txt` 显示补入帧数，保留原有结果目录格式。清理本次批次的空 `map_version` 字段。
- 复用已发布地图 `20261110_184901-8fbf9e88` 完成真实任务：补入 110 帧，最终 380/380 帧、覆盖 100%、最长无视觉 0 秒，有效时长约 12.65 秒。接受段有 3 个共同观测，最大位置差 2.65 mm、旋转差 0.35°；融合视觉一致性 RMS 15.98 mm。这些是内部一致性指标，未获得独立物理真值。最终与初次成功验证的 trajectory.csv 逐字节一致，结果资产哈希、时间单调性和位姿有限性通过检查。

验证：在含标定依赖的 `.deps/calib_env` 中全量 553 项测试通过（106.998 s），包括新增 11 项反向补全与 CSV 顺序检查；覆盖断段、重叠不足、单帧冲突、米制尺度、旋转差、时间错位、正向完整/关闭时跳过补全、原位姿不变和成功结果布局。本机引擎已重编译，真实 CLI 完整流程成功。未重建或覆盖上述地图及相机标定。

## 2026-11-10 场景宽度检测与反向定位加速

- 场景宽度提取先运行既有快速检测，目标缺失时在同帧、同搜索区域调用完整补检；唯一双标签直接使用。保留快速阶段发现的重复 ID；跨阶段的同 ID 位置冲突、同一物理标签不同 ID 均拒绝测量。仍读取原片完整工作半幅的全部帧，保留原始 PTS。完整预览和快速监测入口保持各自模式，检测代码哈希使新策略使用独立缓存。
- `umi-scene run/build` 在任务线程创建前显式将 OpenCV 线程数设为 4，可用正整数环境变量 `UMI_OPENCV_THREADS` 覆盖。实测本机 OpenCV 不采用尝试的原生线程环境变量，因此不依赖它。4/64 线程对同一原片的整段级联距离与缺失掩码逐元素一致。
- 固定地图反向补全优先处理全部缺口及两侧各 1 秒上下文，使用原有 C++ 帧范围接口；范围达全片 75% 时直接运行整段，短范围未完全补齐时回退整段。两次尝试都只用原始正向观测验收，保持重叠数量、位置差和旋转差门限。`frame_range` 按绝对帧号核验引擎 CSV 和结束统计，诊断记录尝试范围、处理帧数与回退状态；`localization_reverse_context_s` 进入策略缓存身份。
- 货架任务 `019.insv` 的 760 帧宽度检测：原流程约 364 秒，4 线程级联用时 65.900 秒（约 5.5 倍吞吐），95 帧进入完整补检。双标签从 370 帧增至 682 帧，原有 370 帧全部保留；共有距离差中位数 0.350、P95 1.142、最大 1.853 原始像素。抽查均匀选取的 8 个新增检出帧及最大距离变化帧，框均落在实际标签上；没有独立坐标真值，检出增加不等于精度保证。
- 同一已发布地图 `20261110_184901-8fbf9e88` 上，反向从 380 帧缩到 140 帧，未触发整段回退；仍保留正向 270 帧、补入 110 帧，最终覆盖 380/380，最长无视觉 0 秒。反向接受段的最大重叠位置差 1.287 mm、旋转差 0.196°；最终 CSV 与此前成功结果的位置差最大 0.001 mm，视觉一致性 RMS 约 15.980 mm。
- 首轮完整场景任务用时 117.54 秒，对照此前约 399 秒，均复用同一地图并实际执行定位、融合和宽度检测；该轮仍是 64 个 OpenCV 线程，随后线程数修复的 4 线程宽度单项计时如上。最终重复 CLI 运行直接复用成功结果（1.72 秒）。保留原结果、原片、地图、设备标定及用户当前宽度锚点。

验证：全量 561 项 Python 测试通过（107.723 秒）；新增级联跳过/回退、原始坐标、重复 ID、跨检测冲突、短范围成功/回退、绝对帧索引与非均匀时间上下文测试。另验证线程配置为 4/2 时任务实际读取对应线程数，非正整数或无效输入拒绝启动。实验记录为 `/tmp/umi-width-cascade-benchmark.json`、`/tmp/umi-width-cascade-samples.jpg`、`/tmp/umi-speed-shelf-run.log`、`/tmp/umi-speed-all-tests.log`。

## 2026-11-10 可配置标签尺寸与双标签三维中心间距

- 在 `feat/tag-distance-3d` 分支实施，保留工作区既有改动。新增 `gripper.tag_distance` 套件资产，JSON 的 `tag_sizes_mm` 分别配置 ID 0/1 黑色外框边长，单位 mm，允许不同尺寸。`kit_002/tag_distance.json` 按用户给定尺寸配置各 6 mm；不借用 AprilGrid 尺寸，不调整原有开口锚点。
- 原始检测缓存增加四角点（原始 1920 工作半幅坐标），像素宽度与三维间距共用一次检测。按 240 像素边距换到已标定的 1440 裁切坐标，使用当前工作镜头 KB4 内参、鱼眼角点反投影和 IPPE_SQUARE 解算两枚标签中心，计算三维欧氏距离。超出标定视野、退化四边形、无正深度解、重投影超限或两解中心歧义的观测保留缺失；不插值填补距离。
- 三维缓存包含相机内参、尺寸文件、尺寸规范值、质量策略与代码身份。尺寸/内参变化复用原始角点并重新求解几何；尺寸快照校验读取与冻结之间的变化。测距资产不进入相机/地图几何身份，单独进入任务输入指纹，运行结果保留本次尺寸副本。
- `trajectory.csv` 增加 `tag_distance_mm`、`tag_distance_valid`、`tag_distance_source`，报告和证据汇入 `episode.json` 的 `tag_distance.<手>`；不将中心间距填入开口、归一化开合量或闭合状态。轨迹交付导出保留新列；训练包的开合通道仍需原有宽度标定。查看器随当前帧显示标签中心间距，双手各按自身时间轴显示，缺失为“—”。
- 使用地图 `20261110_184901-8fbf9e88` 验证货架任务：380 个轨迹时刻中 340 个间距有效（89.47%），39 个缺标签，1 个因重投影误差过大拒绝；有效间距约 40.731–74.079 mm。按 6 mm 名义尺寸解算，尚无尺量真值验证。轨迹覆盖仍为 100%，380 行 TCP 位姿字段与改动前逐项一致，12 个结果资产哈希核验通过，重复运行 SKIPPED。首次包含重新检测角点的完整流程 91.35 秒，缓存后的重发布约 1.96 秒；不能把缓存用时视作首次处理吞吐。

验证：全量 574 项测试通过（111.650 秒）；随后补充尺寸快照一致性、双手查看时间裁切和原片角点坐标检查，61 项场景/测距/宽度定向回归及最终 33 项测距/宽度回归通过。覆盖不同标签尺寸、不同深度、相机刚体运动、毫米尺度、质量拒绝、缓存重算及防篡改、缺失值、仅标签间距的场景运行/导出。Node 查看器测试及 JS 语法检查通过；本机临时只监听回环地址的服务与无头 Chrome 验证实际界面显示 62.43 mm，截图 `/tmp/umi-tag-distance-viewer.png`，服务验证后关闭。全量日志 `/tmp/umi-tag-distance-all-tests.log`，真实任务日志 `/tmp/umi-tag-distance-real.log`，最终回归 `/tmp/umi-tag-distance-last-tests.log`。

## 2026-11-10 全项目架构重构与技术债清理

- 按用户指定的全项目范围执行 `deep-architecture-refactoring`；以本轮开始时的工作区为基线，保留既有未提交修改及设备、地图和录像资产。审计 `software/` 下全部 Python 模块，并检查原生驱动、查看器脚本和命令入口。
- 将算法参数类型从 `config.py` 提取到 `parameters.py`；源代码、规范 JSON、文件和引擎身份统一由 `provenance.py` 管理，删除 registry/原片提取中的重复 SHA256 实现。参数加载保留在 `config.py`，跨模块覆盖调用改用明确的 `merge_overrides()`，完整快照解码使用 `decode_parameters()`。
- 场景的建图/发布/绑定集中到 `pipeline/map_lifecycle.py`，准备和定位缓存及单手融合集中到 `pipeline/reconstruction.py`，设备资产选择和冻结集中到 `management/camera_assets.py`。`workflow.py` 从 652 行降至 283 行，`config.py` 从 520 行降至 249 行。调用方和替代引擎测试改为直接引用职责模块，不保留旧入口透传别名。
- 标签测距拆为 `tag_models.py` 的具名请求、质量策略和时间序列，`tag_geometry.py` 的 KB4 位姿计算，以及 `tag_distance.py` 的缓存编排；尺寸与开口资产读取统一由 `calibration.py` 负责。原始角点成为检测缓存的必需字段，移除无角点缓存分支。
- 查看器的轨迹/约束/测距解析集中到 `readers.py`，公共输入错误位于 `errors.py`，目录发现和装配保留在 `data.py`。浏览器统一使用 recording 领域命名；测距格式和双手时钟选择由 `measurements.mjs` 处理，并验证缺失值、零距离和非零视频时间偏移。
- 清理 39 个模块中的 data/info/res/temp/obj/flag 等模糊绑定及 81 个未使用导入；保留协议键和数学计算顺序。将仍在使用的 main/f331ad9 像素—毫米映射迁到正式 `calibration.pixel_distances_to_widths()` 并保留边界测试，删除未接入正式流程的 8 个历史实验模块及独立测试入口。保留实际任务入口仍调用的目录迁移功能与数据读取协议，避免改变现有输入/成果处理行为。环境变量统一为 UMI_PY / UMI_PY_TOOLS，启动器默认选择当前依赖目录解释器。
- 原生 SLAM 驱动拆出有界帧回放和逐帧位姿序列化，前向/反向帧号、视觉时间、CSV 格式与内存预算保持原值；移除旧 KB4 编译分支，要求当前前向模型，并在本机重编译成功。二进制与源代码身份继续进入复用检查。

结果：`software/` 下 Python 模块从 145 个调整为 146 个，总行数从 17457 降至 16434（净减 1023 行）。静态检查未发现列出的禁用变量绑定或未使用导入。最终 `./test.sh --all` 的 577 项 Python 测试通过（112.209 秒），前端测试和 umi-calib/umi-scene 自检通过；原生 x3_mono_slam 构建成功。使用原地图重放货架任务后，380 行 `trajectory.csv` 与本轮开始前逐字节一致，覆盖率保持 100%，标签间距仍为 340 有效、39 缺失、1 个重投影拒绝；12 个结果资产哈希核验通过，随后重复运行 SKIPPED。无头 Chrome 真实界面仍显示当前帧 62.43 mm，临时服务验证后关闭。

验证记录：`/tmp/umi-architecture-audit.json`、`/tmp/umi-refactor-verified-all.log`、`/tmp/umi-refactor-verified-real.log`、`/tmp/umi-refactor-verified-reuse.log`、`/tmp/umi-refactor-native-build.log`；本轮修改前文本基线保存在 `/tmp/umi-architecture-before`，用于排除既有改动并生成完整重构补丁。

## 2026-11-10 三个程序入口与参数帮助优化

- 精简 umi-view / umi-scene / umi-calib 启动脚本，统一真实脚本路径定位、具名目录变量、严格错误处理和原样 exec 参数；三个启动脚本共 59 行。修复 umi-view 从外部工作目录经软链接启动时找不到 software 模块的问题。
- 删除 umi-calib Shell 中默认启用求解环境的命令黑名单分流。现在 Python CLI 先解析参数并校验输入方式、尺寸、线程数和 force/resume，再由正式 run / calibrate-rear 选择标定解释器并按需补装。帮助、无命令、未知命令、参数解析失败、verify 和 dry-run 不触发安装；check 只使用已存在的解释器并报告缺失。使用 sys.prefix 区分两个 venv，保留 exec 的参数边界、信号和退出码；安装失败或组件仍不完整时终止启动。
- 补齐三个 CLI 的用户参数帮助和常用示例：并发数、CPU 列表、JSON 输出、dry-run、结果目录、验收字段、地图版本和 force 的作用。查看器说明默认 scenes/ 与 9091 端口，并在启动服务前校验 0–65535；0 为自动选择空闲端口。标定尺寸明确为 AprilGrid 标签黑色外框毫米边长，夹爪尺寸仍由 kit 的 gripper.tag_distance 配置。
- 同步日常安装、场景处理、查看器与标定模块说明，明确 check 不再自动安装组件；数值计算与结果格式沿用现状。

验证：新增 13 项入口与环境边界测试，覆盖三入口软链接、空/未知命令、错误参数、只读检查、安装失败、参数中的空格、解释器递归防护、参数帮助完整性和端口边界。最终 `./test.sh --all` 的 590 项 Python 测试通过（111.744 秒），前端测试及两个 CLI 自检通过。实际 umi-calib check 成功；空命令、拼错命令、错误选项及无效输入返回 2，无安装启动或 traceback。真实货架任务仍为 SKIPPED 直接复用。日志 `/tmp/umi-entrypoints-all-tests.log`、`/tmp/umi-entrypoints-tests.log`、`/tmp/umi-entrypoints-scene-smoke.log`；本轮入口基线位于 `/tmp/umi-entrypoints-before`。

## 2026-11-10 建图内容复用与分阶段缓存

- 对货架建图原片 `VID_20261110_181715_00_018.insv` 单独分段计时：47.35 秒录像、1420 个处理帧，准备 15.824 秒、视觉建图 146.644 秒、最终地图定位 48.061 秒、尺度/重力验收 5.972 秒。建图时跟踪 1401 帧，最终地图定位 1420 帧；保留两遍视觉计算及原有验收。
- 新增 `scene/map_cache.py`，集中管理建图代码依赖、内容身份、阶段缓存和计时。完整地图输入仍记录来源路径与时间，复用使用原片 SHA256、相机标定、相关配置和引擎身份；同一场景内相同原片跨批次路径可复用。CLI、查看器及任务编排代码不进入建图源码集合，TCP、宽度、标签测距资产不进入地图身份，遮手图仍参与。原片发布前复核内容，冻结相机资产与计算前身份核对。
- `recordings.prepare_recording()`、`slam.build()` 和 `slam.localize()` 分别缓存数据集/镜头视频、视觉地图与逐图定位。定位按最终地图内容、子地图索引、实际画面和相机设置匹配；读取时使用本次相机时间偏移。缓存校验全部登记资产，复制到本次工作目录后使用，完整成功才发布 `ready.json`；损坏重算，失败刷新保留原有效缓存。`build --force` 全部重算，失败重试刷新视觉阶段。尺度、重力和场景放置每次按当前参数重新验收。
- 真实 CLI 为 `test_shelf/20261109_002` 生成版本 `20261110_212612-0dd2e672`。首次建图记录 218.693 秒；三层缓存建立后，相同批次 API 复用 0.650 秒，另一批次相同原片 API 复用 0.344 秒，实际 `umi-scene build` 含启动约 1.870 秒。只调整可观测性验收门限的临时场景用时 10.866 秒，准备/建图/定位均命中缓存（0.234/0.038/0.047 秒），尺度验收 5.878 秒，冻结导出 3.650 秒。首次计算的 ORB 耗时保持约 145 秒；本轮收益来自减少重复计算。
- 新建图与此前版本的尺度、重力、放置及验收数值完全一致；1420 行定位 CSV 与优化前计时实验、缓存复用结果逐字节一致，两个结果的资产哈希全部有效。真实批次保留原参数，门限变化实验仅在 `/tmp` 进行；旧地图与任务结果保留。

验证：新增 13 项缓存与复用测试，覆盖换路径、同大小/mtime 内容替换、相关参数及实现失效、TCP 独立性、多子地图、时间偏移、损坏/中断、强制重算及失败保护。72 项定向回归通过；最终 `./test.sh --all` 的 603 项 Python 测试通过（109.994 秒），前端测试、CLI 自检、文档链接和 `git diff --check` 通过。计时与对比记录：`/tmp/umi-mapping-cache-benchmark.json`、`/tmp/umi-mapping-cache-cold.log`、`/tmp/umi-mapping-cache-cli-reuse.log`、`/tmp/umi-mapping-cache-all-tests.log`；改动前源码备份为 `/tmp/umi-mapping-cache-before`。

## 2026-11-10 首次建图热点测量与特征提取并行

- 在同一台 Threadripper PRO 5975WX 上，用同一段 47.35 秒货架原片、1420 个处理帧分析原生引擎。原视觉建图约 146.59 秒，其中局部 BA 79.57 秒、ORB 特征 21.23 秒、标签检测 8.65 秒；局部建图总时间包含 BA，不能重复相加。最终地图定位约 46.66 秒，其中特征 20.59 秒、标签 9.47 秒。插桩后逐帧 CSV 与原版完全一致。
- 新增 `orbslam3_feature_parallel.patch`：金字塔各层的角点、方向和描述子使用最多 4 个 OpenMP 工作线程，按层循环分配任务，保持原来的串行输出顺序。线程数显式指定，离线驱动关闭 OpenMP 动态缩减，避免本机 OpenCV 启动时打开动态模式后实际只分配一个线程；OpenMP 编译标志仅作用于 `ORBextractor.cc`，不改变 BA/Eigen 求解。视觉进程默认使用 `OMP_WAIT_POLICY=PASSIVE`，串行建图时让工作线程休眠。
- 无原生查看窗口的纯单目跟踪，在帧创建前根据当前状态决定是否检测标签；初始化、重置后的初始化和原生窗口仍检测。保留原 MONOCULAR 字典和 IMU 单目行为；独立夹爪测距不受影响。新增原生检查覆盖 24 种图像与重叠区组合的串行/并行特征顺序、描述子字节，以及标签检测关闭/恢复时 ORB 特征不变。
- 测试过把最小关键帧间隔从 0.2 秒改为 0.3 秒：视觉建图约 82.62 秒，建图录像自身仍能定位 1420/1420 帧，尺度/重力也通过；但实际左手任务从 332/332 降至 5/332，不能使用。正式代码保留 0.2 秒间隔、BA 迭代次数和验收门限。这说明建图录像自己的验收通过不足以证明地图能支持后续任务。
- 正式引擎已重编译；在全新临时场景中执行完整首次建图，五个阶段全部为 `computed`：准备 16.140 秒、视觉建图 126.842 秒、最终地图定位 28.052 秒、尺度/重力 5.923 秒、冻结 3.634 秒。总耗时从此前同录像的 218.693 秒降至 181.756 秒，节省 36.937 秒（约 16.9%）；原对应建图/定位阶段为 145.139/47.164 秒。这是本机单次完整处理的对照，不能保证其他录像同等收益。建图与定位 CSV 均与原版逐字节一致，尺度、重力和放置完全相同，发布资产哈希有效。
- 使用最终新建地图复测实际任务：`20261109_001` 右手定位 380/380 帧，正向 270、反向补入 110；`20261109_002` 左手定位 332/332 帧，正向 33、反向补入 299。两者最长无视觉均为 0，融合验收通过；视觉一致性 RMS 分别为 15.980/18.267 mm，与基线一致。这些指标不代表独立物理真值精度。

验证：76 项 Python 定向回归通过（20.944 秒）；原生特征、稳定顺序、KB4 和离线暂停/恢复/重置/退出检查全部通过，`umi-scene check` 正常。补丁正向应用与已应用状态检查、Shell 语法和 `git diff --check` 通过。最终计时为 `/tmp/umi-orb-first-build-speed/benchmark.json` 与 `verified-cold.log`，任务证据为同目录的 `installed-task-quality.json`；实验源码、原二进制备份和其他对照也在该目录。实际场景版本和批次绑定不由本次实验改写。

## 2026-11-10 场景结果字段去重与按需辅助轨迹

- 新结果使用 schema 4；未配置 `gripper.width` 时不生成 `width_mm/close_01/closed/valid/source` 五列，也不写空宽度报告。标签中心间距继续使用独立的三列，完整报告只保存在 `tag_distance.<手>`，移除各手元数据中的重复报告。套件结果快照只记录本任务实际使用的手。
- 场景坐标统一保存在 `scene.frame_id`，批次名统一保存在任务层 `session`。输入复用记录使用地图坐标/版本/资产的联合摘要及同步摘要，保留原来的失效判定；场景声明读取时核对摘要。`inputs.code` 合并场景与轨迹源码清单，移除重复的 `trajectory_code` 清单，并检查重复路径内容是否在采集身份期间发生变化。
- 默认不生成可由 TCP 与固定外参重建的 `traj_camera.txt/traj_imu.txt`；排障时可设置 `parameters.export_auxiliary_trajectories: true`。查看器用标准库四元数运算按需重建两条显示轨迹，仍优先读取已保存的历史文件，不增加 NumPy/SciPy 依赖。训练读取器独立读取 TCP 时间轴，允许没有宽度列；缺少实测开口时仍如实标记训练通道缺失。
- 实际重新处理 `test_shelf/20261109_002`，发布地图 `20261110_225400-b48f6971`、任务运行 `d4f8e611bf24435eb5eceb9374991f28`。332 行时间/TCP 位姿字段与清理前逐项一致；267 行标签中心距有效、65 行缺标签，未用插值补测量。新主表仅有 10 个轨迹字段与 3 个间距字段，无宽度资产及默认辅助轨迹文件；9 个登记资产哈希核验通过。元数据由 36011 字节降至 30168 字节，合并后保留 85 个源码哈希。旧运行保留，重复 CLI 为 SKIPPED。

验证：86 项定向回归通过；最终 `./test.sh --all` 的 609 项 Python 测试通过（112.159 秒），前端测试和 CLI 自检通过。新增检查覆盖非平凡旋转/杆臂下的显示轨迹、非法外参、标准库依赖、辅助导出开关、地图复用、旧运行资产完整性、历史 schema 坐标冲突、来源哈希覆盖及可选测量列。实际结果的相机/IMU 显示轨迹与清理前文件差异小于 2 微米/1e-7 弧度；本次未修改 ORB-SLAM3/g2o 源码或正式二进制。临时回环服务验证了实际 HTTP 轨迹/地图/视频读取，无头 Chrome 验证“标签中心间距”显示项可见；第一行缺标签显示“—”，前端格式化首个有效帧为 47.59 mm，临时服务验证后关闭。

记录：`/tmp/umi-result-cleanup-before/`、`/tmp/umi-result-cleanup-targeted.log`、`/tmp/umi-result-cleanup-all-tests.log`、`/tmp/umi-result-cleanup-real.log`、`/tmp/umi-result-cleanup-reuse.log`、`/tmp/umi-result-cleanup-verified.json`、`/tmp/umi-result-cleanup-http.log`、`/tmp/umi-result-cleanup-viewer.png`。

## 2026-11-10 相同轨迹验收失败时停止重复建图

- `test_shelf/20261109_003` 的 `VID_20261110_230749_00_021.insv` 共 61.962 秒、1858 个处理帧；首次与第二次建图/最终定位 CSV 各自逐字节相同。视觉定位覆盖全片，但固定尺度复核中 0–13.981 秒的位置方程 RMS 为 0.932983 m，超过原 0.05 m 门限；后两段分别为 0.047424/0.028332 m。该残差不是实测轨迹误差，不能仅凭它把原因归为录像采集。
- `mapping.MapAcceptanceError` 保留每个被拒子地图的原始拒绝原因及送入验收的观测摘要。摘要包含相机、镜头、子地图索引、帧索引、IMU 时间与完整位姿，排除临时路径和图集序列化差异。固定尺度复核失败还记录实际原片时间区间，并显示在错误日志中。
- 建图允许首轮失败后刷新视觉建图和定位；连续两轮全部候选观测及拒绝原因相同时，第二轮即停止，不再执行第三轮。轨迹变化、拒绝原因变化、缺少候选轨迹证据和引擎临时退出仍保留重试机会。每次构建的成功清单或 `error.json` 中仅保存一份 `retry` 汇总，包含配置上限、每轮证据、重复对象与停止原因；验收数值和门限不变，失败不会替换已有成功地图。
- 在 `/tmp/umi-map-retry-replay/` 复制本次真实两轮视觉产物，运行实际尺度/重力验收及完整 Python 重试编排，未执行新的原生 SLAM。28.315 秒内再次得到两轮原样的验收诊断，并在第二轮以 `repeated_visual_failure` 结束；日志明确显示 0.00–13.98 秒，不发布地图。按原日志每轮约 195 秒建图、39 秒定位、14 秒验收，这一案例可避免第三轮约 4 分钟的重复计算；这不是验收问题已经修复的声明。

验证：新增 8 项回归，覆盖重复失败、变化后第三轮成功、拒绝原因变化、全部候选比较、引擎故障、单次上限、无轨迹证据及原片时间区间；50 项定向回归通过（21.271 秒）。最终 `./test.sh --all` 的 617 项 Python 测试通过（113.393 秒），前端测试与 CLI 自检通过；全量检查中补齐旧共享尺度测试夹具的相机序列号，与真实输入契约一致。正式 ORB-SLAM3/g2o 源码和二进制均未修改，实际场景地图及批次绑定未由回放改写。抽查前段 0/4/8/12 秒未见明显严重模糊或失焦；初始化尺度异常仍需独立排查，不据此认定需要重拍。

记录：`/tmp/umi-map-retry-before/`、`/tmp/umi-map-retry-targeted.log`、`/tmp/umi-map-retry-all-tests.log`、`/tmp/umi-map-retry-replay.log`、`/tmp/umi-map-retry-replay.json`、`/tmp/umi-map-retry-changes.patch`；画面与运动初筛为 `/tmp/umi-003-frames.jpg`、`/tmp/umi-003-motion-check.json`。

## 2026-11-11 标签检测按输出帧、固定 ROI 与双重覆盖验收

- `test_shelf/20261109_004/take_001` 的原片长 18.352 秒、1101 个源帧，旧标签阶段从 23:59:48 持续到次日 00:05:24，约 336 秒。缓存的全部源帧三维解算单独重测仅 0.187 秒。前 120 个源帧插桩重测 101.453 秒，与旧角点缓存逐项一致；完整补检调用 21757 次，其中约 78.641 秒，候选筛选仅 0.801 秒，主要瓶颈是反复对候选图像做标签识别。
- 新增 `gripper_width/sampled_detection.py`，按最终轨迹时间点从原片真实 PTS 选择最近源帧，保留半帧容差和源时钟边界；先快速扫描全段，再按需补检。原始像素不缩小，不沿用前帧测量或插值补缺；当前 30 Hz 输出将本段检测次数从 1101 减到 551。补检从目标前的关键帧解码，保留前邻帧，避免每批都从片头解码。
- 用户明确默认检测范围为原始 1920×1920 工作镜头半幅的 `600 980 1350 1250`，即 750×270 像素。场景使用 `tag_distance_roi_px`，`preview-width`、`detect-width` 使用同一默认值且允许 `--roi` 覆盖。快速扫描、补检子区域及周期重搜均不越过固定范围；保存角点加回 ROI 原点，再减去标定裁切边距进入 1440 KB4 解算。该范围与轨迹视频的 `video.crop` 分别配置。ROI 纳入检测身份，更改范围不会读旧范围缓存。
- 默认每只手同时满足 `tag_distance_min_valid_fraction: 0.5` 和 `tag_distance_max_missing_frames: 15` 才通过。有效分母为该手最终 CSV 全部帧，只有双标签及三维几何检查通过才算有效；开头、结尾和所有无效原因均计入连续缺失。恰好 50% 或 15 帧允许通过，16 帧触发补检；0 帧上限可表达不允许缺失。参数范围、整数和 ROI 边界均在处理前校验。
- 优先补检最长超限缺口的中间窗口；只需提高总比例时，优先已有标签的帧并分散时间点。每帧最多放大补检 16 个候选，保持字典、角点精修、三维重投影和双解门限。两项同时达标才停止；若连续超过上限的帧已完成补检仍无效，其他帧不能消除该缺口，以 `unrecoverable_gap` 提前失败。重复 ID、两轮位置/ID 冲突仍拒绝。日志约每 3 秒显示帧数、速度、耗时、预计剩余及有效比例。
- 快速角点与已尝试补检分别缓存，包含原片、输出时间点、ROI、OpenCV/PyAV 版本、源时钟和实现内容；尺寸、相机参数和验收值不进入角点身份。几何缓存绑定角点内容与当前测量。提高比例或收紧缺失上限可续算未尝试帧；放宽要求不丢已得测量。成功报告和失败 `tag_distance.json` 保存实际有效数、阈值、最长缺口、超限帧号及原片区间；失败不替换旧成功入口。
- 最终正式 CLI 使用已有场景地图 `20261110_235617-2dd0641e` 复测批次 004，标签阶段 **45.724 秒**，相较原约 336 秒减少约 86%。快速扫描仍为 325/551（59.0%）；末尾仍连续缺失 64 帧（16.250–18.352 秒），在中间连续 16 帧补检失败后提前拒绝。本轮实际已尝试补检仅 16 帧；已有角点证据重复验收无需再次补检。旧成功运行 `454c9c3161f34d3c92ff6cd3e511eac1` 保留，当前尝试明确标记 failed；诊断为 `failed-9b9774f1ffd72de3f80b1ea6/right/tag_distance.json`。这些速度数据包含提前拒绝收益，不代表录像缺失已修复。
- 最终固定 ROI 对真实批次 002 的左手录像检测 **21.969 秒**，257/332（77.4%）有效、最长连续缺失恰好 15 帧，通过两项验收；热缓存复测约 0.022 秒。旧完整检测为 267 帧，新策略减少了 10 帧额外补检；共有 257 帧的间距差中位数约 3.4e-7 mm、最大 0.660 mm，来自检测范围/角点变化，不是独立实测误差。004 的时间轴与 TCP 十列和优化前逐字段一致，两版旧资产哈希均有效，正式 ORB-SLAM3/g2o 二进制哈希不变。

验证：最终 120 项定向回归通过（14.796 秒）；`./test.sh --all` 的 644 项 Python 测试通过（117.494 秒），查看器前端测试与 CLI 自检通过，`git diff --check` 通过。新增检查覆盖全段扫描、按比例提前停止、几何无效和重复 ID、阈值/时间轴/ROI 缓存、源时钟变化、真实 PTS/间隙/最近帧和 seek、首尾缺口、15/16 帧边界、长缺口续算与提前拒绝、固定 ROI 不越界及原始坐标、CLI 默认值、双手分别验收与失败保护。

记录：`/tmp/umi-tag-speed-before/`、`/tmp/umi-tag-detection-profile.json`、`/tmp/umi-tag-roi-tests.log`、`/tmp/umi-tag-roi-all-tests.log`、`/tmp/umi-tag-roi-benchmark.json`、`/tmp/umi-tag-roi-final-cli.log`、`/tmp/umi-tag-speed-verified.json`。比例门限和固定 ROI 的不同中间策略对照保留在 `/tmp/umi-tag-speed-*.log`；中间结果不作为最终默认策略的速度保证。

## 2026-11-11 标签入口统一先裁剪再转灰度

- 新增 `gripper_width/frame_processing.py`，场景快速扫描和补检、像素锚点宽度提取、`preview-width`、`detect-width` 共用 `crop_gray()` / `detect_frame()`。移除 monitor 独有的裁剪函数及其他入口整帧转灰度后裁 ROI 的流程。所有入口在源像素格式中先裁剪，再转换 ROI 灰度；彩色预览显示仍单独准备。
- 按像素平面及色度采样边界复制 YUV、NV12/P010 等半平面和 RGB/RGBA 等打包格式，保留颜色空间与亮度量程。奇数坐标先对齐后裁回精确边界；高位深转 8 位保留原图 8×8 抖动相位，避免裁剪引起 1 个灰度级的变化。压缩视频仍需解码源帧，仅减少灰度转换和后续检测的像素量；默认仅转换 750×270 像素。
- `FrameDetection.translated()` 统一坐标恢复。检测器在 ROI 局部坐标中工作，检测结果只加回一次原始半幅偏移；监测 CSV、预览标签框、场景三维角点仍采用原始 1920 半幅坐标。共享转换源文件加入检测及宽度流程身份，更新后不会误读旧实现的检测缓存。默认 ROI、50% 有效比例、最长连续缺失 15 帧以及三维几何门限保持原值。
- 本机单次真实录像对照：004 正式 CLI 标签阶段从上一轮的 45.724 秒降至 **35.307 秒**；002 左手从 21.969 秒降至 **18.147 秒**。两段快速角点、补检角点、源时间戳、重复目标标记、三维距离及来源数组逐项完全相同，验收报告也完全相同。002 仍为 257/332（77.4%）、最长缺失 15 帧，通过；004 仍为 325/551（59.0%）、最长缺失 64 帧，16 帧补检后提前拒绝。速度是本机单次对照，不保证其他录像等同比例收益。
- 实际失败诊断为 `20261109_004/take_001/.umi/runs/failed-61f6b5d1aa25252e3b1deee3/right/tag_distance.json`。旧成功运行及其资产均保留，正式 ORB-SLAM3/g2o 二进制哈希不变。

验证：新增 6 项共享预处理测试，覆盖 13 种像素格式、奇数/末端边界、转换输入尺寸、颜色空间/量程、源平面不变、原始坐标恢复及 ROI 外重复标签。126 项定向回归通过（14.356 秒）；最终 `./test.sh --all` 的 **650 项 Python 测试通过（114.942 秒）**，前端测试、CLI 自检及 `git diff --check` 通过。

记录：`/tmp/umi-crop-gray-before/`、`/tmp/umi-crop-gray-targeted.log`、`/tmp/umi-crop-gray-all-tests.log`、`/tmp/umi-crop-gray-real.log`、`/tmp/umi-crop-gray-benchmark.json`、`/tmp/umi-crop-gray-verified.json`；对照脚本为 `/tmp/umi-crop-gray-verify.py`。

## 2026-11-11 环境海报检测、布局资产与融合接口准备

- 在 `refactor/environment-tags-adaptive-fusion` 分支开始实施。`Scene.localization_lenses()` 将地图定位镜头与已登记标定分开：shared_front 可加载经过相同身份、来源和验收检查的 rear 标定，仍只定位 front；dual 仍定位两镜头。任务预检、单手重建和结果中的 rear 使用标记同步采用实际定位角色。没有把环境观测默认为已参与轨迹。
- 场景 `fuse()` 拆为输入检查、定位采样与初值、惯性图、地图先验、静止因子、求解和一致性度量。保留因子及变量插入顺序、噪声、初值和失败检查顺序。使用修改前工作区实现对 front、dual、短缺口三组 157 节点合成输入逐项比较，位姿、tracked 和 evidence 完全相同；新增 35 节点改动前数值夹具（合成数据，不是实际录像真值）。尚未实施独立 checks 模块、逐帧来源表、海报投影因子或尺度变量融合。
- 新增 `vision/environment_tags.py` 和 `umi-scene detect-tags`：顺序解码 X3 3840×1920 4K60 原片的真实 PTS，默认每两个源帧检测一次环境镜头左半幅。使用 `DICT_4X4_100`，同帧重复 ID 全部拒绝，未知 ID 计数；记录半幅原像素角点、投影边长、标定裁切内标志和采样帧关联。标定视场外观测仅统计，不标为几何可用。角点可显式转换为 1440 裁切坐标及环境镜头 IMU 时间，但不会默认套用工作镜头时间偏移。
- 角点 NPZ 与检测 JSON 绑定原片内容、字典/ID/黑边/采样/筛选参数、实现内容及 OpenCV/PyAV/NumPy 版本，复用时核对角点文件哈希。报告给出每 ID 和整板的采样可见率、最长采样缺口、每帧有效 ID/角点数及重复/未知检测；同帧多 ID 不计作多个时间点。当前只做单次原分辨率检测，未做窗口追踪、放大补检或几何位姿验收。
- 用户提供 `hardware/UMI_ArU..an.pdf`，SHA256 为 `a3da873d895afa719bc0a3021ad1f422c6d67fa54e56d0ca162a80a4095ebaf9`。从 PDF 矢量外框核对 77 个标签坐标，并渲染解码验证：7 列 × 11 行，ID 按行从 12 到 88、朝向一致、黑边 1 格；页面 1000 × 1500 mm，黑框边长 100 mm、间距 130 mm，左右边距 60 mm、上下 50 mm。渲染角点与布局投影最大差 1 px；矢量坐标与设计值差在 0.001 mm 容差内。
- 新增 `hardware/calibration_targets/environment_poster.json` 和 `vision/tag_board.py`，保存显式 ID → 米制角点及 PDF 来源哈希。原点在纸张左上，X 向右、Y 向下、Z 指入纸面，角点对应 OpenCV 解码后的 TL/TR/BR/BL；拒绝重复 ID/字段、未知格式、非平面或非正方形几何等坏资产。尺寸标为 `pdf_design`，不声称打印成品已经实测。CLI 默认读取该资产；布局与检测身份分开，更正物理几何可复用角点，但报告记录当前布局身份。
- 实际批次 006 的任务原片以 `--frame-step 60` 试跑，采样 15 帧、未检出标签；抽查首帧环境镜头未见海报。这里只验证真实原片入口和空观测行为，不证明海报检测召回率。另以左右半幅放置不同真实 ArUco 图案的合成视频验证正式 CLI：2 个采样帧、4 次海报观测（ID 12/88），右半幅 ID 55 未混入；热缓存正常。全部输出位于 `/tmp`，没有重跑或发布场景轨迹。

验证：第一批 41 项定向回归通过；随后 `./test.sh --all` 的 **687 项 Python 测试通过（144.276 秒）**，前端测试和两个 CLI 自检通过。PDF 布局接入后的 23 项检测、布局及入口定向回归通过（2.133 秒），包含直接渲染用户 PDF 验证全部 ID/位置/方向；默认布局的正式 CLI 冒烟通过。`git diff --check` 通过。未提交已有工作区修改，未修改 ORB-SLAM3 二进制、现有相机标定或实际地图/任务结果。

记录：`/tmp/umi-poster-implementation-before/`、`/tmp/umi-poster-targeted.log`、`/tmp/umi-poster-all-tests.log`、`/tmp/umi-poster-layout-tests.log`、`/tmp/umi-poster-pdf-verification.json`、`/tmp/umi-poster-real-detection/`、`/tmp/umi-poster-cli-output/`、`/tmp/umi-poster-default-layout/`。后续仍需含海报的真实录像、打印尺寸实测及有效环境镜头标定来验证几何求解与融合收益。

## 2026-11-11 固定地图下的任务海报联合融合

- 接入 `scene.yaml` 顶层 `tags.layout`；默认只诊断，`parameters.tag_fusion_enabled: true` 才修正任务轨迹。未配置布局不走检测路径。显式融合必须有已验收、身份匹配的 rear 完整标定。每次任务冻结布局到结果 `devices/environment_poster.json`，核对预检哈希；任务身份包含布局内容，计算不再读取可变的原资产。
- 新增 `vision/poster_pose.py`：使用实际 rear KB4 内参反投影、IPPE 平面候选和固定基线多帧重投影优化，得到每手每段录像独立的 `T_scene_board`。77 个 ID 共享一个板位姿变量；没有复用建图/上次任务的板位姿。检查正深度、基础雅可比退化、多解与原始角点 RMS；尚不是实测标定误差或运动可观测性的完整统计检验。
- 新增 `scene/spatial/poster.py`：先通过原有地图 + IMU 融合验收，再加入板位姿与八维四角点 KB4 自定义因子。推导并验证 GTSAM Pose3 右扰动解析导数，采用角点径向噪声和每 ID 共同 Huber 核。数学方法借鉴 specialist 的角点重投影思路；此处为 KB4 模型及刚性整板实现，没有引入其 MEI C++ 实现或逐 ID 自由位姿。
- 融合节点使用原工作镜头 IMU 时刻与环境角点 IMU 时刻的并集；只合并不超过 1 ns 的数值舍入等价时刻，重新积分这些边界间的 IMU。原地图先验仅重编号，不额外插值为新增先验；输出保留工作镜头时间轴。未改变地图尺度、原定位覆盖门槛或 `tracked` 的地图来源语义。
- 无有效观测/有效时刻不足时逐值保留基线；诊断模式几何失败仅报告原因，启用后几何冲突抛出 `PosterGeometryError`，不会被单手场景定位不足的自动局部回退吞掉。保留初始/最终角点 RMS、板位姿、节点数与时刻范围；结果明确区分 rear 用于地图定位或海报融合，并在报告中显示海报状态。`actual_error` 仍为 `not_measured`。
- 默认 8 px 最短边、3 个时刻、1 px 基础角点 sigma、5 px 重投影 RMS 上限等是合成验证的暂定值。未验证真实含海报录像的精度、召回率、卷帘影响和耗时；未测量打印成品尺寸。当前保守地拒绝初始化时与地图轨迹不一致的海报，不借标签挽救原地图验收失败的任务。建图尺度修正、自适应来源权重及可信等级仍未实现。

验证：完整 `./test.sh --all` 的 **698 项 Python 测试通过（142.230 秒）**，前端和两项 CLI 自检通过。后续补齐时间戳舍入及结果来源标记，11 项海报定向测试通过（1.864 秒）：KB4 投影与因子解析导数对数值差分、任意外参/13 ms 时间偏移、部分 ID 可见、合成相对运动改善、录像间海报移动重新初始化、录像内冲突拒绝、无观测和诊断保持基线、配置预检、布局运行快照及 rear 来源标记。最终 35 项场景流程回归通过（19.774 秒），`git diff --check` 通过。完整套件中的原融合数值基线仍通过。没有重跑或发布真实地图/任务，没有修改相机标定或 ORB-SLAM3 二进制。

记录：`/tmp/umi-poster-fusion-tests.log`、`/tmp/umi-poster-joint-all-tests.log`、`/tmp/umi-poster-final-targeted.log`、`/tmp/umi-poster-final-scene-tests.log`。

## 2026-11-11 建图海报尺度诊断、合并与固定尺度惯性复核

- 新增 `scene/spatial/poster_scale.py`，在保持视觉地图形状的前提下优化一个对数尺度和本建图录像的整板位姿。环境角点按本镜头 IMU 时间匹配连续地图段，禁止跨丢帧插值；从建图镜头变换到 rear 时，米制杆臂不乘视觉地图尺度。估计时去除任意地图原点平移，降低尺度与板平移的数值相关性；诊断位姿明确处于这个居中的米制地图坐标系，不作为任务锚点。
- 增加 `tag_mapping_enabled`（诊断）与 `tag_mapping_scale_enabled`（应用），默认都关闭，与任务融合开关独立。先验收原 IMU 定标，再按有效时刻数、相机移动范围、板面角点覆盖、正深度、重投影残差、雅可比退化和统计不确定度筛选海报尺度。统计不确定度由带像素噪声下限的优化雅可比 SVD 得到；不假定 77 个 ID 是独立尺度来源。
- 有效尺度以对数域方差加权平均，两个统计方差分别加地板项；同时检查归一化对数差与绝对尺度比界限。海报不可见、运动不足或不确定度过大时保留原 placement 数值；诊断模式的冲突只记录，应用模式拒绝候选，全部候选失败时保留已有地图指针。没有绕过原地图运动可观测性和可靠段验收。
- `estimate_lens_alignment(..., shared_map=True, fixed_scale=...)` 通过列消元精确固定尺度，重求自由重力、归一化重力和加速度计零偏；不使用极强尺度先验近似固定值。`metric_alignment(fixed_scale=...)` 复用现有可靠段、整图惯性位置方程、重力和 bias 检查。发布的不确定度来自合并结果，而不是固定尺度条件求解里的零方差。未指定 fixed_scale 时保留原数值路径，旧融合基线和既有对齐测试继续通过。
- 建图流程冻结布局与检测角点到地图版本，保存其哈希和诊断。地图输入身份仅在开启建图海报功能时加入布局内容、检测设置和相关门槛；任务融合开关不会改变建图定义。开启修正要求建图设备具有 rear 完整标定；地图尺度修正本身不要求后续所有任务设备都启用标签融合。地图报告显示来源尺度、相容性与采用尺度。
- 暂定门槛：6 个时刻、0.15 m 运动范围、板面 x/y 各 0.08 m 覆盖、统计相对标准差 0.15；IMU/海报标准差地板 0.03/0.02，归一化差阈值 3、尺度比差阈值 0.20。这些仅经合成与流程测试，没有实拍调参。仍需实测打印成品尺寸、尺度不确定度分布、速度和真实精度；任务中的地图尺度变量、自适应来源权重、可信等级和留出验证仍未实施。

验证：新增 10 项数值/流程测试，涵盖 2.3 已知尺度恢复、部分 ID 可见、13 ms 环境时间偏移、米制杆臂、任意地图原点、空观测、纯静止与跟踪缺口、录像内海报移动、两种尺度冲突阈值、固定尺度惯性复核、配置及缓存身份、发布冻结资产、复用以及冲突不替换旧地图。定向测试通过（6.070 秒）；最终 `./test.sh --all` 的 **712 项 Python 测试通过（148.963 秒）**，前端测试、两项 CLI 自检和 `git diff --check` 通过。没有重跑或发布真实地图/任务，未更改相机标定和 ORB-SLAM3 二进制，未提交已有工作区修改。

记录：`/tmp/umi-poster-scale-tests.log`、`/tmp/umi-poster-scale-workflow-tests.log`、`/tmp/umi-poster-scale-all-tests.log`。

## 2026-11-11 独立轨迹与单手回退接入海报约束

- 用户明确要求共享和独立两种模式都接入。新增 `trajectory/algorithms/fusion_poster.py`，把同一个本录像海报位姿与原始角点因子加入独立原生图；轨迹和原有分段地图尺度/旋转/平移共同优化。没有把上一轮融合轨迹当成固定地图再平滑，也没有移除惯性初始化、规范先验或原有尺度变化质量门槛。
- `fusion.solve()` 先走原路径获得基线和海报初值，只有显式启用且观测足够时再求联合图。未声明海报不加载新路径；空观测和仅诊断保持基线位姿逐值一致。海报几何冲突明确失败，失败重试仍需带上相同海报资产，不能静默去掉约束。
- 独立图使用工作镜头视频时钟，角点时刻为原片 PTS 加 `rear_offset - front_offset`。在原优化节点之间插入真实观测时刻，重做对应区间的 Combined IMU 因子；地图因子数量保持原节点采样。全帧恢复使用真实时间邻居，不跨过新增海报节点。只合并不超过 1 ns 的等价时刻。板键使用 `h0`，避开独立图的 rear 杆臂 `l0`。
- `handheld` 与 `dual` 均支持。显式海报 dual 使用已登记 rear 的内参、外参和时间偏移；rear 地图观测换算到工作镜头时刻，不跨丢帧或地图 ID 切换插值。未声明正式 rear 的历史独立 dual 行为保留，不将样例几何当成本设备测量。
- 新增 CLI `--tag-layout FILE --rear-calibration rear_<version>`，通过 `trajectory.yaml` 的 `environment_tags.tag_fusion_enabled: true` 显式开启联合求解；其余角点参数默认与场景一致。声明资产但关闭开关时只诊断；启用却缺资产或 rear 序列号/来源/验收不匹配，在计算前拒绝。`pipeline/poster_assets.py` 冻结布局、rear 和原始角点，逐次读取检查哈希；结果复用身份包含布局与 rear 内容，清理 debug 后仍保留 `environment_poster/`。
- 单手共享定位质量失败后的独立回退继承 scene 的布局、海报开关与七项参数，复用本次冻结 rear 标定。缺 rear 时诊断模式报告 UNKNOWN 原因，启用模式拒绝；正常回退仍输出本录像局部坐标。共享海报几何冲突本身不触发回退；双手不使用单手回退。独立/回退报告保留 `environment_poster` 状态和初始/最终像素 RMS，rear 来源标记区分地图定位与海报融合。
- 尚未用真实海报任务验证收益、耗时和阈值；打印尺寸仍是 PDF 设计值。没有实现仅靠标签初始化尺度/世界原点，也不让无 IMU 可观测性的录像绕过原初始化。共享任务里的地图尺度变量、自适应来源权重与可信等级仍待实现。

验证：原独立融合数值基线 8 项通过（8.903 秒）；新增海报数值、真实融合 CLI 和回退协议测试通过，包括任意 rear 外参、front/rear 17/30 ms 偏移、观测节点插入与恢复、空观测/仅诊断逐值不变、冲突拒绝、资产篡改、错设备拒绝，以及回退开关和诊断传递。完整 `./test.sh --all` 的 **719 项 Python 测试通过（155.492 秒）**，前端与两项 CLI 自检通过；随后补充 native dual 真正联合优化两镜头与板位姿的用例，8 项海报定向测试通过（6.991 秒）。`git diff --check` 通过。未改真实录像、标定、地图/任务结果或 ORB-SLAM3 二进制，未提交工作区修改。

记录：`/tmp/umi-independent-poster-baseline.log`、`/tmp/umi-independent-poster-tests.log`、`/tmp/umi-independent-poster-pipeline-tests.log`、`/tmp/umi-independent-poster-dual-tests.log`、`/tmp/umi-independent-poster-all-tests.log`。


## 2026-11-11 双镜头标定与海报资产使用流程补充

核对现有 `calibrate-rear`、`install-rear`、registry 版本、kit 绑定和各运行快照后，更新标定指南、设备资产说明、套件指南及两份开发计划。明确 front/rear 分别标定和验收、工作镜头编号与 rear 哈希版本的区别、AprilGrid 与 ArUco 海报的用途、打印设计值与实测状态、海报移动/重印/标定更新对应的失效范围，以及标定 C 与任务融合的证据边界。

统一状态查询、真实双镜头组合诊断、海报成品测量证据格式和 C 配置隔离回归列为待完成，不将已有求解入口描述成完整运维工具。本轮仅更新文档，未运行或修改标定、registry、套件配置和真实结果；核对命令与现有 argparse 一致，检查新增本地链接及 `git diff --check`，未重复运行算法测试。

## 2026-11-11 双镜头版本与套件绑定的只读状态查询

新增 `software/calibration/inventory.py`，`umi-calib devices/status` 列出每台登记设备的 front/rear 全部版本、来源及完整性，保留 front 默认与来源选择，不推断 rear 默认。逐版本检查资产哈希、内外清单身份、rear PASS 与 factory 来源证据；损坏版本单独显示，不遮蔽其他版本。原 `registry.list_devices()` 调用接口保留。

`--kit` 沿用套件解析规则检查固定版本；`--intrinsics-source` 选择完整来源配置，缺失时不回退。front/rear 来源需一致；借用、样例和停用设备不通过新采集检查。未绑定 rear 默认显示缺项但允许 front-only 查询通过，`--require-rear` 则失败；显式绑定的无效 rear 始终失败。`--json` 提供结构化结果，退出码 0/1/2 分别表示通过、发现缺项或无效资产、参数或登记表读取错误。套件整体结论仅取决于该套件，清单仍展示其他设备和历史版本的问题。

检查范围仅为相机标定资产与绑定，不读取原片、不检查夹爪安装资产、不评估双镜头组合实测误差，报告保留 `combination_validation: not_checked`。文档已同步，组合诊断、成品测量证据格式和 C 配置隔离回归仍待完成。

验证：新增 10 项库存/套件测试通过，包括多版本多来源、哈希损坏、rear 身份与验收、固定绑定、缺项退出码及文件内容/修改时间不变；连同标定工作流、场景管理和来源切换回归共 **86 项通过（10.989 秒）**，日志 `/tmp/umi-calibration-inventory-tests.log`。实际只读 CLI 确认 `kit_002` 绑定 M4CP front `002`、缺少 rear，登记表中两台设备均尚无 rear；`--require-rear` 返回 1。未修改真实 registry、kit、标定或任务结果，未提交。

## 2026-11-11 C 配置隔离、正式 rear 解耦与统一校验

根据使用链路审查，先修复三项实际耦合：任务海报开关导致 C 缺资产失败，独立正式 rear 必须附带海报，状态查询与共享运行在旧套件来源推断上不一致。

- `Configuration.load_trajectory_validation()` 显式覆盖单镜头模式和关闭任务海报；rear C 同时关闭场景海报融合、建图诊断和尺度修正。视频/融合实验配置和实测 AprilGrid 尺寸继续生效，派生覆盖进入配置快照。没有让海报观测参与标定 C 的待评估轨迹。
- 新增 `trajectory/pipeline/rear_assets.py`：独立 `--mode dual --rear-calibration rear_<hash>` 不再要求海报，正式 rear 独立预检和冻结到结果 `devices/rear/`，包含版本、设备、来源与完整哈希。公开 dual 缺少正式 rear 时在删除旧输出之前失败；handheld 无海报却指定 rear 时明确拒绝未使用的选择。旧数值算法仍可测试，公开入口不再隐式使用 rear 样例。
- 内参用于 rear SLAM，外参/时差通过冻结清单传入融合子进程；`fusion_data.retime_rear()` 不再归属于海报模块，不跨缺帧或地图切换插值。海报与地图定位若收到不同 rear 参数会拒绝。缓存身份包含正式 rear，不再包含未使用的样例 rear；复用时检查冻结文件与原身份，debug 清理不删除标定快照。摘要和报告记录 rear 版本与来源。
- `camera.calibration_source.read_rear()` 统一登记、查询、共享运行和独立运行的身份/验收/来源校验；front 内外身份及必要资产哈希覆盖下沉至 registry。旧格式未写来源时，共享路径从所选 front 推导 rear 预期来源。原片 factory 光学一致性仍在运行时检查；状态页仍不替代实拍组合验收。
- 更新标定、参数、独立轨迹和开发文档；不改宽度实测值、kit 绑定、registry 或真实结果。海报参数两处默认值、成品测量证据格式和真实组合诊断仍待后续工作。

验证：全量 Python 首轮 **740 项中 739 项通过（160.029 秒）**，唯一失败是统一校验后的错误文案与旧测试预期不一致；随后将设备身份与验收缺失分开提示。最终包含该失败用例及全部本轮相关模块的 **196 项回归通过（28.138 秒）**，前端测试和两项 CLI 自检通过，`git diff --check` 通过。新增覆盖 C 前后镜头配置隔离、无海报正式 dual 子进程传递、13 ms 镜头时差下与诊断基线逐值一致、真实融合 CLI 数值输出、冻结资产篡改、错来源、冲突标定和缺 rear 时保留旧结果。未运行实拍双镜头/海报任务，未提交。

记录：`/tmp/umi-calibration-decoupling-all-tests.log`、`/tmp/umi-calibration-decoupling-final-tests.log`、`/tmp/umi-calibration-decoupling-js.log`、`/tmp/umi-calibration-decoupling-checks.log`。

## 2026-11-11 标定资产链路架构收敛与测试迭代提速

按用户显式调用的 `deep-architecture-refactoring` 技能，重构范围聚焦前述资产链路，保持 CLI、JSON 清单格式、数值求解与验收行为。

- `trajectory/pipeline/calibration_assets.py` 集中完成一次环境资产选择；`RearCalibrationIdentity`、`SelectedRearCalibration`、`SelectedPosterLayout` 和 `EnvironmentAssets` 用不可变对象区分身份、文件与布局。删除原 `rear_assets.py` 和海报模块的重复预检；指纹接口显式要求已选资产，不保留隐式重新选择的分支。去除未被后续流程使用的预检测量副本和冻结后的重复读取。
- `poster_assets.py` 只负责已选资产冻结、角点缓存与子进程输入，预检哈希使用 `PosterAssetHashes`；保留快照哈希覆盖和路径边界检查。冻结格式与指纹字段不变，新增回归确认海报与指纹共享一次 rear 选择。
- inventory 输出使用领域 `TypedDict`，检查与终端显示分离，显示移至既有 reports 模块。registry 的资产读取与设备生命周期选择拆开，inventory 不再构造假的 active 记录调用私有函数。移除仅剩旧测试引用、无生产调用的 `registry.list_devices()`，原 CLI 由统一 inventory 提供。实际 `kit_002` JSON 报告与重构前逐字段一致。
- 用户追问测试提速后，`test.sh --unit` 支持指定模块或单用例，无参数仍运行全套；不接受把其他选项混作模块。已有日常相关回归 111 项为 20.016 秒，相比每轮都跑约 159 秒全套，可缩短迭代等待；两者覆盖范围不同，不将此称为相同工作量的性能提升。
- 全量逐例计时发现 `test_fusion_refinement.py` 顶层导入 `tests.test_fusion_stability.SamplingStabilityTests`，同一源码的 4 项测试以不同模块名前缀被重复收集（本次重复耗时合计约 3.471 秒）。改为导入规范模块，保留全部独立测试与断言。初次按测试 ID 查重未发现该别名问题，随后用“源码路径、类名、方法名”核对，确认去重后 737 项均唯一，规范化后的用例集合与去重前完全相同。

验证：架构重构全量 **741 项通过（159.011 秒，包含上述 4 次重复）**；随后测试导入去重相关 **13 项通过（10.675 秒）**，移除旧状态接口后的 **28 项通过（7.336 秒）**；前端、两项 CLI 自检、Shell 语法、选跑单例/非法参数和 diff 空白检查通过。耗时最高的模块为场景流程约 19.846 秒、episode 约 9.463 秒、独立融合约 8.606 秒，尚未降低采样、求解精度或删减数值断言。本轮未修改真实 kit、registry、宽度测量或结果，未提交。

记录：`/tmp/umi-refactor-full-tests.log`、`/tmp/umi-refactor-test-durations.json`、`/tmp/umi-refactor-targeted-tests.log`、`/tmp/umi-test-dedup-check.log`、`/tmp/umi-registry-inventory-cleanup-tests.log`、`/tmp/umi-refactor-js-tests.log`、`/tmp/umi-refactor-cli-checks.log`。


## 2026-11-11：三入口配置统一与批次链路清理

按用户要求梳理 `umi-scene`、`umi-calib`、`umi-view` 的命令行与 YAML 分工，并继续清理相关链路中的重复读取、失效调用和行为不一致。

- 场景 run/build 新增 `--config`，只允许 trajectory / parameters；优先级为默认 YAML → 场景声明 → 本次覆盖。不可变覆盖对象显式传入批次、建图和任务流程；有效数值参与计算身份，文件路径/注释不单独使结果失效。新结果记录覆盖内容、路径与文件哈希，验收和导出按冻结内容恢复，不依赖临时 YAML 仍存在。
- 三入口新增 `--show-config`：配置展示在录像检查、求解环境准备或 HTTP 服务启动之前返回。场景配置展示兼容批次声明和已冻结批次，使用统一套件选择规则；标定展示包含 C 的独立镜头和实测板尺寸覆盖。查看器仍无算法 YAML。帮助按配置、测量、执行分组；标定三个子命令共用选项定义；统一文档位于 config/README.md。
- 去掉可选参数反复拼字典的传递方式和重复套件选择分支；环境镜头 C 的禁用海报规则由展示与执行共用。批次输入直接保存已读取的建图身份记录，生成快照不再重复读取同一原片和计算完整哈希；显式选图不再在解析/生成阶段各加载一次地图。
- 修复有建图原片的旧快照忽略显式覆盖并直接复用地图的问题；修复旧快照 build 忽略 intrinsics-source 的问题。来源检查集中核对固定 front/rear 版本，rear 使用已有完整校验器；同一批次的多任务共用一次来源检查。
- 首轮全量 Python 746 项和前端测试通过，但实际 CLI 自检发现 trajectory.runtime.check 仍导入已删除的 registry.list_devices。已切换到统一 inventory.device_inventory，并补充健康、损坏资产、空档案回归；错误提示区分环境/档案故障，避免一律要求重新安装。全仓 168 个软件模块的绝对与相对本地导入静态扫描未发现剩余未解析项。

分阶段验证：配置/执行 64 项通过；补充旧快照与批次读取清理后 55 项通过；自检与只读配置 11 项通过；最后一轮针对性 48 项通过（10.719 秒）。三个实际启动器的 show-config、修复后的 umi-scene check、Shell 语法及 diff 空白检查通过。所有流程测试使用临时目录与替代求解器，未运行真实录像的 SLAM 或更改真实测量资产。

日志：`/tmp/umi-configuration-all.log`（首轮含自检失效证据）、`/tmp/umi-cleanup-final-targeted.log`、`/tmp/umi-cleanup-final-all.log`。

最终验证：`./test.sh --all` 全部通过，Python **750 项（159.720 秒）**、查看器前端测试、标定与场景 CLI 自检均通过；日志 `/tmp/umi-cleanup-final-all.log`。


## 2026-11-12：按 front / rear 目录统一标定入口

按用户确认的目录约定，标定统一为 `umi-calib run <front或rear目录>`。CLI 由输入末级目录名确定镜头，拒绝日期目录、A 子目录及其他名称；不通过输出路径推断镜头。删除独立 calibrate-rear 命令和 --a/--b/--c/--d 选择方式，不引入 lens 选项或旧命令别名。A/B/C/D 子目录布局与按文件名排序的四原片布局均保留。

配置展示和正式执行共用同一 CalibrationRequest，show-config 也要求 front / rear 路径，展示镜头与实际输出目录，但不读录像或准备求解环境。工作镜头沿用原标定后端，环境镜头接收相同请求并提供独立 C 后端；后端不再重复展开命令行参数或二次读取结果来推断退出码。两镜头默认输出均为各自输入目录下 calibration_result，工厂来源为 calibration_result_factory。旧结果保留原处，可通过 --out 指定。环境镜头依旧要求已登记工作镜头资产。

run --install 按镜头登记：front 安装完整工作镜头结果，rear 在独立验收通过后登记 rear_calibration.json；失败不登记。rear 不支持 compare-sample，提前明确拒绝。入口帮助、Shell 示例、用户指南、配置说明、开发计划和环境镜头缺项提示已更新；历史日志条目不改写。

验证：第一轮 91 项相关回归通过；加入目录/布局、只读配置、真实数值 rear C、成功/失败自动安装等回归后 **131 项通过（19.147 秒）**。实际启动器对 recordings/calibration/IAQEB2609JPAR3/20261102/rear 的 show-config 显示 rear 和该目录下 calibration_result；未运行真实录像求解、安装或搬移真实资产。

日志：/tmp/umi-lens-targeted.log、/tmp/umi-lens-integration.log、/tmp/umi-lens-directory-all.log。

最终验证：`./test.sh --all` 通过，Python **758 项（160.445 秒）**、查看器前端测试、标定与场景 CLI 自检均通过。Shell 语法及 diff 空白检查通过。


## 2026-11-12：缺少正式 rear 绑定时使用 X3 工厂几何兜底

按用户确认，实现共享场景建图/任务及独立 handheld/dual 的 rear 工厂兜底；单手独立回退沿用场景冻结参数。只有未绑定 rear 且需要海报或 dual 时才推导，正式绑定无效、身份或来源错误仍拒绝，不用工厂值掩盖损坏。

双镜头位姿约定参考本机 specialist/src/specialist/record/offset_v3.py：YPR 为 Rz Ry Rx，平移为 -R*[tx,ty,tz]；index0 只对旋转施加机身 z 轴半周修正，平移不随之旋转。组合 T_imu_rear = T_imu_front × T_front_rear，保留本设备 front B；rear KB4 由当前原片 lens0 MEI 转换，与正式工厂内参流程共用前向视野、单调性和 RMS/P99 门槛。rear 时间偏移沿用 front，明确记录两半幅共享帧时钟的假设，未独立测量 rear 延迟。

推导资产在预检内存中生成，运行时才冻结到独立快照；工厂原值、front 内容哈希、转换误差、时间来源及 NOT_EVALUATED 进入资产身份与报告，复用重新校验冻结哈希。不伪造 PASS、不写 registry、不修改真实 kit/测量。dual 使用工厂参数时仍检查 ORB 前向 KB4 能力；shared_front/handheld 只做海报时不新增 rear SLAM 能力要求。CLI 帮助、配置说明、套件、场景、标定指南与计划文档同步更新。

验证：外参组合独立对照 GTSAM 的 specialist 公式（含两镜头非零平移），覆盖 front B/时差继承、只读预检、正式绑定优先与损坏拒绝、冻结/篡改、结果复用、共享与独立 dual 编排、报告来源和转换拒绝。集成回归 71 项通过（31.348 秒）；补充复用和报告校验后针对性 15 项通过（1.460 秒）。首轮全量 765 项中，仅新增报告测试误读替代求解器写出的占位文件而报错，已改为调用真实报告生成函数，生产流程未因此修改。

只读检查现有 IAQEB2609JPAR3 的 20261102 front/B 原片：X3 offset_v3 可解析，基线 0.0264611633 m；rear 前向转换 15114 点，RMS 0.506234 px、P99 1.110218 px，通过当前转换门槛。未运行真实 SLAM、标定求解或安装；这些数值仅验证元数据与光学转换，不代表轨迹精度。

日志：/tmp/umi-rear-integration.log、/tmp/umi-rear-final-targeted.log、/tmp/umi-rear-fallback-all.log（首轮测试夹具问题）、/tmp/umi-rear-fallback-final-all.log（最终全量）。

最终验证：`./test.sh --all` 通过，Python **767 项（162.428 秒）**、查看器前端测试、标定与场景 CLI 自检全部通过；`git diff --check` 通过。


## 2026-11-12：以采集批次为地图和内部存储边界

按用户纠正的边界，scenes/test_shelf 下每个采集批次有自己的地图，场景目录只保留可选参数声明。批次唯一 .umi 收纳 manifest.yaml、import.json、snapshots、maps/{current.json,versions,cache} 和 tasks/<任务>/{episode.yaml,state.json,cache,runs}；外层任务保留原片、可选 task.yaml 和 results/local_results 入口。移除场景级及任务级的空 .umi 与旧锁文件。算法、标定与坐标约定不变。

map_root 随批次解析，建图复用仅查询当前批次；显式 map_version 才允许查找同场景既有版本，并在哈希与标定校验后冻结到当前批次，避免后续依赖来源批次。任务配置与程序原片记录分离，任务配置的规范化哈希参与处理、状态、查看与验收；注释变化不触发重算。发现逻辑覆盖仅有集中元数据的任务，避免重复原片归属被漏查；双手批次可在首次运行前声明单手任务。查看器单独打开任务仅允许访问本任务集中目录，不允许越过该边界读取同批次其他任务。

旧地图依据建图原片路径、批次固定引用及历史结果归还所属批次，不因两个批次使用相同原片哈希而自动共享已发布地图。中断建图可依据提取记录的原片哈希归档，但不生成可用地图指针。迁移修正结果链接、地图相对路径、对应资产哈希及验收结果引用，保留原片与轨迹字节；重复迁移跳过产物哈希和写锁。无法确定归属的已发布地图报错并保留。旧目录只读预览与 dry-run 不生成迁移文件。

验证：首轮全量发现旧测试夹具仍读取场景地图及任务 .umi/runs，修正夹具并加入批次隔离、显式复用后删除源清单、任务配置失效与注释复用、单手任务首次登记、查看边界、未知地图保全、中断地图与链接、迁移幂等及软链接越界回归。最终 `./test.sh --all` 通过，Python **774 项（164.621 秒）**、查看器 JS 测试、标定与场景 CLI 自检通过；Shell 语法与 git diff --check 通过。未重新运行真实 SLAM 或标定求解。

已实际迁移 test_shelf：13 个 .umi 收敛为 6 个批次 .umi；11 个已发布地图分别归入批次 001/002/004/005/006（2/4/2/1/2 个），批次 003 保留 2 个中断建图现场且仍为 pending。迁移前后逐文件核对 13 个建图目录，所有原有普通文件哈希不变；12 段原片 SHA256、inode 和尺寸不变，17 份轨迹表哈希不变，4 个结果入口仍指向原运行，4 个结果的地图展示加载与全部资产校验通过，6 个任务状态可读取。旧任务中的既有失败状态保持原值，不伪造处理成功。所有软链接有效，二次迁移无文件修改，用时约 0.005 秒。

清理可重建的旧全局地图缓存 **1,172,208,997 字节（约 1.09 GiB）**；保留任务阶段缓存、历史地图、成功结果和中断诊断。之后需要重建地图时按批次重新生成缓存。

日志：/tmp/umi-batch-final-focused.log、/tmp/umi-batch-final-all.log；迁移保全快照 /tmp/umi-batch-storage-before.json；验证报告 /tmp/umi-batch-storage-verified.json；实际只读入口检查 /tmp/umi-batch-storage-status.json、/tmp/umi-batch-storage-config.json。


## 2026-11-12：退役旧场景布局与 schema 2 输入

将共享场景主夹具与复用它的流程测试改为批次 `.umi/manifest.yaml`、`.umi/tasks/<任务>/episode.yaml` 和外层 `task.yaml`；夹具显式区分原片目录与内部清单目录。建图原片及测量固定在批次 mapping 和 kit，场景参数改用 schema 3 内联配置，双手同步、指令、使用手的回归继续覆盖处理、复用、验收和训练导出。

删除旧清单位置、外层 manifest.yaml 批次识别、旧状态与运行目录回退、episode.yaml 人工字段合并与重新登记时的提取。人工字段出现在 episode.yaml 时明确拒绝；新批次也拒绝旧 videos/ 及混合目录，原片发现只遍历 left/right。场景 schema 2 与不经批次的 build_scene 入口退役；独立引擎结果、历史 episode.json、设备不可变标定包及当前布局下的批次/任务 schema 4 协议继续读取。移除工厂内参流程测试中未使用的 source 构造和未使用包装函数。操作指南及清理遗留事项同步更新。

验证：清理前 `./test.sh --all` 的 769 项 Python 测试、查看器前端及 CLI 自检通过。迁移后补充旧清单、状态、运行路径、人工字段与场景建图入口拒绝回归；最后一轮相关测试 55 项通过（15.129 秒）。

真实运行 `./umi-scene run scenes/test_shelf/20261109_002 --json`，使用批次当前配置完成新地图建图、原片定位、融合、标签中心距测量和结果发布，状态 ok；地图版本 `20261112_161509-296e7735`，任务区间 0～11.044366667 秒。重复运行状态 skipped 并复用同一地图；发布结果的 9 项资产校验通过，查看器成功加载 332 帧。结果未人工验收，运行成功不代表已完成独立精度评估；历史地图与运行记录保留。

日志：`/tmp/umi-cleanup-baseline.log`、`/tmp/umi-cleanup-last-focused.log`、`/tmp/umi-cleanup-final-all.log`、`/tmp/umi-cleanup-real.log`、`/tmp/umi-cleanup-real-reuse.log`。

最终验证：`./test.sh --all` 全部通过，Python **774 项（165.945 秒）**、查看器前端测试、标定与场景 CLI 自检和 KB4 C++ 数值回归均通过；`git diff --check` 通过。


## 2026-11-12：合并任务原片的三类检测解码

新增 `vision.decoding.VideoDecode`，一次顺序解码向像素宽度、采样标签中心距与环境海报检测器交付原始帧。场景同一只手配置至少两类检测时，在重建前填充各自缓存，后续融合和测量复用；单类检测保留原入口。命中的检测器不参与扫描，全命中不打开原片。各消费者完成校验后才开始保存缓存，解码或检测完整性校验失败不会发布本轮检测缓存。

保留各自 ROI、逐源帧/最近 PTS/源帧号采样、独立检测状态和缓存身份。中心距最近 PTS 等距取前帧，范围外及超出半帧容差仍缺失；有界补检的关键帧定位、优先级和质量门限不变。ffmpeg 镜头视频生成和按需补检仍分别解码，本次仅合并三类检测的首次扫描。

真实原片 `scenes/test_shelf/20261109_002/take_001/left/VID_20261110_181810_00_049.insv`：664 个源帧、332 个查询时刻。修改前独立检测耗时：宽度 40.518 秒、中心距快速扫描 17.924 秒、海报 24.911 秒，合计 **83.353 秒**；合并后 **51.771 秒**，仅打开解码器一次，减少约 **37.9%**。三份压缩 `.npz` 与修改前逐字节一致，数组类型、形状与原始字节亦一致。这是本机单段录像、三类检测首次执行的对比，不代表整条重建流程的加速比例。

新增 7 项回归覆盖共享/独立输出一致、全缓存命中、海报参数或查询时间变化时的独立失效、最近 PTS 边界、解码/完整性校验失败不发布，以及场景准备后消费者无须再次解码。已有相关 104 项回归通过。首轮全量仅新增 PTS 测试失败：Matroska 将时间戳取整到毫秒，使 17 ms 间隔中点超出既有半帧容差；将测试中点改为 16 ms 间隔后通过，未修改生产选帧容差。

对比脚本、原实现副本与结果：`/tmp/umi-shared-decode/{baseline,before,after}`、`baseline_run.py`、`compare_run.py`；对比报告 `after/comparison.json`。测试日志：`focused.log`、`shared-tests-final.log`、`all.log`（首轮夹具失败）、`all-final.log`。以上临时路径均位于 `/tmp/umi-shared-decode/`。

真实场景 `./umi-scene run scenes/test_shelf/20261109_002 --json` 成功，发布地图 `20261112_163931-09b580cd` 和运行 `15dfa2840760458a9def7216806836b8`；重复运行返回 skipped。该批次仅配置中心距，本次实跑验证单类检测入口与完整流程；三类共享路径由上述真实原片对比和场景缓存集成测试验证。新结果 9 项资产校验通过，查看器加载 332 帧及对应地图，中心距有效 257/332 帧（77.4%）、最长连续缺失 15 帧。日志与校验报告：`scene-real.log`、`scene-reuse.log`、`scene-validation.json`，均位于 `/tmp/umi-shared-decode/`。

最终验证：`./test.sh --all` 全部通过，Python **781 项（167.186 秒）**、查看器前端测试、标定与场景 CLI 自检及 KB4 C++ 数值回归通过；`git diff --check` 通过。


## 2026-11-12：清理清单收尾与批次、双手调度优化

按用户“继续，都做干净”的要求完成剩余代码项；用户明确选择批次发现原片问题时整批停止。`processing.run()` 对入库错误、待配对/待补齐和忽略原片立即结束该批次的处理规划；显式任务入口沿用相同规则，其他批次仍可处理。已登记清单和旧成果保留供修正后续跑。

`run_episodes()` 为每批创建独立 `BatchMap`，首次合格任务在锁内构建/校验地图并缓存绑定或失败，后续任务复用，不反复构建、读取地图资产。每个任务仍选择自身标定并与地图核对；复用前检查固定地图、批次/场景/套件配置、建图原片的文件系统身份；发布前核对地图和建图原片，地图引擎变化也拒绝。已开始任务继续使用冻结配置，外部配置修改只阻止后续任务复用。地图构建收到原先遗漏的 `cores` 参数。不同批次锁互不依赖，缓存只存在于本次运行。

`--jobs 1` 且双手可用 CPU 至少 8 个时拆分子进程核心集合，并行定位、融合、测量和各手结果写入；单手、核心不足或多任务并发时串行。单手独立回退语义保留。主线程等待所有工作线程退出，双手全部成功才合并并发布；失败时先等待再归档现场，旧成功入口不变。数值夹具验证串行与并行的两手 trajectory.csv 和 bimanual.csv 逐字节一致，并用同步屏障验证确有执行重叠；不据此宣称真实双手耗时改善。

`source_hash()` 对 Python 无位置信息的 AST 求版本化 SHA256，去掉模块/类/函数文档字符串并忽略注释及排版，覆盖完整算法、地图阶段、宽度/中心距/海报及标定准备/板检测的缓存身份。按源码字节缓存 AST 解析，源码变化即时生效。资产仍用原始字节 SHA256；算法不得依赖源码行号或文档字符串作数值输入。旧指纹首次重新计算，不删除旧地图或结果。

删除批次快照归档写入及无用包装函数；批次只维护当前 manifest，已有成果中的输入和设备副本继续完整保存。确认全仓无归档读取者后，将 scenes 下 26 份旧 snapshots（20,968 字节）移至 `/tmp/umi-clean-final-snapshots/`，保留 inventory.json 并逐文件验证哈希。真实原片、历史地图与运行资产没有删除。rear 工厂推导读取 front 字节一次，同时解析和计算身份，成果导出核对已验收的资产哈希及结果清单哈希；安装/冻结后的副本完整性校验仍保留。

现行用户指南、模块文档、维护说明和清理清单同步更新。训练开合通道缺少实物宽度标定已移至当前待办，中心距不冒充开口；独立结果读取、历史标定包及仍有引用的夹具/PDF 保留并说明用途。

验证期间发现并处理两类测试假设：原测距失败夹具按“第二次调用”指定右手，改为按原片身份定位，允许并发执行顺序变化；新增第二任务夹具补齐外层任务目录。首轮全量 791 项中已有“运行中修改配置仍使用冻结副本”回归暴露新批次检查过严，修正为只在后续绑定前检查外部配置，当前任务发布时仍核对原片与固定地图。针对冻结配置、并行调度、成果导出和标定检测的 33 项回归通过。审查补齐 `vision/decoding.py` 在建图缓存中的依赖，防止海报解码改动后复用旧地图。

首轮修正后的完整检查通过：`./test.sh --all` Python 792 项（172.364 秒）、查看器前端、标定/场景 CLI 自检及 KB4 C++ 数值回归全部通过，日志 `/tmp/umi-clean-complete-all.log`。最后补充建图解码依赖后再次执行完整检查及真实批次回归，最终记录见下。

最终收尾验证：补齐建图解码依赖后，`./test.sh --all` Python **792 项（171.789 秒）**、查看器前端、标定与场景 CLI 自检及 KB4 C++ 数值回归通过；`git diff --check` 通过。真实 `test_shelf/20261109_002` 重新运行成功，地图 `20261112_185204-1060af12`、运行目录 `e360cd2de9f846c2a6d0b4ea21c2cec8`，重复运行返回 `skipped`。结果 9 项资产校验通过，查看器加载 332 帧并成功加载地图；`/tmp/umi-clean-verified-validation.json`。无 `.umi/snapshots/` 目录。


## 2026-11-12：补齐三个入口的命令行参数参考

新增 `docs/命令行参数.md`，覆盖查看器与场景/标定的全部 21 个子命令，列明位置参数、必填项、默认值、取值范围、互斥关系、输出位置和启动环境变量。统一说明配置预览与输入检查、各命令的 force 行为、训练导出的参数作用域、标定来源和资源预算。根 README、文档导航、操作指南、配置说明、模块 README、启动脚本注释和顶层帮助均链接到该参考页。

按实际实现修正帮助文本：批次内部清单路径和 scene.yaml 输入、双手并行的 CPU 条件、skip-preflight 保留批次入库检查、训练格式与 with-video 冲突、宽度诊断拒绝覆盖和数值范围、标定 workers 默认预算、resume/force 互斥。设备升级 dry-run 实际会使用临时目录，改为明确不写设备档案。此次修改集中在帮助与文档。

验证：实际调用三个启动脚本的顶层及全部子命令 `--help`，共 **24 条命令**退出码为 0；对照 argparse 确认所有公开选项有帮助并出现在各程序参考章节，核对 **21 处**相关本地链接与章节锚点。`./test.sh --unit test_effective_configuration test_mapping_cli test_calibration_directory test_calibration_execution test_calibration_inventory test_calibration_workflow` 的 **79 项测试通过（4.392 秒）**；`./test.sh --checks` 标定/场景 CLI 自检和 KB4 C++ 回归通过，启动脚本 Shell 语法与 `git diff --check` 通过。本轮未重新执行真实标定或轨迹重建。

检查脚本与记录：`/tmp/umi-cli-reference-audit.py`、`/tmp/umi-cli-reference-audit.json`、`/tmp/umi-cli-reference-help.log`、`/tmp/umi-cli-reference-tests.log`、`/tmp/umi-cli-reference-checks.log`。


## 2026-11-12：按镜头选择当前参数，简化套件标定配置

套件新增 `intrinsics.front/rear: measured|factory`，默认 front measured、rear factory；用户不再填写标定版本号。读取当前套件时解析为固定参数身份，读取批次/结果快照时保留已冻结的身份，不重新查询当前选择。旧的固定版本套件仍可读取供项目迁移，新旧字段不能混用。

工作镜头按来源维护当前完整参数；环境镜头安装在 registry 锁内发布资产并更新该来源的当前选择。安装一类参数保留另一镜头和来源的选择；工作镜头安装状态也按来源检查，安装 factory 不把仍有效的 measured 报成未启用。前后镜头可以选择不同来源；同一相机的建图与任务仍核对来源、参数哈希和完整身份，不要求不同相机使用同一来源。显式固定地图也提前核对已登记 rear 的参数哈希。

rear measured 缺少当前参数时拒绝；rear factory 优先使用当前已独立验收的整包，未登记时使用既有原片工厂几何与 front B 推导，继续标明 NOT_EVALUATED。当前参数损坏或身份/来源错误不允许退回推导。设备查询默认只显示每个镜头、每种来源的一份当前参数；历史资产在按旧绑定读取时继续校验，未删除原片、旧测量或历史结果。手动 upgrade-device / activate-calibration 命令及无调用的实现已移除；安装自动更新当前选择。

迁移 kit_001、kit_002 为 front measured / rear factory。6 个 test_shelf 批次移除重复的统一 measured 声明，由 kit 指定各镜头来源；迁移前声明备份在 `/tmp/umi-current-calibration-batches-before.json`。首次真实检查发现旧批次的统一 measured 会覆盖新的 rear factory，补齐批次迁移后实跑成功。kit、场景、标定、命令行、配置和开发文档均同步更新。

新增 13 项回归覆盖当前参数自动更新、安装期间配置变化拒绝、历史批次冻结、每镜头/来源单一当前选择、损坏当前参数不兜底、历史损坏不影响当前查询、混合镜头来源、rear 独立验收边界、各来源安装状态、地图参数匹配和已退役命令。旧手动升级/激活专用测试随入口退役；相关 99 项回归通过。最终 `./test.sh --all` **804 项 Python 测试通过（177.628 秒）**，查看器前端、标定/场景 CLI 自检和 KB4 C++ 回归通过。全部 **22 条**顶层/子命令帮助和相关参数文档链接检查通过；`git diff --check` 通过。

真实 `test_shelf/20261109_002` 成功发布地图 `20261112_200536-ace281be`；重复运行返回 skipped。结果 9 项资产哈希通过，查看器加载 332 帧与地图。本轮未执行新相机实物标定，当前设备仍无独立 rear 验收参数。日志 `/tmp/umi-current-calibration-all.log`、`/tmp/umi-current-lens-tests.log`、`/tmp/umi-current-calibration-real.log`、`/tmp/umi-current-calibration-reuse.log`；参数与结果核对 `/tmp/umi-current-calibration-inventory.json`、`/tmp/umi-current-calibration-config.json`、`/tmp/umi-current-calibration-validation.json`。


## 2026-11-12：运行、查看器和夹爪检测默认值集中化

- 新增 `config/operations/runtime.yaml`、`viewer.yaml`、`tag_detection.yaml`，由 `software/configuration/operational_settings.py` 严格校验与进程内冻结。可选同名 `.local.yaml` 已忽略；优先级为项目默认 → 本机覆盖 → 既有环境变量 → CLI。算法 `--config` 含义保持不变。
- 运行配置接入共享/采样解码、场景 OpenCV、标定 worker/预留核/bag 线程、双手并行阈值、SLAM 和地图导出超时。`--show-config` 补充逐项来源、CPU affinity 与实际线程预算。场景/标定主流程输出精简资源摘要；宽度诊断保持原有 CSV/stdout 与 JSON/stderr 契约。
- 查看器端口默认 9092；预览/画面编码、并发、排队、内存缓存与超时集中配置，资源延迟创建且并发首次请求只创建一组工作资源。画质/尺寸影响对应缓存身份，线程/超时不影响；画面 URL 同时包含画质身份。地图显示预算仅抽样响应，不改写已发布资产。
- 夹爪搜索间隔、候选数和补检批量进入原始检测与最终任务结果身份；运行预算不进入数值缓存身份。X3 图像约定、标签字典、低层 ArUco 阈值及内部流水线策略仍由代码管理。
- 全量 `./test.sh --all`：814 项 Python 测试（176.358 秒）、前端测试、CLI 环境检查、KB4 C++ 回归全部通过；日志 `/tmp/umi-settings-all.log`。新增 10 项配置/冻结/资源/缓存边界测试。最后的查看器配置展示补充另经相关模块复验，日志 `/tmp/umi-settings-final-focused.log`。
- 22 个 CLI 帮助入口及参数文档/链接审计通过，记录 `/tmp/umi-cli-reference-audit.json`。
- 真实批次 `scenes/test_shelf/20261109_002` 处理成功，地图 `20261112_205556-d6f6b37f`，任务运行 `b06319dbe0934ee2b84da35b31fd3ece`。9 项发布资产哈希通过，查看器读取 332 帧；地图 5259 点，显示预算降为 1000 时返回 877 点，发布地图哈希保持不变。JPEG 首帧和 WebM 预览有效，重复预览未改写缓存。
- 默认重复运行与将解码/OpenCV 线程改为 2、SLAM 基础超时改为 600 秒后的运行均为 `skipped`，结果指针不变。验证记录 `/tmp/umi-settings-validation.json`，处理和复用日志 `/tmp/umi-settings-real.log`、`/tmp/umi-settings-reuse.log`、`/tmp/umi-settings-runtime-reuse.log`。

## 2026-11-13：批次 .umi 去冗余与自动清理

- 录像准备缓存键只含原片、视频参数、镜头列表、解码代码（`map_cache.DECODE_SOURCES`）和 ffmpeg/ffprobe/numpy；去掉 `registry.py`、标定代码、相机身份与 ORB 参数。此前改这些文件会让同一原片重新解码，`20261109_008` 中建图与任务预处理各有一组逐字节相同的副本（约 360 MB）。任务预处理改用与建图相同的 `MapBuildCache.stage`，键统一为完整 SHA256；只读产物以硬链接放入运行期 `work/` 并设为只读，结束即删，结果 `front.mp4` 与缓存共享 inode。建图、视觉、定位阶段的代码依赖集合保持不变。
- 新增 `scene/management/cleanup.py`：任务结束后只保留 results / local_results 指向的运行、最近一次尝试和验收运行；成功计算后删除本次未命中的缓存条目（命中时 `storage.mark_used` 刷新目录时间）；建图成功后同样清理建图缓存；批次处理结束后删除未被 manifest.yaml 或保留运行引用的地图版本，最新一次建图保留。新增 `umi-scene clean [--cache] [--dry-run]` 对已有数据执行同一规则。被取代的历史运行不再保留，相关测试改为断言其被删除。
- 删除 `maps/current.json`：当前地图只记录在批次 `manifest.yaml` 的 `map`；`_build_scene` 按新到旧扫描已有版本复用，绑定仍由 `build_scene` / `bind_map` 写入清单。`inspect` 改为接受批次目录（此前传场景或批次目录均无法读取）。
- 真实批次 `20261109_008`：首次运行因建图代码指纹变化重建地图（`20261113_160741-f27b5ae9`），`.umi` 由 919 MB 降至 461 MB，各缓存与运行目录各剩一项；再次运行 2 秒全部复用、目录不变；`--force` 重算 11 秒，录像准备命中缓存、缓存条目保持不变、仅删除被取代的运行。

## 2026-11-13：批次内部目录 .umi/ 更名为 .pipeline/

- 原名看不出职责，且与 `results` 的关系不清。新名表示“生产过程的内部记录”，与作为交付入口的 `take_*/results` 对应：`results` 是唯一需要查看的结果，`.pipeline/` 由程序维护。上一条记录中的 `.umi` 即本目录。
- 名称只在 `layout.INTERNAL` 定义；查看器、批次发现、地图导入等原先写死 `'.umi'` 的路径改为引用该常量。测试、用户文档与 `.gitignore` 同步更新；历史记录保持原文。
- 现有 8 个批次用一次性脚本迁移（未入库）：持锁改名，改写 `state.json` 的 `attempt.run` / `review.run` 与 `results` 软链接，`import.json` 路径同步。运行产物中的绝对路径只作追溯且计入资产哈希，未改写。迁移前后任务状态一致，5 份结果资产哈希与查看器读取通过。改名改变了建图代码指纹，`20261109_008` 重跑生成新版本 `20261113_162624-b4b6434e`，各阶段均命中缓存，耗时 52 秒。

## 2026-11-13：.pipeline 内两处残留

- 地图版本 `devices/` 只冻结建图依赖的 `front`、`front_manifest`、`rear`、`mask`（`map_cache.MAP_CAMERA_ASSETS`，与地图相机身份共用）；TCP、宽度和标签测距不再复制进地图。任务运行仍冻结全部资产。
- 任务定位缓存的引擎工作目录移出条目（`.<键>-<uuid>`），成功后删除；失败时保留供错误信息中的日志排查，下次成功处理由自动清理删除。缓存条目只含 `ready.json` 列出的文件。
- 修正自动清理选择“最新一次建图”的规则：版本名为秒级时间加随机后缀，同秒版本此前可能按后缀误判，现按时间前缀再按目录修改时间排序。
- 全量 830 项测试无新增失败；`20261109_008` 实跑通过，地图 `20261113_163707-c4f953d6` 的设备目录与定位缓存条目符合上述结构。

## 2026-11-19：poster_vio 联合优化夹爪 base tag 安装关系

- 在 `feature/cross-camera-base-tags` 中对齐 specialist `da29719` 的 base-tag 图结构：每手一个刚性安装 Pose3，自身角点因子约束安装变量，跨相机角点因子同时连接观察者位姿、拥有者位姿与拥有者安装变量。复用现有 KB4 投影、角点噪声、Huber 和 GTSAM CPU LM；新增因子使用解析 Jacobian，未引入 specialist 依赖或 GPU 求解器。
- 修正首版的环境镜头接线，基座标签从 UMI 工作镜头 front 检测，原始角点缓存与环境海报隔离；基座标签区域不作为自然点重复入图。kit 声明内的 ID 必须唯一，且不能与海报或已配置宽度标签冲突。
- 自身多帧重投影拟合提供安装种子；联合图的软先验来自同一份种子，明确为正则化而非独立测量。先验强度、最大旋转/平移修正与最少自身时刻数集中在 `poster_vio.base_tag` 配置中。相机标定和 TCP 几何保持冻结，不回写登记表。
- 跨手时钟使用 `t_task = t_native_imu - front_offset + video_sync_offset`；双方链显式插入观测物理时刻的节点，使用真实 IMU 约束，不按最近曝光帧吸附。共同范围外观测记录 skipped；纳秒舍入时保留原有观测到状态的对应关系。
- 基座标签异常按完整四角点和原始观测数累计预算剔除，有限轮热启动重解；非有限投影、非正深度、修正超限或残差不合格拒绝结果。冻结 NPZ/JSON 保存原像素、归属、时钟、初始安装与哈希；结果保存每手自身/跨手残差、逐条使用状态、初始/优化安装和修正量。
- 新增实跑 LM、三变量解析 Jacobian 数值差分、非零同步偏移、工作镜头半幅、冻结输入、原始剔除预算、孤立坏观测、安装修正拒绝与纳秒边界回归。完整 Python 回归 **1045 项通过（246.215 秒）**，日志 `/tmp/umi_all_unit.log`；最后的纳秒边界修正及相关流程 **36 项通过（15.178 秒）**，日志 `/tmp/umi_final_targeted.log`。Ruff、compileall、`git diff --check` 与脚本语法检查通过。新增模块加入 CI lint，新增测试加入 quick 档。
- 本次未执行真实双夹爪录像或独立 TCP 真值验收；数值收敛和像素一致性不能替代物理精度验证。当前能力只接入 poster_vio，未把 specialist 的手指关节变量、镜头座在线修正或 GPU LM 一并移植。


## 2026-11-19：base tag 实现与文档去冗余

- 将安装种子字段及冻结 NPZ 键统一为 `T_imu_base_tag`，明确从标签到 IMU 的变换方向；移除跨手因子名中不再有区分意义的 `_variable` 后缀。
- 冻结准备和数值求解入口保留各自的时间节点检查，节点已齐全时直接复用输入，避免重复插值和复制。跨手因子通过伴随变换复用已有位姿 Jacobian，三个变量只进行一次投影。
- 移除准备阶段被覆盖前从未读取的索引计算；标签归属使用单一 ID 反查表；掩膜构造只生成一次整组角点，避免按观测反复分配整组数组。畸形标量时间输入现在由契约层明确拒绝。
- 文档补齐同步偏移对跨手因子的作用、双方精确时间节点及 base-tag 配置入口，删除重复的模块职责段落。
- 清理后 `./test.sh --quick` **222 项通过（42.728 秒）**，日志 `/tmp/umi_base_review_quick.log`；相关求解与输入测试此前 **36 项通过（15.112 秒）**。新增回归验证重复节点扩展不重采样、三个 Jacobian 共用一次投影。Ruff、compileall 与 `git diff --check` 通过。本轮没有重新进行真实录像或独立精度验收。


## 2026-11-19：仅按有效跨手观测启用 base tag 安装变量

- 单手 poster_vio 跳过 base tag 检测、安装初值拟合和安装外参优化。kit 声明仍随冻结相机资产保存；相机标定与实测 TCP 不依赖自身标签拟合。
- 双手继续检测自身/对方标签以发现跨手用途；只有在双方本机时间范围内存在同步跨手观测时，才为被观察方估计安装初值、创建 Pose3 安装变量和相关因子。单向观测只启用被观察方的安装变量。所需标签仍须具备至少配置数量的有效自身观测；缺少初值时明确拒绝。
- 输入准备和数值求解共用有效拥有者判定；即使回放携带自身安装种子，没有有效跨手用途时也不创建孤立安装变量。未使用的自身观测记录 `no_cross_camera_observations`，共同范围外的跨手观测记录 `outside_common_time_span`，不因无用安装的像素残差或自身时刻不足而拒绝轨迹。
- 自身标签仅连接静态安装变量，不再为其增加或保留额外轨迹节点；有效跨手观测继续在双方链中插入精确物理时间节点。原始观测、启用状态、数值输入和质量证据保留可追溯关系。kit 与后端文档及配置注释同步更新。
- 新增六项回归覆盖单手检测/拟合跳过、稀疏自身观测、单向跨手用途、共同范围外观测、所需拥有者缺少自身观测，以及回放中无用安装种子和异常自身残差。相关 **32 项通过（11.866 秒）**；完整 `./test.sh --unit` **1054 项通过（251.321 秒）**，日志分别为 `/tmp/umi_base_activation_targeted.log`、`/tmp/umi_base_activation_all_unit.log`。Ruff、compileall 和 `git diff --check` 通过；未执行真实双夹爪精度验收。


## 2026-11-19：统一使用手解析与配置权威边界

- 新增无 IO 的 `software/scene/arms.py`，集中左右手枚举、子集校验、规范顺序与声明继承；移除 kit 中的旧实现和 collection schema 的重复成员/去重校验，入库、任务读取、批次快照与配置展示直接使用同一规则。
- 当前 collection 的 `recording.arms` 继续必填，套件选择入口移除不可达的 kit 默认回退。任务 `arms` 必须是批次使用手的非空子集，schema 提前拒绝；入库与读取共用 `resolve_arms()`，任务读取仅从 collection 任务声明解析使用手。
- 新批次快照始终写出已解析 `expected_arms`；旧快照省略时由统一解析器从快照内的冻结 kit 恢复，不查询当前设备配置。原片不足继续保持不完整，不推断成单手。
- 修正场景处理文档中根级 arms 和可省略使用手的旧说明；示例任务省略重复的双手声明，明确批次继承；SSOT 和代码指南补充使用手权威边界。顺带移除修改到的测试文件中的重复/未用导入和无效局部变量。
- 新增 schema 必填/子集/畸形值检查，以及同一 take 修改覆盖后入库和读取同时恢复批次继承的回归；确认任务清单不保存第二份使用手声明。相关 **67 项通过（10.516 秒）**，场景/管理/配置/回退/地图/海报发布 **97 项通过（38.398 秒）**，最终规则复验 **17 项通过（2.061 秒）**。日志 `/tmp/umi_arms_ssot_targeted.log`、`/tmp/umi_arms_ssot_workflows.log`、`/tmp/umi_arms_ssot_final_rules.log`；Ruff、compileall 和 `git diff --check` 通过。


## 2026-11-19：收敛 YAML 配置入口、用途权限和生效来源

- 修正文档中误标的 collection schema、已退役的 scene.tags 海报示例、旧用户内参来源字段及 kit 固定版本说明；文档海报示例新增真实 public schema 解析检查。多地图采集顺序明确由 Organizer 交接或 plan-epochs --write 登记，map_epochs.yaml 是生成规划快照，不再指导用户手工填写内部状态。
- 配置展示保留 take 目标，消费统一的使用手继承规则，展示任务后端、同步、地图请求及任务/批次范围；原片目录、集中清单及清单目录三个入口给出一致结果，不读取原片、不生成清单或运行目录。批次展示继续保留批次默认值。
- 查看器传递逐字段覆盖来源，区分 user.yaml 与 command_line，并将 user_config 纳入操作配置的优先级报告；CLI 继续优先于本机偏好，本机偏好继续优先于 operation/local 默认值。
- 当前 expert-config 拒绝覆盖海报建图/尺度/融合用途及融合方式，统一由 collection 声明；数值调参仍可用。历史 SceneOverrides.from_snapshot 继续读取已冻结覆盖以复现旧结果。
- 当前 kit 只接受 serial、intrinsics 和 gripper，来源按 front/rear 当前选择解析；固定 calibration/calibrations 字段与旧来源协议只留在资产快照读取分支。迁移公共测试夹具为当前 kit 格式，保持旧快照、来源校验、错误资产拒绝与独立镜头选择覆盖。
- 新增 user.defaults.yaml 作为用户偏好的唯一默认值来源，user.yaml 只保存本机覆盖；移除 Python 中重复的默认值和 v1 当前用户入口。现有工作区 user.yaml 原有值均为默认值，迁移后运行选择不变；用户配置报告记录默认值来源。
- 更新仍依赖旧人工 kit/专家用途开关的测试，以及使用手快照显式 expected_arms 的旧断言。相关入口 **76 项通过（11.354 秒）**、最终配置展示 **8 项通过（0.423 秒）**、最后修正的设备清单/快照 **17 项通过（1.268 秒）**。最终完整 ./test.sh --unit **1061 项通过（251.622 秒）**，日志 /tmp/umi_yaml_cleanup_final_all_unit.log；修改文件 Ruff、compileall 和 git diff --check 通过。配置读取及报告相关源文件已加入 CI lint。


## 2026-11-19：任务配置共用解析结果与后端能力推导

- 新增 management/task_configuration.py 的 ResolvedTaskConfiguration，统一解析使用手、指令、后端、同步和地图请求/选择；同步解析从 episode reader 移入该边界，执行读取与 --show-config 均消费同一结果。collection 与运行策略共用后端枚举。
- 地图选择保留 latest-prior、显式版本、显式 epoch 与批次默认规则；展示选中版本、epoch、选择原因及 ready/pending/unavailable 状态。任务展示读取批次已登记的默认地图，修正过去只展示请求或遗漏冻结默认值的问题。解析不读取原片或地图资产。
- epoch_selection 返回未构建选择的 pending 状态，正式 scene_map 执行在读取原片前拒绝未就绪地图；auto 保留决策，并由既定 MapUnavailable 回退边界处理 pending/unavailable。损坏的规划清单仍直接拒绝。命令行 epoch 与任务版本/epoch 声明冲突时明确报错。
- poster_vio、poster_independent 与 auto 从任务 backend 推导必需的海报检测和任务约束能力，只要求提供有效 layout_path；显式 detection.enabled:false 或 task_constraint.enabled:false 与后端冲突，不静默覆盖。任务级推导不启用其他 scene_map 任务的海报用途；scene_map 声明用途时可省略检测总开关。相机参数与 TCP 规则未改。
- 任务与批次配置展示均显示推导后的能力；无地图后端继续忽略地图规划。模板和操作指南提供可通过公共 schema 的最小海报配置，不再重复要求必需开关。
- 新增八项对照回归，覆盖选择规则与批次默认、pending 展示/执行拒绝、auto 决策、最小海报配置、显式关闭冲突、任务隔离与无效 epoch。完整 ./test.sh --unit **1069 项通过（251.886 秒）**，日志 /tmp/umi_task_resolution_all_unit.log。最后补齐批次级能力展示后，相关 **106 项通过（19.932 秒）**，日志 /tmp/umi_task_resolution_final.log；修改文件 Ruff、compileall 与 git diff --check 通过。解析模块加入 CI lint，测试加入 quick 档。


## 2026-11-19：补齐 YAML 字段说明与候选值注释

- 为 `config/user.defaults.yaml` 和 `config/user.yaml` 补充用户偏好、默认/覆盖关系、布尔候选、端口范围、CPU 范围及特殊值说明。
- 为 kit 与 collection 示例补充设备来源、资产路径、单/双手、地图布局、地图选择、海报用途、后端、同步、任务覆盖和 base tag 的含义、候选值、单位、继承与冲突规则；实际 test_shelf collection 同步补充注释。
- 为轨迹、标定、scene_parameters 的未充分说明字段补充单位、正数/范围、枚举、零值语义和字段关系说明；运行、查看器和夹爪检测 YAML 的资源范围与特殊值也明确写在字段旁。
- 新增文档契约：扫描 config、kit、scene collection 的所有 scalar YAML key，若未提供行尾注释则测试失败；现有 YAML 解析后语义与注释前快照完全一致。
- YAML 注释检查相关 **37 项通过（1.231 秒）**，quick **242 项通过（46.900 秒）**；Ruff、compileall 与 `git diff --check` 通过。注释不会改变配置指纹或运行行为。


## 2026-11-19：SSOT 最终核对与 epoch 地图绑定修正

- 对照 SSOT 表逐项核查 collection、kit/registry、user defaults、task resolution、map epochs、manifest、episode 与结果边界。确认当前用户声明、设备来源、使用手、任务后端/同步和地图选择均有单一解析入口；快照/结果/缓存只承担冻结、审计和复现。
- 发现并修正 workflow 的一个边界：`Episode.map_selection` 由 epoch 规划选出 `collection_sequence` 地图版本时，旧 `_requested_map_version()` 没有把该版本传给 scene-map 绑定层，可能回退到批次旧地图。现在 collection sequence、显式 task version/epoch 都传递同一个解析版本；batch_default 仍由批次绑定层管理。
- 扫描并修正文档中残留的旧 `collection.yaml.scene.tags`、旧 `config/user.yaml.calibration.intrinsics_source` 和过时的 scene/vision/标定说明；代码内部 `scene.tags` 仅作为 normalize 后的内部结构，不再作为用户入口。
- 相关 task resolution、地图规划、scene workflow 与文档契约 **62 项通过（25.622 秒）**，此前完整 Python 回归 **1069 项通过**；工作流修正后执行/配置展示对照通过。Ruff、compileall、`git diff --check` 通过。
