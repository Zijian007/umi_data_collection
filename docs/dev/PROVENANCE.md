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
- `docs/轨迹/录制.md`、`docs/轨迹/使用.md`、`docs/轨迹/原理.md`、`docs/标定/说明.md`（当初拷来时是四份平铺的说明，后来按两块功能收进这两个目录）
- `docs/dev/QUICKSTART_X3_STREAM.md`

初次导入时，上述路径里的算法文件与该提交逐字节相同；此后的修改按下文记录。本仓库另外加了 `docs/dev/todo.md`、本文件、`.gitignore`，以及 `software/trajectory/calib/README.md`（说明 `calibration_r1440.json` 是实验室给某一台 X3 标定的样例，不是厂家出厂标定，也不是众包默认标定）。

此后如果改融合、ORB-SLAM3 封装、X3 读取或 Basalt 标定，在本文件追加一笔：日期、改了什么、为什么改。

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
`software.registry.UnknownCameraError` 是 `RegistryError` 的子类，既有调用者仍可捕获父类。
验证报告的 `calibration` 对象增加显示用的 `name` 字段，原字段保留。

外围回归检查（不运行真实视频求解）：

```bash
.deps/calib_env/bin/python -m unittest discover -s tests -v
./umi-calib check
```

## 2026-10-22 标定目录自动识别

`umi-calib run <目录>` 支持 A/B/C/D 子文件夹各放一个原始 `.insv`，保留平铺四段按文件名排序的输入方式。布局混用、缺段或同段多个视频时明确报错，输入发现逻辑位于 `software/calibration/inputs.py`，求解与验收流程不变。

## 2026-10-22 标定产物精简

结果顶层保留 `report.txt`、`bundle/`、`diagnostics/`；详细 JSON、板配置、A 阶段结果、D 检查、日志与可选工作目录集中到 `diagnostics/`。
B 阶段成功后将最终参数移入 `bundle/calibration_r1440.json`，不再保存顶层和 B 阶段的重复副本；保留求解器、阈值、标定包格式和手动启用方式。
报告优先展示失败项、测量或覆盖提醒，再展示样例参数差异；加入生成时的设备档案核对快照。安装兼容旧 `report.json` 位置，并刷新本地和设备档案中的文本报告。
当前 M4CP 的既有输出也按相同布局整理；原始录像和标定包不变。

## 2026-10-22 轨迹外围流程与结果复用

`software/trajectory/cli.py` 负责参数和批量调度，单视频流程移入 `workflow.py`。
`models.py` 定义运行选项，`selection.py` 将设备档案转换为有明确字段的标定选择及资产路径，`reuse.py` 集中判断旧结果是否匹配。
重试和评分位于 `runner.py`，诊断发布与清理位于 `artifacts.py`，输入发现位于 `inputs.py`，环境诊断位于 `runtime.py`。
标定选择优先级保持为 `UMI_TRAJ_CALIB`、`--device`、视频内部序列号；`software.registry.resolve_insv` 的原有字典 API 保留供标定模块使用。

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

标定和轨迹使用一致的请求、序列号、保留中间文件及失败信息命名。当前单段轨迹入口为 `software.trajectory.workflow.run_recording`，请求类型为 `TrajectoryRequest`。`software/console.py` 统一终端格式，各阶段与报告模块负责业务文案；移除旧打印函数、设备状态转发函数及数据对象中的终端描述方法。CLI 参数与 JSON 字段保持兼容，数值算法和引擎日志协议保持原样。

## 2026-10-24 文档整理

README 按安装、标定并手动启用、任务录制、抽取、查看排列。公共安装与设备规则集中到 docs/；轨迹命令与结果字段分开，查看器操作移至 docs/查看.md。代码命名、终端约定和测试入口集中到 docs/dev/README.md。旧建仓计划和尚未迁入的实时预览教程移入 archive/，当前待办只列尚未完成事项。修正过时的三步进度、仅重建图的重试描述和旧终端报错；运行代码、设备档案和既有产物未改动。
