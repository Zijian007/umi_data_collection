# 标定代码阅读指南

先读 [`cli.py`](cli.py) 的 `main()`，再进入 [`workflow.py`](workflow.py) 的 `run()` 或 `verify()`。跨模块调用写出模块名，阶段间通过 [`models.py`](models.py) 的数据结构传递结果。

## 按执行顺序阅读

```text
__main__.py
  └─ cli.main()
      ├─ inputs.parse_run_args()          命令行值 → CalibrationRequest
      └─ workflow.run(request)
          ├─ runtime.prepare()            检查依赖，设置 CPU
          ├─ inputs.prepare_inputs()      确认 A/B/C/D、序列号和板尺寸
          ├─ runtime.prepare_output()     检查并准备输出目录
          ├─ recording_data.extract_recordings()  INSV → 时间戳、IMU、单镜头图像
          ├─ intrinsics.solve_intrinsics()        A → 镜头内参
          ├─ camera_imu.solve_camera_imu()         B + A 内参 → 外参、时间差
          ├─ validation.validate_gravity()        D → 重力一致性测量
          ├─ camera_imu.build_bundle()            参数 → 标定包
          ├─ validation.validate_trajectory()     C → 轨迹误差
          ├─ checks.verdict()                     检查项 → 最终结论
          └─ report.finish()                      写入并展示报告
```

`verify` 同样直接进入 `workflow.verify()`，其中依次展示 `recording_data.build()` → `board_pnp.detect_all()` → `validation.trajectory_check()`。录像提取、板检测和误差测量各自的调用位置都在流程中可见。

轨迹比较直接接收 `StageContext`，标定 JSON、标定包和本次比较名称使用命名参数：

```python
validation.trajectory_check(
    video, detections, context,
    calibration_json=calibration_json, bundle=bundle, tag="current",
)
```

路径、板配置和 CPU 设置从 `context` 读取，调用方不再拆成一长串位置参数。

`prepare_inputs()` 先读取录像身份信息；真正解析时间戳、IMU 并解码画面的是 `extract_recordings()`。A/B/C 解码画面，D 只提取时间戳和 IMU。

每个阶段返回结果后，`workflow.py` 调用 `reports.py` 记录证据、调用 `checks.py` 生成检查项。检查项中的 FAIL 会影响最终结论；只要阶段本身还能执行，流程继续收集后续验证结果。求解或读取失败则停止，保留已完成的证据并生成 FAIL 报告。

## 文件职责

| 文件 | 负责什么 |
|---|---|
| [`cli.py`](cli.py) | 参数解析、子命令分发 |
| [`workflow.py`](workflow.py) | 阶段顺序、结果衔接、失败报告和工作目录清理 |
| [`models.py`](models.py) | 请求、路径、阶段结果和阶段异常 |
| [`inputs.py`](inputs.py) | 找到 A/B/C/D，检查是否同一台相机，读取板尺寸 |
| [`recording_data.py`](recording_data.py) | 录像转 EuRoC 数据，统计 IMU 运动量；原 `dataset.py` |
| [`intrinsics.py`](intrinsics.py) | A 段内参求解及覆盖度测量 |
| [`camera_imu.py`](camera_imu.py) | B 段相机–IMU 求解、生成标定包 |
| [`validation.py`](validation.py) | D 重力检查、C 轨迹误差测量、单段 `verify` |
| [`checks.py`](checks.py) | A/B/C/D 验收阈值、检查项和最终 PASS/FAIL |
| [`reports.py`](reports.py) | 阶段结果转报告字段，写入 TXT/JSON，展示启用状态 |
| [`runtime.py`](runtime.py) | 依赖、子进程、日志、CPU 设置和输出目录准备 |
| [`devices.py`](devices.py) | 显式安装、查看、停用设备档案 |

辅助代码和安装资源分为三个目录，顶层保留上述主流程模块、`README.md` 和 Python 包入口：

```text
software/calibration/
├── ...                         上表中的 12 个主流程模块
├── README.md
├── __init__.py
├── __main__.py
├── algorithms/                 底层计算，供阶段模块导入
│   ├── __init__.py
│   ├── aprilgrid_detect.py     AprilGrid 检测
│   └── board_pnp.py            板位姿、覆盖度及轨迹对齐
├── tools/                      独立运行，由阶段模块调用
│   ├── __init__.py
│   ├── calib_folder_to_bag.py  EuRoC 转 ROS bag
│   └── imu_calib_physical_check.py  重力一致性检查
└── setup/                      安装依赖与求解器资源
    ├── install.sh
    ├── requirements.txt
    └── basalt/                 预编译包、补丁和来源说明
```

需要理解某项计算时再进入 [`algorithms/`](algorithms/)；需要排查转 bag 或物理检查进程时进入 [`tools/`](tools/)。安装程序、依赖版本和第三方二进制集中在 [`setup/`](setup/)。

## 改动从哪里下手

- 改四段录像的查找方式：`inputs.py`。
- 改 INSV 提取或画面裁剪：`recording_data.py`；底层时间戳与 IMU 解析在 `software/camera/x3/extraction.py`。
- 改 Basalt 参数：`intrinsics.py`、`camera_imu.py`。
- 改通过标准：`checks.py`；改误差测量方法：`validation.py`、[`algorithms/board_pnp.py`](algorithms/board_pnp.py)。
- 改报告字段或排版：`reports.py`。
- 改执行顺序或失败收尾：`workflow.py`。

`StageContext` 传递路径、已解析的算法参数和执行配置；阶段结果显式传给下一阶段。阶段函数不修改报告，也不写设备档案。

CLI 负责将参数转换成请求，`models.py` 只定义数据。样例参考参数由 workflow 加载后显式传给检查与报告；生成标定包直接使用输入录像信息。报告不作为求解参数的来源，最终结论和退出码由 workflow 决定。

跨模块命名、终端输出与验证约定见 [代码维护](../../docs/dev/README.md)。

`run()` 和 `verify()` 各自维护 `check_results`，直接用 `checks.verdict()` 判定；报告对象只在 `finish(..., check_results=...)` 时接收检查项。
`CalibrationReport` 与 `VerificationReport` 都由 `reports.py` 负责组装、计时、落盘和展示，workflow 不再手工拼验证报告字典。

## 输出与失败处理

输出保持 `report.txt`、`bundle/`、`diagnostics/`。默认清理 `diagnostics/work/`，`--keep-work` 可保留。阶段执行出错时，JSON 报告的 `failed_stage` 标明阶段，`diagnostics/logs/stage_error.log` 保存异常堆栈；已生成的求解器日志继续保留。

两个入口均按“记录阶段结果或失败 → 判定 → 写报告 → 最后清理工作目录”的顺序执行。清理位于外层 `finally`，中断时也会执行；`--keep-work` 仍会保留工作数据。
单段验证分别记录“读取 C 段”“C 标定板检测”“C 轨迹验证”，TXT 和 JSON 都展示失败阶段。全套标定中，验收函数等阶段外的异常也会保留已完成的测量并生成 FAIL 报告。

输入身份检查发生在准备输出目录之前，四段来自不同相机时，即使指定 `--force` 也不会清空已有结果。

运行方式不变：

```bash
./umi-calib run recordings/IAQEB26097M4CP/calib_20261022
# 查看 report.txt 后，手动启用：
./umi-calib install recordings/IAQEB26097M4CP/calib_20261022/umi_calib_out
```

`run` 默认不启用标定。原有显式 `--install` 选项保留，仅在检查通过时执行。

C 段默认只验证新标定。需要额外跑样例对照时加 `--compare-sample`；两次计算共用板检测结果。样例结果只作 INFO 对照，不参与验收；对照失败单独记入报告和日志。
不运行样例对照时，详细报告记录 `compare_sample: false`，不生成 `C_factory` 字段；原有含 `C_factory` 的结果仍可读取和安装。

手动安装依赖使用 `bash software/calibration/setup/install.sh`。正常使用 `./umi-calib` 时仍会按需调用安装程序。单独排查辅助工具可运行：

```bash
.deps/calib_env/bin/python -m software.calibration.tools.calib_folder_to_bag --help
.deps/calib_env/bin/python -m software.calibration.tools.imu_calib_physical_check --help
```

回归测试模拟录像提取和外部求解器，实际执行编排、验收、标定包生成和报告写入：

```bash
.deps/env/bin/python -m unittest discover -s tests -v
./umi-calib check
```

真实标定的录制和操作见 [`docs/标定/说明.md`](../../docs/标定/说明.md)。

## 配置入口

[统一参数配置](../../config/README.md) 由 workflow 在所有副作用前加载并校验，A/B/C/D 阶段接收相应的参数组。`checks` 将原始指标按当前门限判定，`reports` 保存完整配置和原始指标。`--dry-run` 只读预检；`recheck` 仅从完整报告重新验收，不调用算法或安装标定。新设备包只含测量 JSON 与来源元数据。
