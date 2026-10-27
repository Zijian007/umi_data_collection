# 代码维护

[文档导航](../README.md) · [当前待办](todo.md) · [来源与变更历史](PROVENANCE.md)

## 从入口沿流程阅读

| 流程 | 阅读入口 |
|---|---|
| 标定 | [标定代码指南](../../software/calibration/README.md)：`cli.main()` → `workflow.run()` / `verify()` |
| 共享场景 | [场景代码指南](../../software/scene/README.md)：`cli.main()` → `workflow.build_scene()` / `run_episode()`（架构子包：`management/`, `spatial/`, `pipeline/`） |
| 轨迹抽取 | [轨迹代码指南](../../software/trajectory/README.md)：`cli.main()` → `workflow.run_batch()` → `run_recording()`（架构子包：`calibration/`, `pipeline/`, `reporting/`, `algorithms/`） |
| 三维查看 | [查看器代码指南](../../software/viewer/README.md)：`server.main()` → `RecordingCatalog` → 浏览器播放状态 |
| 共用视觉 | [视觉代码指南](../../software/vision/README.md)：录像准备、ORB 建图/定位、视觉上下文与执行验收 |
| 相机数据 | [extraction.py](../../software/camera/x3/extraction.py) 提取时间戳和 IMU；[frame.py](../../software/camera/x3/frame.py) 定义录制几何、裁剪、帧率和时钟约定 |
| 设备选择 | [registry.py](../../software/registry.py) 读取内部序列号、选择标定与夹爪并核对哈希 |

`workflow.py` 展示顺序、阶段衔接与失败收尾；阶段函数返回具名结果。标定的 `checks.py` 决定验收，`reports.py` 负责展示。第三方进程、依赖与日志放在 `runtime.py`，数值计算在 `algorithms/`，独立命令在 `tools/`，安装资源在 `setup/`。

## 参数管理

见 [统一参数配置](../../config/README.md)。数值默认值写在 `config/*.yaml`，共享加载、校验、快照和指纹在 `software/config.py`。workflow 加载一次，向阶段传递不可变的参数对象。修改参数后同时核对复用、报告与默认数值回归。

## 命名和终端约定

外围代码使用 `request`、`camera_serial`、`keep_work`、`return_code`、`reason`、`error`。数学符号使用对应领域的惯例。既有 CLI、JSON 与第三方接口保留其约定名称，例如 `--serial`、JSON 的 `serial` / `why`、`subprocess.returncode`。

[console.py](../../software/console.py) 统一输出格式，调用为 `runtime.console.write(...)`。阶段负责进度，报告负责结果；标签依次表示工具、录像、步骤、重建次数和执行状态。标定验收 `PASS` / `FAIL` 与执行“完成 / 失败”分别展示。引擎原始日志保留原文。

## 验证

完成[安装](../安装.md)与标定组件安装后，可直接通过根目录的统一脚本一键测试：

```bash
./test.sh        # 一键测试：主单元测试 (unittest) + 夹爪历史测试 (pytest) + 前端测试 (node)
./test.sh --all  # 包含 CLI check 运行时检查
```

也可分项执行各环境原生命令：

```bash
.deps/calib_env/bin/python -m unittest discover -s tests -q
.deps/env/bin/python -m pytest -q -p no:cacheprovider software/gripper_width/legacy/tests
node --test tests/js/viewer.test.mjs
./umi-calib check
./umi-traj check
```

Node 只用于查看器前端测试，运行查看器不需要 Node。Python 测试覆盖输入、标定验收、流程、复用、重建协议、产物布局与查看器；涉及求解器和 SLAM 的单元测试使用替代引擎。真实录像端到端回归和独立误差评估需另外记录输入、设备标定、版本及结果。

## 文档怎么维护

操作步骤写在用户指南，字段定义写在结果说明，模块关系写在对应代码目录的 README。修改参数、终端文案、目录或失败行为时，同步对应指南；用明确链接代替“见第几节”。历史变更只写入 `PROVENANCE.md`，不作为当前操作入口。

历史资料：[建仓计划](archive/建仓计划.md)、[未迁入的实时预览方案](archive/QUICKSTART_X3_STREAM.md)。实时预览与当前 `umi-view` 的离线轨迹回放是两套功能。
