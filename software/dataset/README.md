# 训练中间包代码

操作与字段见 [训练数据指南](../../docs/训练数据.md)。由 `./umi-scene export <路径> --format training --out <输出目录>` 调用，无独立启动程序。

| 文件 | 职责 |
|---|---|
| `source.py` | 通过 scene 的只读接口取得当前已验收结果，检查指令/同步，发布前再核验 |
| `sampling.py` | 一次读取位姿/宽度统一表、帧映射及视频 PTS，构造共同时间轴、按位置支持筛选并在缺口处分段 |
| `gripper.py` | 读取场景宽度结果并对齐训练时间，保留有效性、来源及缺失值 |
| `coordinates.py` | 校验坐标声明并提供带参考系的 `read_pose_series()` 位姿视图 |
| `policy.py` | 质量筛选策略的公共取值与入口校验 |
| `schema.py` | 训练包格式、版本、时间轴和数组字段的共享契约 |
| `reader.py` | 只依赖 NumPy 与标准库读取训练包、校验哈希和坐标；不加载场景或求解器 |
| `export.py` | `build()` 管理整包校验、事务与发布；`_write_episode()` 写入一集样本、视频、元数据及哈希 |

`scene.management.episodes.episode_paths()` 和 `accepted_run()` 是共用读取边界；本模块不导入场景 CLI、workflow、融合算法或 wholebody。几何插值复用 `camera.geometry`，历史 TUM 读取复用 `trajectory.io.tum`；新结果只读取一次 trajectory.csv，由 ArmSamples 携带宽度序列供 gripper 模块采样。

样本保留实际 TCP 定义和本集参考系中的位姿（固定场景或任务海报坐标），不提前绑定模型的动作预测长度。宽度标定未配置或导出时间轴上仍有无效样本时标记为未就绪；读取时可用 `require_training_ready=True` 检查。测试使用实际编码、解码的小视频检查索引、时间、可移动读取及失败不发布。

支持 `quality_policy="all" / "position_supported"` 与 `unknown_support="drop" / "error"`。默认保留旧采样行为；策略与删除原因随包保存。导出还可用 `--start-s/--end-s` 限定人工确认的训练时间窗口，窗口外样本不会进入训练包。磁盘位姿数组名保持兼容，消费端通过 `coordinates.read_pose_series(metadata, arrays, arm)` 获取位姿及参考系，避免从历史字段名推断坐标。
