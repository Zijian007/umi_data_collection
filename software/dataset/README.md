# 训练中间包代码

操作与字段见 [训练数据指南](../../docs/训练数据.md)。入口为根目录 `umi-dataset`，不增加 `umi-scene` 的训练格式参数。

| 文件 | 职责 |
|---|---|
| `cli.py` | `build` 参数、结果与错误输出 |
| `source.py` | 通过 scene 的只读接口取得当前已验收结果，检查指令/同步，发布前再核验 |
| `sampling.py` | 读取 TUM、帧映射及视频 PTS，构造共同时间轴和质量数组 |
| `gripper.py` | 读取场景宽度结果并对齐训练时间，保留有效性、来源及缺失值 |
| `export.py` | 原子发布中间包、文件哈希、独立读取 |

`scene.episodes.episode_paths()` 和 `accepted_run()` 是共用读取边界；本模块不导入场景 CLI、workflow、融合算法或 wholebody。几何插值复用 `camera.geometry`，TUM 读取复用 `pose_io`。

样本保留实际 TCP 定义和绝对场景位姿，不提前绑定模型的动作预测长度。宽度标定未配置、未核验或导出时间轴上仍有无效样本时标记为未就绪；读取时可用 `require_training_ready=True` 检查。测试使用实际编码、解码的小视频检查索引、时间、可移动读取及失败不发布。
