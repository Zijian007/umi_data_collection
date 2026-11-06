# 训练中间包

通过同一个 `umi-scene` 程序导出训练包，自动携带实际使用手的视频：

```bash
./umi-scene export scenes/shelf/20261104_001 --format training --out datasets/shelf_20261104_001
```

输出示意：

```text
datasets/shelf_20261104_001/
├── manifest.json
├── export_report.json
└── episodes/shelf/20261104_001/take_001/
    ├── episode.json
    ├── samples.npz
    └── videos/left.mp4
```

上面是目录示意。真实数据由导出器生成，输入需通过人工验收并补齐操作指令和双手同步信息；宽度缺失或无效时包会标记训练未就绪。

视频由原始 INSV 提取，处理选项见[视频参数](../config/README.md#视频处理参数)。训练包内的帧索引与 TCP 共享样本时间轴；字段和读取方式见[训练数据](../docs/训练数据.md)。
