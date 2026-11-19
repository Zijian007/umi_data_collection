# 轨迹交付包

| 格式 | 用途 | 内容与入口 |
|---|---|---|
| `trajectory`（默认） | 轨迹查看、交付与排障 | 轨迹 CSV、报告、元数据，视频可选；见下方示例 |
| `training` | 供训练适配器读取的中间包 | 统一采样的 `samples.npz`、视频、坐标声明与质量数组；见[训练数据指南](../docs/训练数据.md) |

两种格式都要求当前成功结果已人工接收。训练格式还检查指令、双手同步与 TCP；`training_ready` 的具体含义见训练指南。

通过 `umi-scene export` 导出已验收的轨迹结果；需要同时携带视频时使用 `--with-video`：

```bash
./umi-scene export scenes/shelf/20261104_001 --with-video --out exports/shelf_20261104_001
```

输出示意：

```text
exports/shelf_20261104_001/
├── manifest.json
└── shelf/20261104_001/take_001/
    ├── report.txt
    ├── episode.json
    └── left/
        ├── trajectory.csv
        └── front.mp4
```

上面是目录示意。真实文件需要从已人工验收的成功结果导出，指定的具体输出目录应尚不存在。
