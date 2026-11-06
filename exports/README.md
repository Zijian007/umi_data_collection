# 轨迹交付包

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
        ├── traj_tcp.txt
        └── front.mp4
```

上面是目录示意。真实文件需要从已人工验收的成功结果导出，指定的具体输出目录应尚不存在。
