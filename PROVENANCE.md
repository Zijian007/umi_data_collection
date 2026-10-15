# 引擎来源

轨迹引擎从实验室仓库原样拷贝，不带该仓库的提交历史。

- 仓库：`github.com/Torbjorn-s-Apple-Americano/umi`
- 分支：`dev_Zijian`
- 提交：`71a124d640b998dbbfcf35504dea830e179eddfc`（短号 `71a124d`）

拷贝的路径：

- `software/camera/`
- `software/trajectory/`
- `software/calibration/`
- `software/__init__.py`
- `install.sh`、`umi-traj`、`umi-calib`、`Dockerfile`、`.dockerignore`
- `docs/录制说明.md`、`docs/使用说明.md`、`docs/标定说明.md`、`docs/原理与精度.md`
- `QUICKSTART_X3_STREAM.md`

这些路径里的算法文件与该提交逐字节相同。本仓库另外加了 `todo.md`、本文件、`.gitignore`，以及 `software/trajectory/calib/README.md`（说明 `calibration_r1440.json` 是实验室给某一台 X3 标定的样例，不是厂家出厂标定，也不是众包默认标定）。

此后如果改融合、ORB-SLAM3 封装、X3 读取或 Basalt 标定，在本文件追加一笔：日期、改了什么、为什么改。
