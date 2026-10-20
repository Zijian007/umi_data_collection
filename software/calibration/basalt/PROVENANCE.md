# 自带的 Basalt 标定程序是哪来的

`umi-basalt-0.1.7-fsoffset-linux-x86_64.tar.xz`（22 MB，sha256 `c531b065fde3d51b4a188168eb3867cef3ec17338ae11f9397d0a5236c5aa4c2`）
由 `install_calib.sh` 解压到 `<deps>/basalt/`。里面是：

| 文件 | sha256 |
|---|---|
| `bin/basalt_calibrate` | `99146bfa9b731ce10bf876db8baa270e3166bc182b00874114aa28cee1195f9f` |
| `bin/basalt_calibrate_imu` | `8b437691e12455798488c22c246ac0f34a90e2e98db0eb40fe8772b9b6d4a43e` |
| `lib/libbasalt.so` | `febf3d3d0f53cb5fd761bf9f4024783675c0990adebac9b98d15514a753ff884` |

这两个程序就是算出样例标定（`software/trajectory/calib/calibration_r1440.json`，2026-09-04）的那两个文件，逐字节相同。

## 源码

- 官方 Basalt：<https://gitlab.com/VladyslavUsenko/basalt>，标签 `0.1.7`（提交 `6d8637b9d68ea18a1a63c1baa72818779e156932`），BSD 3-Clause。
- 本仓库的补丁：同目录 `umi_basalt_patch.diff`（`git diff 6d8637b9..7573abd`，460 行），内容：
  1. `src/calibrate_imu.cpp`：新增 `--fixed-scale-opt-offset`（以及 `--fixed-accel-scale`）。官方流程在第二阶段同时放开 IMU 尺度和相机时间差，
     在 X3 的 IMU 上会把加速度计尺度压扁，得到错误的相机–IMU 平移（偏十几厘米）；这个选项让第一阶段收敛后**只**放开相机时间差，IMU 尺度始终不优化。
  2. `include/basalt/optimization/*.h`：在日志里额外打印精确的重投影 RMS（`RMS_REPROJ_EVIDENCE` 行），不改变优化结果。
  3. 构建相关：去掉 `-Werror`、去掉可选的 realsense2 依赖、opengv / pangolin 的编译选项调整（用较新的 gcc 才能编过）。

## 怎么编的

2026-08-26/27，Ubuntu 20.04（glibc 2.35）x86_64，conda-forge gcc 12，Basalt 自带的 vcpkg 流程，Release。
从源码重编：检出官方 `0.1.7`，`git apply umi_basalt_patch.diff`，按官方 README 的 vcpkg 步骤构建（依赖多，下载量大，耗时一小时量级）。

## 为什么自带运行库

程序依赖 glibc 2.34，Ubuntu 20.04 自带的是 2.31，直接运行会报找不到符号。所以压缩包里带了编译机上的动态加载器和全部依赖库
（`lib/ld-linux-x86-64.so.2`、glibc、libstdc++、libgcc、X11 / Wayland / EGL 的客户端库），`umi-calib` 用
`lib/ld-linux-x86-64.so.2 --library-path lib bin/<程序>` 的方式启动，不使用系统的库，也不改系统。已在 glibc 2.31 的机器上实测可运行。
标定只用 `--no-gui` 模式，不需要显示器。

各库的许可证：glibc（LGPL-2.1+）、libstdc++ / libgcc（GPL-3.0 带运行时库例外）、libX11 / libxcb / libXau / libXdmcp / wayland / libxkbcommon / libffi / libglvnd（MIT 类）。
这些库未经修改，对应源码可从各自上游或 conda-forge / Ubuntu 的源码包获得。
