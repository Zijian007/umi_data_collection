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
- `docs/轨迹/录制.md`、`docs/轨迹/使用.md`、`docs/轨迹/原理.md`、`docs/标定/说明.md`（当初拷来时是四份平铺的说明，后来按两块功能收进这两个目录）
- `docs/dev/QUICKSTART_X3_STREAM.md`

这些路径里的算法文件与该提交逐字节相同。本仓库另外加了 `docs/dev/todo.md`、本文件、`.gitignore`，以及 `software/trajectory/calib/README.md`（说明 `calibration_r1440.json` 是实验室给某一台 X3 标定的样例，不是厂家出厂标定，也不是众包默认标定）。

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
- `software/calibration/basalt/umi-basalt-0.1.7-fsoffset-linux-x86_64.tar.xz`：sha256 `c531b065fde3d51b4a188168eb3867cef3ec17338ae11f9397d0a5236c5aa4c2`，与 `basalt/PROVENANCE.md` 一致。

## 2026-10-15 入口

根目录只留 `README.md` 作为使用者的五步说明。`todo.md`、本文件、`QUICKSTART_X3_STREAM.md` 挪到 `docs/dev/`。实时预览程序仍不在仓库里，录制说明不再指向它。

## 2026-10-18 入口按两块功能

给人看的文档收成 `docs/标定/` 和 `docs/轨迹/`。根目录 `README.md` 先写标定、轨迹两条命令，再写同一台相机的五步。`software/trajectory/calib/` 的说明改成：这里是轨迹要读的样例标定和夹爪 CAD，不是 `umi-calib` 的产出。算法路径、`kiwi_fuse3s.py`、`KIWI_*`、`source: factory` 都没改。
