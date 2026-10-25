# 场景代码阅读指南

用户操作见 [共享场景与双手任务](../../docs/场景.md)。保留 `umi-traj` 的单段流程，场景流程由本模块单独编排。

```text
./umi-scene build scene.yaml
  cli.main() → workflow.build_scene()
    models.read_scene() / select_camera()   预检清单与实际设备测量
    engine.prepare()                       共用 X3 时间戳、IMU 和镜头视频工具
    mapping.build_maps()
      engine.build() / atlas_candidates()  纯视觉建图、逐图定位
      geometry.metric_alignment()          恢复尺度、重力并验收可观测性
      geometry.place_front()               前地图放入共享后地图坐标
    mapping.freeze_maps()                  固定地图、测量、场景变换
    storage.write_json(current.json)       全部完成后才发布版本

./umi-scene run episode.yaml
  cli.main() → workflow.run_episode()
    models.read_episode() / offsets()      显式绑定左右手和时间偏移
    workflow.load_published()              校验地图版本、资产和引擎
    engine.prepare() / localize()          只查询已有地图，不重新建图
    Observations.transformed()             转成固定场景系的视觉位姿
    fusion.fuse()                          每只手的 IMU 图，共享固定场景位姿约束
    results.write_arm() / write_bimanual()  同坐标 TUM、共同时间上的双手表
    storage.publish_link(results)          发布当前成功结果
```

## 模块边界

| 文件 | 负责什么 |
|---|---|
| `cli.py` | 参数与命令分发，创建人可读清单 |
| `models.py` | 场景、任务、设备资产、参数与只读校验 |
| `workflow.py` | 执行顺序、版本复用、失败收尾与发布 |
| `engine.py` | X3 准备、ORB 建图/定位与地图导出边界 |
| `mapping.py` | 地图选择、米制对齐、场景放置和资产冻结 |
| `geometry.py` | 显式方向的 SE(3)/尺度运算、惯性对齐、放置优化 |
| `fusion.py` | 固定场景系中的单手 GTSAM 联合优化 |
| `synchronization.py` | 音频相关性、音视频延迟换算和时间证据 |
| `results.py` | 轨迹、帧约束、双手对齐表、报告 |
| `storage.py` | 原子文件、进程锁和版本指针 |
| `rear_calibration.py` | 复用现有 A/B/D 和独立 C 验收，生成后镜头测量 |

场景数学与融合参考 KIWI 公开参考实现，具体来源、修改和许可证见 [NOTICE.md](NOTICE.md)。本项目运行时不依赖本机的另一个 `KIWI_software` 仓库；所需数值代码已适配在本模块内。

## 坐标和时间

- `camera_to_imu()` 返回 Basalt `T_imu_cam`；融合输入外参取其逆 `T_cam_imu`。
- 地图放置保存尺度 `s` 和刚体 `T_scene_metric_map`：`p_scene = R (s p_map) + t`。相机–IMU 的物理杆臂不乘视觉地图尺度。
- `Observations.times` 在本机 IMU 时钟：`t_video + cam_time_offset_ns / 1e9`。读取 EuRoC 时移除容器工具人为加的 1 秒。
- 输出时间为 `t_episode = t_video + offset_s`。跨相机同步只改变输出时间，不混入本机惯性积分。
- 融合使用场景中的绝对视觉位姿约束，不加入每段起点置零或地图重放置变量。没有可靠地图锚定时直接失败。
- ORB 的建图与定位只使用图像；原始 IMU 由对齐和融合阶段读取。

## 验证

`tests/test_scene.py` 包含尺度、重力退化、物理杆臂、固定坐标融合、三地图双手、非零同步偏移、多任务复用、资产变化和失败发布测试。数值融合测试使用真实 GTSAM；流程中的 ORB 使用可控替代引擎，真实录像验证另行记录。
